import os
import numpy as np
import scipy.io
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import matplotlib.pyplot as plt

# =====================================
# Configuration / Hyperparameters
# =====================================

BLOCK_SIZE = 128       # Size of sub-block (e.g., 128x128)
STRIDE = 128           # Stride to move the sliding window (could be < BLOCK_SIZE for overlap)
LATENT_DIM = 32        # Latent dimensionality of the VAE
EPOCHS = 50            # Number of training epochs
BATCH_SIZE = 16        # Reduce if memory issues persist
VAL_SPLIT = 0.2
DATA_DIRECTORY = 'matrices_sample'
MAX_MATRICES = None    # Set to an integer to limit the number of .mtx files processed
MODEL_PATH = 'saved_vae_model'  # Directory to save/load the model

# =====================================
# 0. Handle GPU Configuration
# =====================================

# Suppress TensorFlow GPU warnings if GPUs are not available
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    print("No GPU detected. Running on CPU.")
else:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Detected {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(e)

# =====================================
# 1. Data Generator for Efficient Loading
# =====================================

def extract_subblocks(dense_matrix, block_size=128, stride=128):
    """
    Given a dense matrix of shape (rows, cols),
    extract 2D sub-blocks of shape (block_size, block_size).
    """
    subblocks = []
    rows, cols = dense_matrix.shape

    # Slide over matrix with the given stride
    for r in range(0, rows - block_size + 1, stride):
        for c in range(0, cols - block_size + 1, stride):
            block = dense_matrix[r:r+block_size, c:c+block_size]
            subblocks.append(block)
    
    return subblocks

def data_generator(directory, block_size=128, stride=128, batch_size=16, max_matrices=None):
    """
    Generator that yields batches of sub-blocks from .mtx files.
    Ensures all batches have the specified `batch_size`.
    """
    mtx_files = [f for f in os.listdir(directory) if f.endswith('.mtx')]
    if max_matrices is not None:
        mtx_files = mtx_files[:max_matrices]
    
    all_blocks = []

    for filename in mtx_files:
        path = os.path.join(directory, filename)
        print(f"Processing {filename}")
        try:
            # Load sparse matrix from .mtx
            matrix = scipy.io.mmread(path).tocsr()
            dense_mat = matrix.toarray()

            # Binarize: 1 if nonzero, 0 otherwise
            binarized = (dense_mat != 0).astype(np.float32)

            # Extract sub-blocks
            blocks = extract_subblocks(binarized, 
                                       block_size=block_size, 
                                       stride=stride)
            all_blocks.extend(blocks)

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    # Shuffle the blocks
    np.random.shuffle(all_blocks)

    # Yield batches
    while True:
        for i in range(0, len(all_blocks), batch_size):
            batch = all_blocks[i:i+batch_size]
            if len(batch) == batch_size:
                batch = np.array(batch)[..., np.newaxis]
                yield batch, batch  # Inputs and targets are the same
            else:
                # Ignore smaller batches at the end
                continue


# =====================================
# 2. Build the Variational Autoencoder
# =====================================

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], LATENT_DIM))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def build_encoder(input_shape):
    """
    Build a CNN encoder for sub-blocks of shape `input_shape`.
    """
    encoder_inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(encoder_inputs)
    x = layers.MaxPooling2D(2, padding='same')(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    z_mean = layers.Dense(LATENT_DIM, name='z_mean')(x)
    z_log_var = layers.Dense(LATENT_DIM, name='z_log_var')(x)
    z = layers.Lambda(sampling, output_shape=(LATENT_DIM,), name='z')([z_mean, z_log_var])

    encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
    return encoder

def build_decoder(output_shape):
    """
    Build a CNN decoder that reconstructs sub-blocks back to `output_shape`.
    Example: (128, 128, 1).
    """
    latent_inputs = layers.Input(shape=(LATENT_DIM,))
    reduced_h = output_shape[0] // 4
    reduced_w = output_shape[1] // 4
    x = layers.Dense(32 * reduced_h * reduced_w, activation='relu')(latent_inputs)
    x = layers.Reshape((reduced_h, reduced_w, 32))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)
    decoder_outputs = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)

    decoder = models.Model(latent_inputs, decoder_outputs, name='decoder')
    return decoder

# Build encoder/decoder
input_shape = (BLOCK_SIZE, BLOCK_SIZE, 1)
encoder = build_encoder(input_shape)
decoder = build_decoder(input_shape)

# Define the VAE Model
class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed

vae = VAE(encoder, decoder)

# Define VAE Loss
def vae_loss(inputs, outputs):
    reconstruction_loss = tf.keras.losses.binary_crossentropy(
        K.flatten(inputs), K.flatten(outputs)
    )
    reconstruction_loss *= (BLOCK_SIZE * BLOCK_SIZE)
    z_mean, z_log_var, _ = encoder(inputs)
    kl_loss = -0.5 * K.sum(
        1 + z_log_var - K.square(z_mean) - K.exp(z_log_var),
        axis=-1
    )
    return K.mean(reconstruction_loss + kl_loss)

vae.compile(optimizer='adam', loss=vae_loss)

# =====================================
# 3. Check for Existing Model
# =====================================

if os.path.exists(MODEL_PATH):
    print(f"Model found at '{MODEL_PATH}'.")
    use_existing = input("Do you want to use the existing model? (yes/no): ").strip().lower()
    if use_existing == 'yes':
        print("Loading existing model...")
        vae = tf.keras.models.load_model(MODEL_PATH, custom_objects={'vae_loss': vae_loss})
    else:
        print("Proceeding to train a new model...")
else:
    print("No existing model found. Proceeding to train a new model...")

# =====================================
# 4. Train the Model and Save
# =====================================

if not os.path.exists(MODEL_PATH) or use_existing == 'no':
    # Create the data generator
    gen = data_generator(
        directory=DATA_DIRECTORY,
        block_size=BLOCK_SIZE,
        stride=STRIDE,
        batch_size=BATCH_SIZE,
        max_matrices=MAX_MATRICES
    )

    steps_per_epoch = 1000  # Adjust based on your dataset size
    print("Starting training...")
    history = vae.fit(
        tf.data.Dataset.from_generator(
            lambda: gen,
            output_types=(tf.float32, tf.float32),
            output_shapes=(
                (BATCH_SIZE, BLOCK_SIZE, BLOCK_SIZE, 1),
                (BATCH_SIZE, BLOCK_SIZE, BLOCK_SIZE, 1),
            )
        ).prefetch(tf.data.AUTOTUNE),
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
    )

    # Save the trained model
    print(f"Saving model to '{MODEL_PATH}'...")
    vae.save(MODEL_PATH)


# =====================================
# 5. Generate New Blocks / Large Matrix
# =====================================

def generate_new_blocks(num_samples):
    """
    Generate new sub-blocks by sampling from the latent space.
    """
    z_samples = np.random.normal(size=(num_samples, LATENT_DIM))
    generated_blocks = decoder.predict(z_samples)
    return generated_blocks

def stitch_blocks_to_large_matrix(blocks, num_rows, num_cols, block_size=128):
    """
    Stitch a list (or array) of block_size×block_size blocks into a 
    large matrix of shape (num_rows*block_size, num_cols*block_size).
    Blocks are assumed to be in row-major order:
      blocks[0]   blocks[1]   ... blocks[num_cols-1]
      blocks[num_cols] ...           ...
    """
    large_matrix = np.zeros((num_rows * block_size, num_cols * block_size))
    idx = 0
    for r in range(num_rows):
        for c in range(num_cols):
            if idx >= len(blocks):
                break
            block = blocks[idx].squeeze()  # shape: (block_size, block_size)
            row_start = r * block_size
            col_start = c * block_size
            large_matrix[row_start:row_start+block_size, 
                         col_start:col_start+block_size] = block
            idx += 1
    return large_matrix

# Example: Generate 9 new blocks (3×3) and stitch into a 384×384 matrix
num_blocks = 9
generated_blocks = generate_new_blocks(num_blocks)
stitched_matrix = stitch_blocks_to_large_matrix(
    generated_blocks, 
    num_rows=3, 
    num_cols=3, 
    block_size=BLOCK_SIZE
)

# Visualize the stitched matrix
plt.figure(figsize=(6,6))
plt.imshow(stitched_matrix, cmap='gray')
plt.title('Stitched Large Matrix (3x3 blocks)')
plt.axis('off')
plt.show()

# =====================================
# 6. Evaluate Sparsity
# =====================================

def calculate_sparsity(matrices):
    """
    Calculate the average sparsity of a batch of sub-blocks or 
    a single large matrix.
    """
    if len(matrices.shape) == 2:
        # Single large 2D matrix (H x W)
        total_elements = matrices.size
        non_zero_elements = np.count_nonzero(matrices)
        sparsity = 1 - (non_zero_elements / total_elements)
        return sparsity
    elif len(matrices.shape) == 3:
        # Single sub-block (H x W x 1)
        total_elements = matrices.size
        non_zero_elements = np.count_nonzero(matrices)
        sparsity = 1 - (non_zero_elements / total_elements)
        return sparsity
    elif len(matrices.shape) == 4:
        # Batch of blocks: (N, H, W, 1)
        total_elements = np.prod(matrices.shape[1:])
        non_zero_elements = np.count_nonzero(matrices)
        return 1 - (non_zero_elements / total_elements)
    else:
        raise ValueError("Matrices must be 2D, 3D, or 4D with channel dimension.")

# Calculate sparsity for generated blocks
generated_sparsity = calculate_sparsity(generated_blocks)
print(f"Generated Block Sparsity: {generated_sparsity:.4f}")

# Optionally, calculate sparsity for a sample from the dataset
# Note: Implementing this efficiently with the generator requires fetching a batch
# Here's a simple way to fetch a single batch for analysis

try:
    sample_batch, _ = next(iter(gen))
    original_sparsity = calculate_sparsity(sample_batch)
    print(f"Original Sub-Block Sparsity: {original_sparsity:.4f}")
except StopIteration:
    print("No data available to calculate original sparsity.")






# --- After Stitching Generated Blocks ---

# Print the generated matrix as a grid of 1s and 0s
def print_matrix(matrix):
    """
    Print the matrix with `1` representing non-empty and `0` representing empty entries.
    """
    binary_matrix = (matrix != 0).astype(int)
    for row in binary_matrix:
        print(" ".join(map(str, row)))

# Visualize the matrix with empty (white) and non-empty (black) regions
def visualize_matrix(matrix, title="Generated Matrix"):
    """
    Visualize the binary representation of the matrix.
    """
    binary_matrix = (matrix != 0).astype(int)
    plt.figure(figsize=(6, 6))
    plt.imshow(binary_matrix, cmap='gray', interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    plt.show()

# --- Example: Generate, Print, and Visualize ---

# Example: Generate a 384×384 stitched matrix
num_blocks = 9  # 3x3 grid
generated_blocks = generate_new_blocks(num_blocks)
stitched_matrix = stitch_blocks_to_large_matrix(
    generated_blocks, 
    num_rows=3, 
    num_cols=3, 
    block_size=BLOCK_SIZE
)

# Print the binary matrix to the console
print("Generated Matrix:")
print_matrix(stitched_matrix)

# Visualize the binary matrix
visualize_matrix(stitched_matrix, title="Generated Matrix Visualization")

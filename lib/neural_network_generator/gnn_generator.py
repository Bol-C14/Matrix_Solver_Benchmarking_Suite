import os
import numpy as np
import scipy.io
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import matplotlib.pyplot as plt

# =====================================
# Configuration / Hyperparameters
# =====================================
BLOCK_SIZE = 128
STRIDE = 128
LATENT_DIM = 32
EPOCHS = 50
BATCH_SIZE = 16
VAL_SPLIT = 0.2
DATA_DIRECTORY = 'matrices_sample'
MAX_MATRICES = 5
MODEL_PATH = 'saved_vae_model'
MODEL_PATH_KERAS = 'saved_vae_model.keras'
MODEL_PATH_SAVEDMODEL = 'saved_vae_model_dir'

# =====================================
# 0. Handle GPU Configuration
# =====================================
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    print("No GPU detected. Running on CPU.")
else:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Detected {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(e)

# =====================================
# 1. Data Generator for Efficient Loading
# =====================================
def extract_subblocks(dense_matrix, block_size=128, stride=128):
    subblocks = []
    rows, cols = dense_matrix.shape
    for r in range(0, rows - block_size + 1, stride):
        for c in range(0, cols - block_size + 1, stride):
            block = dense_matrix[r:r+block_size, c:c+block_size]
            subblocks.append(block)
    return subblocks

def data_generator(directory, block_size=128, stride=128, batch_size=16, max_matrices=None):
    mtx_files = [f for f in os.listdir(directory) if f.endswith('.mtx')]
    if max_matrices is not None:
        mtx_files = mtx_files[:max_matrices]
    
    all_blocks = []
    for filename in mtx_files:
        path = os.path.join(directory, filename)
        print(f"Processing {filename}")
        try:
            matrix = scipy.io.mmread(path).tocsr()
            dense_mat = matrix.toarray()
            binarized = (dense_mat != 0).astype(np.float32)
            blocks = extract_subblocks(binarized, block_size=block_size, stride=stride)
            all_blocks.extend(blocks)
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    np.random.shuffle(all_blocks)
    while True:
        for i in range(0, len(all_blocks), batch_size):
            batch = all_blocks[i:i+batch_size]
            if len(batch) == batch_size:
                batch = np.array(batch)[..., np.newaxis]
                yield batch, batch
            else:
                continue

# =====================================
# 2. Build the Variational Autoencoder
# =====================================
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], LATENT_DIM))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def build_encoder(input_shape):
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

input_shape = (BLOCK_SIZE, BLOCK_SIZE, 1)
encoder = build_encoder(input_shape)
decoder = build_decoder(input_shape)

@tf.keras.utils.register_keras_serializable()
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
use_existing = 'no'
if os.path.exists(MODEL_PATH_KERAS):
    print(f"Model found at '{MODEL_PATH_KERAS}'.")
    use_existing = input("Do you want to use the existing model? (yes/no): ").strip().lower()
    if use_existing == 'yes':
        print("Loading existing model in Keras format...")
        vae = tf.keras.models.load_model(MODEL_PATH_KERAS, custom_objects={'vae_loss': vae_loss})
    else:
        print("Proceeding to train a new model...")
else:
    print("No existing model found. Proceeding to train a new model...")

# =====================================
# 4. Train the Model and Save
# =====================================
if not os.path.exists(MODEL_PATH_KERAS) or use_existing == 'no':
    print("Starting training...")
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(
            directory=DATA_DIRECTORY,
            block_size=BLOCK_SIZE,
            stride=STRIDE,
            batch_size=BATCH_SIZE,
            max_matrices=MAX_MATRICES,
        ),
        output_signature=(
            tf.TensorSpec(shape=(BATCH_SIZE, BLOCK_SIZE, BLOCK_SIZE, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(BATCH_SIZE, BLOCK_SIZE, BLOCK_SIZE, 1), dtype=tf.float32),
        )
    ).prefetch(tf.data.AUTOTUNE)
    history = vae.fit(dataset, epochs=EPOCHS, steps_per_epoch=1000)

    print(f"Saving model to '{MODEL_PATH_KERAS}' in Keras format...")
    vae.save(MODEL_PATH_KERAS)
    print(f"Saving model to '{MODEL_PATH_SAVEDMODEL}' in SavedModel format...")
    # Use the low-level SavedModel API for directory-based saving in TF
    tf.saved_model.save(vae, MODEL_PATH_SAVEDMODEL)

# =====================================
# 5. Generate New Blocks / Large Matrix
# =====================================
def generate_new_blocks(num_samples):
    z_samples = np.random.normal(size=(num_samples, LATENT_DIM))
    generated_blocks = decoder.predict(z_samples)
    return generated_blocks

def stitch_blocks_to_large_matrix(blocks, num_rows, num_cols, block_size=128):
    large_matrix = np.zeros((num_rows * block_size, num_cols * block_size))
    idx = 0
    for r in range(num_rows):
        for c in range(num_cols):
            if idx >= len(blocks):
                break
            block = blocks[idx].squeeze()
            row_start = r * block_size
            col_start = c * block_size
            large_matrix[row_start:row_start+block_size, col_start:col_start+block_size] = block
            idx += 1
    return large_matrix

num_blocks = 9
generated_blocks = generate_new_blocks(num_blocks)
stitched_matrix = stitch_blocks_to_large_matrix(generated_blocks, 3, 3, BLOCK_SIZE)

plt.figure(figsize=(6,6))
plt.imshow(stitched_matrix, cmap='gray')
plt.title('Stitched Large Matrix (3x3 blocks)')
plt.axis('off')
plt.show()

# =====================================
# 6. Evaluate Sparsity
# =====================================
def calculate_sparsity(matrices):
    if len(matrices.shape) == 2:
        total_elements = matrices.size
        non_zero_elements = np.count_nonzero(matrices)
        sparsity = 1 - (non_zero_elements / total_elements)
        return sparsity
    elif len(matrices.shape) == 3:
        total_elements = matrices.size
        non_zero_elements = np.count_nonzero(matrices)
        return 1 - (non_zero_elements / total_elements)
    elif len(matrices.shape) == 4:
        total_elements = np.prod(matrices.shape[1:])
        non_zero_elements = np.count_nonzero(matrices)
        return 1 - (non_zero_elements / total_elements)
    else:
        raise ValueError("Matrices must be 2D, 3D, or 4D with channel dimension.")

generated_sparsity = calculate_sparsity(generated_blocks)
print(f"Generated Block Sparsity: {generated_sparsity:.4f}")

# Example: fetch a single batch for analysis
try:
    gen = data_generator(DATA_DIRECTORY, block_size=BLOCK_SIZE, stride=STRIDE, batch_size=BATCH_SIZE, max_matrices=MAX_MATRICES)
    sample_batch, _ = next(gen)
    original_sparsity = calculate_sparsity(sample_batch)
    print(f"Original Sub-Block Sparsity: {original_sparsity:.4f}")
except StopIteration:
    print("No data available to calculate original sparsity.")

def print_matrix(matrix):
    binary_matrix = (matrix != 0).astype(int)
    for row in binary_matrix:
        print(" ".join(map(str, row)))

def visualize_matrix(matrix, title="Generated Matrix"):
    binary_matrix = (matrix != 0).astype(int)
    plt.figure(figsize=(6, 6))
    plt.imshow(binary_matrix, cmap='gray', interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    plt.show()

num_blocks = 9
generated_blocks = generate_new_blocks(num_blocks)
stitched_matrix = stitch_blocks_to_large_matrix(generated_blocks, 3, 3, BLOCK_SIZE)

print("Generated Matrix:")
print_matrix(stitched_matrix)
print(stitched_matrix.shape)
visualize_matrix(stitched_matrix, title="Generated Matrix Visualization")

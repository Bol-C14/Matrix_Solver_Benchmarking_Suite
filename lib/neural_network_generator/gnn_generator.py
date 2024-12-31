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
EPOCHS = 10
BATCH_SIZE = 16
VAL_SPLIT = 0.2
DATA_DIRECTORY = 'matrices_sample'
MAX_MATRICES = 5
MODEL_PATH_KERAS = 'saved_vae_model.keras'
BETA = 1.0  # Weight for KL divergence

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
                batch = np.array(batch)
                if batch.ndim == 3:
                    batch = np.expand_dims(batch, axis=-1)
                yield batch, batch
            else:
                continue  # Discard incomplete batches

# =====================================
# 2. Build the Variational Autoencoder
# =====================================

@tf.keras.utils.register_keras_serializable()
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], LATENT_DIM))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

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
    def __init__(self, encoder, decoder, beta=1.0, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed

    def train_step(self, data):
        inputs, _ = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(inputs)
            reconstructed = self.decoder(z)
            # Reconstruction loss
            reconstruction_loss = tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(inputs, reconstructed), axis=[1, 2]
            )
            # KL Divergence
            kl_loss = -0.5 * tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
            )
            total_loss = tf.reduce_mean(reconstruction_loss + self.beta * kl_loss)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": total_loss}

    def get_config(self):
        config = super(VAE, self).get_config()
        config.update({
            "encoder": tf.keras.utils.serialize_keras_object(self.encoder),
            "decoder": tf.keras.utils.serialize_keras_object(self.decoder),
            "beta": self.beta
        })
        return config

    @classmethod
    def from_config(cls, config):
        encoder = tf.keras.utils.deserialize_keras_object(config.pop("encoder"))
        decoder = tf.keras.utils.deserialize_keras_object(config.pop("decoder"))
        beta = config.pop("beta", 1.0)
        return cls(encoder=encoder, decoder=decoder, beta=beta, **config)

vae = VAE(encoder, decoder, beta=BETA)

# No external loss function needed since it's handled in train_step
vae.compile(optimizer='adam')

# =====================================
# 3. Check for Existing Model
# =====================================
use_existing = 'no'
if os.path.exists(MODEL_PATH_KERAS):
    print(f"Model found at '{MODEL_PATH_KERAS}'.")
    use_existing = input("Do you want to use the existing model? (yes/no): ").strip().lower()
    if use_existing == 'yes':
        print("Loading existing model in Keras format...")
        vae = tf.keras.models.load_model(
            MODEL_PATH_KERAS, 
            custom_objects={
                "VAE": VAE, 
                "sampling": sampling  # Register the 'sampling' function
            }
        )
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

    # Estimate steps per epoch
    mtx_files = [f for f in os.listdir(DATA_DIRECTORY) if f.endswith('.mtx')]
    if MAX_MATRICES is not None:
        mtx_files = mtx_files[:MAX_MATRICES]
    total_blocks = 0
    for filename in mtx_files:
        path = os.path.join(DATA_DIRECTORY, filename)
        try:
            matrix = scipy.io.mmread(path).tocsr()
            dense_mat = matrix.toarray()
            binarized = (dense_mat != 0).astype(np.float32)
            blocks = extract_subblocks(binarized, block_size=BLOCK_SIZE, stride=STRIDE)
            total_blocks += len(blocks)
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    steps_per_epoch = total_blocks // BATCH_SIZE
    print(f"Total blocks: {total_blocks}, Steps per epoch: {steps_per_epoch}")

    history = vae.fit(
        dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
    )

    print(f"Saving model to '{MODEL_PATH_KERAS}' in Keras format...")
    vae.save(MODEL_PATH_KERAS)

# =====================================
# Verify Model Outputs
# =====================================
gen = data_generator(DATA_DIRECTORY, block_size=BLOCK_SIZE, stride=STRIDE, batch_size=BATCH_SIZE, max_matrices=MAX_MATRICES)
sample_batch, _ = next(gen)
print(f"Sample Batch Shape: {sample_batch.shape}")
sample_output = vae(sample_batch)
print(f"Model Output Shape: {sample_output.shape}")

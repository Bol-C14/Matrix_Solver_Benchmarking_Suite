import os
import gzip
import numpy as np
import scipy.io
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import matplotlib.pyplot as plt


# --- Step 1: Load and Preprocess Matrices ---
def load_and_preprocess_matrices(directory, max_size=128):
    """
    Load sparse circuit matrices, convert to dense, and preprocess for model training.
    """
    matrices = []
    for filename in os.listdir(directory):
        if filename.endswith('.mtx.gz'):
            print(f"Processing {filename}")
            with gzip.open(os.path.join(directory, filename), 'rb') as f:
                try:
                    matrix = scipy.io.mmread(f).tocsr()
                    rows, cols = matrix.shape
                    if rows > max_size or cols > max_size:
                        continue
                    dense_matrix = matrix.toarray()
                    padded_matrix = np.zeros((max_size, max_size))
                    padded_matrix[:rows, :cols] = dense_matrix
                    binarized_matrix = (padded_matrix != 0).astype(np.float32)
                    matrices.append(binarized_matrix)
                except Exception as e:
                    print(f"Failed to process {filename}: {e}")
    return np.array(matrices)


# Directory containing downloaded .mtx.gz matrices
data_directory = 'matrices_sample'
os.makedirs(data_directory, exist_ok=True)
processed_matrices = load_and_preprocess_matrices(data_directory)

# Add channel dimension for CNN
processed_matrices = processed_matrices[..., np.newaxis]
print(f"Data shape: {processed_matrices.shape}")


# --- Step 2: Build the Variational Autoencoder ---
latent_dim = 32  # Latent space dimensionality


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def build_encoder(input_shape):
    encoder_inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(encoder_inputs)
    x = layers.MaxPooling2D(2, padding='same')(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)

    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
    return encoder


def build_decoder(output_shape):
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(32 * (output_shape[0] // 4) * (output_shape[1] // 4), activation='relu')(latent_inputs)
    x = layers.Reshape((output_shape[0] // 4, output_shape[1] // 4, 32))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)
    decoder_outputs = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)

    decoder = models.Model(latent_inputs, decoder_outputs, name='decoder')
    return decoder


input_shape = (128, 128, 1)
encoder = build_encoder(input_shape)
decoder = build_decoder(input_shape)


# Define VAE Model
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

# Define Loss Function
def vae_loss(inputs, outputs):
    reconstruction_loss = tf.keras.losses.binary_crossentropy(K.flatten(inputs), K.flatten(outputs))
    reconstruction_loss *= input_shape[0] * input_shape[1]
    z_mean, z_log_var, _ = encoder(inputs)
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(reconstruction_loss + kl_loss)


vae.compile(optimizer='adam', loss=vae_loss)


# --- Step 3: Train the Model ---
vae.fit(processed_matrices, processed_matrices, epochs=50, batch_size=16, validation_split=0.2)


# --- Step 4: Generate New Matrices ---
def generate_new_matrices(num_samples):
    z_samples = np.random.normal(size=(num_samples, latent_dim))
    generated_matrices = decoder.predict(z_samples)
    return generated_matrices


# Generate and Visualize Matrices
new_matrices = generate_new_matrices(5)

for i, matrix in enumerate(new_matrices):
    plt.imshow(matrix.squeeze(), cmap='gray')
    plt.title(f'Generated Matrix {i+1}')
    plt.axis('off')
    plt.show()


# --- Step 5: Evaluate Sparsity ---
def calculate_sparsity(matrices):
    total_elements = np.prod(matrices.shape[1:])
    non_zero_elements = np.count_nonzero(matrices)
    sparsity = 1 - (non_zero_elements / (matrices.shape[0] * total_elements))
    return sparsity


original_sparsity = calculate_sparsity(processed_matrices)
generated_sparsity = calculate_sparsity(new_matrices)

print(f"Original Matrices Sparsity: {original_sparsity:.4f}")
print(f"Generated Matrices Sparsity: {generated_sparsity:.4f}")

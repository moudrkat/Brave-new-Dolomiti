import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from src.utils import save_generated_images, show_images_in_streamlit
from src.data_preprocessing import denormalize_images
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def encoder(latent_dim=64):
    inputs = layers.Input(shape=(256, 256, 3))
    
    # Add more Conv2D layers to make the model deeper, with Batch Normalization
    x = layers.Conv2D(32, 3, activation=None, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)  # Apply ReLU after BatchNormalization
    
    x = layers.Conv2D(64, 3, activation=None, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(128, 3, activation=None, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(256, 3, activation=None, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(512, 3, activation=None, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)  # Reduced dense layer size
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    
    return models.Model(inputs, [z_mean, z_log_var])

def decoder(latent_dim=64):
    latent_inputs = layers.Input(shape=(latent_dim,))
    
    # Initial dense layer to reshape the latent vector
    x = layers.Dense(16 * 16 * 512, activation='relu')(latent_inputs)  # Increased size
    x = layers.Reshape((16, 16, 512))(x)  # Reshape to 16x16x512
    
    # Add more Conv2DTranspose layers to make the decoder deeper
    x = layers.Conv2DTranspose(512, 3, activation='relu', strides=2, padding='same')(x)  # 32x32
    x = layers.Conv2DTranspose(256, 3, activation='relu', strides=2, padding='same')(x)  # 64x64
    x = layers.Conv2DTranspose(128, 3, activation='relu', strides=2, padding='same')(x)  # 128x128
    x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)   # 256x256

    decoded = layers.Conv2D(3, 3, activation='tanh', padding='same')(x)  # Output shape (256, 256, 3)
    
    return models.Model(latent_inputs, decoded)

# Load and preprocess image function
def load_and_preprocess_image(image_path, target_size=(256, 256)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    img_array = 2 * img_array - 1  # Scale to [-1, 1]
    return img_array

def load_dataset(dataset_dir, target_size=(256, 256)):
    image_paths = [os.path.join(dataset_dir, fname) for fname in os.listdir(dataset_dir) if fname.endswith('.jpg') or fname.endswith('.png')]
    images = [load_and_preprocess_image(img_path, target_size) for img_path in image_paths]
    dataset = tf.data.Dataset.from_tensor_slices(np.array(images))
    return dataset

@tf.function
def train_step(encoder, decoder, images, optimizer):
    with tf.GradientTape() as tape:
        # Forward pass through the encoder
        z_mean, z_log_var = encoder(images)
        
        # Sampling from the latent space
        z = sampling([z_mean, z_log_var])
        
        # Forward pass through the decoder
        reconstructed = decoder(z)
        
        # Calculate VAE loss (reconstruction + KL divergence)
        reconstruction_loss = tf.reduce_mean(tf.square(images - reconstructed))

        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )

        vae_loss = reconstruction_loss + kl_loss

    # Compute gradients and update the model
    grads = tape.gradient(vae_loss, encoder.trainable_variables + decoder.trainable_variables)
    optimizer.apply_gradients(zip(grads, encoder.trainable_variables + decoder.trainable_variables))

    return vae_loss
# , reconstruction_loss, kl_loss

# Training loop
def train_vae( strategy, sketch_type,optimizer, dataset, encoder,decoder, image_placeholder, freq_show=10, freq_save=100, epochs=100,latent_dim=100):
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        for batch_images in dataset:
            batch_loss = train_step(encoder,decoder, batch_images, optimizer)
            epoch_loss += batch_loss
            num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss.numpy()}')

        
        if epoch % freq_show == 0:
            # print(f"Showing generated images at epoch {epoch + 1}...")
            random_latent_vectors = tf.random.normal(shape=(64, latent_dim))  # Generate 20 random latent vectors
            generated_images = decoder(random_latent_vectors)  # Decode them to generate images
            # print("going to denormalize method")
            denormalized_generated_images = denormalize_images(generated_images) 
            # print("going to ahow method")
            show_images_in_streamlit(strategy, batch_images, denormalized_generated_images, epoch, image_placeholder)

        # Show generated images every 'display_interval' epochs
        if epoch % freq_save == 0:
            # print(f"Saving generated images at epoch {epoch + 1}...")
            # Generate random latent vectors
            random_latent_vectors = tf.random.normal(shape=(64, latent_dim))  # Generate 20 random latent vectors
            # print("saving before")
            generated_images = decoder(random_latent_vectors)  # Decode them to generate images
            denormalized_generated_images = denormalize_images(generated_images) 
            save_generated_images(strategy, sketch_type, denormalized_generated_images, epoch, path=f"./generated_images_{strategy}_{sketch_type}")
            # print("saving done")



def generate_image(decoder_model, latent_dim):
    random_latent_vectors = np.random.normal(size=(1, latent_dim))
    generated_image = decoder_model.predict(random_latent_vectors)
    
    return generated_image[0]
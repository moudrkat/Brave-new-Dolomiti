import tensorflow as tf
import numpy as np
from src.utils import save_generated_images, show_images_in_streamlit, show_loss_acc_in_streamlit
from src.data_preprocessing import denormalize_images
import streamlit as st
from keras import backend as K

# Define Wasserstein loss
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

# Building the generator
def build_generator(latent_dim=100, drop=0.4):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Reshape(target_shape=[1, 1, 4096], input_shape=[4096]))
    assert model.output_shape == (None, 1, 1, 4096)

    model.add(tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=4))
    model.add(tf.keras.layers.Activation('relu'))
    assert model.output_shape == (None, 4, 4, 256)

    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=4, padding='same'))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.7))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.UpSampling2D())
    assert model.output_shape == (None, 8, 8, 256)

    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=4, padding='same'))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.7))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.UpSampling2D())
    assert model.output_shape == (None, 16, 16, 128)

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same'))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.7))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.UpSampling2D())
    assert model.output_shape == (None, 32, 32, 64)

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same'))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.7))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.UpSampling2D())
    assert model.output_shape == (None, 64, 64, 32)

    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same'))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.7))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.UpSampling2D())
    assert model.output_shape == (None, 128, 128, 16)

    model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.UpSampling2D())
    assert model.output_shape == (None, 256, 256, 8)

    model.add(tf.keras.layers.Conv2D(filters=3, kernel_size=3, padding='same'))
    model.add(tf.keras.layers.Activation('tanh'))
    assert model.output_shape == (None, 256, 256, 3)

    return model

# Building the critic (formerly discriminator)
def build_critic(img_width=256, img_height=256, p=0.4):
    model = tf.keras.Sequential()

    # Add Gaussian noise to prevent overfitting
    model.add(tf.keras.layers.GaussianNoise(0.2, input_shape=[img_width, img_height, 3]))

    # First convolutional layer
    model.add(tf.keras.layers.Conv2D(64, kernel_size=5, strides=2, padding='same', input_shape=(img_width, img_height, 3)))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(p))

    # Second convolutional layer
    model.add(tf.keras.layers.Conv2D(128, kernel_size=5, strides=2, padding='same'))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(p))

    # Third convolutional layer
    model.add(tf.keras.layers.Conv2D(256, kernel_size=5, strides=2, padding='same'))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(p))

    # Fourth convolutional layer
    model.add(tf.keras.layers.Conv2D(512, kernel_size=5, strides=2, padding='same'))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(p))

    # Flatten layer to feed into a dense output layer
    model.add(tf.keras.layers.Flatten())

    # Output layer: No sigmoid for WGAN, raw output
    model.add(tf.keras.layers.Dense(1))  # No sigmoid activation

    # Compile model
    opt = tf.keras.optimizers.RMSprop(lr=0.00005)
    model.compile(loss=wasserstein_loss, optimizer=opt)

    return model

# Compiling the WGAN with Wasserstein loss and RMSprop optimizer
def compile_gan(generator, critic):
    # Freeze the critic's weights during the GAN training
    critic.trainable = False

    # Use RMSprop optimizer for stability
    lr_gen = 0.00005
    optimizer_gen = tf.keras.optimizers.RMSprop(lr=lr_gen, decay=3e-8, clipvalue=1.0)

    # GAN is a combined model of generator and critic
    gan = tf.keras.Sequential([generator, critic])

    # Compile the GAN model with the optimizer for the generator
    gan.compile(loss=wasserstein_loss, optimizer=optimizer_gen)  

    return gan

# WGAN Weight Clipping
def clip_weights(model, clip_value=0.01):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            weights = layer.get_weights()
            weights = [np.clip(w, -clip_value, clip_value) for w in weights]
            layer.set_weights(weights)

# Training the WGAN
def train_gan(strategy, sketch_type, generator, critic, gan, images, image_placeholder, image_placeholder_loss, epochs=100, batch_size=64, latent_dim=100,n_critic =5):
    g_losses = []
    c_losses = []  
    c_accuracies = [] 

    half_batch = batch_size // 2
    for epoch in range(epochs):
        noise = np.random.normal(0, 1, (batch_size, latent_dim))  # Latent noise for generator
        generated_images = generator.predict(noise)  # Generate fake images

        # update the critic
        for _ in range(n_critic):
            # 1. Train the critic with real and fake images
            critic.trainable = True  # Unfreeze the critic to train it
            idx = np.random.randint(0, images.shape[0], half_batch)
            real_images = images[idx]
            real_labels = -np.ones((half_batch, 1))  # Label for real images 

            fake_images = generated_images[:half_batch]  # Use first half of generated images
            fake_labels = np.ones((half_batch, 1))  # Label for fake images 

            c_loss_real = critic.train_on_batch(real_images, real_labels)
            c_loss_fake = critic.train_on_batch(fake_images, fake_labels)
            c_loss = np.mean([c_loss_real, c_loss_fake])  # Average of both losses

            # 2. Clip the weights of the critic
            clip_weights(critic)

        # 3. Train the generator (through the GAN model)
        critic.trainable = False  # Freeze the critic (now only the generator will be trained)
        valid_labels = np.ones((batch_size, 1))  # Fake images are labeled as real for the generator
        g_loss = gan.train_on_batch(noise, valid_labels)

        # Append losses
        g_losses.append(g_loss)
        c_losses.append(c_loss)

        # Every few epochs, save the model and show images
        if epoch % 10 == 0:
            generator.save(f"./trained_generators_{strategy}_{sketch_type}/trained_generator_{strategy}_epoch_{epoch}.h5")
            save_generated_images(strategy, sketch_type, generated_images, epoch, path=f"./generated_images_{strategy}_{sketch_type}")
            denormalized_fake_images = denormalize_images(fake_images)
            denormalized_real_images = denormalize_images(real_images)
            show_images_in_streamlit(strategy, denormalized_real_images, denormalized_fake_images, epoch, image_placeholder)
            show_loss_acc_in_streamlit(strategy, g_losses, c_losses, c_accuracies, epoch, epochs, image_placeholder_loss)

    # Save the final model
    generator.save(f"trained_generator_{strategy}_{sketch_type}_final.h5")



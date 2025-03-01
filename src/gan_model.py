import tensorflow as tf
import numpy as np
from src.utils import save_generated_images, show_images_in_streamlit, show_loss_acc_in_streamlit
from src.data_preprocessing import denormalize_images
import streamlit as st

def build_generator(latent_dim=100, drop=0.4):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Reshape(target_shape = [1, 1, 4096], input_shape = [4096]))
    assert model.output_shape == (None, 1, 1, 4096)
        
    model.add(tf.keras.layers.Conv2DTranspose(filters = 256, kernel_size = 4))
    model.add(tf.keras.layers.Activation('relu'))
    assert model.output_shape == (None, 4, 4, 256)
    
    model.add(tf.keras.layers.Conv2D(filters = 256, kernel_size = 4, padding = 'same'))
    model.add(tf.keras.layers.BatchNormalization(momentum = 0.7))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.UpSampling2D())
    assert model.output_shape == (None, 8, 8, 256)
    
    model.add(tf.keras.layers.Conv2D(filters = 128, kernel_size = 4, padding = 'same'))
    model.add(tf.keras.layers.BatchNormalization(momentum = 0.7))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.UpSampling2D())
    assert model.output_shape == (None, 16, 16, 128)
    
    model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, padding = 'same'))
    model.add(tf.keras.layers.BatchNormalization(momentum = 0.7))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.UpSampling2D())
    assert model.output_shape == (None, 32, 32, 64)
    
    model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, padding = 'same'))
    model.add(tf.keras.layers.BatchNormalization(momentum = 0.7))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.UpSampling2D())
    assert model.output_shape == (None, 64, 64, 32)
    
    model.add(tf.keras.layers.Conv2D(filters = 16, kernel_size = 3, padding = 'same'))
    model.add(tf.keras.layers.BatchNormalization(momentum = 0.7))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.UpSampling2D())
    assert model.output_shape == (None, 128, 128, 16)
    
    model.add(tf.keras.layers.Conv2D(filters = 8, kernel_size = 3, padding = 'same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.UpSampling2D())
    assert model.output_shape == (None, 256, 256, 8)
    
    model.add(tf.keras.layers.Conv2D(filters = 3, kernel_size = 3, padding = 'same'))
    model.add(tf.keras.layers.Activation('tanh'))
    assert model.output_shape == (None, 256, 256, 3)
    
    return model


def build_discriminator(img_width=256, img_height=256, p=0.4):
    model = tf.keras.Sequential()

    #add Gaussian noise to prevent Discriminator overfitting
    model.add(tf.keras.layers.GaussianNoise(0.2, input_shape = [img_width, img_height, 3]))
    
    # First convolutional layer
    model.add(tf.keras.layers.Conv2D(64, kernel_size=5, strides=2, padding='same', input_shape=(img_width, img_height, 3)))  # RGB input
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(p))
    
    # Second convolutional layer
    model.add(tf.keras.layers.Conv2D(128, kernel_size=5, strides=2, padding='same'))
    #model.add(tf.keras.layers.BatchNormalization(momentum = 0.7))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(p))
    
    # Third convolutional layer
    model.add(tf.keras.layers.Conv2D(256, kernel_size=5, strides=2, padding='same'))
    #model.add(tf.keras.layers.BatchNormalization(momentum = 0.7))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(p))
    
    # Fourth convolutional layer
    model.add(tf.keras.layers.Conv2D(512, kernel_size=5, strides=2, padding='same'))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(p))

    # # Fifth convolutional layer (to further downscale)
    # model.add(tf.keras.layers.Conv2D(1024, kernel_size=4, strides=2, padding='same'))
    # model.add(tf.keras.layers.LeakyReLU(0.2))
    # model.add(tf.keras.layers.Dropout(p))

    # Flatten layer to feed into a dense output layer
    model.add(tf.keras.layers.Flatten())
    
    # Output layer: Single neuron with sigmoid activation
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Use 0 to 1 output range
    
    # Compile model
    opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)  # Use Adam optimizer
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

def compile_gan(generator, discriminator):
    
    # # Freeze the discriminator's weights during the GAN training
    discriminator.trainable = False
    
    # # Optimizer for the generator
    lr_gen = 0.0004
    optimizer_gen = tf.keras.optimizers.RMSprop(lr=lr_gen, clipvalue=1.0)
    #optimizer_gen = tf.keras.optimizers.Adam(lr=lr_gen)

    # GAN is a combined model of generator and discriminator
    gan = tf.keras.Sequential([generator, discriminator])

    # Compile the GAN model with the optimizer for the generator
    gan.compile(loss='binary_crossentropy', optimizer=optimizer_gen)

    return gan


def train_gan(strategy, sketch_type, generator, discriminator, gan, images, image_placeholder,image_placeholder_loss, epochs=100, batch_size=64, latent_dim=100):
    # Initialize lists to store losses
    g_losses = []
    d_losses = []
    d_accuracies = []

    #half_batch = batch_size // 2  # Half batch size for real/fake images
    for epoch in range(epochs):

        print(f"processing epoch {epoch}")
        
        noise = np.random.normal(0, 1, (batch_size, latent_dim))  # Latent noise for generator

        generated_images = generator.predict(noise)  # Generate fake images

        # 1. Train the discriminator with real and fake images
        discriminator.trainable = True  # Unfreeze the discriminator to train it
        idx = np.random.randint(0, images.shape[0], batch_size)
        real_images = images[idx]  # Select real images from the dataset
        real_labels = np.ones((batch_size, 1))  # Label for real images: 1
        fake_images = generated_images  
        fake_labels = np.zeros((batch_size, 1))  # Label for fake images: 0

        d_loss_real, d_acc_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake, d_acc_fake = discriminator.train_on_batch(fake_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)  # Average of both losses
        d_acc = 0.5 * np.add(d_acc_real, d_acc_fake)  # Average of both accuracies
        
        # 2. Train the generator (through the GAN model) every epoch
        discriminator.trainable = False  # Freeze the discriminator (now only the generator will be trained)
        valid_labels = np.ones((batch_size, 1))  # Fake images are labeled as real for the generator
        g_loss = gan.train_on_batch(noise, valid_labels)

        # Append losses for this epoch to the respective lists
        g_losses.append(g_loss)
        d_losses.append(d_loss)  
        d_accuracies.append(d_acc)  

        # Every few epochs, print the progress and save the model
        if epoch % 10 == 0:  # Save model and show images every 10 epochs
            generator.save(f"./trained_generators_{strategy}_{sketch_type}/trained_generator_{strategy}_epoch_{epoch}.h5")  # Save model

            save_generated_images(strategy, sketch_type,generated_images, epoch, path=f"./generated_images_{strategy}_{sketch_type}")

            denormalized_fake_images = denormalize_images(fake_images)  # Denormalize for display
            denormalized_real_images = denormalize_images(real_images)  # Denormalize for display
            show_images_in_streamlit(strategy, denormalized_real_images, denormalized_fake_images, epoch, image_placeholder)  # Show images in Streamlit
            
            show_loss_acc_in_streamlit(strategy, g_losses, d_losses, d_accuracies, epoch,epochs, image_placeholder_loss)

    # After training completes, save the final generator model
    generator.save(f"trained_generator_{strategy}_{sketch_type}_final.h5")



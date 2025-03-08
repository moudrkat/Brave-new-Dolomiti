import tensorflow as tf
import numpy as np
from src.utils import save_generated_images, show_images_in_streamlit, show_loss_acc_in_streamlit
from src.data_preprocessing import denormalize_images
import streamlit as st
from keras.initializers import RandomNormal

# Define the label flipping function
def noisy_labels(y, p_flip):
    # determine the number of labels to flip
    n_select = int(p_flip * y.shape[0])
    # choose labels to flip
    flip_ix = np.random.choice([i for i in range(y.shape[0])], size=n_select, replace=False)
    # invert the labels in place
    y[flip_ix] = 1 - y[flip_ix]
    return y


# def build_generator(noise_dim):
#     model = tf.keras.Sequential()

#     model.add(tf.keras.layers.Dense(4 * 4 * 1024, input_dim=noise_dim))
#     model.add(tf.keras.layers.BatchNormalization())
#     model.add(tf.keras.layers.ReLU())

#     # Reshape the feature vector into [batch, channels, height, width] for further processing
#     model.add(tf.keras.layers.Reshape((4, 4, 1024)))

#     # Upsample to 8x8
#     model.add(tf.keras.layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding='same'))
#     # model.add(tf.keras.layers.BatchNormalization())
#     model.add(tf.keras.layers.ReLU())

#     # Upsample to 16x16
#     model.add(tf.keras.layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same'))
#     # model.add(tf.keras.layers.BatchNormalization())
#     model.add(tf.keras.layers.ReLU())

#     # Upsample to 32x32
#     model.add(tf.keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))
#     # model.add(tf.keras.layers.BatchNormalization())
#     model.add(tf.keras.layers.ReLU())

#     # Upsample to 64x64
#     model.add(tf.keras.layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'))
#     # model.add(tf.keras.layers.BatchNormalization())
#     model.add(tf.keras.layers.ReLU())

#     # Upsample to 128x128
#     model.add(tf.keras.layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding='same'))
#     # model.add(tf.keras.layers.BatchNormalization())
#     model.add(tf.keras.layers.ReLU())

#     # Upsample to 256x256
#     model.add(tf.keras.layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same'))
#     model.add(tf.keras.layers.Activation('tanh'))  # Output image in range [-1, 1]

#     return model


# Building the generator
def build_generator(latent_dim=100, drop=0.45):

    # weight initialization
    init = RandomNormal(stddev=0.02)

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Reshape(target_shape=[1, 1, 4096], input_shape=[4096]))
    assert model.output_shape == (None, 1, 1, 4096)

    model.add(tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=4,kernel_initializer=init))
    model.add(tf.keras.layers.Activation('relu'))
    assert model.output_shape == (None, 4, 4, 256)

    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',kernel_initializer=init))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.UpSampling2D())
    assert model.output_shape == (None, 8, 8, 256)

    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same',kernel_initializer=init))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.UpSampling2D())
    assert model.output_shape == (None, 16, 16, 128)

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same',kernel_initializer=init))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.UpSampling2D())
    assert model.output_shape == (None, 32, 32, 64)

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same',kernel_initializer=init))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.UpSampling2D())
    assert model.output_shape == (None, 64, 64, 32)

    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same',kernel_initializer=init))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.UpSampling2D())
    assert model.output_shape == (None, 128, 128, 16)

    model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='same',kernel_initializer=init))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.UpSampling2D())
    assert model.output_shape == (None, 256, 256, 8)

    model.add(tf.keras.layers.Conv2D(filters=3, kernel_size=3, padding='same',kernel_initializer=init))
    model.add(tf.keras.layers.Activation('tanh'))
    assert model.output_shape == (None, 256, 256, 3)

    return model


def build_discriminator(img_width=256, img_height=256, p=0.45):
    # weight initialization
    init = RandomNormal(stddev=0.02)

    model = tf.keras.Sequential()

    #add Gaussian noise to prevent Discriminator overfitting
    # model.add(tf.keras.layers.GaussianNoise(0.2, input_shape = [img_width, img_height, 3]))

    # model.add(tf.keras.layers.Conv2D(32, kernel_size=4, strides=2, padding='same',kernel_initializer=init))
    # # model.add(tf.keras.layers.GaussianNoise(0.1))  # Add noise after Conv2D layer
    # # model.add(tf.keras.layers.BatchNormalization(momentum = 0.7))
    # model.add(tf.keras.layers.LeakyReLU(0.2))
    # model.add(tf.keras.layers.Dropout(p))

    model.add(tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, padding='same',kernel_initializer=init))
    # model.add(tf.keras.layers.GaussianNoise(0.1))  # Add noise after Conv2D layer
    # model.add(tf.keras.layers.BatchNormalization(momentum = 0.7))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(p))
    
    model.add(tf.keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same',kernel_initializer=init))
    # model.add(tf.keras.layers.GaussianNoise(0.1))  # Add noise after Conv2D layer
    # model.add(tf.keras.layers.BatchNormalization(momentum = 0.7))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(p))
    
    model.add(tf.keras.layers.Conv2D(256, kernel_size=4, strides=2, padding='same',kernel_initializer=init))
    # model.add(tf.keras.layers.GaussianNoise(0.1))  # Add noise after Conv2D layer
    # model.add(tf.keras.layers.BatchNormalization(momentum = 0.7))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(p))
    
    # model.add(tf.keras.layers.Conv2D(512, kernel_size=4, strides=2, padding='same'))
    # # model.add(tf.keras.layers.GaussianNoise(0.1))  # Add noise after Conv2D layer
    # # model.add(tf.keras.layers.BatchNormalization(momentum = 0.7))
    # model.add(tf.keras.layers.LeakyReLU(0.2))
    # model.add(tf.keras.layers.Dropout(p))

    # Flatten layer to feed into a dense output layer
    model.add(tf.keras.layers.Flatten())
    
    # Output layer: Single neuron with sigmoid activation
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Use 0 to 1 output range
    
    return model

def compile_gan(generator, discriminator, optimizer_gan):
    
    # # # Freeze the discriminator's weights during the GAN training
    discriminator.trainable = False

    # GAN is a combined model of generator and discriminator
    gan = tf.keras.Sequential([generator, discriminator])

    # Compile the GAN model with the optimizer for the generator
    gan.compile(loss='binary_crossentropy', optimizer=optimizer_gan)

    return gan

# Define the label smoothing function
def smooth_labels(labels, smoothing_factor=0.1):
    return labels * (1 - smoothing_factor) + smoothing_factor * 0.5


def train_gan(strategy, sketch_type, dataset, generator, discriminator, gan, image_placeholder,image_placeholder_loss,freq_show = 10, freq_save = 100,epochs=100,latent_dim=100):

    # Initialize lists to store the average losses for each epoch
    avg_D_losses = []
    avg_G_losses = []
    avg_D_accuracies = []

    for epoch in range(epochs):
        # Initialize loss accumulators for each epoch
        sum_D_losses = 0
        sum_G_losses = 0
        sum_D_acc = 0
        num_batches = 0

        for real_images in dataset:

            batch_size = tf.shape(real_images)[0]
            half_batch = batch_size//2

            # Latent noise for generator
            noise = np.random.normal(0, 1, (batch_size, latent_dim))

            # Generate fake images using the generator
            generated_images = generator.predict(noise)

            # 1. Train the discriminator with real and fake images
            discriminator.trainable = True  # Unfreeze the discriminator to train it

            # Select real images from the dataset
            batch_real_images = real_images[:half_batch]   # Directly use real images from the batch
            batch_fake_images = generated_images[:half_batch]  # Use first half of generated images

            # Initial labels
            real_labels = np.zeros((half_batch, 1))  + np.random.uniform(low=0.0, high=0.1, size=(half_batch, 1)) # Label for real images: 1
            fake_labels = np.ones((half_batch, 1))  - np.random.uniform(low=0.0, high=0.1, size=(half_batch, 1)) # Label for fake images: 0

            # # # # Flip 1/3 of the labels (using 0.33 as p_flip)
            # flip_fraction = 0.05
            # real_labels = noisy_labels(real_labels, flip_fraction)  # Flip some real labels
            # fake_labels = noisy_labels(fake_labels, flip_fraction)  # Flip some fake labels

            # Train the discriminator on real images (with smoothed labels) and fake images (with original labels)
            d_loss_real, d_acc_real = discriminator.train_on_batch(batch_real_images, real_labels)
            d_loss_fake, d_acc_fake = discriminator.train_on_batch(batch_fake_images, fake_labels)

            # Average the losses and accuracies
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)  # Average of both losses
            d_acc = 0.5 * np.add(d_acc_real, d_acc_fake)  # Average of both accuracies

            # print(d_acc)
            
            # 2. Train the generator (through the GAN model) every epoch
            discriminator.trainable = False  # Freeze the discriminator (now only the generator will be trained)
            valid_labels = np.zeros((batch_size, 1))  # Fake images are labeled as real for the generator
            g_loss = gan.train_on_batch(noise, valid_labels)

            # Append losses for this batch to the respective lists
            sum_D_losses += d_loss
            sum_G_losses += g_loss
            sum_D_acc += d_acc
            num_batches += 1

        # Compute and store the average losses for the epoch
        avg_D_loss = sum_D_losses / num_batches
        avg_G_loss = sum_G_losses / num_batches

        avg_D_acc = sum_D_acc / num_batches

        avg_D_losses.append(avg_D_loss)
        avg_G_losses.append(avg_G_loss)
        avg_D_accuracies.append( avg_D_acc)

        # Every few epochs, print the progress and save the model
        if epoch % freq_save == 0:  # Save model images every `freq_save` epochs
            generator.save(f"./trained_generators_{strategy}_{sketch_type}/trained_generator_{strategy}_epoch_{epoch}.h5")  # Save model

            denormalized_generated_images = denormalize_images(generated_images)  # Denormalize for display

            save_generated_images(strategy, sketch_type, denormalized_generated_images, epoch, path=f"./generated_images_{strategy}_{sketch_type}")

        if epoch % freq_show == 0:  # Show images every `freq_show` epochs
            denormalized_fake_images = denormalize_images(batch_fake_images)  # Denormalize for display
            denormalized_real_images = denormalize_images(batch_real_images)  # Denormalize for display
            show_images_in_streamlit(strategy, denormalized_real_images, denormalized_fake_images, epoch, image_placeholder)  # Show images in Streamlit
            
            show_loss_acc_in_streamlit(strategy, avg_G_losses, avg_D_losses,avg_D_accuracies, epoch, epochs, image_placeholder_loss)

    # After training completes, save the final generator model
    generator.save(f"trained_generator_{strategy}_{sketch_type}_final.h5")


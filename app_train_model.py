import streamlit as st
import tensorflow as tf
import numpy as np
from src.data_preprocessing import load_data, normalize_images,extract_last_word_from_filename, create_dataset
from src.gan_model_wasserstein import build_generator_WGAN, build_critic, compile_gan_WGAN, train_gan_WGAN, wasserstein_loss
from src.gan_model import build_generator, build_discriminator, compile_gan, train_gan
from src.vae_model import vae_model, train_vae
import matplotlib.pyplot as plt
import argparse


# Function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Choose the model type (DCGAN or WGAN or VAE)")
    parser.add_argument('--model', type=str, choices=['DCGAN', 'WGAN','VAE'], required=True, help="Choose between 'DCGAN' or 'WGAN' or 'VAE'")
    args = parser.parse_args()
    return args

# Get the parsed arguments
args = parse_args()

# Use the argument
strategy = args.model

# Streamlit UI setup
st.title("Brave new Dolomiti")
st.write(f"Generate Dolomiti-like landscapes using a {strategy} model")

# Set a local file path 
# local_data_file = "./data/data_kaggle.npz"  # Change this to your local file path
local_data_file = "./data/data_dolomiti.npz"  # Change this to your local file path

try:
    # Load data from the .npz file
    data = np.load(local_data_file)
    images = data['images']
    
    # Get the sketch type from the file name 
    sketch_type = extract_last_word_from_filename(local_data_file)
    
    # Ensure the images are RGB (3 channels) and 256x256
    if images.shape[-1] != 3 or images.shape[1] != 256 or images.shape[2] != 256:
        st.error("Images should be of shape (256, 256, 3) for RGB images.")
    else:
        images = normalize_images(images)

        st.write("Dataset loaded, training model...")

        # Create a placeholder for images before training starts
        image_placeholder = st.empty()
        image_placeholder_loss = st.empty()

        if(strategy == "DCGAN"):
            # GAN setup
            latent_dim = 4096
            learning_rate = 0.0004
            # optimizer_gan = tf.keras.optimizers.RMSprop(lr=learning_rate, decay = 3e-8, clipvalue=1.0)
            # optimizer_disc = tf.keras.optimizers.RMSprop(lr=0.00004,decay = 6e-8, clipvalue=1.0)

            optimizer_gan = tf.keras.optimizers.RMSprop(lr=learning_rate, weight_decay = 3e-8, clipvalue=1.0)
            optimizer_disc = tf.keras.optimizers.RMSprop(lr=0.00008,weight_decay = 6e-8, clipvalue=1.0)

            print('bulding generator')
            generator = build_generator(latent_dim)

            print('bulding discriminator')
            discriminator = build_discriminator()
            discriminator.compile(loss='binary_crossentropy', optimizer=optimizer_disc, metrics=['accuracy'])

            print('bulding GAN')
            gan = compile_gan(generator, discriminator,optimizer_gan)

            # GAN Training
            n_epochs = 3000
            batch_size = 64  

            dataset = create_dataset(images, batch_size, 5000)

            num_batches = sum(1 for _ in dataset)  # Count the number of batches
            # print(f"Number of batches in dataset: {num_batches}")

            # how often are results saved and displayed
            n_freq_show = 1
            n_freq_save = 1000

            # Start training process
            train_gan(strategy,
                    sketch_type,
                    dataset, 
                    generator, 
                    discriminator, 
                    gan, 
                    image_placeholder, 
                    image_placeholder_loss,
                    freq_show = n_freq_show, 
                    freq_save = n_freq_save,
                    epochs=n_epochs, 
                    latent_dim=latent_dim)
            
        if(strategy == "WGAN"):
            # GAN setup
            latent_dim = 4096
            learning_rate_gan = 0.00004
            learning_rate_disc = 0.00004
            optimizer_gan = tf.keras.optimizers.RMSprop(lr=learning_rate_gan)
            optimizer_crit = tf.keras.optimizers.RMSprop(lr=learning_rate_disc)

            # optimizer_gan = tf.keras.optimizers.RMSprop(lr=learning_rate_gan, decay = 3e-8, clipvalue=1.0)
            # optimizer_crit = tf.keras.optimizers.RMSprop(lr=learning_rate_disc,decay = 6e-8, clipvalue=1.0)

            print('bulding generator')
            generator = build_generator_WGAN(latent_dim)

            print('bulding critic')
            critic = build_critic()
            critic.compile(loss=wasserstein_loss, optimizer=optimizer_crit, metrics=['accuracy'])

            print('bulding GAN')
            gan = compile_gan_WGAN(generator, critic,optimizer_gan)

            # GAN Training
            n_epochs = 30000
            batch_size = 64  

            dataset = create_dataset(images, batch_size, 5000)
            # dataset = dataset.skip(len(dataset) - 1)  # Skip the last batch

            # how often are results saved and displayed
            n_freq_show = 1
            n_freq_save = 1000

            # Start training process
            train_gan_WGAN(strategy,
                    sketch_type,
                    dataset, 
                    generator, 
                    critic, 
                    gan,  
                    image_placeholder, 
                    image_placeholder_loss,
                    freq_show = n_freq_show, 
                    freq_save = n_freq_save,
                    epochs=n_epochs,  
                    latent_dim=latent_dim)
            

        if(strategy == "VAE"):

            latent_dim = 100

            vae, encoder_model, decoder_model = vae_model(latent_dim)
                        # Define optimizer
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

            # GAN Training
            n_epochs = 1000
            batch_size = 64  

            dataset = create_dataset(images, batch_size, 5000)
            # dataset = dataset.skip(len(dataset) - 1)  # Skip the last batch

            # how often are results saved and displayed
            n_freq_show = 1
            n_freq_save = 1000

            # Start training
            train_vae( strategy,
                    sketch_type,
                    optimizer,
                    dataset,
                    encoder_model,
                    decoder_model,
                    image_placeholder,
                    n_freq_show,
                    n_freq_save,
                    n_epochs,
                    latent_dim)


    st.write("Training complete!")

except Exception as e:
    st.error(f"An error occurred: {e}")

print('DONE')
import streamlit as st
import tensorflow as tf
import numpy as np
from src.data_preprocessing import load_data, normalize_images
from src.gan_model_wasserstein import build_generator, build_critic, compile_gan, train_gan
import matplotlib.pyplot as plt
from src.utils import extract_last_word_from_filename

# Streamlit UI setup
st.title("Dolomiti GAN")
st.write("Generate Dolomite landscapes using a simple GAN")

# Set a local file path 
local_data_file = "./data/data_dolomiti.npz"  # Change this to your local file path

strategy = 'WGAN'

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

        # GAN setup
        latent_dim = 4096  # Latent dimension (size of noise vector)
        generator = build_generator(latent_dim)  # Create the generator model
        critic = build_critic()  # Create the critic (formerly discriminator) model
        gan = compile_gan(generator, critic)  # Compile the GAN model (generator + critic)

        # WGAN Training Setup
        epochs = 1000  
        batch_size = 64  

        # Start training process
        train_gan(strategy, sketch_type, generator, critic, gan, images, image_placeholder, image_placeholder_loss, epochs=epochs, batch_size=batch_size, latent_dim=latent_dim)

        st.write("Training complete!")  # Display message when training is complete

except Exception as e:
    st.error(f"An error occurred: {e}")


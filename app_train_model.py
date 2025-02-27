import streamlit as st
import tensorflow as tf
import numpy as np
from src.data_preprocessing import load_data, normalize_images
from src.gan_model import build_generator, build_discriminator, compile_gan, train_gan
import matplotlib.pyplot as plt
from src.utils import extract_last_word_from_filename

# Streamlit UI setup
st.title("Dolomiti GAN")
st.write("Generate Dolomite landscapes using a simple GAN")

# Load and preprocess data
data_file = st.file_uploader("Upload your dataset (.npz)", type=["npz"])

if data_file is not None:
    try:
        # Split the file name by underscore and get the last part
        sketch_type = extract_last_word_from_filename(data_file)

        # Load data from the uploaded .npz file
        data = np.load(data_file)
        images = data['images']
        
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
            latent_dim = 4096
            #print("going to build gen")
            generator = build_generator(latent_dim)
            #print("going to build disc")
            discriminator = build_discriminator()

            #print("going to build gan")
            gan = compile_gan(generator, discriminator)
            #print("gan done")

            # GAN Training
            epochs = 1000
            batch_size = 64  # Can adjust based on available resources

            # Start training process
            train_gan(sketch_type, generator, discriminator, gan, images, image_placeholder, image_placeholder_loss, epochs=epochs, batch_size=batch_size, latent_dim=latent_dim)

            st.write("Training complete!")

    except Exception as e:
        st.error(f"An error occurred: {e}")


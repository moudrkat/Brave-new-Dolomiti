import streamlit as st
import tensorflow as tf
import numpy as np
from src.data_preprocessing import normalize_images,extract_last_word_from_filename
import argparse

from model_setup import dcgan_setup,wgan_setup,vae_setup

# Function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Choose the model type (DCGAN or WGAN or VAE or VQVAE)")
    parser.add_argument('--model', type=str, choices=['DCGAN', 'WGAN','VAE','VQVAE'], required=True, help="Choose between 'DCGAN' or 'WGAN' or 'VAE' or 'VQVAE'")
    args = parser.parse_args()
    return args

# Get the parsed arguments
args = parse_args()

# Use the argument
strategy = args.model

# rndn seed for TensorFlow:
tf.random.set_seed(42)

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
            dcgan_setup(strategy,sketch_type,images,image_placeholder,image_placeholder_loss)
            
        if(strategy == "WGAN"):
            wgan_setup(strategy,sketch_type,images,image_placeholder,image_placeholder_loss)
            
        if(strategy == "VAE"):
            vae_setup(strategy,sketch_type,images,image_placeholder,image_placeholder_loss)

        if(strategy == "VQVAE"):
            vqvae_setup(strategy,sketch_type,images,image_placeholder,image_placeholder_loss)

    st.write("Training complete!")

except Exception as e:
    st.error(f"An error occurred: {e}")

print('DONE')
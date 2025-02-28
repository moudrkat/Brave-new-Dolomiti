import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import matplotlib.cm as cm
import os

# # Function to extract the last word from the file name
# def extract_last_word_from_filename(file_path):
#     # Extract the base name (without extension)
#     file_name = os.path.splitext(file_path.name)[0]
    
#     # Split the file name by underscore and get the last part
#     last_word = file_name.split("_")[-1]
    
#     return last_word


# Modify your function or logic to handle file paths directly
def extract_last_word_from_filename(file_path):
    # Get the filename from the path
    filename = os.path.basename(file_path)
    # Extract the last word based on your logic (for example, splitting by underscore)
    last_word = filename.split('_')[-1].replace('.npz', '')
    return last_word

def save_generated_images(strategy, sketch_type, images, epoch,  path=f"./generated_images"):

    images = (images + 1) * 127.5  # Rescale to [0, 255]
    images = np.clip(images, 0, 255)  # Ensure values are in the valid range [0, 255]

    plt.figure(figsize=(10.8, 13.5))
    
    # Add the epoch number as the title
    plt.suptitle(f"{strategy} generated {sketch_type} - epoch {epoch}", fontsize=35, color = 'black', fontfamily = 'Lucida Console', fontweight='bold')

    for i in range(20):  # Show only the first 24 images
        plt.subplot(5, 4, i + 1)  # Create a 4x6 grid of images
        #plt.imshow(images[i, :, :, 1])
        plt.imshow(images[i].astype(np.uint8))  # Display the full RGB image (ensure it's uint8)
        plt.axis('off')

    plt.savefig(f"{path}/generated_{strategy}_epoch_{epoch}.png")
    plt.close()

def show_images_in_streamlit(strategy, real_images, fake_images, epoch,image_placeholder):

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Display the real image (first image from the batch)
    axes[0].imshow(real_images[0]) 
    axes[0].set_title("Real Dolomiti")
    axes[0].axis('off')

    # Display the fake image (first generated image)
    axes[1].imshow(fake_images[0])  
    axes[1].set_title("Brave new Dolomiti")
    axes[1].axis('off')

    image_placeholder.pyplot(fig) 

def show_loss_acc_in_streamlit(strategy, g_losses, d_losses, d_accuracies, epoch,epochs,image_placeholder_loss,path="./generated_images"):

    fig, ax1 = plt.subplots()

    ax1.plot(g_losses, label='Generator Loss', color='blue')
    ax1.plot(d_losses, label='Discriminator Loss', color='red')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, 2)

    # ax2 = ax1.twinx()  # Create a second y-axis for accuracy
    # ax2.plot(d_accuracies, label='Discriminator Accuracy', color='green', linestyle='dashed')
    # ax2.set_ylabel("Accuracy")
    # ax2.legend(loc='upper right')

    plt.title(f"Losses and Accuracy at Epoch {epoch}")
    image_placeholder_loss.pyplot(fig)  # Display plot in Streamlit

    if epoch == epochs-1:
        plt.savefig(f"{path}/model_metrics.png")



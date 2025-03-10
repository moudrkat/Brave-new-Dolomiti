import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import matplotlib.cm as cm
import os
import tensorflow as tf


def save_generated_images(strategy, sketch_type, images, epoch,  path=f"./generated_images"):

    # images = (images + 1) * 127.5  # Rescale to [0, 255]
    # images = np.clip(images, 0, 255)  # Ensure values are in the valid range [0, 255]

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

def show_images_in_streamlit(strategy, real_images, fake_images, epoch, image_placeholder):
    # Ensure the images are 4D arrays 
    batch_size = real_images.shape[0]
    # batch_size = 64

    # Randomly select 10 unique indices from the batch
    random_indices = np.random.choice(batch_size, 5, replace=False)

    # Select 5 random real and fake images using the selected indices
    # random_real_images = real_images[random_indices]
    random_fake_images = fake_images[random_indices]
    
    # Create a figure with 2 columns and 10 rows
    fig, axes = plt.subplots(5, 2, figsize=(5, 10))

    # Loop through the 10 rows and display real and fake images in the respective columns
    for i in range(5):
        # Display the real image in the first column
        # axes[i, 0].imshow(random_real_images[i])  
        axes[i, 0].set_title(f"Real Dolomiti {i+1}")
        axes[i, 0].axis('off')

        # Display the fake image in the second column
        axes[i, 1].imshow(random_fake_images[i])  
        axes[i, 1].set_title(f"Brave new Dolomiti {i+1}")
        axes[i, 1].axis('off')

    # Adjust layout for better spacing between images
    #plt.tight_layout()
    
    # Show the images in the Streamlit placeholder
    image_placeholder.pyplot(fig)

def show_loss_acc_in_streamlit(strategy, g_losses, d_losses,d_accuracies, epoch,epochs,image_placeholder_loss,path="./generated_images"):

    fig, ax1 = plt.subplots()

    ax1.plot(g_losses, label='Generator Loss', color='blue')
    ax1.plot(d_losses, label='Discriminator Loss', color='red')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, 8)

    ax2 = ax1.twinx()  # Create a second y-axis for accuracy
    ax2.plot(d_accuracies, label='Discriminator Accuracy', color='green', linestyle='dashed')
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc='upper right')

    plt.title(f"Losses and Accuracy at Epoch {epoch}")
    image_placeholder_loss.pyplot(fig)  # Display plot in Streamlit

    if epoch == epochs-1:
        plt.savefig(f"{path}/model_metrics.png")


# Function to plot the losses
def plot_vae_losses(avg_reconstruction_losses, avg_kl_losses):
    plt.figure(figsize=(12, 6))

    # Plot Reconstruction Loss
    plt.subplot(1, 2, 1)
    plt.plot(avg_reconstruction_losses, label='Reconstruction Loss', color='b')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Reconstruction Loss')
    plt.legend()

    # Plot KL Divergence Loss
    plt.subplot(1, 2, 2)
    plt.plot(avg_kl_losses, label='KL Divergence', color='r')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('KL Divergence Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()



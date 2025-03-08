import numpy as np
import tensorflow as tf
import os

def extract_last_word_from_filename(file_path):
    # Get the filename from the path
    filename = os.path.basename(file_path)
    # Extract the last word based on your logic (for example, splitting by underscore)
    last_word = filename.split('_')[-1].replace('.npz', '')
    return last_word

def create_dataset(images_array, batch_size, image_num, shuffle=True):
    # Ensure the images_array is a NumPy array and has the shape (num_images, height, width, channels)
    images_array = np.array(images_array)
    
    # Create TensorFlow dataset from the images array
    dataset = tf.data.Dataset.from_tensor_slices(images_array)
    
    # Shuffle the dataset if needed
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    # Limit dataset to 'image_num' images (this is optional)
    # dataset = dataset.take(image_num // batch_size)
    
    # Batch the dataset
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    return dataset

def normalize_images(images):
    images = images.astype(np.float32)
    # Normalizes images from 0-255 to -1 to 1
    images = (images - 127.5) / 127.5
    return images

def load_data(file_path):
    data = np.load(file_path)
    images = data['images']
    return images

def denormalize_images(images):
    # Convert from [-1, 1] to [0, 255]
    images = (images + 1) * 127.5
    images = np.clip(images, 0, 255).astype(np.uint8)
    return images
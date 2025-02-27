import numpy as np

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
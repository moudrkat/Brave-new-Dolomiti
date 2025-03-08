import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Define image preprocessing/transformations
def preprocess_image(image):
    # Example transformation: resize and normalize the image
    image = tf.image.resize(image, (256, 256))  # Resize to 256x256
    image = image / 255.0  # Normalize to [0, 1]
    return image

def create_dataset(image_dir, batch_size, image_num, shuffle=True):
    # Load images from the directory
    dataset = image_dataset_from_directory(
        image_dir,
        image_size=(256, 256),  # Resize images to 256x256
        batch_size=batch_size,
        shuffle=shuffle,
        label_mode='int',  # Labels are integer encoded (this can be changed to 'categorical' if needed)
        validation_split=0.0,  # No validation split, all data will be used
        subset=None
    )

    # Apply image preprocessing to the dataset
    dataset = dataset.map(lambda x, y: (preprocess_image(x), y))  # Apply preprocessing to each image

    # Limit the dataset to 'image_num' images (e.g., 4000)
    dataset = dataset.take(image_num // batch_size)

    return dataset

# Example usage
image_dir = 'path/to/your/images'  # Replace with your image folder path
batch_size = 32
image_num = 4000
epochs = 10  # Number of epochs to train for

# Create the dataset
dataset = create_dataset(image_dir, batch_size, image_num)

# Iterate over epochs
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    
    # Iterate over batches in the dataset
    for batch_images, batch_labels in dataset:
        # Here, you can process the batch (e.g., forward pass, loss calculation, etc.)
        print(f"Batch shape: {batch_images.shape}, Labels shape: {batch_labels.shape}")
        
        # Example: Do something with the batch (e.g., training step)
        # For now, we just break to avoid unnecessary processing
        # Add your model training code here
        # model.train_step(batch_images, batch_labels)
    
    print(f"End of epoch {epoch + 1}\n")



or


import tensorflow as tf
from tensorflow.keras import layers

# Generator Model
def create_generator():
    model = tf.keras.Sequential([
        layers.Dense(8 * 8 * 256, input_shape=(100,)),  # 100-dimensional noise vector
        layers.Reshape((8, 8, 256)),  # Reshape to 8x8x256

        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', activation='relu'),
        layers.BatchNormalization(),
        
        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', activation='relu'),
        layers.BatchNormalization(),
        
        layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh'),  # RGB output
    ])
    return model

# Discriminator Model
def create_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, kernel_size=4, strides=2, padding='same', activation='relu', input_shape=(256, 256, 3)),
        layers.Dropout(0.3),
        
        layers.Conv2D(128, kernel_size=4, strides=2, padding='same', activation='relu'),
        layers.Dropout(0.3),
        
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')  # Output: 1 (real or fake)
    ])
    return model

# Create models
generator = create_generator()
discriminator = create_discriminator()

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Loss function
binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# Create dataset (you can adapt the dataset creation as shown in previous responses)
def create_dataset(image_dir, batch_size, image_num, shuffle=True):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        image_dir,
        image_size=(256, 256),
        batch_size=batch_size,
        shuffle=shuffle,
        label_mode=None  # No labels since this is unsupervised
    )
    
    #dataset = dataset.map(lambda x: x / 255.0)  # Normalize images to [0, 1]
    dataset = dataset.take(image_num // batch_size)  # Limit dataset to 'image_num' images
    
    return dataset

# Training loop using train_on_batch
epochs = 10
image_dir = 'path/to/your/images'  # Replace with your image folder path
batch_size = 32
image_num = 4000

dataset = create_dataset(image_dir, batch_size, image_num)

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    
    for real_images in dataset:
        batch_size = tf.shape(real_images)[0]
        
        # Generate fake images from the Generator
        noise = tf.random.normal([batch_size, 100])  # Generate random noise
        fake_images = generator(noise, training=False)
        
        # Train the Discriminator
        # - Real images are labeled 1
        # - Fake images are labeled 0
        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))
        
        # Train Discriminator on real images
        d_loss_real, d_acc_real = discriminator.train_on_batch(real_images, real_labels)
        
        # Train Discriminator on fake images
        d_loss_fake, d_acc_fake = discriminator.train_on_batch(fake_images, fake_labels)
        
        # Calculate total Discriminator loss
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        d_acc = 0.5 * (d_acc_real + d_acc_fake)

        # Train the Generator by trying to fool the Discriminator
        # The Generator wants the Discriminator to classify its fake images as "real"
        g_labels = tf.ones((batch_size, 1))  # Label fake images as "real" for the Generator
        g_loss = discriminator.train_on_batch(fake_images, g_labels)  # Generator loss based on discriminator's feedback
        
        # Print losses (you can choose to print less often)
        print(f"Discriminator Loss: {d_loss:.4f}, Discriminator Accuracy: {d_acc:.4f}, Generator Loss: {g_loss:.4f}")
    
    print(f"End of Epoch {epoch+1}")

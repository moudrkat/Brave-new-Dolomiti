import tensorflow as tf
from src.data_preprocessing import create_dataset
from src.gan_model import build_generator, build_discriminator, compile_gan, train_gan

def dcgan_setup(strategy,sketch_type,images,image_placeholder,image_placeholder_loss):
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
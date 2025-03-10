            
import tensorflow as tf
from src.data_preprocessing import  create_dataset
from src.gan_model_wasserstein import build_generator_WGAN, build_critic, compile_gan_WGAN, train_gan_WGAN, wasserstein_loss


def wgan_setup(strategy,sketch_type,images,image_placeholder,image_placeholder_loss):
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
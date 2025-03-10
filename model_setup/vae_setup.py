import tensorflow as tf
from src.data_preprocessing import create_dataset
from src.vae_model import train_vae, encoder, decoder

def vae_setup(strategy,sketch_type,images,image_placeholder,image_placeholder_loss):
            latent_dim = 256
          
            encoder_model = encoder(latent_dim)
            decoder_model = decoder(latent_dim)

            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.96, staircase=True)
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

            # GAN Training
            n_epochs = 1000
            batch_size = 64  

            dataset = create_dataset(images, batch_size, 5000)

            # how often are results saved and displayed
            n_freq_show = 1
            n_freq_save = 1000

            # Start training
            train_vae( strategy,
                    sketch_type,
                    optimizer,
                    dataset,
                    encoder_model,
                    decoder_model,
                    image_placeholder,
                    n_freq_show,
                    n_freq_save,
                    n_epochs,
                    latent_dim)
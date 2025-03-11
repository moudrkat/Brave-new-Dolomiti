import tensorflow as tf
from src.data_preprocessing import create_dataset
from src.vqvae_model import train_vqvae, encoder, decoder, VectorQuantizer


def vqvae_setup(strategy,sketch_type,images,image_placeholder,image_placeholder_loss):
    latent_dim = 64
    num_codes = 256  # Number of codebook entries (discrete latents)
    code_dim = 64     # Dimension of each codebook vector

    encoder_model = encoder(latent_dim, code_dim)
    vector_quantizer = VectorQuantizer(num_codes, code_dim)
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
    train_vqvae( strategy,
            sketch_type,
            optimizer,
            dataset,
            encoder_model,
            vector_quantizer,
            decoder_model,
            image_placeholder,
            n_freq_show,
            n_freq_save,
            n_epochs,
            latent_dim)

The aim of this project is to train a model capable of generating new Dolomiti landscapes, in honor of the author of thousands of stunning photographs used for training.

I am currently working on implementing DCGAN, WGAN and VAE. Later, I plan to explore the VQGAN approach as well.

Hopefully, one of these methods will converge successfully.

Sadly, nothing gives sufficient results yet. As I said, it is work in progress.

DCGAN and WGAN are weirdly tuned so they fall to mode collapse and currently produce weird images.

VAE gives blurry results because its VAE ... but I am trying to improve it.

Run with argument: streamlit run app_train_model.py -- --model DCGAN
                   streamlit run app_train_model.py -- --model WGAN
                   streamlit run app_train_model.py -- --model VAE


Resources: 

        GAN:
        https://machinelearningmastery.com/how-to-code-generative-adversarial-network-hacks/

        DCGAN:

        https://arxiv.org/abs/1511.06434

        https://distill.pub/2016/deconv-checkerboard/

        WGAN:

        https://medium.com/@m.naufalrizqullah17/exploring-conditional-gans-with-wgan-4f13e91d30eb

        https://machinelearningmastery.com/how-to-code-a-wasserstein-generative-adversarial-network-wgan-from-scratch/


        VQVAE:

        https://keras.io/examples/generative/vq_vae/



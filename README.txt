The aim of this project is to train a model capable of generating new Dolomiti landscapes, in honor of the author of thousands of stunning photographs used for training.

I am currently working on implementing DCGAN, WGAN and VAE. Later, I plan to explore the VQGAN approach as well.

Hopefully, one of these methods will converge successfully.

Sadly, nothing gives sufficient results yet. As I said, it is work in progress.

Run with argument: streamlit run app_train_model.py -- --model DCGAN
                   streamlit run app_train_model.py -- --model WGAN
                   streamlit run app_train_model.py -- --model VAE


Resources: https://machinelearningmastery.com/how-to-code-generative-adversarial-network-hacks/

        https://arxiv.org/abs/1511.06434

        https://distill.pub/2016/deconv-checkerboard/

        https://medium.com/@m.naufalrizqullah17/exploring-conditional-gans-with-wgan-4f13e91d30eb

        https://machinelearningmastery.com/how-to-code-a-wasserstein-generative-adversarial-network-wgan-from-scratch/

        https://machinelearningmastery.com/how-to-code-generative-adversarial-network-hacks/



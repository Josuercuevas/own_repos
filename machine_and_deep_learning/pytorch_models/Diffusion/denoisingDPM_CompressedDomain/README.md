# Compress Domain DDPM

The idea is to use [compress domain of images](https://www.researchgate.net/publication/358137310_Usage_of_compressed_domain_in_fast_frameworks) to train and evaluate DDPMs. Therefore ideally the only thing that should change is how the input images are fed to the model, while maintaning performance in terms of accuracy and generative power. Additionally, we improvement in terms of training/inference time should be obeserved.

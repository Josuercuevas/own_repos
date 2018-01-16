# Classification or Feature Extraction models implementation in Tensorflow

Tensorflow implementations of classification models mainly, for now I have completed 3/5 models which I have been researching, they all have different parameters settings and number of layers compared with the original paper, the reason is because I adapted them to my needs. You are encouraged to do the same. The models are specified as follows:

1. [SqueezeNet](https://arxiv.org/abs/1602.07360), the final model is illustrated below:

![squeezenet](bodynet_squeeznet.png)

2. [Resnet](https://arxiv.org/abs/1512.03385), the final model is illustrated below:

![resnet](bodynet_resnet.png)

3. [Helnet]: similar to resnet but no skip connections, the final model is illustrated below:

![helnet](bodynet_helnet.png)

4. [ShuffleNet](https://arxiv.org/abs/1707.01083) --> not ready

5. [MobileNet](https://arxiv.org/abs/1704.04861) --> not ready

### How to run it

Just type "python main.py --help" for details on how to execute this code

### Important

1. **Training dataset**: training dataset can be in hd5 format or pure images (jpegs). Images have to be places in separated folders by class inside the "train/", "test/", and "val/" subfolders.

2. **Features supported**: [Dense Sparse Dense](https://arxiv.org/abs/1607.04381) training

3. **Model visualization**: a dot file and png files are created in order to visualize the models.

4. **Models and Logs**: models and tensorboard logs can be saved inside "models/" and "tfboard/" subfolder respectively.

5. **Configurations**: configurations of the different models and for the training/testing engines are specified inside the "configs/" subfolder

### Contributor

Josue R. Cuevas

josuercuevas@gmail.com

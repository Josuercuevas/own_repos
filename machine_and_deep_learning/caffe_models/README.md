# Caffe framework tests

Two models are used in this repository:

Inside subfolder **model_viz/**:

1. **SqueezeDet**: based on the manuscript [here](https://arxiv.org/abs/1612.01051), I have written the prototxt which could be used to determine computation cost, or as in here I used it for visualization purposes. The python module "draw_net.py" will read the prototxt and write an image file specified by the User. Consequently, this code could be easily adapted to any other network as long as the prototxt is provided and the Caffe repository supports the layers defined inside the prototxt file.

Inside the subfolder **shuffleNet/**:

2. **ShuffleNet**: Based on the paper [here](https://arxiv.org/abs/1707.01083), the training and inference prototxt are provided inside this subfolder. I have trained the weights with Imagenet and tinyImagenet. Some testing samples are provided as well.

**Note**: more models are planned to be added in the future ...

### Contributor

Josue R. Cuevas

josuercuevas@gmail.com

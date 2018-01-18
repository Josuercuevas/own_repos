# Caffe repo with own modification for different ideas proposed over the years

In here I have modified the original source code from [CAFFE](http://caffe.berkeleyvision.org/), the original README can be found in the file **"README_original"**. This repository is in essence the same as the original CAFFE, except for some extra layers that I have found to be very useful over the years.

### Extra layers added

1. **Permutation layer**: a way to arrange the data axes in a different order, so as to match the one in Tensorflow. This was very useful when comparing models from CAFFE vs. Tensorflow. [reference](http://www.mathworks.com/help/matlab/ref/permute.html) **find it as permute_layer**

2. **Group convolution**: Used to learn better representation of the input data, it was firstly proposed [here](https://arxiv.org/pdf/1605.06489.pdf). **Now supported in cuDNN 7**

3. **Shuffle channels layer**: Proposed in this [paper](https://arxiv.org/abs/1707.01083) with the objective of reducing model Flops, this layer what actually does is to avoid using the same groups all the time, which in my opinion is the biggest problem when just using Group Convolutions.**find it as shuffle_channel_layer**

4. **Depthwise separable convolution Layer**: Basically to handle a huge problem that most network have, which is **"reduntant"** filters. The idea behind DW_sep_conv is that you can drop or reduce irrelevant filters that will provide the same information when inference is performed. [reference](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d) **find it as conv_dw_layer**, and a very interesting paper to take a look at is [this](https://arxiv.org/pdf/1704.04861.pdf)

### Contributors

Josue R. Cuevas

josuercuevas@gmail.com

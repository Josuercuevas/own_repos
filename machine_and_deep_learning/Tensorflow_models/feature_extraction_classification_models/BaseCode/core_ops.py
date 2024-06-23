'''
 Copyright (C) <2016>  <Josue R. Cuevas>

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

'''
    All the functions in this script are related to the models or topologies
    used in this project, which is basically training feature extraction models
    such as Resnet, Helnet, SqueezeNet, ShuffleNet, and others.

    All the core operations for model training and testing should be in this file.
'''

import math
import numpy as np
import tensorflow as tf

'''
    General information to be extracted from each layer
'''
def _get_dims(shape):
    """ Get the fan-in and fan-out of a Tensor. """
    fan_in = np.prod(shape[:-1])
    fan_out = shape[-1]
    return fan_in, fan_out

def _get_shape(x):
    """ Get the shape of a Tensor. """
    return x.get_shape().as_list()

'''
    To generate the weights we need for training the model
    on the head parts
'''
def weight_module(name, shape, init='normal', range=0.1, stddev=0.001, init_val=None, group_id=0):
    """ Get a weight variable. """
    if init_val != None:
        initializer = tf.constant_initializer(init_val)
    elif init == 'uniform':
        initializer = tf.random_uniform_initializer(-range, range)
    elif init == 'normal':
        initializer = tf.random_normal_initializer(stddev=stddev)
    elif init == 'he':
        fan_in, _ = _get_dims(shape)
        std = math.sqrt(2.0 / fan_in)
        initializer = tf.random_normal_initializer(stddev=std)
    elif init == 'xavier':
        fan_in, fan_out = _get_dims(shape)
        range = math.sqrt(6.0 / (fan_in + fan_out))
        initializer = tf.random_uniform_initializer(-range, range)
    else:
        initializer = tf.truncated_normal_initializer(stddev=stddev)

    var = tf.get_variable(name, shape, initializer=initializer)

    # this could be used if the user wants in order to penalize the loss function
    tf.add_to_collection('l2_'+str(group_id), tf.nn.l2_loss(var))
    return var


'''
    Bias initializer for the head parts
'''
def bias_module(name, dim, init_val=0.0):
    """ Get a bias variable. """
    dims = dim if isinstance(dim, list) else [dim]
    return tf.get_variable(name, dims, initializer = tf.constant_initializer(init_val))


'''
    Layer containing all the activations
'''
def nonlinear_module(x, nl=None):
    """ Apply a nonlinearity layer. """
    if nl == 'relu':
        return tf.nn.relu(x)
    elif nl == 'tanh':
        return tf.tanh(x)
    elif nl == 'sigmoid':
        return tf.sigmoid(x)
    else:
        return x

'''
    Core operation to be performed on a particular layer to extract the corresponding
    features when data has been feed to the network
'''
def convolution_module(x, k_h, k_w, c_o, s_h, s_w, name, init_w='normal', init_b=0.0, stddev=0.001,
                       padding='SAME', re_use=None, group_id=0):
    """ Apply a convolutional layer (with bias). """
    c_i = _get_shape(x)[-1]
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    with tf.variable_scope(name, reuse=re_use) as scope:
        w = weight_module('weights', [k_h, k_w, c_i, c_o], init=init_w, stddev=stddev, group_id=group_id)
        z = convolve(x, w)
        b = bias_module('biases', c_o, init_b)
        z = tf.nn.bias_add(z, b)
    return z


'''
    Fully connected layer to be used for prediction/segmentation/classification
'''
def fully_connected_module(x, output_size, name, init_w='normal', init_b=0, stddev=0.001, group_id=0):
    """ Apply a fully-connected layer (with bias). """
    x_shape = _get_shape(x)
    input_dim = x_shape[-1]

    with tf.variable_scope(name) as scope:
        w = weight_module('weights', [input_dim, output_size], init=init_w, stddev=stddev, group_id=group_id)
        b = bias_module('biases', [output_size], init_b)
        z = tf.nn.xw_plus_b(x, w, b)
    return z

'''
    In case we want to use batch normalization in the network to be trained
'''
def batch_norm_module(x, name, is_train, bn_axis=3, bn=True):
    """ Apply a batch normalization layer and a nonlinearity layer. """
    if bn:
        x = tf.layers.batch_normalization(inputs=x, name=name, training=is_train, axis=bn_axis)

    return x

'''
    to be used when training is activated
'''
def dropout_module(x, keep_prob, is_train):
    """ Apply a dropout layer. """
    return tf.cond(is_train, lambda: tf.nn.dropout(x, keep_prob), lambda: x)

'''
    Pooling layer to be used during fastforwards convolution
'''
def max_pool_module(x, k_h, k_w, s_h, s_w, name, padding='SAME'):
    """ Apply a max pooling layer. """
    return tf.nn.max_pool(x, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name=name)

'''
    In case no max pooling is to be used, but average
'''
def avg_pool(x, k_h, k_w, s_h, s_w, name, padding='SAME'):
    """ Apply an average pooling layer. """
    return tf.nn.avg_pool(x, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name=name)

'''
    SqueeNet topology main module
'''
def fire_module(x, c_s, c_e, name, padd1='SAME', padd2='SAME', padd3='SAME', debug_level=None):
    net_s = convolution_module(x, 1, 1, c_s, 1, 1, name + '_squeeze1x1', padding=padd1, init_w='truncated',
                               init_b=0.01, stddev=1.0 / c_s)
    net_s = nonlinear_module(x=net_s, nl='relu')

    if debug_level is not None and debug_level > 2:
        print("Squeeze Layer shape is:")
        print(net_s.get_shape())

    net_e1x1 = convolution_module(net_s, 1, 1, c_e, 1, 1, name + '_expand1x1', padding=padd2, init_w='truncated',
                                  init_b=0.01, stddev=1.0 / c_e)
    net_e1x1 = nonlinear_module(x=net_e1x1, nl='relu')

    if debug_level is not None and debug_level > 2:
        print("Expand 1x1 Layer shape is:")
        print(net_e1x1.get_shape())

    net_e3x3 = convolution_module(net_s, 3, 3, c_e, 1, 1, name + '_expand3x3', padding=padd3, init_w='truncated',
                                  init_b=0.01, stddev=1.0 / c_e)
    net_e3x3 = nonlinear_module(x=net_e3x3, nl='relu')

    if debug_level is not None and debug_level > 2:
        print("Expand 3x3 Layer shape is:")
        print(net_e3x3.get_shape())

    net = tf.concat(axis=3, values=[net_e1x1, net_e3x3], name=(name + '_concat'))

    return net #, net_s, net_e1x1, net_e3x3

'''
    Average pooling to be used for prediction of the classes in some of the topologies used in this
    project
'''
def Build_average_pool_layer(network, shared_shape, n_classes, pool_type, reshape=True, debug_level=None):
    # To be used later
    net = network
    info_layer = shared_shape

    if debug_level is not None and debug_level > 2:
        print("Last layer resulted in:")
        print(net.get_shape())

    # doing average pooling
    net = tf.nn.avg_pool(net, ksize=[1, info_layer[1], info_layer[2], 1],
                         strides=[1, info_layer[1], info_layer[2], 1], padding=pool_type)
    if debug_level is not None and debug_level > 2:
        print("Average pooling layer was:")
        print(net.get_shape())

    # reshaping to match the classes
    if reshape:
        net = tf.reshape(net, [-1, n_classes])
        if debug_level is not None and debug_level > 2:
            print("Model reshaped and new resulted in:")
            print(net.get_shape())

    return net

'''
    Modules required for topologies like Resnet
'''
def res_conv_skip(input_tensor, kernel_size, filters, stage, block, stride=2, name=None, is_training=True,
                  debug_level=None):
    """
    This block is in charge to creating the branches for the residual network structure
    where skip layer cannot be deactivated and downsampling is performed in the input
    layer
  """
    filters1, filters2, filters3 = filters
    conv_name_base = name + '_res' + str(stage) + block + '_branch'
    bn_name_base = name + '_bn' + str(stage) + block + '_branch'

    # a-branch of the residual module
    net = convolution_module(x=input_tensor, k_h=1, k_w=1, c_o=filters1, s_h=stride, s_w=stride,
                             name=(conv_name_base+'2a'), init_w='truncated', init_b=0.01, stddev=(1.0/filters1),
                             re_use=None, group_id=0)

    net = batch_norm_module(x=net, name=(bn_name_base+'2a'), is_train=is_training, bn_axis=3, bn=True)

    net = nonlinear_module(x=net, nl='relu')

    if debug_level is not None and debug_level > 2:
        print("%s shape is (branch-a): " % conv_name_base)
        print(net.get_shape())

    # b-branch of the residual module
    net = convolution_module(x=net, k_h=kernel_size, k_w=kernel_size, c_o=filters2, s_h=1, s_w=1,
                             name=(conv_name_base + '2b'), init_w='truncated', init_b=0.01, stddev=(1.0 / filters2),
                             padding='SAME', re_use=None, group_id=0)

    net = batch_norm_module(x=net, name=(bn_name_base + '2b'), is_train=is_training, bn_axis=3, bn=True)

    net = nonlinear_module(x=net, nl='relu')

    if debug_level is not None and debug_level > 2:
        print("%s shape is (branch-b): " % conv_name_base)
        print(net.get_shape())

    # c-branch of the residual module
    net = convolution_module(x=net, k_h=1, k_w=1, c_o=filters3, s_h=1, s_w=1,
                             name=(conv_name_base + '2c'), init_w='truncated', init_b=0.01, stddev=(1.0 / filters3),
                             re_use=None, group_id=0)

    net = batch_norm_module(x=net, name=(bn_name_base + '2c'), is_train=is_training, bn_axis=3, bn=True)

    if debug_level is not None and debug_level > 2:
        print("%s shape is (branch-c): " % conv_name_base)
        print(net.get_shape())

    # end of branch
    shortcut = convolution_module(x=input_tensor, k_h=1, k_w=1, c_o=filters3, s_h=stride, s_w=stride,
                                  name=(conv_name_base + 'shortcut'), init_w='truncated', init_b=0.01,
                                  stddev=(1.0 / filters3), re_use=None, group_id=0)
    shortcut = batch_norm_module(x=shortcut, name=(bn_name_base + 'shortcut'), is_train=is_training, bn_axis=3,
                                 bn=True)

    if debug_level is not None and debug_level > 2:
        print("%s shape is (shortcut): " % conv_name_base)
        print(shortcut.get_shape())

    # adding the shortcut and the branch
    net = tf.add(shortcut, net, name=(conv_name_base+'_residual_layer_withConv'))
    net = nonlinear_module(x=net, nl='relu')

    return net


def res_conv(input_tensor, kernel_size, filters, stage, block, name=None, is_training=True, debug_level=None,
             add_skip=True):
  """
    This block is in charge to creating the branches for the residual network structure
    where skip layer could be deactivated and no downsampling is performed in the input
    layer
  """
  filters1, filters2, filters3 = filters
  conv_name_base = name + '_res' + str(stage) + block + '_branch'
  bn_name_base = name + '_bn' + str(stage) + block + '_branch'

  # a-branch of the residual module
  net = convolution_module(x=input_tensor, k_h=1, k_w=1, c_o=filters1, s_h=1, s_w=1, name=(conv_name_base + '2a'),
                           init_w='truncated', init_b=0.01, stddev=(1.0 / filters1), re_use=None, group_id=0)

  net = batch_norm_module(x=net, name=(bn_name_base + '2a'), is_train=is_training, bn_axis=3, bn=True)

  net = nonlinear_module(x=net, nl='relu')

  if debug_level is not None and debug_level > 2:
      print("%s shape is (branch-a): " % conv_name_base)
      print(net.get_shape())

  # b-branch of the residual module
  net = convolution_module(x=net, k_h=kernel_size, k_w=kernel_size, c_o=filters2, s_h=1, s_w=1,
                           name=(conv_name_base + '2b'), init_w='truncated', init_b=0.01, stddev=(1.0 / filters2),
                           padding='SAME', re_use=None, group_id=0)

  net = batch_norm_module(x=net, name=(bn_name_base + '2b'), is_train=is_training, bn_axis=3, bn=True)

  net = nonlinear_module(x=net, nl='relu')

  if debug_level is not None and debug_level > 2:
      print("%s shape is (branch-b): " % conv_name_base)
      print(net.get_shape())

  # c-branch of the residual module
  net = convolution_module(x=net, k_h=1, k_w=1, c_o=filters3, s_h=1, s_w=1, name=(conv_name_base + '2c'),
                           init_w='truncated', init_b=0.01, stddev=(1.0 / filters3), re_use=None, group_id=0)

  net = batch_norm_module(x=net, name=(bn_name_base + '2c'), is_train=is_training, bn_axis=3, bn=True)

  # end of branch adding net and input without strided convolution
  if add_skip:
      net = tf.add(input_tensor, net, name=(conv_name_base + '_residual_layer_noConv'))
  net = nonlinear_module(x=net, nl='relu')

  if debug_level is not None and debug_level > 2:
      print("%s shape is (branch-c): " % conv_name_base)
      print(net.get_shape())

  return net

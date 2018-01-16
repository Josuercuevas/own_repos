'''
Copyright (C) <2017>  <Josue R. Cuevas>

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

"""
    Main core operations are to be handled in this part of the code
    where this one have to called from other modules where model is
    to be constructed or trained, tested.

    Extra operations are present just in case we want to test the model
    behavior under different settings or architectures.
"""
import math
import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages

'''
    To generate the weights we need for training the model
    on the head parts
'''
def weight_module(name, shape, init='xavier', range=0.1, stddev=0.001, init_val=None, group_id=0, trainable=True):
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
        initializer = tf.truncated_normal_initializer(stddev = stddev)

    var = tf.get_variable(name, shape, initializer = initializer, trainable=trainable)
    tf.add_to_collection('l2_'+str(group_id), tf.nn.l2_loss(var))
    return var

'''
    Bias initializer for the head parts
'''
def bias_module(name, dim, init_val=0.0, trainable=True):
    """ Get a bias variable. """
    dims = dim if isinstance(dim, list) else [dim]
    return tf.get_variable(name, dims, initializer = tf.constant_initializer(init_val), trainable=trainable)


'''
    Layer containing all the activations
'''
def nonlinear_module(x, name, nl=None, re_use=None):
    """ Apply a nonlinearity layer. """
    with tf.variable_scope(name, reuse=re_use) as scope:
        if nl == 'relu':
            z = tf.nn.relu(x)
        elif nl == 'tanh':
            z = tf.tanh(x)
        elif nl == 'sigmoid':
            z = tf.sigmoid(x)
        else:
            z = x
    return z

'''
    Core operation to be performed on a particular layer to extract the corresponding
    features when data has been feed to the network
'''
def convolution_module(x, k_h, k_w, c_o, s_h, s_w, name, init_w='normal', init_b=0, stddev=0.001, padding='SAME', re_use=None,
                group_id=0, trainable=True):
    """ Apply a convolutional layer (with bias). """
    c_i = _get_shape(x)[-1]
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    with tf.variable_scope(name, reuse=re_use) as scope:
        w = weight_module('weights', [k_h, k_w, c_i, c_o], init=init_w, stddev=stddev, group_id=group_id,
                          trainable=trainable)
        z = convolve(x, w)
        b = bias_module('biases', c_o, init_b, trainable=trainable)
        z = tf.nn.bias_add(z, b)
    return z

'''
    Just in case we dont want to feed any bias for the convolutional layer
'''
def convolution_no_bias_module(x, k_h, k_w, c_o, s_h, s_w, name, init_w='normal', stddev=0.001, padding='SAME',
                               group_id=0, trainable=True):
    """ Apply a convolutional layer (without bias). """
    c_i = _get_shape(x)[-1]
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    with tf.variable_scope(name) as scope:
        w = weight_module('weights', [k_h, k_w, c_i, c_o], init=init_w, stddev=stddev, group_id=group_id,
                          trainable=trainable)
        z = convolve(x, w)
    return z

def atrous_convolution_module(x, k_h, k_w, c_o, rate, name, init_w='normal', init_b=0, stddev=0.001, padding='SAME', re_use=None,
                group_id=0, trainable=True):
    """ Apply a convolutional layer (with bias). """
    c_i = _get_shape(x)[-1]
    convolve = lambda i, k: tf.nn.atrous_conv2d(i, k, rate, padding=padding)

    with tf.variable_scope(name, reuse=re_use) as scope:
        w = weight_module('weights', [k_h, k_w, c_i, c_o], init=init_w, stddev=stddev, group_id=group_id,
                          trainable=trainable)
        z = convolve(x, w)
        b = bias_module('biases', c_o, init_b, trainable=trainable)
        z = tf.nn.bias_add(z, b)
    return z


'''
    Fully connected layer to be used for prediction/segmentation/classification
'''
def fully_connected_module(x, output_size, name, init_w='normal', init_b=0, stddev=0.001, group_id=0, trainable=True):
    """ Apply a fully-connected layer (with bias). """
    x_shape = _get_shape(x)
    input_dim = x_shape[-1]

    with tf.variable_scope(name) as scope:
        w = weight_module('weights', [input_dim, output_size], init=init_w, stddev=stddev, group_id=group_id,
                          trainable=trainable)
        b = bias_module('biases', [output_size], init_b, trainable=trainable)
        z = tf.nn.xw_plus_b(x, w, b)
    return z

'''
    Fully connected layer to be used for prediction/segmentation/classification
    no bias to be used
'''
def fully_connected_no_bias(x, output_size, name, init_w='normal', stddev=0.001, group_id=0, trainable=True):
    """ Apply a fully-connected layer (without bias). """
    x_shape = _get_shape(x)
    input_dim = x_shape[-1]

    with tf.variable_scope(name) as scope:
        w = weight_module('weights', [input_dim, output_size], init=init_w, stddev=stddev,
                          group_id=group_id, trainable=trainable)
        z = tf.matmul(x, w)
    return z

'''
    In case we want to use batch normalization in the network to be trained
'''
def batch_norm_module(x, name, is_train, bn=True, nl='relu', trainable=True):
    """ Apply a batch normalization layer and a nonlinearity layer. """
    if bn:
        x = _batch_norm_module(x, name, is_train)
    x = nonlinear_module(x, nl)
    return x

def _batch_norm_module(x, name, is_train):
    """ Apply a batch normalization layer. """
    with tf.variable_scope(name):
        inputs_shape = x.get_shape()
        axis = list(range(len(inputs_shape) - 1))
        param_shape = int(inputs_shape[-1])

        moving_mean = tf.get_variable('mean', [param_shape], initializer=tf.constant_initializer(0.0), trainable=is_train)
        moving_var = tf.get_variable('variance', [param_shape], initializer=tf.constant_initializer(1.0), trainable=is_train)

        beta = tf.get_variable('offset', [param_shape], initializer=tf.constant_initializer(0.0))
        gamma = tf.get_variable('scale', [param_shape], initializer=tf.constant_initializer(1.0))

        control_inputs = []

        def mean_var_with_update():
            mean, var = tf.nn.moments(x, axis)
            update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, 0.99)
            update_moving_var = moving_averages.assign_moving_average(moving_var, var, 0.99)
            control_inputs = [update_moving_mean, update_moving_var]
            return tf.identity(mean), tf.identity(var)

        def mean_var():
            mean = moving_mean
            var = moving_var
            return tf.identity(mean), tf.identity(var)

        mean, var = tf.cond(is_train, mean_var_with_update, mean_var)

        with tf.control_dependencies(control_inputs):
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

    return normed

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
    In case we want to use other architectures such as FastMask
'''
def neck_module(x, c_o1, c_o2, re_use, groupid, trainable=True):
    net = convolution_module(x, 3, 3, c_o1, 1, 1, 'resneck3x3', re_use=re_use, group_id=groupid, trainable=trainable)
    net = convolution_module(net, 1, 1, c_o2, 1, 1, 'resneck1x1', re_use=re_use, group_id=groupid, trainable=trainable)
    net = avg_pool(net, 2, 2, 2, 2, 'resneckavgpool')

    net1 = avg_pool(x, 2, 2, 2, 2, 'neckavgpool')
    net = tf.add(net, net1)
    return net

'''
    Squeeznet architecture main module
'''
def fire_module(x, c_s, c_e, name, padd1='SAME', padd2='SAME', padd3='SAME', trainable=True):
    net_s = convolution_module(x, 1, 1, c_s, 1, 1, name + '_squeeze1x1', padding=padd1, trainable=trainable)
    net_s = nonlinear_module(net_s, name + '_s1x1_relu')
    net_e1x1 = convolution_module(net_s, 1, 1, c_e, 1, 1, name + '_expand1x1', padding=padd2, trainable=trainable)
    net_e1x1 = nonlinear_module(net_e1x1, name + '_e1x1_relu')
    net_e3x3 = convolution_module(net_s, 3, 3, c_e, 1, 1, name + '_expand3x3', padding=padd3, trainable=trainable)
    net_e3x3 = nonlinear_module(net_e3x3, name + '_e3x3_relu')
    net = tf.concat(axis=3, values=[net_e1x1, net_e3x3], name=(name + '_concat'))
    return net, net_s, net_e1x1, net_e3x3

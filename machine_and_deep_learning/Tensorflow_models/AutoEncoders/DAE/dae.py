"""
    Denoising Autoencoders implementation in Tensorflow, for defect detection

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
"""
import tensorflow as tf
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from glob import glob
import os
from scipy import misc
import time
import matplotlib.patches as rect
import matplotlib.colors as colors
import cv2
import json
import argparse
import ntpath
import cv2
from PIL import ImageEnhance
from scipy.signal.signaltools import convolve2d
from scipy.signal.windows import gaussian

parser = argparse.ArgumentParser(description='Autoencoders implementation in Tensorflow')
def parse_args():
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device Id to be used for processing [0]',
                        default=0, type=int)
    parser.add_argument('--mode', dest='execution_mode',
                        help='mode in which the program to be executed (train/test/visualize model) [train]',
                        default='train', type=str)
    parser.add_argument('--test_img', dest='test_image',
                        help='test image, default [None]',
                        default=None, type=str)
    parser.add_argument('--thres', dest='threshold', help='threshold defect',
                        default=10000, type=float)
    parser.add_argument('--model', dest='model', help='model to be loaded for testing', default='models/dae-99',
                        type=str)
    parser.add_argument('--load_checkpoint', dest='load_checkpoint', help='model checkpoint to be used, '
                                                                          'FLAG true or false, default is false',
                        default='false', type=str)
    parser.add_argument('--output_file', dest='output_file', help='file name to be used to output visualization '
                                                                  'results',
                        default="results/final_results.bmp", type=str)
    parser.add_argument('--confidence', dest='confidence', help='threshold defect % confidence, default 0.85',
                        default=0.85, type=float)
    parser.add_argument('--nruns', dest='nruns', help='statistical analysis',
                        default=10, type=int)
    return parser.parse_args()

# different sizes for the input image
ACTIVATE_2K = True
ACTIVATE_4K = False
ACTIVATE_8K = False

# uses half precision for the network only not postprocessing
USE_HALF = False

# trace performance
DUMP_COMPUTING_TIME = False
if DUMP_COMPUTING_TIME:
    from tensorflow.python.client import timeline

patch_size_x = 128
patch_size_y = 100
N_CHANNELS = 1

# default is 4K
img_height = 2000
img_width = 4096

# resetting image resolution
if ACTIVATE_2K:
    rescale_x = img_width / 2
    rescale_y = img_height / 2
elif ACTIVATE_4K:
    rescale_x = img_width
    rescale_y = img_height
elif ACTIVATE_8K:
    rescale_x = img_width * 2
    rescale_y = img_height * 2

'''
    Defining the number of channels we have in order to determine
    how many patches are to be extracted later
'''
if N_CHANNELS == 1:
    n_patches_global = ((rescale_x / patch_size_x) * (rescale_y / patch_size_y)) * 3
else:
    n_patches_global = ((rescale_x / patch_size_x) * (rescale_y / patch_size_y))

# to avoid artifacts at the border of each patch
border_mask = np.zeros((patch_size_y, patch_size_x, 3), np.float)

for row in range(patch_size_y):
    for col in range(patch_size_x):
        if row > 1 and col > 1:
            border_mask[row, col, 0] = 1.
            border_mask[row, col, 1] = 1.
            border_mask[row, col, 2] = 1.

# creating a small patch with noise
def masking_noise(X, f):

    X_noise = X.copy()
    n_samples = X.shape[0]

    for i in range(n_samples):
        for j in range(f):
            # leave limits of the patch alone
            k = np.random.randint(4,32)

            # choosing the patch color for all the samples
            if N_CHANNELS == 1:
                mask_Y = np.ones(shape=[k, k], dtype=np.uint8)
                mask_Y = mask_Y * np.random.randint(0, 255)
            else:
                mask_r = np.ones(shape=[k, k], dtype=np.uint8)
                mask_g = np.ones(shape=[k, k], dtype=np.uint8)
                mask_b = np.ones(shape=[k, k], dtype=np.uint8)

                mask_r = mask_r * np.random.randint(0, 255)
                mask_g = mask_g * np.random.randint(0, 255)
                mask_b = mask_b * np.random.randint(0, 255)

            r = np.random.randint(0,patch_size_y - k)
            c = np.random.randint(0,patch_size_x - k)

            if N_CHANNELS == 1:
                X_noise[i, r:r + k, c:c + k, 0] = \
                    (np.multiply(X_noise[i,r:r+k,c:c+k, 0], 0.6) + np.multiply(mask_Y, 0.4)) / 2
            else:
                X_noise[i,r:r+k,c:c+k, 0] = \
                    (np.multiply(X_noise[i,r:r+k,c:c+k, 0], 0.6) + np.multiply(mask_r, 0.4)) / 2
                X_noise[i, r:r + k, c:c + k, 1] = \
                    (np.multiply(X_noise[i,r:r+k,c:c+k, 0], 0.6) + np.multiply(mask_g, 0.4)) / 2
                X_noise[i, r:r + k, c:c + k, 2] = \
                    (np.multiply(X_noise[i,r:r+k,c:c+k, 0], 0.6) + np.multiply(mask_b, 0.4)) / 2

    return X_noise

# handles image transformation
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# saves the current status of the model
def save(checkpoint_dir, step, sess, saver, model_name):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess,os.path.join(checkpoint_dir, model_name),global_step=step)

# reads previously saved models
def load(checkpoint_dir, sess, saver, restore_particular_checkpoint=False, checkpointpath=None, istrain=False):
    print(" [*] Reading checkpoints...")

    if restore_particular_checkpoint == False:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False
    else:
        if checkpointpath is not None:
            global_step = checkpointpath.split('/')[-1].split('-')[-1]
            saver.restore(sess, checkpointpath)
            print('Model at step %d has been successfully restored' % int(global_step))
            if istrain:
                return int(global_step)
            else:
                return True
        else:
            return False






# ============================================== FOR TRAINING
def lrelu(x, leak=0.2, name="lrelu"):
    """Leaky rectifier.
    Parameters
    ----------
    x : Tensor
        The tensor to apply the nonlinearity to.
    leak : float, optional
        Leakage parameter.
    name : str, optional
        Variable scope to use.
    Returns
    -------
    x : Tensor
        Output of the nonlinearity.
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

# %%
def GaussianNoise(X, sd=0.5):
    # Injecting small gaussian noise
    X += np.random.normal(0, sd, X.shape)
    return X

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y


def autoencoder_train(input_shape=[None, patch_size_y, patch_size_x, N_CHANNELS],
                n_filters=[N_CHANNELS, 16, 32, 64, 128],
                filter_sizes=[3, 3, 3, 3, 3],
                filter_shape = [9, 9, None, 16],
                corruption=True):
    """Build a deep denoising autoencoder w/ tied weights.
    """
    # %%
    # input to the network

    # To reduce memory transferring from CPU to GPU
    x_temp = tf.placeholder(tf.uint8, input_shape, name='x_int8')
    x_noise_temp = tf.placeholder(tf.uint8, input_shape, name='x_noise_int8')

    x = tf.cast(x_temp, dtype=tf.float32, name='x')
    x_noise = tf.cast(x_noise_temp, dtype=tf.float32, name='x_noise')

    x = tf.subtract(tf.divide(x, tf.ones_like(x)*127.5), tf.ones_like(x)) #np.array(batch_xs) / 127.5 - 1.
    x_noise = tf.subtract(tf.divide(x_noise, tf.ones_like(x_noise) * 127.5), tf.ones_like(x_noise))  # np.array(batch_xs_noise) / 127.5 - 1.

    # %%
    # ensure 2-d is converted to square tensor.
    if len(x.get_shape()) == 2:
        x_dim = np.sqrt(x.get_shape().as_list()[1])
        if x_dim != int(x_dim):
            raise ValueError('Unsupported input dimensions')
        x_dim = int(x_dim)
        x_tensor = tf.reshape(
            x, [-1, x_dim, x_dim, n_filters[0]])
    elif len(x.get_shape()) == 4:
        x_tensor = x
    else:
        raise ValueError('Unsupported input dimensions')

    current_input = x_tensor

    # %%
    # Optionally apply denoising autoencoder
    if corruption:
        current_input = x_noise
    #    current_input = masking_noise(current_input, 10, 16)

    tf.summary.image("Clean_input", x_tensor, max_outputs=2)
    tf.summary.image("Corrupted_input", current_input, max_outputs=2)

    # %%
    # Build the encoder
    encoder = []
    shapes = []
    for layer_i, n_output in enumerate(n_filters[1:]):
        n_input = current_input.get_shape().as_list()[3]
        shapes.append(current_input.get_shape().as_list())

        # # Dilated Convolution Stack
        # W_A_1 = tf.Variable(tf.random_uniform([filter_sizes[layer_i], filter_sizes[layer_i], n_input, n_input],
        #                                       -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
        # current_input = tf.nn.atrous_conv2d(value=current_input, filters=W_A_1, rate=2, padding='SAME',
        #                     name=('dilated_encoder_' + str(layer_i)+'_a'))
        # print 'Atrous_1', layer_i, ":", current_input.get_shape()
        #
        # W_A_2 = tf.Variable(tf.random_uniform([filter_sizes[layer_i], filter_sizes[layer_i], n_input, n_input],
        #                                       -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
        # current_input = tf.nn.atrous_conv2d(value=current_input, filters=W_A_2, rate=2, padding='SAME',
        #                                     name=('dilated_encoder_' + str(layer_i)+'_b'))
        # print 'Atrous_2', layer_i, ":", current_input.get_shape()
        #
        # W_A_3 = tf.Variable(tf.random_uniform([filter_sizes[layer_i], filter_sizes[layer_i], n_input, n_input],
        #                                       -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
        # current_input = tf.nn.atrous_conv2d(value=current_input, filters=W_A_3, rate=2, padding='SAME',
        #                                     name=('dilated_encoder_' + str(layer_i)+'_c'))
        # print 'Atrous_3', layer_i, ":", current_input.get_shape()


        W_A_4 = tf.Variable(tf.random_uniform([filter_sizes[layer_i], filter_sizes[layer_i], n_input, n_output],
                                              -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
        # W = tf.Variable(tf.random_uniform([filter_sizes[layer_i], filter_sizes[layer_i], n_input, n_output],
        #                                   -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W_A_4)
        output = lrelu(tf.add(tf.nn.conv2d(current_input, W_A_4, strides=[1, 2, 2, 1], padding='SAME'), b),
                       name='encoder_'+str(layer_i))

        print 'encoder', layer_i, ":", output.get_shape()

        current_input = output


    #current_input, W, strides=[1, 2, 2, 1], padding='SAME')
    # %% tf.nn.atrous_conv2d(i, k, rate, padding=padding)
    # store the latent representation
    z = current_input
    encoder.reverse()
    shapes.reverse()

    # %%
    # Build the decoder using the same weights
    for layer_i, shape in enumerate(shapes):
        W = encoder[layer_i]
        b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
        output = lrelu(tf.add(tf.nn.conv2d_transpose(current_input, W,
                                                     tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                                                     strides=[1, 2, 2, 1], padding='SAME'), b),
                       name='decoder_'+str(layer_i))
        print 'decoder', layer_i, ":", output.get_shape()
        current_input = output

    # %%
    # now have the reconstruction through the network
    y = current_input
    tf.summary.image("reconstructed_image", y, max_outputs=2)
    diff = tf.abs(y-x_tensor)
    tf.summary.image("diff image", diff, max_outputs=2)

    # cost function measures pixel-wise difference
    cost = tf.reduce_sum(tf.square(y - x_tensor))
    tf.summary.scalar("Loss", cost)

    # To reduce memory transferring from GPU to CPU
    y = (y + 1.) / 2.
    y = tf.cast(tf.minimum(tf.multiply(y, tf.ones_like(y)*255.), 255.), dtype=tf.uint8)

    # %%
    return {'x': x, 'x_noise':x_noise, 'x_int8': x_temp, 'x_noise_int8':x_noise_temp, 'z': z, 'y': y, 'cost': cost}






# %%
def train(arguments):
    """Train the convolutional autoencder."""
    # %%


    data = glob(os.path.join("training_dataset/", "*.jpg"))
    # data = glob(os.path.join("training_dataset/good_samples_100x128_2k_rgb", "*.jpg"))
    #mean_img = np.mean(mnist.train.images, axis=0)
    ae = autoencoder_train(corruption=True)

    learning_rate = tf.placeholder(tf.float32)
    # %%
    #learning_rate = 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    logs_dir = "logs/"

    # %%
    # We create a session to use the graph
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25, allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(logs_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=100)

    if arguments.load_checkpoint == 'true':
        print("====================> Loading latest checkpoint ...")
        global_step = load("models/", sess, saver, restore_particular_checkpoint=True,
                        checkpointpath=arguments.model, istrain=True)

    # %%
    # Fit all training data
    batch_size = 10
    n_epochs = 100

    itr = 0
    total_time = 0
    for epoch_i in range(n_epochs):
        total_batch_i = len(data) // batch_size
        s = time.time()
        for batch_i in range(total_batch_i):
            batch_files = data[batch_i * batch_size:(batch_i + 1) * batch_size]

            if N_CHANNELS == 1:
                batch_xs = [np.expand_dims(misc.imread(batch_file, mode='L').astype(np.float), axis=3)
                         for batch_file in batch_files]
            else:
                batch_xs = [misc.imread(batch_file, mode='RGB').astype(np.float) for batch_file in batch_files]

            #batch_xs = [prewhiten(batch_file)
            #            for batch_file in batch_xs]

            batch_xs_noise = masking_noise(np.array(batch_xs), 10)
            # batch_xs = np.array(batch_xs) / 127.5 - 1.
            #batch_xs = global_contrast_normalization(batch_xs, 1, 10, 0.000000001)
            train = np.array(batch_xs).astype(np.uint8)

            # batch_xs_noise = np.array(batch_xs_noise) / 127.5 - 1.
            #batch_xs_noise = global_contrast_normalization(batch_xs_noise, 1, 10, 0.000000001)
            train_noise = np.array(batch_xs_noise).astype(np.uint8)

            # print(np.max(train), np.min(train))
            # print(np.max(train_noise), np.min(train_noise))

            if epoch_i < 10:
                #train_noise = batch_xs.astype(np.float32)
                #train = np.array([img - mean_img for img in batch_xs])
                sess.run(optimizer, feed_dict={ae['x_int8']: train, ae['x_noise_int8']:train_noise, learning_rate:0.001})

                if (itr % 50 == 0) and itr > 0:
                    cost, lr, summary_str = sess.run([ae['cost'], learning_rate, summary_op],
                                                     feed_dict={ae['x_int8']: train, ae['x_noise_int8']:train_noise,
                                                                learning_rate:0.001})
                else:
                    cost, lr = sess.run([ae['cost'], learning_rate],
                                                     feed_dict={ae['x_int8']: train, ae['x_noise_int8']:train_noise,
                                                                learning_rate:0.001})
            else:

                # train = np.array([img - mean_img for img in batch_xs])
                sess.run(optimizer, feed_dict={ae['x_int8']: train, ae['x_noise_int8']:train_noise,learning_rate: 0.0001})

                if (itr % 50 == 0) and itr > 0:
                    cost, lr, summary_str = sess.run([ae['cost'], learning_rate, summary_op],
                                                     feed_dict={ae['x_int8']: train, ae['x_noise_int8']: train_noise,
                                                                learning_rate: 0.0001})
                else:
                    cost, lr = sess.run([ae['cost'], learning_rate],
                                                     feed_dict={ae['x_int8']: train, ae['x_noise_int8']: train_noise,
                                                                learning_rate: 0.0001})

            print("epoch_i: %d [%d/%d], loss: %g, lr: %f" % (epoch_i, batch_i, total_batch_i, cost, lr))

            if (itr % 50 == 0) and itr > 0:
                summary_writer.add_summary(summary_str, itr)

            itr += 1
        e = time.time() - s
        total_time += e
        print("time: %s" % total_time)
        if arguments.load_checkpoint == 'true':
            global_step += 1
            save("models/", global_step, sess, saver, "dae")
        else:
            save("models/", epoch_i, sess, saver, "dae")

        tf.train.write_graph(sess.graph.as_graph_def(), "models/", "graph_dae.pb", as_text=True)
# ============================================== FOR TRAINING


def autoencoder_inference(input_shape=[n_patches_global, patch_size_y, patch_size_x, N_CHANNELS],
                      n_filters=[N_CHANNELS, 16, 32, 64, 128],
                      filter_sizes=[3, 3, 3, 3, 3],
                      filter_shape=[9, 9, None, 16],
                      corruption=True):
    """Build a deep denoising autoencoder w/ tied weights.
    """
    # %%
    # input to the network

    # To reduce memory transferring from CPU to GPU
    x_temp = tf.placeholder(tf.uint8, input_shape, name='x_int8')
    x_noise_temp = tf.placeholder(tf.uint8, input_shape, name='x_noise_int8')

    if ACTIVATE_4K or ACTIVATE_8K or USE_HALF:
        x = tf.cast(x_temp, dtype=tf.half, name='x')
        x_noise = tf.cast(x_noise_temp, dtype=tf.half, name='x_noise')
    else:
        x = tf.cast(x_temp, dtype=tf.float32, name='x')
        x_noise = tf.cast(x_noise_temp, dtype=tf.float32, name='x_noise')

    x = tf.subtract(tf.divide(x, tf.ones_like(x) * 127.5), tf.ones_like(x))  # np.array(batch_xs) / 127.5 - 1.
    x_noise = tf.subtract(tf.divide(x_noise, tf.ones_like(x_noise) * 127.5),
                          tf.ones_like(x_noise))  # np.array(batch_xs_noise) / 127.5 - 1.

    # %%
    # ensure 2-d is converted to square tensor.
    if len(x.get_shape()) == 2:
        x_dim = np.sqrt(x.get_shape().as_list()[1])
        if x_dim != int(x_dim):
            raise ValueError('Unsupported input dimensions')
        x_dim = int(x_dim)
        x_tensor = tf.reshape(
            x, [-1, x_dim, x_dim, n_filters[0]])
    elif len(x.get_shape()) == 4:
        x_tensor = x
    else:
        raise ValueError('Unsupported input dimensions')

    current_input = x_tensor

    # %%
    # Optionally apply denoising autoencoder
    if corruption:
        current_input = x_noise
    # current_input = masking_noise(current_input, 10, 16)

    # %%
    # Build the encoder
    encoder = []
    shapes = []
    for layer_i, n_output in enumerate(n_filters[1:]):
        n_input = current_input.get_shape().as_list()[3]
        shapes.append(current_input.get_shape().as_list())

        # # Dilated Convolution Stack
        # W_A_1 = tf.Variable(tf.random_uniform([filter_sizes[layer_i], filter_sizes[layer_i], n_input, n_input],
        #                                       -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
        # current_input = tf.nn.atrous_conv2d(value=current_input, filters=W_A_1, rate=2, padding='SAME',
        #                     name=('dilated_encoder_' + str(layer_i)+'_a'))
        # print 'Atrous_1', layer_i, ":", current_input.get_shape()
        #
        # W_A_2 = tf.Variable(tf.random_uniform([filter_sizes[layer_i], filter_sizes[layer_i], n_input, n_input],
        #                                       -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
        # current_input = tf.nn.atrous_conv2d(value=current_input, filters=W_A_2, rate=2, padding='SAME',
        #                                     name=('dilated_encoder_' + str(layer_i)+'_b'))
        # print 'Atrous_2', layer_i, ":", current_input.get_shape()
        #
        # W_A_3 = tf.Variable(tf.random_uniform([filter_sizes[layer_i], filter_sizes[layer_i], n_input, n_input],
        #                                       -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
        # current_input = tf.nn.atrous_conv2d(value=current_input, filters=W_A_3, rate=2, padding='SAME',
        #                                     name=('dilated_encoder_' + str(layer_i)+'_c'))
        # print 'Atrous_3', layer_i, ":", current_input.get_shape()


        W_A_4 = tf.Variable(tf.random_uniform([filter_sizes[layer_i], filter_sizes[layer_i], n_input, n_output],
                                              -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
        # W = tf.Variable(tf.random_uniform([filter_sizes[layer_i], filter_sizes[layer_i], n_input, n_output],
        #                                   -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))

        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W_A_4)

        # casting to speed up and reduce memory footprint
        if ACTIVATE_4K or ACTIVATE_8K or USE_HALF:
            W_A_4 = tf.cast(W_A_4, dtype=tf.half)
            b = tf.cast(b, dtype=tf.half)

        output = lrelu(tf.add(tf.nn.conv2d(current_input, W_A_4, strides=[1, 2, 2, 1], padding='SAME'), b),
                       name='encoder_' + str(layer_i))

        print 'encoder', layer_i, ":", output.get_shape()

        current_input = output

    # %%
    # store the latent representation
    z = current_input
    encoder.reverse()
    shapes.reverse()

    # %%
    # Build the decoder using the same weights
    for layer_i, shape in enumerate(shapes):
        W = encoder[layer_i]
        b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))

        # casting to improve performance
        if ACTIVATE_4K or ACTIVATE_8K or USE_HALF:
            W = tf.cast(W, dtype=tf.half)
            b = tf.cast(b, dtype=tf.half)


        output = lrelu(tf.add(tf.nn.conv2d_transpose(current_input, W, tf.stack([tf.shape(x)[0],
                                                                                      shape[1], shape[2], shape[3]]),
                                                          strides=[1, 2, 2, 1], padding='SAME'), b),
                            name='decoder_' + str(layer_i))

        print 'decoder', layer_i, ":", output.get_shape()
        current_input = output

    # %%
    # now have the reconstruction through the network
    y = current_input
    diff = tf.abs(y - x_tensor)

    # cost function measures pixel-wise difference
    cost = tf.reduce_sum(tf.square(y - x_tensor))

    # To reduce memory transferring from GPU to CPU
    y = (y + 1.) / 2.
    y = tf.cast(tf.minimum(tf.multiply(y, tf.ones_like(y) * 255.), 255.), dtype=tf.uint8)

    if N_CHANNELS == 1:  # GRAY IMAGES or PATCHES
        '''
            Perform the postprocessing in this part of the graph and since they are not weights
            we dont need retrain just activate this part of the code during inference
            recon = recon / 127.5 - 1
            patches = patches / 127.5 - 1
        '''
        temp_input_x = (x + 1.) / 2.
        temp_input_x = tf.cast(tf.minimum(tf.multiply(temp_input_x,
                                                      tf.ones_like(temp_input_x) * 255.), 255.), dtype=tf.uint8)

        shift_patches = temp_input_x.get_shape()[0] / 3
        rgb_input_datalist = []
        rgb_recon_datalist = []

        # original input image
        rgb_input_datalist.append(temp_input_x[0:shift_patches, :, :, 0])
        rgb_input_datalist.append(temp_input_x[shift_patches:shift_patches * 2, :, :, 0])
        rgb_input_datalist.append(temp_input_x[shift_patches * 2:shift_patches * 3, :, :, 0])

        # reconstruction image
        rgb_recon_datalist.append(y[0:shift_patches, :, :, 0])
        rgb_recon_datalist.append(y[shift_patches:shift_patches * 2, :, :, 0])
        rgb_recon_datalist.append(y[shift_patches * 2:shift_patches * 3, :, :, 0])

        # put back together all the channels
        original_image_patches = tf.cast(tf.stack(rgb_input_datalist, axis=3), dtype=tf.float32)
        reconstruction_image_patches = tf.cast(tf.stack(rgb_recon_datalist, axis=3), dtype=tf.float32)

        # normalize the channels in order to compute differences
        raw_input_converted = original_image_patches / 127.5 - 1
        reconstruction_converted = reconstruction_image_patches / 127.5 - 1

        # masking difference to avoid false alarms in patch borders
        difference = tf.multiply(tf.abs(raw_input_converted - reconstruction_converted), border_mask)

        # ===================================================
        # Mapping patches to single Image
        patch_id = 0
        rows_original = []
        rows_reconstructed = []
        for yid in range(rescale_y / patch_size_y):
            for xid in range((rescale_x / patch_size_x)):
                if xid == 0:
                    cols_original = tf.concat(values=(original_image_patches[patch_id],
                                                      original_image_patches[patch_id + 1]), axis=1)
                    cols_reconstructed = tf.concat(values=(difference[patch_id],
                                                           difference[patch_id + 1]), axis=1)
                elif xid > 1:
                    cols_original = tf.concat(values=(cols_original, original_image_patches[patch_id]), axis=1)
                    cols_reconstructed = tf.concat(values=(cols_reconstructed, difference[patch_id]),
                                                   axis=1)
                patch_id += 1

            rows_original.append(cols_original)
            rows_reconstructed.append(cols_reconstructed)

        for rows_id in range(len(rows_original)):
            if rows_id == 0:
                original_image = tf.concat(values=(rows_original[rows_id], rows_original[rows_id + 1]), axis=0)
                reconstructed_image = tf.concat(
                    values=(rows_reconstructed[rows_id], rows_reconstructed[rows_id + 1]),
                    axis=0)
            elif rows_id > 1:
                original_image = tf.concat(values=(original_image, rows_original[rows_id]), axis=0)
                reconstructed_image = tf.concat(values=(reconstructed_image, rows_reconstructed[rows_id]),
                                                axis=0)

        original_image = tf.cast(original_image, tf.uint8)
        reconstructed_image = tf.cast((((reconstructed_image + 1.) / 2.) * 255.), tf.uint8)
        # ===================================================


        # calculating the total difference per patch which is going to tell me where to put attention
        # after using Gabor Filter banks
        diff_scores = tf.reduce_mean(tf.reshape(difference, [-1, patch_size_y * patch_size_x * 3]), 1)

        filters = tf.placeholder(tf.float32, filter_shape, name='filters')

        # =============================================== Extract Defect
        grayscaled = tf.image.rgb_to_grayscale(reconstructed_image, name='gray_scaler')

        temp_gray = tf.expand_dims(grayscaled, axis=0)

        # NMS ROUND 1
        # Gabor
        gabor_filtered1 = tf.nn.conv2d(input=tf.cast(temp_gray, dtype=tf.float32), filter=filters, strides=[1, 1, 1, 1],
                                       padding='SAME', name="Gabor_1")
        gabor_filtered1 = tf.reduce_max(gabor_filtered1, axis=3, keep_dims=True)
        to_filter = tf.identity(gabor_filtered1)  # to be eroded and dilated several times

        # Erode
        kernel1 = tf.ones((3, 3, 1), tf.float32)
        erode1 = tf.nn.erosion2d(to_filter, kernel=kernel1, strides=[1, 1, 1, 1], padding='SAME',
                                 rates=[1, 1, 1, 1], name="Erode_1")

        # Dilate
        kernel2 = tf.ones((7, 7, 1), tf.float32)
        dilate1 = tf.nn.dilation2d(erode1, filter=kernel2, strides=[1, 1, 1, 1], padding='SAME',
                                   rates=[1, 1, 1, 1], name="Dilate_1")

        # NMS ROUND 2
        # Gabor
        gabor_filtered2 = tf.nn.conv2d(input=tf.cast(dilate1, dtype=tf.float32), filter=filters, strides=[1, 1, 1, 1],
                                       padding='SAME', name="Gabor_2")
        gabor_filtered2 = tf.reduce_max(gabor_filtered2, axis=3, keep_dims=True)
        to_filter2 = tf.identity(gabor_filtered2)  # to be eroded and dilated several times

        # Erode
        kernel3 = tf.ones((7, 7, 1), tf.float32)
        erode2 = tf.nn.erosion2d(to_filter2, kernel=kernel3, strides=[1, 1, 1, 1], padding='SAME',
                                 rates=[1, 1, 1, 1], name="Erode_2")

        # normalization
        tensor_to_normalize = tf.squeeze(erode2[0], axis=2)
        max_val = tf.reduce_max(tensor_to_normalize)
        min_val = tf.reduce_min(tensor_to_normalize)

        tensor_to_normalize = (tensor_to_normalize - min_val) / (max_val - min_val)
        tensor_to_normalize *= 255

        # histogram
        histogram = tf.histogram_fixed_width(values=tf.to_float(tensor_to_normalize),
                                             value_range=[0., 255.], nbins=255, name="Histogram_extractor")
        # ========================================================================================

        # Thresholding image
        theshold_percent = tf.placeholder(tf.float32, name='theshold_percent')

        histo = tf.cast(histogram, tf.float32)
        histo = histo / tf.reduce_sum(histo) # normalizing

        accumulator = tf.cumsum(histo, name="Prob_accumulator")
        locations = (tf.where(tf.greater_equal(accumulator, theshold_percent)))
        threshold_from_histo = locations[0]
        threshold_from_histo = threshold_from_histo[0] + 1
        accumulator = accumulator[threshold_from_histo]

        thresholded_image = tf.where(
            tf.greater_equal(tensor_to_normalize, tf.cast(threshold_from_histo, tf.float32)),
            255 * tf.ones_like(tensor_to_normalize), tf.zeros_like(tensor_to_normalize))

        kernelfinal = tf.ones((11, 11, 1), tf.float32)
        thresholded_image = tf.nn.erosion2d(tf.expand_dims(tf.expand_dims(thresholded_image, axis=0), axis=3),
                                            kernel=kernelfinal, strides=[1, 1, 1, 1], padding='SAME',
                                            rates=[1, 1, 1, 1], name="Erode_3")

        thresholded_image = tf.cast(tf.squeeze(thresholded_image[0], axis=2), tf.uint8)

        return {'x': temp_input_x, 'x_noise': x_noise, 'x_int8': x_temp, 'x_noise_int8': x_noise_temp, 'z': z,
                'y': y, 'cost': cost, 'input_image': original_image_patches,
                'reconstruction': reconstruction_image_patches,
                'differences': difference, 'diff_scores': diff_scores, 'original_image': original_image,
                'reconstructed_image': reconstructed_image, 'filters': filters, 'grayscaled': grayscaled,
                'gabor_filtered1': gabor_filtered1[0], 'erode1': erode1[0], 'dilate1': dilate1[0],
                'gabor_filtered2': gabor_filtered2[0], 'erode2': erode2[0], 'histogram': histogram,
                'tensor_to_normalize': tensor_to_normalize, 'threshold_confidence': theshold_percent,
                'theshold_percent': theshold_percent, 'thresholded_image': thresholded_image,
                'threshold_from_histo': threshold_from_histo, 'aread_accumulated': accumulator
                }
    else: # RGB images
        '''
            Perform the postprocessing in this part of the graph and since they are not weights
            we dont need retrain just activate this part of the code during inference
            recon = recon / 127.5 - 1
            patches = patches / 127.5 - 1
        '''
        temp_input_x = (x + 1.) / 2.
        temp_input_x = tf.cast(tf.minimum(tf.multiply(temp_input_x,
                                                      tf.ones_like(temp_input_x) * 255.), 255.), dtype=tf.uint8)

        # put back together all the channels they are in RGB already
        original_image_patches = tf.cast(temp_input_x, dtype=tf.float32)
        reconstruction_image_patches = tf.cast(y, dtype=tf.float32)

        # normalize the channels in order to compute differences
        raw_input_converted = original_image_patches / 127.5 - 1
        reconstruction_converted = reconstruction_image_patches / 127.5 - 1

        # masking difference to avoid false alarms in patch borders
        difference = tf.multiply(tf.abs(raw_input_converted - reconstruction_converted), border_mask)/3

        # ===================================================
        # Mapping patches to single Image
        patch_id = 0
        rows_original = []
        rows_reconstructed = []
        for yid in range(rescale_y / patch_size_y):
            for xid in range((rescale_x / patch_size_x)):
                if xid == 0:
                    cols_original = tf.concat(values=(original_image_patches[patch_id],
                                                      original_image_patches[patch_id + 1]), axis=1)
                    cols_reconstructed = tf.concat(values=(difference[patch_id],
                                                           difference[patch_id + 1]), axis=1)
                elif xid > 1:
                    cols_original = tf.concat(values=(cols_original, original_image_patches[patch_id]), axis=1)
                    cols_reconstructed = tf.concat(values=(cols_reconstructed, difference[patch_id]),
                                                   axis=1)
                patch_id += 1

            rows_original.append(cols_original)
            rows_reconstructed.append(cols_reconstructed)

        for rows_id in range(len(rows_original)):
            if rows_id == 0:
                original_image = tf.concat(values=(rows_original[rows_id], rows_original[rows_id + 1]), axis=0)
                reconstructed_image = tf.concat(
                    values=(rows_reconstructed[rows_id], rows_reconstructed[rows_id + 1]),
                    axis=0)
            elif rows_id > 1:
                original_image = tf.concat(values=(original_image, rows_original[rows_id]), axis=0)
                reconstructed_image = tf.concat(values=(reconstructed_image, rows_reconstructed[rows_id]),
                                                axis=0)

        original_image = tf.cast(original_image, tf.uint8)
        reconstructed_image = tf.cast((((reconstructed_image + 1.) / 2.) * 255.), tf.uint8)
        # ===================================================


        # calculating the total difference per patch which is going to tell me where to put attention
        # after using Gabor Filter banks
        diff_scores = tf.reduce_mean(tf.reshape(difference, [-1, patch_size_y * patch_size_x * 3]), 1)

        filters = tf.placeholder(tf.float32, filter_shape, name='filters')

        # =============================================== Extract Defect
        grayscaled = tf.image.rgb_to_grayscale(reconstructed_image, name='gray_scaler')

        temp_gray = tf.expand_dims(grayscaled, axis=0)

        # NMS ROUND 1
        # Gabor
        gabor_filtered1 = tf.nn.conv2d(input=tf.cast(temp_gray, dtype=tf.float32), filter=filters, strides=[1, 1, 1, 1],
                                       padding='SAME', name="Gabor_1")
        gabor_filtered1 = tf.reduce_max(gabor_filtered1, axis=3, keep_dims=True)
        to_filter = tf.identity(gabor_filtered1)  # to be eroded and dilated several times

        # Erode
        kernel1 = tf.ones((3, 3, 1), tf.float32)
        erode1 = tf.nn.erosion2d(to_filter, kernel=kernel1, strides=[1, 1, 1, 1], padding='SAME',
                                 rates=[1, 1, 1, 1], name="Erode_1")

        # Dilate
        kernel2 = tf.ones((7, 7, 1), tf.float32)
        dilate1 = tf.nn.dilation2d(erode1, filter=kernel2, strides=[1, 1, 1, 1], padding='SAME',
                                   rates=[1, 1, 1, 1], name="Dilate_1")

        # NMS ROUND 2
        # Gabor
        gabor_filtered2 = tf.nn.conv2d(input=tf.cast(dilate1, dtype=tf.float32), filter=filters, strides=[1, 1, 1, 1],
                                       padding='SAME', name="Gabor_2")
        gabor_filtered2 = tf.reduce_max(gabor_filtered2, axis=3, keep_dims=True)
        to_filter2 = tf.identity(gabor_filtered2)  # to be eroded and dilated several times

        # Erode
        kernel3 = tf.ones((7, 7, 1), tf.float32)
        erode2 = tf.nn.erosion2d(to_filter2, kernel=kernel3, strides=[1, 1, 1, 1], padding='SAME',
                                 rates=[1, 1, 1, 1], name="Erode_2")

        # normalization
        tensor_to_normalize = tf.squeeze(erode2[0], axis=2)
        max_val = tf.reduce_max(tensor_to_normalize)
        min_val = tf.reduce_min(tensor_to_normalize)

        tensor_to_normalize = (tensor_to_normalize - min_val) / (max_val - min_val)
        tensor_to_normalize *= 255

        # histogram
        histogram = tf.histogram_fixed_width(values=tf.to_float(tensor_to_normalize), value_range=[0., 255.],
                                             nbins=255, name="Histogram_extractor")
        # ========================================================================================

        # Thresholding image
        theshold_percent = tf.placeholder(tf.float32, name='theshold_percent')

        histo = tf.cast(histogram, tf.float32)
        histo = histo / tf.reduce_sum(histo)  # normalizing

        accumulator = tf.cumsum(histo, name='Prob_accumulator')
        locations = tf.where(tf.greater_equal(accumulator, theshold_percent))
        threshold_from_histo = locations[0]
        threshold_from_histo = threshold_from_histo[0] + 1
        accumulator = accumulator[threshold_from_histo]

        thresholded_image = tf.where(
            tf.greater_equal(tensor_to_normalize, tf.cast(threshold_from_histo, tf.float32)),
            255 * tf.ones_like(tensor_to_normalize), tf.zeros_like(tensor_to_normalize))

        kernelfinal = tf.ones((11, 11, 1), tf.float32)
        thresholded_image = tf.nn.erosion2d(tf.expand_dims(tf.expand_dims(thresholded_image, axis=0), axis=3),
                                            kernel=kernelfinal, strides=[1, 1, 1, 1], padding='SAME',
                                            rates=[1, 1, 1, 1], name="Erode_3")

        thresholded_image = tf.cast(tf.squeeze(thresholded_image[0], axis=2), tf.uint8)

        return {'x': temp_input_x, 'x_noise': x_noise, 'x_int8': x_temp, 'x_noise_int8': x_noise_temp, 'z': z,
                'y': y, 'cost': cost, 'input_image': original_image_patches,
                'reconstruction': reconstruction_image_patches,
                'differences': difference, 'diff_scores': diff_scores, 'original_image': original_image,
                'reconstructed_image': reconstructed_image, 'filters': filters, 'grayscaled': grayscaled,
                'gabor_filtered1': gabor_filtered1[0], 'erode1': erode1[0], 'dilate1': dilate1[0],
                'gabor_filtered2': gabor_filtered2[0], 'erode2': erode2[0], 'histogram': histogram,
                'tensor_to_normalize': tensor_to_normalize, 'threshold_confidence': theshold_percent,
                'theshold_percent': theshold_percent, 'thresholded_image': thresholded_image,
                'threshold_from_histo': threshold_from_histo, 'aread_accumulated': accumulator
                }


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def build_filters():
    filters = []
    ksize = 9

    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 15.0, theta, 30.0, 1., 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)

    return filters



def inference(testimg, thres, arguments):
    def sliding_window(image, stepSize, windowSize):
        for y in range(0, image.shape[0], stepSize[0]):
            for x in range(0, image.shape[1], stepSize[1]):
                if (y + windowSize[0]) <= (image.shape[0]) and (x + windowSize[1]) <= (image.shape[1]):
                    yield (x, y, image[y:y + windowSize[0], x:x + windowSize[1]])



    img_path = testimg#"samples/"+testimg
    img = misc.imread(img_path,mode='RGB')
    if not ACTIVATE_4K and not ACTIVATE_8K:
        img = misc.imresize(img, (1000, 2048))
    ae = autoencoder_inference(corruption=False)

    # for GABOR
    filters = build_filters()

    if arguments.load_checkpoint == "false":
        # before enhancement
        misc.imsave("results/Before_enhancement.png", img)

    enhanced_image = img.copy()

    if arguments.load_checkpoint == "false":
        # after enhancement
        # enhanced_image = colors.rgb_to_hsv(enhanced_image)
        misc.imsave("results/After_enhancement_R_H.png", enhanced_image[:, :, 0])
        misc.imsave("results/After_enhancement_G_S.png", enhanced_image[:, :, 1])
        misc.imsave("results/After_enhancement_B_V.png", enhanced_image[:, :, 2])

        misc.imsave("results/After_enhancement.png", enhanced_image)

        # sys.exit(0)


    # re-asigning
    img = enhanced_image

    # memory restrictions
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1)

    if DUMP_COMPUTING_TIME:
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

    if arguments.load_checkpoint == "true":
        isLoaded = load("models/latest", sess, saver, restore_particular_checkpoint=True,
                        checkpointpath=arguments.model)
    else:
        isLoaded = load("models/latest", sess, saver)

    assert (isLoaded)

    elapsed_inference = 0.
    elapsed_postprocessing = 0.
    elapsed_total = 0.
    elapsed_patch_extraction = 0.

    # number of runs
    n_runs = arguments.nruns

    for i in range(n_runs):
        total_start = time.time() # Global

        # ===================== PATCHES
        started = time.time()

        patches = []
        xs = []
        ys = []
        for (x, y, window) in sliding_window(img, [patch_size_y, patch_size_x], (patch_size_y, patch_size_x)):
            # patches.append(np.expand_dims(np.array(window),0))
            # print x, y, np.array(window).shape
            # plt.imshow(np.array(window))
            # plt.show()
            patches.append(window)
            xs.append(x)
            ys.append(y)
        patches = np.vstack(patches)
        patches = np.reshape(patches, [-1, patch_size_y, patch_size_x, 3])
        # patches = patches / 127.5 - 1.
        patches = patches.astype(np.uint8)

        if i > 0:
            elapsed_patch_extraction += 1000.0 * (time.time() - started)

        print("Patch Extraction %f ms in run %d" % (1000.0 * (time.time() - started), i))
        # ===================== PATCHES


        # =============================== inference only
        started = time.time()
        if N_CHANNELS==1:
            # concatenating all in a single bunch
            concatenated_patches = np.expand_dims(np.concatenate((patches[:, :, :, 0], patches[:, :, :, 1],
                                                                  patches[:, :, :, 2]), axis=0), axis=3)

            print(concatenated_patches.shape)

            if DUMP_COMPUTING_TIME:
                reconALL, InputAll, score, input_img_patches, recon_img_patches, difference, diff_scores, original_img, \
                reconstructed_img, background, gabor_filtered1, erode1, dilate1, gabor_filtered2, erode2, histogram, \
                tensor_to_normalize, thresholded_image, threshold_from_histo, accumulator = sess.run(
                    [ae['y'], ae['x'], ae['cost'], ae['input_image'], ae['reconstruction'], ae['differences'],
                     ae['diff_scores'], ae['original_image'], ae['reconstructed_image'], ae['grayscaled'],
                     ae['gabor_filtered1'], ae['erode1'], ae['dilate1'], ae['gabor_filtered2'], ae['erode2'],
                     ae['histogram'], ae['tensor_to_normalize'], ae['thresholded_image'], ae['threshold_from_histo'],
                     ae['aread_accumulated']],
                    feed_dict={
                        ae['x_int8']: concatenated_patches,
                        ae['filters']: np.expand_dims(np.swapaxes(filters, 0, 2), axis=2),
                        ae['threshold_confidence']: arguments.confidence
                    },
                    options=options, run_metadata=run_metadata
                )
            else:
                reconALL, InputAll, score, input_img_patches, recon_img_patches, difference, diff_scores, original_img, \
                reconstructed_img, background, gabor_filtered1, erode1, dilate1, gabor_filtered2, erode2, histogram, \
                tensor_to_normalize, thresholded_image, threshold_from_histo, accumulator = sess.run(
                    [ae['y'], ae['x'], ae['cost'], ae['input_image'], ae['reconstruction'], ae['differences'],
                     ae['diff_scores'], ae['original_image'], ae['reconstructed_image'], ae['grayscaled'],
                     ae['gabor_filtered1'], ae['erode1'], ae['dilate1'], ae['gabor_filtered2'], ae['erode2'],
                     ae['histogram'], ae['tensor_to_normalize'], ae['thresholded_image'], ae['threshold_from_histo'],
                     ae['aread_accumulated']],
                    feed_dict={
                        ae['x_int8']: concatenated_patches,
                        ae['filters']: np.expand_dims(np.swapaxes(filters, 0, 2), axis=2),
                        ae['threshold_confidence']: arguments.confidence
                    }
                )
        else:
            # RGB
            if DUMP_COMPUTING_TIME:
                reconALL, InputAll, score, input_img_patches, recon_img_patches, difference, diff_scores, original_img, \
                reconstructed_img, background, gabor_filtered1, erode1, dilate1, gabor_filtered2, erode2, histogram, \
                tensor_to_normalize, thresholded_image, threshold_from_histo, accumulator = sess.run(
                    [ae['y'], ae['x'], ae['cost'], ae['input_image'], ae['reconstruction'], ae['differences'],
                     ae['diff_scores'], ae['original_image'], ae['reconstructed_image'], ae['grayscaled'],
                     ae['gabor_filtered1'], ae['erode1'], ae['dilate1'], ae['gabor_filtered2'], ae['erode2'],
                     ae['histogram'], ae['tensor_to_normalize'], ae['thresholded_image'], ae['threshold_from_histo'],
                     ae['aread_accumulated']],
                    feed_dict={
                        ae['x_int8']: patches,
                        ae['filters']: np.expand_dims(np.swapaxes(filters, 0, 2), axis=2),
                        ae['threshold_confidence']: arguments.confidence
                    },
                    options=options, run_metadata=run_metadata
                )
            else:
                reconALL, InputAll, score, input_img_patches, recon_img_patches, difference, diff_scores, original_img, \
                reconstructed_img, background, gabor_filtered1, erode1, dilate1, gabor_filtered2, erode2, histogram, \
                tensor_to_normalize, thresholded_image, threshold_from_histo, accumulator = sess.run(
                    [ae['y'], ae['x'], ae['cost'], ae['input_image'], ae['reconstruction'], ae['differences'],
                     ae['diff_scores'], ae['original_image'], ae['reconstructed_image'], ae['grayscaled'],
                     ae['gabor_filtered1'], ae['erode1'], ae['dilate1'], ae['gabor_filtered2'], ae['erode2'],
                     ae['histogram'], ae['tensor_to_normalize'], ae['thresholded_image'], ae['threshold_from_histo'],
                     ae['aread_accumulated']],
                    feed_dict={
                        ae['x_int8']: patches,
                        ae['filters']: np.expand_dims(np.swapaxes(filters, 0, 2), axis=2),
                        ae['threshold_confidence']: arguments.confidence
                    }
                )

        if i > 0:
            elapsed_inference += 1000.0*(time.time() - started)
        print("Inference %f ms in run %d" % (1000.0*(time.time() - started), i))

        # =============================== inference only


        # ============================================================= START POST PROCESSING
        started = time.time()  # +++++++++++++++++++++++++++++++++++
        # casting to correct representation
        original_input_patches = np.array(input_img_patches)
        reconstructed_image_patches = np.array(recon_img_patches)
        original_input = np.array(original_img)
        reconstructed_image = np.array(reconstructed_img)
        diff = difference
        score_diff = diff_scores

        # # thresholding
        background = tensor_to_normalize

        # ======================
        # ======================
        before_threshold = background.copy()  # ====================== remove when no NEED, DEBUG ONLY
        # ======================
        # ======================

        background = np.array(thresholded_image)
        # print(background)
        # asdasd

        # ======================
        # ======================
        after_threshold = background.copy()  # ====================== remove when no NEED, DEBUG ONLY
        # ======================
        # ======================

        # edge extraction
        background = cv2.Canny(np.array(background).astype(np.uint8), 0, 255)

        detection_res = np.zeros_like(img)
        detection_res[:, :, 0] = background

        idx_over = np.where(detection_res[:, :, 0]>0)
        detection_res += img
        detection_res[idx_over[0], idx_over[1], 1] = 0
        detection_res[idx_over[0], idx_over[1], 2] = 0

        if i > 0:
            elapsed_total += 1000.0 * (time.time() - total_start)
            elapsed_postprocessing += 1000.0 * (time.time() - started)  # +++++++++++++++++++++++++++++++++++

        print("Detection & PostProcessing %f ms in run %d" % (1000.0 * (time.time() - started), i))
        print("Total %f ms in run %d" % (1000.0 * (time.time() - total_start), i))
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")

        if DUMP_COMPUTING_TIME:
            # Create the Timeline object, and write it to a json file
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open('timeline_inference.json', 'w') as f:
                f.write(chrome_trace)
        # =========================================================================== END POST PROCESSING


    print("===================== SUMMARY ========================")
    # ---------------- just for visualization purposes
    print(original_input_patches.shape)
    print(reconstructed_image_patches.shape)
    print(original_input.shape)
    print(reconstructed_image.shape)

    recon_dump = reconstructed_image_patches[0]
    raw_input_dump = original_input_patches[0]
    misc.imsave("results/reconstruction_stitched.jpg", reconstructed_image)
    misc.imsave("results/original_stitched.jpg", original_input)
    misc.imsave("results/reconstruction.jpg", recon_dump)
    misc.imsave("results/patch_original.jpg", raw_input_dump)
    print("recong", np.max(recon_dump), np.min(recon_dump))
    print("real input", np.max(raw_input_dump), np.min(raw_input_dump))
    print(diff.shape)
    print(score_diff)

    print("Threshold: %d, confidence: %f%%" % (threshold_from_histo, accumulator * 100.))

    misc.imsave("results/before_threshold.bmp", before_threshold)

    misc.imsave("results/after_threshold.bmp", after_threshold)

    misc.imsave(arguments.output_file, detection_res)

    print("Inference time on average over %d runs was: %f ms" %
          (i, float(elapsed_inference)/float(i)))

    print("Postprocessing & Detection time  on average over %d runs was: %f ms" %
          (i, float(elapsed_postprocessing) / float(i)))

    print("Total processing Image time  on average over %d runs was: %f ms" %
          (i, float(elapsed_total) / float(i)))

    print("Patch Extraction time  on average over %d runs was: %f ms" %
          (i, (float(elapsed_patch_extraction) / float(i))))
    # ---------------- just for visualization purposes
    print("===================== SUMMARY ========================")


# %%
if __name__ == "__main__":
    arguments = parse_args()
    assigned = False
    for id_name in range(20):
        if arguments.gpu_id == id_name:
            print("assigning GPU %d for processing ..." % arguments.gpu_id)
            os.environ['CUDA_VISIBLE_DEVICES'] = str(id_name)
            assigned = True
            break

    if not assigned:
        print("Pure CPU assigned for processing ...")
        os.environ['CUDA_VISIBLE_DEVICES'] = ' '

    if arguments.execution_mode == 'train':
        train(arguments=arguments)
    else:
        inference(testimg=arguments.test_image, thres=arguments.threshold, arguments=arguments)

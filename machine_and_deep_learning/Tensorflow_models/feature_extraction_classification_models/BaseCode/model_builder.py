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
    This is the core code to be used to build the models and to call all the required
    modules for training, testing, feature extraction when using any of the BodyNets supported
    in this project.
'''

import tensorflow as tf
from core_ops import *
from model_ops import *

import sys
sys.path.append('../')
from configs import common_configs, resnet_configs,
from configs import squeeznet_configs, helnet_configs
from configs import shufflenet_configs, mobilenet_configs

'''
    Model class builder to be used for constructing the models and designing the
    computation graph for TensorFlow
'''
class ModelNetwork(BaseOps):
    '''
        main function in charge of storing all the necessary information and building
        the computation graph for TensorFlow
    '''
    def build(self, input_args, db_size=None):
        if input_args.debug_info >= 2:
            print("Builder for class ModelNetwork")

        # storing the information
        self.input_args = input_args
        self.execution_mode = input_args.execution_mode
        self.db_size = db_size
        self.debug_level = input_args.debug_info
        self.model_name = input_args.model_name
        self.solver = input_args.solver
        self.max_epochs = input_args.max_epochs
        self.enable_tensorboard = input_args.enable_tensorboard
        self.model_dir = input_args.model_dir

        # Body network loading, these are the topologies to be used to extract the features from the input
        # data or images processed in this project
        if self.model_name == 'helnet':
            self.NVHelNet()
        elif self.model_name == 'squeeznet':
            self.SqueezeNet()
        elif self.model_name == 'resnet':
            self.RESNET_50()
        elif self.model_name == 'shufflenet':
            self.ShuffleNet()
        elif self.model_name == 'mobilenet':
            self.MobileNet()
        else:
            print("This network (%s) type is not yet supported, quitting this application" % input_args.model_name)
            sys.exit(1)

        if self.execution_mode == 'train':
            # Training preparation
            print("This model is to be trained ...")
            self.Prepare_Engine()
        else:
            # Should be test
            print("This model is to be tested ...")
            self.Prepare_Engine_inference()

        # Base operations builder
        self.basic_ops = BaseOps()
        self.basic_ops.build(input_args=input_args, db_size=self.db_size)

        return

    ## ============ Body Nets
    def ShuffleNet(self):
        print("========================= Not Yet Implemented =====================")
        sys.exit(0)
        return

    def MobileNet(self):
        print("========================= Not Yet Implemented =====================")
        sys.exit(0)
        return

    def SqueezeNet(self):
        if self.debug_level >= 2:
            print("Building SqueezNet for Body Network...")

        if common_configs.ENABLE_RANDOM_CROP == True:
            # User Inputs
            pinputimage = tf.placeholder(tf.float32, shape=[None, common_configs.crop_height,
                                                            common_configs.crop_width,
                                                            common_configs.CHANNELS_G], name="input_image")
        else:
            # User Inputs
            pinputimage = tf.placeholder(tf.float32, shape=[None, common_configs.HEIGHT_GLOBAL,
                                                            common_configs.WIDTH_GLOBAL,
                                                            common_configs.CHANNELS_G], name="input_image")

        true_labels = tf.placeholder(tf.float32, shape=[None, common_configs.N_CLASSES])


        # determine if we are training or testing
        istrain = tf.placeholder(tf.bool, name="istrain")

        # for dropout in case we are traaining or not
        keepprob = tf.placeholder(tf.float32, name="keep_probabilty")

        # learning rate reduction if DSD is activated
        reduce_lr = tf.placeholder(tf.bool, name="reduce_learning_rate")

        # ===============================================================================
        # First part of SqueezeNet
        if self.debug_level >= 2:
            print("Simplest Squeeznet model layers...")

        conv1_feats = convolution_module(x=pinputimage, k_h=3, k_w=3, c_o=64, s_h=2, s_w=2, name='conv1',
                                         init_b=0.0, init_w='xavier', stddev=0.01, padding='SAME', re_use=None,
                                         group_id=0)
        conv1_feats = nonlinear_module(x=conv1_feats, nl='relu')
        if self.debug_level >= 2:
            print("conv1_feats layer")
            print(conv1_feats.get_shape())

        pool1_feats = max_pool_module(x=conv1_feats, k_h=3, k_w=3, s_h=2, s_w=2, name='pool1')
        if self.debug_level >= 2:
            print("pool1_feats layer")
            print(pool1_feats.get_shape())
        # ===============================================================================


        # ===============================================================================
        # Second part of SqueezeNet
        fire2_feats = fire_module(x=pool1_feats, c_s=16, c_e=64, name='fire2', padd1='SAME', padd2='SAME', padd3='SAME',
                             debug_level=self.debug_level)
        fire2_feats = batch_norm_module(x=fire2_feats, name='fire2_batchnorm', is_train=istrain, bn_axis=3, bn=True)
        if self.debug_level >= 2:
            print("fire2_feats layer")
            print(fire2_feats.get_shape())

        fire3_feats = fire_module(x=fire2_feats, c_s=16, c_e=64, name='fire3', padd1='SAME', padd2='SAME', padd3='SAME',
                                  debug_level=self.debug_level)
        fire3_feats = batch_norm_module(x=fire3_feats, name='fire3_batchnorm', is_train=istrain, bn_axis=3, bn=True)
        fire3_feats = tf.add(x=fire2_feats, y=fire3_feats, name='fire3_add')
        if self.debug_level >= 2:
            print("fire3_feats layer")
            print(fire3_feats.get_shape())

        pool3_feats = max_pool_module(x=fire3_feats, k_h=3, k_w=3, s_h=2, s_w=2, name='pool3')
        if self.debug_level >= 2:
            print("pool3_feats layer")
            print(pool3_feats.get_shape())
        # ===============================================================================

        # ===============================================================================
        # Third part of SqueezeNet
        fire4_feats = fire_module(x=pool3_feats, c_s=32, c_e=128, name='fire4', padd1='SAME', padd2='SAME',
                                  padd3='SAME', debug_level=self.debug_level)
        fire4_feats = batch_norm_module(x=fire4_feats, name='fire4_batchnorm', is_train=istrain, bn_axis=3, bn=True)
        if self.debug_level >= 2:
            print("fire4_feats layer")
            print(fire4_feats.get_shape())

        fire5_feats = fire_module(x=fire4_feats, c_s=32, c_e=128, name='fire5', padd1='SAME', padd2='SAME',
                                  padd3='SAME', debug_level=self.debug_level)
        fire5_feats = batch_norm_module(x=fire5_feats, name='fire5_batchnorm', is_train=istrain, bn_axis=3, bn=True)
        fire5_feats = tf.add(x=fire4_feats, y=fire5_feats, name='fire5_add')
        if self.debug_level >= 2:
            print("fire5_feats layer")
            print(fire5_feats.get_shape())

        pool5_feats = max_pool_module(x=fire5_feats, k_h=3, k_w=3, s_h=2, s_w=2, name='pool5')
        if self.debug_level >= 2:
            print("pool5_feats layer")
            print(pool5_feats.get_shape())
        # ===============================================================================

        # ===============================================================================
        # Fourth part of SqueezeNet
        fire6_feats = fire_module(x=pool5_feats, c_s=48, c_e=192, name='fire6', padd1='SAME', padd2='SAME',
                                  padd3='SAME', debug_level=self.debug_level)
        fire6_feats = batch_norm_module(x=fire6_feats, name='fire6_batchnorm', is_train=istrain, bn_axis=3, bn=True)
        if self.debug_level >= 2:
            print("fire6_feats layer")
            print(fire6_feats.get_shape())

        fire7_feats = fire_module(x=fire6_feats, c_s=48, c_e=192, name='fire7', padd1='SAME', padd2='SAME',
                                  padd3='SAME', debug_level=self.debug_level)
        fire7_feats = batch_norm_module(x=fire7_feats, name='fire7_batchnorm', is_train=istrain, bn_axis=3, bn=True)
        fire7_feats = tf.add(x=fire6_feats, y=fire7_feats, name='fire6_add')

        if self.debug_level >= 2:
            print("fire7_feats layer")
            print(fire7_feats.get_shape())

        fire8_feats = fire_module(x=fire7_feats, c_s=64, c_e=256, name='fire8', padd1='SAME', padd2='SAME',
                                  padd3='SAME', debug_level=self.debug_level)
        fire8_feats = batch_norm_module(x=fire8_feats, name='fire8_batchnorm', is_train=istrain, bn_axis=3, bn=True)
        if self.debug_level >= 2:
            print("fire8_feats layer")
            print(fire8_feats.get_shape())

        fire9_feats = fire_module(x=fire8_feats, c_s=64, c_e=256, name='fire9', padd1='SAME', padd2='SAME',
                                  padd3='SAME', debug_level=self.debug_level)
        fire9_feats = batch_norm_module(x=fire9_feats, name='fire9_batchnorm', is_train=istrain, bn_axis=3, bn=True)
        fire9_feats = tf.add(x=fire8_feats, y=fire9_feats, name='fire9_add')

        if self.debug_level >= 2:
            print("fire9_feats layer")
            print(fire9_feats.get_shape())
        # ===============================================================================

        # ===============================================================================
        # Fifth part of SqueezeNet
        fire10_feats = fire_module(x=fire9_feats, c_s=96, c_e=384, name='fire10', padd1='SAME', padd2='SAME',
                                  padd3='SAME', debug_level=self.debug_level)
        fire10_feats = batch_norm_module(x=fire10_feats, name='fire10_batchnorm', is_train=istrain, bn_axis=3, bn=True)
        if self.debug_level >= 2:
            print("fire10_feats layer")
            print(fire10_feats.get_shape())

        fire11_feats = fire_module(x=fire10_feats, c_s=96, c_e=384, name='fire11', padd1='SAME', padd2='SAME',
                                  padd3='SAME', debug_level=self.debug_level)
        fire11_feats = batch_norm_module(x=fire11_feats, name='fire11_batchnorm', is_train=istrain, bn_axis=3, bn=True)
        fire11_feats = tf.add(x=fire10_feats, y=fire11_feats, name='fire11_add')

        if self.debug_level >= 2:
            print("fire11_feats layer")
            print(fire11_feats.get_shape())

        fire12_feats = convolution_module(x=fire11_feats, k_h=1, k_w=1, c_o=common_configs.N_CLASSES, s_h=1, s_w=1,
                                          name="fire12", init_w='xavier', stddev=0.01, padding='SAME',
                                          re_use=None, group_id=0)
        if self.debug_level >= 2:
            print("fire12_feats layer")
            print(fire12_feats.get_shape())

        # to add some generalization characteristic to the model
        dropout_layer = dropout_module(x=fire12_feats, keep_prob=keepprob, is_train=istrain)
        # ===============================================================================

        info_layer = dropout_layer.get_shape()
        predictions = Build_average_pool_layer(network=dropout_layer, shared_shape=info_layer,
                                               n_classes=common_configs.N_CLASSES, pool_type='SAME',
                                               debug_level=self.debug_level)
        if self.debug_level >= 2:
            print("predictions layer")
            print(predictions.get_shape())

        self.pinputimage = pinputimage
        self.true_labels = true_labels
        self.istrain = istrain
        self.keepprob = keepprob
        self.predictions = predictions
        self.reduce_lr = reduce_lr

        return

    def NVHelNet(self):
        if self.debug_level >= 2:
            print("Building NV-Helnet for Body Network...")

        if common_configs.ENABLE_RANDOM_CROP == True:
            # User Inputs
            pinputimage = tf.placeholder(tf.float32, shape=[None, common_configs.crop_height,
                                                            common_configs.crop_width,
                                                            common_configs.CHANNELS_G], name="input_image")
        else:
            # User Inputs
            pinputimage = tf.placeholder(tf.float32, shape=[None, common_configs.HEIGHT_GLOBAL,
                                                            common_configs.WIDTH_GLOBAL,
                                                            common_configs.CHANNELS_G], name="input_image")

        # Ground Truth for labels of images
        true_labels = tf.placeholder(tf.float32, shape=[None, common_configs.N_CLASSES])

        # determine if we are training or testing
        istrain = tf.placeholder(tf.bool, name="istrain")

        # for dropout in case we are traaining or not
        keepprob = tf.placeholder(tf.float32, name="keep_probabilty")

        # learning rate reduction if DSD is activated
        reduce_lr = tf.placeholder(tf.bool, name="reduce_learning_rate")

        # ==============================================================================================
        # First part of the network where the input is reduced in dimension
        conv1_feats = convolution_module(x=pinputimage, k_h=7, k_w=7, c_o=64, s_h=2, s_w=2, name='Layer_1',
                                         init_b=0.0, init_w='truncated', stddev=0.001, padding='SAME', re_use=None,
                                         group_id=0)
        conv1_feats = nonlinear_module(x=conv1_feats, nl='relu')
        if self.debug_level >= 2:
            print("conv1_feats layer")
            print(conv1_feats.get_shape())

        pool1_feats = max_pool_module(x=conv1_feats, k_h=3, k_w=3, s_h=2, s_w=2, name='MaxPool_Layer1')
        if self.debug_level >= 2:
            print("pool1_feats layer")
            print(pool1_feats.get_shape())
        # ==============================================================================================

        # ==============================================================================================
        # Second part of the nvHelnet
        Layer2 = res_conv(input_tensor=pool1_feats, kernel_size=3, filters=[64, 64, 256], stage=2, block='a',
                          name='Layer2', is_training=istrain, debug_level=self.debug_level, add_skip=False)
        if self.debug_level >= 2:
            print("Layer2 layer")
            print(Layer2.get_shape())

        pool2_feats = max_pool_module(x=Layer2, k_h=3, k_w=3, s_h=2, s_w=2, name='MaxPool_Layer2')
        if self.debug_level >= 2:
            print("pool2_feats layer")
            print(pool2_feats.get_shape())

        Layer3 = res_conv(input_tensor=pool2_feats, kernel_size=3, filters=[128, 128, 512], stage=2, block='b',
                          name='Layer3', is_training=istrain, debug_level=self.debug_level, add_skip=False)
        if self.debug_level >= 2:
            print("Layer3 layer")
            print(Layer3.get_shape())

        pool3_feats = max_pool_module(x=Layer3, k_h=3, k_w=3, s_h=2, s_w=2, name='MaxPool_Layer3')
        if self.debug_level >= 2:
            print("pool3_feats layer")
            print(pool3_feats.get_shape())
        # ==============================================================================================

        # ==============================================================================================
        # Third part of the nvHelnet
        Layer4 = res_conv(input_tensor=pool3_feats, kernel_size=3, filters=[256, 256, 1024], stage=3, block='a',
                          name='Layer4', is_training=istrain, debug_level=self.debug_level, add_skip=False)
        if self.debug_level >= 2:
            print("Layer4 layer")
            print(Layer4.get_shape())

        Layer5 = res_conv(input_tensor=Layer4, kernel_size=3, filters=[256, 256, 1024], stage=3, block='b',
                          name='Layer5', is_training=istrain, debug_level=self.debug_level, add_skip=False)
        if self.debug_level >= 2:
            print("Layer5 layer")
            print(Layer5.get_shape())

        pool4_feats = max_pool_module(x=Layer5, k_h=3, k_w=3, s_h=2, s_w=2, name='MaxPool_Layer4')
        if self.debug_level >= 2:
            print("pool4_feats layer")
            print(pool4_feats.get_shape())
        # ==============================================================================================

        # ==============================================================================================
        # fourth part of the nvHelnet
        Layer6 = res_conv(input_tensor=pool4_feats, kernel_size=3, filters=[512, 512, 2048], stage=4, block='a',
                          name='Layer6', is_training=istrain, debug_level=self.debug_level, add_skip=False)
        if self.debug_level >= 2:
            print("Layer6 layer")
            print(Layer6.get_shape())

        Layer7 = res_conv(input_tensor=Layer6, kernel_size=3, filters=[512, 512, 2048], stage=4, block='b',
                          name='Layer7', is_training=istrain, debug_level=self.debug_level, add_skip=False)
        if self.debug_level >= 2:
            print("Layer7 layer")
            print(Layer7.get_shape())

        Layer8 = convolution_module(x=Layer7, k_h=1, k_w=1, c_o=common_configs.N_CLASSES, s_h=1, s_w=1,
                                    name="Layer8", init_w='truncated', stddev=1.0 / common_configs.N_CLASSES,
                                    padding='SAME', re_use=None, group_id=0)
        if self.debug_level >= 2:
            print("Layer8 layer")
            print(Layer8.get_shape())

        # to add some generalization characteristic to the model
        dropout_layer = dropout_module(x=Layer8, keep_prob=keepprob, is_train=istrain)
        # ==============================================================================================

        info_layer = dropout_layer.get_shape()
        predictions = Build_average_pool_layer(network=dropout_layer, shared_shape=info_layer,
                                               n_classes=common_configs.N_CLASSES, pool_type='SAME',
                                               debug_level=self.debug_level)
        if self.debug_level >= 2:
            print("predictions layer")
            print(predictions.get_shape())

        self.pinputimage = pinputimage
        self.true_labels = true_labels
        self.istrain = istrain
        self.keepprob = keepprob
        self.predictions = predictions
        self.reduce_lr = reduce_lr

        return

    def RESNET_50(self):
        if self.debug_level >= 2:
            print("Building RESNET-18 for Body Network...")

        if common_configs.ENABLE_RANDOM_CROP == True:
            # User Inputs
            pinputimage = tf.placeholder(tf.float32, shape=[None, common_configs.crop_height,
                                                            common_configs.crop_width,
                                                            common_configs.CHANNELS_G], name="input_image")
        else:
            # User Inputs
            pinputimage = tf.placeholder(tf.float32, shape=[None, common_configs.HEIGHT_GLOBAL,
                                                            common_configs.WIDTH_GLOBAL,
                                                            common_configs.CHANNELS_G], name="input_image")

        # Ground Truth for labels of images
        true_labels = tf.placeholder(tf.float32, shape=[None, common_configs.N_CLASSES])

        # determine if we are training or testing
        istrain = tf.placeholder(tf.bool, name="istrain")

        # for dropout in case we are traaining or not
        keepprob = tf.placeholder(tf.float32, name="keep_probabilty")

        # learning rate reduction if DSD is activated
        reduce_lr = tf.placeholder(tf.bool, name="reduce_learning_rate")

        # ==============================================================================================
        # First part of the network where the input is reduced in dimension
        conv1_feats = convolution_module(x=pinputimage, k_h=7, k_w=7, c_o=64, s_h=2, s_w=2, name='Layer_1',
                                         init_b=0.0, init_w='truncated', stddev=0.001, padding='SAME', re_use=None, group_id=0)
        conv1_feats = batch_norm_module(x=conv1_feats, name='conv1_feats_batchnorm', is_train=istrain, bn_axis=3, bn=True)
        conv1_feats = nonlinear_module(x=conv1_feats, nl='relu')
        if self.debug_level >= 2:
            print("conv1_feats layer")
            print(conv1_feats.get_shape())

        pool1_feats = max_pool_module(x=conv1_feats, k_h=3, k_w=3, s_h=2, s_w=2, name='MaxPool_Layer1')
        if self.debug_level >= 2:
            print("pool1_feats layer")
            print(pool1_feats.get_shape())
        # ==============================================================================================

        # ==============================================================================================
        # Second part of the Resnet-50
        Layer2 = res_conv_skip(input_tensor=pool1_feats, kernel_size=3, filters=[64, 64, 256], stage=2, block='a',
                               stride=1, name='Layer2', is_training=istrain, debug_level=self.debug_level)
        if self.debug_level >= 2:
            print("Layer2 layer")
            print(Layer2.get_shape())

        Layer3 = res_conv(input_tensor=Layer2, kernel_size=3, filters=[64, 64, 256], stage=2, block='b', name='Layer3',
                          is_training=istrain, debug_level=self.debug_level)
        if self.debug_level >= 2:
            print("Layer3 layer")
            print(Layer3.get_shape())

        Layer4 = res_conv(input_tensor=Layer3, kernel_size=3, filters=[64, 64, 256], stage=2, block='c', name='Layer4',
                          is_training=istrain, debug_level=self.debug_level)
        if self.debug_level >= 2:
            print("Layer4 layer")
            print(Layer4.get_shape())
        # ==============================================================================================

        # ==============================================================================================
        # Third part of the Resnet-50
        Layer5 = res_conv_skip(input_tensor=Layer4, kernel_size=3, filters=[128, 128, 512], stage=3, block='a',
                               name='Layer5', is_training=istrain, debug_level=self.debug_level)
        if self.debug_level >= 2:
            print("Layer5 layer")
            print(Layer5.get_shape())

        Layer6 = res_conv(input_tensor=Layer5, kernel_size=3, filters=[128, 128, 512], stage=3, block='b',
                          name='Layer6', is_training=istrain, debug_level=self.debug_level)
        if self.debug_level >= 2:
            print("Layer6 layer")
            print(Layer6.get_shape())

        Layer7 = res_conv(input_tensor=Layer6, kernel_size=3, filters=[128, 128, 512], stage=3, block='c',
                          name='Layer7', is_training=istrain, debug_level=self.debug_level)
        if self.debug_level >= 2:
            print("Layer7 layer")
            print(Layer7.get_shape())

        Layer8 = res_conv(input_tensor=Layer7, kernel_size=3, filters=[128, 128, 512], stage=3, block='d',
                          name='Layer8', is_training=istrain, debug_level=self.debug_level)
        if self.debug_level >= 2:
            print("Layer8 layer")
            print(Layer8.get_shape())
        # ==============================================================================================

        # ==============================================================================================
        # Fourth part of the Resnet-50 and heaviest one
        Layer9 = res_conv_skip(input_tensor=Layer8, kernel_size=3, filters=[256, 256, 1024], stage=4, block='a',
                               name='Layer9', is_training=istrain, debug_level=self.debug_level)
        if self.debug_level >= 2:
            print("Layer9 layer")
            print(Layer9.get_shape())

        Layer10 = res_conv(input_tensor=Layer9, kernel_size=3, filters=[256, 256, 1024], stage=4, block='b',
                           name='Layer10', is_training=istrain, debug_level=self.debug_level)
        if self.debug_level >= 2:
            print("Layer10 layer")
            print(Layer10.get_shape())
        Layer11 = res_conv(input_tensor=Layer10, kernel_size=3, filters=[256, 256, 1024], stage=4, block='c',
                           name='Layer11', is_training=istrain, debug_level=self.debug_level)
        if self.debug_level >= 2:
            print("Layer11 layer")
            print(Layer11.get_shape())

        Layer12 = res_conv(input_tensor=Layer11, kernel_size=3, filters=[256, 256, 1024], stage=4, block='d',
                           name='Layer12', is_training=istrain, debug_level=self.debug_level)
        if self.debug_level >= 2:
            print("Layer12 layer")
            print(Layer12.get_shape())

        Layer13 = res_conv(input_tensor=Layer12, kernel_size=3, filters=[256, 256, 1024], stage=4, block='e',
                           name='Layer13', is_training=istrain, debug_level=self.debug_level)
        if self.debug_level >= 2:
            print("Layer13 layer")
            print(Layer13.get_shape())

        Layer14 = res_conv(input_tensor=Layer13, kernel_size=3, filters=[256, 256, 1024], stage=4, block='f',
                           name='Layer14', is_training=istrain, debug_level=self.debug_level)
        if self.debug_level >= 2:
            print("Layer14 layer")
            print(Layer14.get_shape())
        # ==============================================================================================

        # ==============================================================================================
        # Final stage of Resnet-50 Fatest one
        Layer15 = res_conv_skip(input_tensor=Layer14, kernel_size=3, filters=[512, 512, 2048], stage=5, block='a',
                                name='Layer15', is_training=istrain, debug_level=self.debug_level)
        if self.debug_level >= 2:
            print("Layer15 layer")
            print(Layer15.get_shape())

        Layer16 = res_conv(input_tensor=Layer15, kernel_size=3, filters=[512, 512, 2048], stage=5, block='b',
                           name='Layer16', is_training=istrain, debug_level=self.debug_level)
        if self.debug_level >= 2:
            print("Layer16 layer")
            print(Layer16.get_shape())

        Layer17 = res_conv(input_tensor=Layer16, kernel_size=3, filters=[512, 512, 2048], stage=5, block='c',
                           name='Layer17', is_training=istrain, debug_level=self.debug_level)
        if self.debug_level >= 2:
            print("Layer17 layer")
            print(Layer17.get_shape())

        Layer18 = convolution_module(x=Layer17, k_h=1, k_w=1, c_o=common_configs.N_CLASSES, s_h=1, s_w=1,
                                     name="Layer18", init_w='truncated', stddev=1.0 / common_configs.N_CLASSES,
                                     padding='SAME', re_use=None, group_id=0)
        Layer18 = batch_norm_module(x=Layer18, name='Layer18_batchnorm', is_train=istrain, bn_axis=3, bn=True)
        Layer18 = nonlinear_module(x=Layer18, nl='relu')
        if self.debug_level >= 2:
            print("Layer18 layer")
            print(Layer18.get_shape())

        # to add some generalization characteristic to the model
        dropout_layer = dropout_module(x=Layer18, keep_prob=keepprob, is_train=istrain)
        # ==============================================================================================

        info_layer = dropout_layer.get_shape()
        predictions = Build_average_pool_layer(network=dropout_layer, shared_shape=info_layer,
                                               n_classes=common_configs.N_CLASSES, pool_type='SAME',
                                               debug_level=self.debug_level)
        if self.debug_level >= 2:
            print("predictions layer")
            print(predictions.get_shape())

        self.pinputimage = pinputimage
        self.true_labels = true_labels
        self.istrain = istrain
        self.keepprob = keepprob
        self.predictions = predictions
        self.reduce_lr = reduce_lr

        return

    ## ============ Body Nets

    # ================================================== TRAIN/TEST ENGINES
    '''
        Will generate the loss function and initialize the corresponding optimizer
        to be used for training
    '''
    def Prepare_Engine(self):
        print("Preparing engine for training process ...")

        # Training computation: logits + cross-entropy loss.
        class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.predictions,
                                                                            labels=self.true_labels))
        # weight_decay = 0.00005
        class_loss = class_loss  # + weight_decay*(tf.add_n(tf.get_collection('l2_0')))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.name_scope('optimization'):
            if self.model_name == 'squeeznet':
                if self.solver == 'moment':
                    if self.debug_level >= 2:
                        print("*********** Using Momentum Optimizer ***********")
                    batch = tf.Variable(0)

                    learning_rate = tf.cond(self.reduce_lr, lambda: tf.train.exponential_decay(
                        squeeznet_configs.MOMENT_Lr / 10, tf.multiply(batch, tf.ones_like(batch)*common_configs.BATCH_SIZE),
                        self.db_size, squeeznet_configs.MOMENT_DECAY_RATE, staircase=True),
                                            lambda: tf.train.exponential_decay(
                                                squeeznet_configs.MOMENT_Lr,
                                                tf.multiply(batch, tf.ones_like(batch) * common_configs.BATCH_SIZE),
                                                self.db_size, squeeznet_configs.MOMENT_DECAY_RATE, staircase=True)
                                            )

                    with tf.control_dependencies(update_ops):
                        self.optimizer = tf.train.MomentumOptimizer(learning_rate,
                                                               squeeznet_configs.MOMENTUM).minimize(class_loss,
                                                                                                    global_step=batch)
                else:
                    if self.debug_level >= 2:
                        print("*********** Using Adam algorithm ***********")
                    with tf.control_dependencies(update_ops):
                        self.optimizer = tf.cond(self.reduce_lr,
                                                 lambda: tf.train.AdamOptimizer(squeeznet_configs.ADAM_START_Lr/10).minimize(class_loss),
                                                 lambda: tf.train.AdamOptimizer(squeeznet_configs.ADAM_START_Lr).minimize(class_loss))

            elif self.model_name == 'helnet':
                if self.solver == 'moment':
                    if self.debug_level >= 2:
                        print("*********** Using Momentum Optimizer ***********")
                    batch = tf.Variable(0)
                    learning_rate = tf.cond(self.reduce_lr, lambda: tf.train.exponential_decay(
                        helnet_configs.MOMENT_Lr / 10,
                        tf.multiply(batch, tf.ones_like(batch) * common_configs.BATCH_SIZE),
                        self.db_size, helnet_configs.MOMENT_DECAY_RATE, staircase=True),
                                            lambda: tf.train.exponential_decay(
                                                helnet_configs.MOMENT_Lr,
                                                tf.multiply(batch, tf.ones_like(batch) * common_configs.BATCH_SIZE),
                                                self.db_size, helnet_configs.MOMENT_DECAY_RATE, staircase=True)
                                            )
                    with tf.control_dependencies(update_ops):
                        self.optimizer = tf.train.MomentumOptimizer(learning_rate,
                                                               helnet_configs.MOMENTUM).minimize(class_loss,
                                                                                                 global_step=batch)
                else:
                    if self.debug_level >= 2:
                        print("*********** Using Adam algorithm ***********")
                    with tf.control_dependencies(update_ops):
                        self.optimizer = tf.cond(self.reduce_lr,
                                                 lambda: tf.train.AdamOptimizer(
                                                     helnet_configs.ADAM_START_Lr / 10).minimize(class_loss),
                                                 lambda: tf.train.AdamOptimizer(helnet_configs.ADAM_START_Lr).minimize(
                                                     class_loss))
            elif self.model_name == 'resnet':
                if self.solver == 'moment':
                    if self.debug_level >= 2:
                        print("*********** Using Momentum Optimizer ***********")
                    batch = tf.Variable(0)
                    learning_rate = tf.cond(self.reduce_lr, lambda: tf.train.exponential_decay(
                        resnet_configs.MOMENT_Lr / 10,
                        tf.multiply(batch, tf.ones_like(batch) * common_configs.BATCH_SIZE),
                        self.db_size, resnet_configs.MOMENT_DECAY_RATE, staircase=True),
                                            lambda: tf.train.exponential_decay(
                                                resnet_configs.MOMENT_Lr,
                                                tf.multiply(batch, tf.ones_like(batch) * common_configs.BATCH_SIZE),
                                                self.db_size, resnet_configs.MOMENT_DECAY_RATE, staircase=True)
                                            )
                    with tf.control_dependencies(update_ops):
                        self.optimizer = tf.train.MomentumOptimizer(learning_rate,
                                                               resnet_configs.MOMENTUM).minimize(class_loss,
                                                                                                 global_step=batch)
                else:
                    if self.debug_level >= 2:
                        print("*********** Using Adam algorithm ***********")
                    with tf.control_dependencies(update_ops):
                        self.optimizer = tf.cond(self.reduce_lr,
                                                 lambda: tf.train.AdamOptimizer(
                                                     resnet_configs.ADAM_START_Lr / 10).minimize(class_loss),
                                                 lambda: tf.train.AdamOptimizer(resnet_configs.ADAM_START_Lr).minimize(
                                                     class_loss))
            elif self.model_name == 'shufflenet':
                if self.solver == 'moment':
                    if self.debug_level >= 2:
                        print("*********** Using Momentum Optimizer ***********")
                    batch = tf.Variable(0)
                    learning_rate = tf.cond(self.reduce_lr, lambda: tf.train.exponential_decay(
                        shufflenet_configs.MOMENT_Lr / 10,
                        tf.multiply(batch, tf.ones_like(batch) * common_configs.BATCH_SIZE),
                        self.db_size, shufflenet_configs.MOMENT_DECAY_RATE, staircase=True),
                                            lambda: tf.train.exponential_decay(
                                                shufflenet_configs.MOMENT_Lr,
                                                tf.multiply(batch, tf.ones_like(batch) * common_configs.BATCH_SIZE),
                                                self.db_size, shufflenet_configs.MOMENT_DECAY_RATE, staircase=True)
                                            )
                    with tf.control_dependencies(update_ops):
                        self.optimizer = tf.train.MomentumOptimizer(learning_rate,
                                                               shufflenet_configs.MOMENTUM).minimize(class_loss,
                                                                                                 global_step=batch)
                else:
                    if self.debug_level >= 2:
                        print("*********** Using Adam algorithm ***********")
                    with tf.control_dependencies(update_ops):
                        self.optimizer = tf.cond(self.reduce_lr,
                                                 lambda: tf.train.AdamOptimizer(
                                                     shufflenet_configs.ADAM_START_Lr / 10).minimize(class_loss),
                                                 lambda: tf.train.AdamOptimizer(shufflenet_configs.ADAM_START_Lr).minimize(
                                                     class_loss))
            elif self.model_name == 'mobilenet':
                if self.solver == 'moment':
                    if self.debug_level >= 2:
                        print("*********** Using Momentum Optimizer ***********")
                    batch = tf.Variable(0)
                    learning_rate = tf.cond(self.reduce_lr, lambda: tf.train.exponential_decay(
                        mobilenet_configs.MOMENT_Lr / 10,
                        tf.multiply(batch, tf.ones_like(batch) * common_configs.BATCH_SIZE),
                        self.db_size, mobilenet_configs.MOMENT_DECAY_RATE, staircase=True),
                                            lambda: tf.train.exponential_decay(
                                                mobilenet_configs.MOMENT_Lr,
                                                tf.multiply(batch, tf.ones_like(batch) * common_configs.BATCH_SIZE),
                                                self.db_size, mobilenet_configs.MOMENT_DECAY_RATE, staircase=True)
                                            )
                    with tf.control_dependencies(update_ops):
                        self.optimizer = tf.train.MomentumOptimizer(learning_rate,
                                                               mobilenet_configs.MOMENTUM).minimize(class_loss,
                                                                                                 global_step=batch)
                else:
                    if self.debug_level >= 2:
                        print("*********** Using Adam algorithm ***********")
                    with tf.control_dependencies(update_ops):
                        self.optimizer = tf.cond(self.reduce_lr,
                                                 lambda: tf.train.AdamOptimizer(
                                                     mobilenet_configs.ADAM_START_Lr / 10).minimize(class_loss),
                                                 lambda: tf.train.AdamOptimizer(mobilenet_configs.ADAM_START_Lr).minimize(
                                                     class_loss))



        with tf.name_scope('output'):
            # Performing softmax to determine class prediction
            softmax_pred = tf.nn.softmax(self.predictions)

        if self.solver == 'moment':
            self.learning_rate = learning_rate
            self.batch = batch

        self.loss = class_loss
        self.softmax_pred = softmax_pred

        with tf.name_scope('training_accuracy'):
            accuracy = tf.equal(tf.argmax(self.softmax_pred, 1), tf.argmax(self.true_labels, 1))
            accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

        self.training_accuracy = accuracy

        return

    '''
        Will generate the loss function and initialize the corresponding optimizer
        to be used for training
    '''
    def Prepare_Engine_inference(self):
        print("Preparing engine for inference only ...")
        with tf.name_scope('output'):
            # Performing softmax to determine class prediction
            softmax_pred = tf.nn.softmax(self.predictions)

        self.softmax_pred = softmax_pred

        with tf.name_scope('testing_accuracy'):
            accuracy = tf.equal(tf.argmax(self.softmax_pred, 1), tf.argmax(self.true_labels, 1))
            accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

        self.testing_accuracy = accuracy

        return
    # =====================================================================================

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

from model_ops import *
import sys
import config
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops

class ModelNetwork(BaseOps):
    def build(self, input_args):
        if input_args.debug_info >= 2:
            print("Builder for class ModelNetwork")

        # Body network loading
        if input_args.model_name == 'helnet':
            self.NVHelNet(input_args=input_args)
        elif input_args.model_name == 'squeeznet':
            self.squeezenet(input_args=input_args)
        elif input_args.model_name == 'resnet':
            self.Resnet(input_args=input_args)
        else:
            print("This network (%s) type is not yet supported, quitting this application"%input_args.model_name)
            sys.exit(1)

        # Heads from hierarchy
        self.Hierarchy(input_args=input_args)

        # Base operations builder
        self.basic_ops = BaseOps()
        self.basic_ops.build(input_args=input_args)

    def load_pretrained(self, model_path, session, input_args, ignore_missing=True):
        """ Load the pretrained CNN model. """
        if input_args.debug_info >= 2:
            print("Loading basic model from %s..." % model_path)

        data_dict = np.load(model_path).item()
        count = 0
        miss_count = 0
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    print(np.array(data).shape)
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                        count += 1
                        if input_args.debug_info >= 2:
                            print("Variable %s:%s loaded" % (op_name, param_name))
                            print(np.array(data).sum())
                    except ValueError:
                        miss_count += 1
                        if input_args.debug_info >= 2:
                            print("Variable %s:%s missed, not needed for this architecture" % (op_name, param_name))
                        if not ignore_missing:
                            raise

    def modified_smooth_l1(self, sigma, bbox_pred, bbox_targets, bbox_inside_weights,
                           bbox_outside_weights):
        """
            ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        """
        sigma2 = sigma * sigma

        inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

        smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
        smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
        smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                                  tf.multiply(smooth_l1_option2,
                                              tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

        outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

        return outside_mul

    # BODY Net types
    def Resnet(self, input_args):
        if input_args.debug_info >= 2:
            print("Building RESNET-18 for Body Network...")

        # User Inputs
        pinputimage = tf.placeholder(tf.float32, shape=[None, config.image_size_y, config.image_size_x, 3],
                                     name="input_image")

        conv1_feats = convolution_module(pinputimage, 7, 7, 32, 2, 2, 'Layer1',
                               init_b=0.1, init_w='truncated', stddev=0.1)
        if input_args.debug_info >= 2:
            print("conv1_feats layer")
            print(conv1_feats.get_shape())

        pool1_feats = max_pool_module(conv1_feats, 2, 2, 2, 2, 'Layer2')
        if input_args.debug_info >= 2:
            print("pool1_feats layer")
            print(pool1_feats.get_shape())

        conv2_feats = convolution_module(pool1_feats, 3, 3, 56, 2, 2, 'Layer3',
                               init_b=0.1, init_w='truncated', stddev=0.1)
        if input_args.debug_info >= 2:
            print("conv2_feats layer")
            print(conv2_feats.get_shape())

        conv3_feats = convolution_module(conv2_feats, 3, 3, 56, 1, 1, 'Layer4',
                               init_b=0.1, init_w='truncated', stddev=0.1)

        # skipping layer, for the real resnet arch
        conv3_feats = tf.add(conv3_feats, conv2_feats, name='skip_layer1')

        if input_args.debug_info >= 2:
            print("conv3_feats layer")
            print(conv3_feats.get_shape())

        # conv4_feats = convolution_module(conv3_feats, 3, 3, 64, 2, 2, 'Layer5',
        #                        init_b=0.1, init_w='truncated', stddev=0.1)
        # if input_args.debug_info >= 2:
        #     print("conv4_feats layer")
        #     print(conv4_feats.get_shape())
        #
        # conv5_feats = convolution_module(conv4_feats, 3, 3, 56, 2, 2, 'Layer6',
        #                        init_b=0.1, init_w='truncated', stddev=0.1)
        # if input_args.debug_info >= 2:
        #     print("conv5_feats layer")
        #     print(conv5_feats.get_shape())
        #
        # conv6_feats = convolution_module(conv5_feats, 3, 3, 80, 2, 2, 'Layer7',
        #                        init_b=0.1, init_w='truncated', stddev=0.1)
        # if input_args.debug_info >= 2:
        #     print("conv6_feats layer")
        #     print(conv6_feats.get_shape())
        #
        # conv7_feats = convolution_module(conv6_feats, 3, 3, 96, 1, 1, 'Layer8',
        #                        init_b=0.1, init_w='truncated', stddev=0.1)
        # if input_args.debug_info >= 2:
        #     print("conv7_feats layer")
        #     print(conv7_feats.get_shape())
        #
        # conv8_feats = convolution_module(conv7_feats, 3, 3, 96, 1, 1, 'Layer9',
        #                        init_b=0.1, init_w='truncated', stddev=0.1)
        #
        # # skipping layer, for the real resnet arch
        # conv8_feats = tf.add(conv8_feats, conv7_feats, name='skip_layer2')
        #
        # if input_args.debug_info >= 2:
        #     print("conv8_feats layer")
        #     print(conv8_feats.get_shape())
        #
        # conv9_feats = convolution_module(conv8_feats, 3, 3, 80, 2, 2, 'Layer10',
        #                        init_b=0.1, init_w='truncated', stddev=0.1)
        # if input_args.debug_info >= 2:
        #     print("conv9_feats layer")
        #     print(conv9_feats.get_shape())
        #
        # conv10_feats = convolution_module(conv9_feats, 3, 3, 96, 1, 1, 'Layer11',
        #                        init_b=0.1, init_w='truncated', stddev=0.1)
        # if input_args.debug_info >= 2:
        #     print("conv10_feats layer")
        #     print(conv10_feats.get_shape())
        #
        # conv11_feats = convolution_module(conv10_feats, 3, 3, 104, 1, 1, 'Layer12',
        #                        init_b=0.1, init_w='truncated', stddev=0.1)
        # if input_args.debug_info >= 2:
        #     print("conv11_feats layer")
        #     print(conv11_feats.get_shape())
        #
        # conv12_feats = convolution_module(conv11_feats, 3, 3, 96, 1, 1, 'Layer13',
        #                        init_b=0.1, init_w='truncated', stddev=0.1)
        #
        # # skipping layer, for the real resnet arch
        # conv12_feats = tf.add(conv12_feats, conv10_feats, name='skip_layer3')
        #
        # if input_args.debug_info >= 2:
        #     print("conv12_feats layer")
        #     print(conv12_feats.get_shape())
        #
        # conv13_feats = convolution_module(conv12_feats, 3, 3, 88, 2, 2, 'Layer14',
        #                        init_b=0.1, init_w='truncated', stddev=0.1)
        # if input_args.debug_info >= 2:
        #     print("conv13_feats layer")
        #     print(conv13_feats.get_shape())
        #
        # conv14_feats = convolution_module(conv13_feats, 3, 3, 72, 2, 2, 'Layer15',
        #                        init_b=0.1, init_w='truncated', stddev=0.1)
        # if input_args.debug_info >= 2:
        #     print("conv14_feats layer")
        #     print(conv14_feats.get_shape())
        #
        # conv15_feats = convolution_module(conv14_feats, 3, 3, 72, 1, 1, 'Layer16',
        #                        init_b=0.1, init_w='truncated', stddev=0.1)
        #
        # # skipping layer, for the real resnet arch
        # conv15_feats = tf.add(conv15_feats, conv14_feats, name='skip_layer4')
        #
        # if input_args.debug_info >= 2:
        #     print("conv15_feats layer")
        #     print(conv15_feats.get_shape())
        #
        # conv16_feats = convolution_module(conv15_feats, 3, 3, 56, 1, 1, 'Layer17',
        #                        init_b=0.1, init_w='truncated', stddev=0.1)
        # if input_args.debug_info >= 2:
        #     print("Dropout layer")
        #     print(conv16_feats.get_shape())
        #
        # conv17_feats = convolution_module(conv16_feats, 3, 3, 208, 2, 2, 'Layer18',
        #                        init_b=0.1, init_w='truncated', stddev=0.1)
        # if input_args.debug_info >= 2:
        #     print("conv17_feats layer")
        #     print(conv17_feats.get_shape())


        self.pinputimage = pinputimage
        self.feats = conv3_feats
        self.conv1_feats = conv1_feats
        self.pool1_feats = pool1_feats
        self.conv2_feats = conv2_feats
        self.conv3_feats = conv3_feats
        # self.conv4_feats = conv4_feats
        # self.conv5_feats = conv5_feats
        # self.conv6_feats = conv6_feats
        # self.conv7_feats = conv7_feats
        # self.conv8_feats = conv8_feats
        # self.conv9_feats = conv9_feats
        # self.conv10_feats = conv10_feats
        # self.conv11_feats = conv11_feats
        # self.conv12_feats = conv12_feats
        # self.conv13_feats = conv13_feats
        # self.conv14_feats = conv14_feats
        # self.conv15_feats = conv15_feats
        # self.conv16_feats = conv16_feats
        # self.conv17_feats = conv17_feats

    def NVHelNet(self, input_args):
        if input_args.debug_info >= 2:
            print("Building NVHelNet for Body Network...")

        pinputimage = tf.placeholder(tf.float32, shape=[None, config.image_size_y, config.image_size_x, 3],
                                     name="input_image")

        if input_args.add_skyplayer == 'false':
            if input_args.debug_info >= 2:
                print("Simplest NVHelNet layers...")

            conv1_feats = convolution_module(pinputimage, 7, 7, 32, 2, 2, 'Layer1')
            if input_args.debug_info >= 2:
                print("conv1_feats:")
                print(conv1_feats.get_shape())

            pool1_feats = max_pool_module(conv1_feats, 2, 2, 2, 2, 'Layer2')
            if input_args.debug_info >= 2:
                print("pool1_feats:")
                print(pool1_feats.get_shape())

            conv2_feats = convolution_module(pool1_feats, 3, 3, 56, 2, 2, 'Layer3')
            if input_args.debug_info >= 2:
                print("conv2_feats:")
                print(conv2_feats.get_shape())

            conv3_feats = convolution_module(conv2_feats, 3, 3, 56, 2, 2, 'Layer4')
            if input_args.debug_info >= 2:
                print("conv3_feats:")
                print(conv3_feats.get_shape())

            # conv4_feats = convolution_module(conv3_feats, 3, 3, 64, 2, 2, 'Layer5')
            # if input_args.debug_info >= 2:
            #     print("conv4_feats:")
            #     print(conv4_feats.get_shape())
            #
            # conv5_feats = convolution_module(conv4_feats, 3, 3, 56, 2, 2, 'Layer6')
            # if input_args.debug_info >= 2:
            #     print("conv5_feats:")
            #     print(conv5_feats.get_shape())
            #
            # conv6_feats = convolution_module(conv5_feats, 3, 3, 80, 2, 2, 'Layer7')
            # if input_args.debug_info >= 2:
            #     print("conv6_feats:")
            #     print(conv6_feats.get_shape())
            #
            # conv7_feats = convolution_module(conv6_feats, 3, 3, 96, 2, 2, 'Layer8')
            # if input_args.debug_info >= 2:
            #     print("conv7_feats:")
            #     print(conv7_feats.get_shape())
            #
            # conv8_feats = convolution_module(conv7_feats, 3, 3, 96, 2, 2, 'Layer9')
            # if input_args.debug_info >= 2:
            #     print("conv8_feats:")
            #     print(conv8_feats.get_shape())
            #
            # conv9_feats = convolution_module(conv8_feats, 3, 3, 80, 2, 2, 'Layer10')
            # if input_args.debug_info >= 2:
            #     print("conv9_feats:")
            #     print(conv9_feats.get_shape())
            #
            # conv10_feats = convolution_module(conv9_feats, 3, 3, 96, 2, 2, 'Layer11')
            # if input_args.debug_info >= 2:
            #     print("conv10_feats:")
            #     print(conv10_feats.get_shape())
            #
            # conv11_feats = convolution_module(conv10_feats, 3, 3, 104, 2, 2, 'Layer12')
            # if input_args.debug_info >= 2:
            #     print("conv11_feats:")
            #     print(conv11_feats.get_shape())
            #
            # conv12_feats = convolution_module(conv11_feats, 3, 3, 96, 2, 2, 'Layer13')
            # if input_args.debug_info >= 2:
            #     print("conv12_feats:")
            #     print(conv12_feats.get_shape())
            #
            # conv13_feats = convolution_module(conv12_feats, 3, 3, 88, 2, 2, 'Layer14')
            # if input_args.debug_info >= 2:
            #     print("conv1_feats:")
            #     print(conv13_feats.get_shape())
            #
            # conv14_feats = convolution_module(conv13_feats, 3, 3, 72, 2, 2, 'Layer15')
            # if input_args.debug_info >= 2:
            #     print("conv14_feats:")
            #     print(conv14_feats.get_shape())
            #
            # conv15_feats = convolution_module(conv14_feats, 3, 3, 72, 2, 2, 'Layer16')
            # if input_args.debug_info >= 2:
            #     print("conv15_feats:")
            #     print(conv15_feats.get_shape())
            #
            # conv16_feats = convolution_module(conv15_feats, 3, 3, 56, 2, 2, 'Layer17')
            # if input_args.debug_info >= 2:
            #     print("conv16_feats:")
            #     print(conv16_feats.get_shape())
            #
            # conv17_feats = convolution_module(conv16_feats, 3, 3, 208, 2, 2, 'Layer18')
            # if input_args.debug_info >= 2:
            #     print("conv17_feats:")
            #     print(conv17_feats.get_shape())

            self.feats = conv3_feats
            self.pinputimage = pinputimage
            self.conv1_feats = conv1_feats
            self.pool1_feats = pool1_feats
            self.conv2_feats = conv2_feats
            self.conv3_feats = conv3_feats
            # self.conv4_feats = conv4_feats
            # self.conv5_feats = conv5_feats
            # self.conv6_feats = conv6_feats
            # self.conv7_feats = conv7_feats
            # self.conv8_feats = conv8_feats
            # self.conv9_feats = conv9_feats
            # self.conv10_feats = conv10_feats
            # self.conv11_feats = conv11_feats
            # self.conv12_feats = conv12_feats
            # self.conv13_feats = conv13_feats
            # self.conv14_feats = conv14_feats
            # self.conv15_feats = conv15_feats
            # self.conv16_feats = conv16_feats
            # self.conv17_feats = conv17_feats
        else:
            if input_args.debug_info >= 2:
                print("Inserting skipping layers NVHelNet, NOT YET IMPLEMENTED...")

    def squeezenet(self, input_args):
        if input_args.debug_info >= 2:
            print("Building SqueezNet for Body Network...")

        pinputimage = tf.placeholder(tf.float32, shape=[None, config.image_size_y, config.image_size_x, 3], name="input_image")

        if input_args.add_skyplayer == 'false':
            if input_args.debug_info >= 2:
                print("Simplest Squeeznet model layers...")


            if config.FREEZE_WEIGHTS or config.FREEZE_BODYNET:
                print("freezing weights at conv1_feats")
                conv1_feats = convolution_module(pinputimage, 3, 3, 64, 2, 2, 'conv1', trainable=False)
            else:
                conv1_feats = convolution_module(pinputimage, 3, 3, 64, 2, 2, 'conv1', trainable=True)
            if input_args.debug_info >= 2:
                print(conv1_feats.get_shape())

            if config.FREEZE_WEIGHTS or config.FREEZE_BODYNET:
                print("freezing weights at pool1_feats")
                pool1_feats = max_pool_module(conv1_feats, 2, 2, 2, 2, 'pool1', padding='VALID')
            else:
                pool1_feats = max_pool_module(conv1_feats, 2, 2, 2, 2, 'pool1', padding='VALID')
            if input_args.debug_info >= 2:
                print(pool1_feats.get_shape())

            # Main Modules for Squeeznet
            if config.FREEZE_WEIGHTS or config.FREEZE_BODYNET:
                print("freezing weights at fire2_feats")
                fire2_feats, fire2_s, fire2_e1x1, fire2_e3x3 = fire_module(pool1_feats, 16, 64, 'fire2', 'VALID',
                                                                           'VALID', 'SAME', trainable=False)
            else:
                fire2_feats, fire2_s, fire2_e1x1, fire2_e3x3 = fire_module(pool1_feats, 16, 64, 'fire2', 'VALID',
                                                                           'VALID', 'SAME', trainable=True)
            if input_args.debug_info >= 2:
                print(fire2_feats.get_shape())

            if config.FREEZE_WEIGHTS or config.FREEZE_BODYNET:
                print("freezing weights at fire3_feats")
                fire3_feats, fire3_s, fire3_e1x1, fire3_e3x3 = fire_module(fire2_feats, 16, 64, 'fire3', 'VALID',
                                                                           'VALID', 'SAME', trainable=False)
            else:
                fire3_feats, fire3_s, fire3_e1x1, fire3_e3x3 = fire_module(fire2_feats, 16, 64, 'fire3', 'VALID',
                                                                           'VALID', 'SAME', trainable=True)
            if input_args.debug_info >= 2:
                print(fire3_feats.get_shape())

            if config.FREEZE_WEIGHTS or config.FREEZE_BODYNET:
                print("freezing weights at pool3_feats")
                pool3_feats = max_pool_module(fire3_feats, 2, 2, 2, 2, 'pool3', padding='VALID')
            else:
                pool3_feats = max_pool_module(fire3_feats, 2, 2, 2, 2, 'pool3', padding='VALID')
            if input_args.debug_info >= 2:
                print(pool3_feats.get_shape())

            if config.FREEZE_WEIGHTS or config.FREEZE_BODYNET:
                print("freezing weights at fire4_feats")
                fire4_feats, fire4_s, fire4_e1x1, fire4_e3x3 = fire_module(pool3_feats, 32, 128, 'fire4', 'VALID',
                                                                           'VALID', 'SAME', trainable=False)
            else:
                fire4_feats, fire4_s, fire4_e1x1, fire4_e3x3 = fire_module(pool3_feats, 32, 128, 'fire4', 'VALID',
                                                                           'VALID', 'SAME', trainable=True)
            if input_args.debug_info >= 2:
                print(fire4_feats.get_shape())

            if config.FREEZE_WEIGHTS or config.FREEZE_BODYNET:
                print("freezing weights at fire5_feats")
                fire5_feats, fire5_s, fire5_e1x1, fire5_e3x3 = fire_module(fire4_feats, 32, 128, 'fire5', 'VALID',
                                                                           'VALID', 'SAME', trainable=False)
            else:
                fire5_feats, fire5_s, fire5_e1x1, fire5_e3x3 = fire_module(fire4_feats, 32, 128, 'fire5', 'VALID',
                                                                           'VALID', 'SAME', trainable=True)
            if input_args.debug_info >= 2:
                print(fire5_feats.get_shape())

            if config.FREEZE_WEIGHTS or config.FREEZE_BODYNET:
                print("freezing weights at pool5_feats")
                pool5_feats = max_pool_module(fire5_feats, 2, 2, 2, 2, 'pool5', padding='VALID')
            else:
                pool5_feats = max_pool_module(fire5_feats, 2, 2, 2, 2, 'pool5', padding='VALID')
            if input_args.debug_info >= 2:
                print(pool5_feats.get_shape())

            if config.FREEZE_WEIGHTS or config.FREEZE_BODYNET:
                print("freezing weights at fire6_feats")
                fire6_feats, fire6_s, fire6_e1x1, fire6_e3x3 = fire_module(pool5_feats, 48, 192, 'fire6', 'VALID',
                                                                           'VALID', 'SAME', trainable=False)
            else:
                fire6_feats, fire6_s, fire6_e1x1, fire6_e3x3 = fire_module(pool5_feats, 48, 192, 'fire6', 'VALID',
                                                                           'VALID', 'SAME', trainable=True)
            if input_args.debug_info >= 2:
                print(fire6_feats.get_shape())

            if config.FREEZE_WEIGHTS or config.FREEZE_BODYNET:
                print("freezing weights at fire7_feats")
                fire7_feats, fire7_s, fire7_e1x1, fire7_e3x3 = fire_module(fire6_feats, 48, 192, 'fire7', 'VALID',
                                                                           'VALID', 'SAME', trainable=False)
            else:
                fire7_feats, fire7_s, fire7_e1x1, fire7_e3x3 = fire_module(fire6_feats, 48, 192, 'fire7', 'VALID',
                                                                           'VALID', 'SAME', trainable=True)
            if input_args.debug_info >= 2:
                print(fire7_feats.get_shape())

            if config.FREEZE_WEIGHTS or config.FREEZE_BODYNET:
                print("freezing weights at fire8_feats")
                fire8_feats, fire8_s, fire8_e1x1, fire8_e3x3 = fire_module(fire7_feats, 64, 256, 'fire8', 'VALID',
                                                                           'VALID', 'SAME', trainable=False)
            else:
                fire8_feats, fire8_s, fire8_e1x1, fire8_e3x3 = fire_module(fire7_feats, 64, 256, 'fire8', 'VALID',
                                                                           'VALID', 'SAME', trainable=True)
            if input_args.debug_info >= 2:
                print(fire8_feats.get_shape())

            if config.FREEZE_WEIGHTS or config.FREEZE_BODYNET:
                print("freezing weights at fire9_feats")
                fire9_feats, fire9_s, fire9_e1x1, fire9_e3x3 = fire_module(fire8_feats, 64, 256, 'fire9', 'VALID',
                                                                           'VALID', 'SAME', trainable=False)
            else:
                fire9_feats, fire9_s, fire9_e1x1, fire9_e3x3 = fire_module(fire8_feats, 64, 256, 'fire9', 'VALID',
                                                                           'VALID', 'SAME', trainable=True)
            if input_args.debug_info >= 2:
                print(fire9_feats.get_shape())

            self.feats = fire9_feats
            self.pinputimage = pinputimage
        else:
            if input_args.debug_info >= 2:
                print("Inserting skipping layers squeezenet, NOT YET IMPLEMENTED...")

    def Hierarchy(self, input_args):
        if input_args.debug_info >= 2:
            print("+++++++++++ Building Hierarchy part ...")

        if input_args.model_name == 'helnet':
            input_feat = tf.placeholder(tf.float32, shape=[None, None, None, 56], name="input_feat")
        elif input_args.model_name == 'resnet':
            input_feat = tf.placeholder(tf.float32, shape=[None, None, None, 56], name="input_feat")
        else:
            input_feat = tf.placeholder(tf.float32, shape=[None, None, None, 512], name="input_feat")

        pgtcoverage = tf.placeholder(tf.float32, shape=[None, None, None, config.TOTAL_CLASSES],
                                     name="gt_coverage")
        pgtedge = tf.placeholder(tf.float32, shape=[None, config.image_size_y, config.image_size_x, 1],
                                 name="gt_edge")
        # config.SEMANTIC_CLASSES
        pgtmask = tf.placeholder(tf.float32, shape=[None, config.image_size_y, config.image_size_x, 1],
                                 name="gt_mask")
        pgtbbox = tf.placeholder(tf.float32, shape=[None, None, None, config.BBOX_DIM],
                                 name="gt_bbox")
        pgtcovblock = tf.placeholder(tf.float32, shape=[None, None, None, config.BBOX_DIM],
                                     name="gt_covblock")
        pgtsizeblock = tf.placeholder(tf.float32, shape=[None, None, None, config.BBOX_DIM],
                                  name="gt_sizeblock")
        pgtobjblock = tf.placeholder(tf.float32, shape=[None, None, None, config.BBOX_DIM],
                                     name="gt_objblock")

        transformed_input = tf.placeholder(tf.float32, shape=[None, config.image_size_y, config.image_size_x, 3],
                                     name="transformed_input")

        self.transformed_input = tf.subtract(transformed_input, config.mean_value)

        if input_args.debug_info >= 2:
            print("input_feat:")
            print(input_feat.get_shape())
            print("gt_coverage:")
            print(pgtcoverage.get_shape())
            print("gt_edge:")
            print(pgtedge.get_shape())
            print("gt_mask:")
            print(pgtmask.get_shape())
            print("gt_bbox:")
            print(pgtbbox.get_shape())
            print("gt_covblock:")
            print(pgtcovblock.get_shape())
            print("gt_sizeblock:")
            print(pgtsizeblock.get_shape())
            print("gt_objblock:")
            print(pgtobjblock.get_shape())

        # Edge features
        if input_args.debug_info >= 2:
            print("+++++++++++ Edge features ...")

        if config.FREEZE_WEIGHTS:
            edge_feat1 = convolution_module(input_feat, 3, 3, 512, 1, 1, 'edge_feat1', group_id=1, trainable=False)
            edge_feat1 = nonlinear_module(edge_feat1, name='edge_feat1_relu', nl='relu')
        else:
            edge_feat1 = convolution_module(input_feat, 3, 3, 512, 1, 1, 'edge_feat1', group_id=1, trainable=True)
            edge_feat1 = nonlinear_module(edge_feat1, name='edge_feat1_relu', nl='relu')
        if input_args.enable_tensorboard == 'true':
            tf.summary.histogram("edge_feat1", edge_feat1)
        if input_args.debug_info >= 2:
            print("edge_feat1:")
            print(edge_feat1.get_shape())

        if config.FREEZE_WEIGHTS:
            edge_feat2 = convolution_module(edge_feat1, 1, 1, config.EDGES_CLASSES, 1, 1, 'edge_feat2', group_id=1,
                                            trainable=False)
            edge_feat2 = nonlinear_module(edge_feat2, name='edge_feat2_relu', nl='relu')
        else:
            edge_feat2 = convolution_module(edge_feat1, 1, 1, config.EDGES_CLASSES, 1, 1, 'edge_feat2', group_id=1,
                                            trainable=True)
            edge_feat2 = nonlinear_module(edge_feat2, name='edge_feat2_relu', nl='relu')
        if input_args.enable_tensorboard == 'true':
            tf.summary.histogram("edge_feat1", edge_feat2)
        if input_args.debug_info >= 2:
            print(edge_feat2.get_shape())

        if config.FREEZE_WEIGHTS:
            edge_influence = convolution_module(edge_feat2, 1, 1, config.SEMANTIC_CLASSES, 1, 1, 'edge_influence',
                                                trainable=False)
            edge_influence = nonlinear_module(edge_influence, name='edge_influence_relu', nl='relu')
        else:
            edge_influence = convolution_module(edge_feat2, 1, 1, config.SEMANTIC_CLASSES, 1, 1, 'edge_influence',
                                                trainable=True)
            edge_influence = nonlinear_module(edge_influence, name='edge_influence_relu', nl='relu')
        if input_args.enable_tensorboard == 'true':
            tf.summary.histogram("edge_influence", edge_influence)
        if input_args.debug_info >= 2:
            print("edge_influence:")
            print(edge_influence.get_shape())

        with tf.variable_scope('edge_upscores', reuse=None) as scope:
            w = weight_module('edge_weights', [32, 32, config.EDGES_CLASSES, config.EDGES_CLASSES], init='normal', group_id=1)
            if input_args.model_name == 'helnet':
                # has to match Imgsize/inputfeat_size
                edge_upscores = tf.nn.conv2d_transpose(edge_feat2, w, [tf.shape(input_feat)[0], config.image_size_y,
                                                                       config.image_size_x, config.EDGES_CLASSES],
                                                       [1, int(config.image_size_y/(config.image_size_y/config.stride)),
                                                        int(config.image_size_x/(config.image_size_x/config.stride)), 1],
                                                       padding='SAME')
            elif input_args.model_name == 'resnet':
                # has to match Imgsize/inputfeat_size
                edge_upscores = tf.nn.conv2d_transpose(edge_feat2, w, [tf.shape(input_feat)[0], config.image_size_y,
                                                                       config.image_size_x, config.EDGES_CLASSES],
                                                       [1, int(config.image_size_y/(config.image_size_y/config.stride)),
                                                        int(config.image_size_x/(config.image_size_x/config.stride)), 1],
                                                       padding='SAME')
            else:
                # has to match Imgsize/inputfeat_size
                edge_upscores = tf.nn.conv2d_transpose(edge_feat2, w, [tf.shape(input_feat)[0], config.image_size_y,
                                                                       config.image_size_x, config.EDGES_CLASSES],
                                                       [1, int(config.image_size_y/(config.image_size_y/config.stride)),
                                                        int(config.image_size_x/(config.image_size_x/config.stride)), 1],
                                                       padding='SAME')

        # Rectify unit
        if config.FREEZE_WEIGHTS:
            edge_upscores = tf.nn.relu(edge_upscores)
        else:
            edge_upscores = tf.nn.relu(edge_upscores)

        if input_args.enable_tensorboard == 'true':
            tf.summary.histogram("edge_upscores", edge_upscores)
        if input_args.debug_info >= 2:
            print("edge_upscores:")
            print(edge_upscores.get_shape())

        if input_args.execution_mode == 'train':
            cross_entropy_edge = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=edge_upscores,
                                                                                labels=tf.squeeze(tf.to_int32(pgtedge),
                                                                                                  axis=[3]),
                                                                                name='edge_entropy')
            loss_edge = tf.reduce_mean(cross_entropy_edge, name='edge_entropy_mean')

        pred_edge = tf.argmax(edge_upscores, axis=3, name="pred_edge")

        # ----------------------------------------------------------------------------------------------------------------------------
        # Segmentation features
        if input_args.debug_info >= 2:
            print("+++++++++++ Segmentation features ...")

        if config.FREEZE_WEIGHTS:
            seg_feat1 = convolution_module(input_feat, 3, 3, 512, 1, 1, 'seg_feat1', group_id=2, trainable=False)
            seg_feat1 = nonlinear_module(seg_feat1, name='seg_feat1_relu', nl='relu')
        else:
            seg_feat1 = convolution_module(input_feat, 3, 3, 512, 1, 1, 'seg_feat1', group_id=2, trainable=True)
            seg_feat1 = nonlinear_module(seg_feat1, name='seg_feat1_relu', nl='relu')
        if input_args.enable_tensorboard == 'true':
            tf.summary.histogram("seg_feat1", seg_feat1)
        if input_args.debug_info >= 2:
            print("seg_feat1:")
            print(seg_feat1.get_shape())

        if config.FREEZE_WEIGHTS:
            seg_feat2 = convolution_module(seg_feat1, 3, 3, 512, 1, 1, 'seg_feat2', group_id=2, trainable=False)
            seg_feat2 = nonlinear_module(seg_feat2, name='seg_feat2_relu', nl='relu')
        else:
            seg_feat2 = convolution_module(seg_feat1, 3, 3, 512, 1, 1, 'seg_feat2', group_id=2, trainable=True)
            seg_feat2 = nonlinear_module(seg_feat2, name='seg_feat2_relu', nl='relu')
        if input_args.enable_tensorboard == 'true':
            tf.summary.histogram("seg_feat2", seg_feat2)
        if input_args.debug_info >= 2:
            print("seg_feat2:")
            print(seg_feat2.get_shape())

        if config.FREEZE_WEIGHTS:
            seg_feat3 = convolution_module(seg_feat2, 1, 1, config.SEMANTIC_CLASSES, 1, 1, 'seg_feat3', group_id=2,
                                           trainable=False)
            seg_feat3 = nonlinear_module(seg_feat3, name='seg_feat3_relu', nl='relu')
        else:
            seg_feat3 = convolution_module(seg_feat2, 1, 1, config.SEMANTIC_CLASSES, 1, 1, 'seg_feat3', group_id=2,
                                           trainable=True)
            seg_feat3 = nonlinear_module(seg_feat3, name='seg_feat3_relu', nl='relu')
        if input_args.enable_tensorboard == 'true':
            tf.summary.histogram("seg_feat3", seg_feat3)
        if input_args.debug_info >= 2:
            print("seg_feat3:")
            print(seg_feat3.get_shape())

        if config.FREEZE_WEIGHTS:
            seg_concat_feat = tf.concat(axis=3, values=[edge_influence, seg_feat3], name='seg_concat_feat')
        else:
            seg_concat_feat = tf.concat(axis=3, values=[edge_influence, seg_feat3], name='seg_concat_feat')
        if input_args.debug_info >= 2:
            print("seg_concat_feat:")
            print(seg_concat_feat.get_shape())

        if config.FREEZE_WEIGHTS:
            seg_feat4 = convolution_module(seg_concat_feat, 1, 1, config.SEMANTIC_CLASSES, 1, 1, 'seg_feat4',
                                           group_id=2, trainable=False)
            seg_feat4 = nonlinear_module(seg_feat4, name='seg_feat4_relu', nl='relu')
        else:
            seg_feat4 = convolution_module(seg_concat_feat, 1, 1, config.SEMANTIC_CLASSES, 1, 1, 'seg_feat4',
                                           group_id=2, trainable=True)
            seg_feat4 = nonlinear_module(seg_feat4, name='seg_feat4_relu', nl='relu')
        if input_args.enable_tensorboard == 'true':
            tf.summary.histogram("seg_feat4", seg_feat4)
        if input_args.debug_info >= 2:
            print("seg_feat4:")
            print(seg_feat4.get_shape())

        if config.FREEZE_WEIGHTS:
            seg_influence = convolution_module(seg_feat4, 1, 1, config.SEMANTIC_CLASSES, 1, 1, 'seg_influence',
                                               group_id=2, trainable=False)
            seg_influence = nonlinear_module(seg_influence, name='seg_influence_relu', nl='relu')
        else:
            seg_influence = convolution_module(seg_feat4, 1, 1, config.SEMANTIC_CLASSES, 1, 1, 'seg_influence',
                                               group_id=2, trainable=True)
            seg_influence = nonlinear_module(seg_influence, name='seg_influence_relu', nl='relu')
        if input_args.enable_tensorboard == 'true':
            tf.summary.histogram("seg_influence", seg_influence)
        if input_args.debug_info >= 2:
            print("seg_influence:")
            print(seg_influence.get_shape())

        with tf.variable_scope('seg_upscores', reuse=None) as scope:
            w = weight_module('seg_weights', [32, 32, config.SEMANTIC_CLASSES, config.SEMANTIC_CLASSES], init='normal', group_id=2)
            if input_args.model_name == 'helnet':
                # has to match Imgsize/inputfeat_size
                seg_upscores = tf.nn.conv2d_transpose(seg_feat4, w, [tf.shape(input_feat)[0], config.image_size_y,
                                                                       config.image_size_x, config.SEMANTIC_CLASSES],
                                                       [1, int(config.image_size_y/(config.image_size_y/config.stride)),
                                                        int(config.image_size_x/(config.image_size_x/config.stride)), 1],
                                                       padding='SAME')
            elif input_args.model_name == 'resnet':
                # has to match Imgsize/inputfeat_size
                seg_upscores = tf.nn.conv2d_transpose(seg_feat4, w, [tf.shape(input_feat)[0], config.image_size_y,
                                                                       config.image_size_x, config.SEMANTIC_CLASSES],
                                                       [1, int(config.image_size_y/(config.image_size_y/config.stride)),
                                                        int(config.image_size_x/(config.image_size_x/config.stride)), 1],
                                                       padding='SAME')
            else:
                # has to match Imgsize/inputfeat_size
                seg_upscores = tf.nn.conv2d_transpose(seg_feat4, w, [tf.shape(input_feat)[0], config.image_size_y,
                                                                       config.image_size_x, config.SEMANTIC_CLASSES],
                                                       [1, int(config.image_size_y/(config.image_size_y/config.stride)),
                                                        int(config.image_size_x/(config.image_size_x/config.stride)), 1],
                                                       padding='SAME')

        # rectify this layer before calculating losses or everything can explode
        if config.FREEZE_WEIGHTS:
            seg_upscores = tf.nn.relu(seg_upscores)
        else:
            seg_upscores = tf.nn.relu(seg_upscores)

        if input_args.enable_tensorboard == 'true':
            tf.summary.histogram("seg_upscores", seg_upscores)
        if input_args.debug_info >= 2:
            print("seg_upscores:")
            print(seg_upscores.get_shape())

        pred_seg = tf.argmax(seg_upscores, axis=3, name="pred_seg")

        if input_args.execution_mode == 'train':
            cross_entropy_segmentation = (tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_upscores,
                                                                                         labels=tf.squeeze(
                                                                                             tf.to_int32(pgtmask),
                                                                                             axis=[3]),
                                                                                         name="seg_entropy"))
            # cross_entropy_segmentation = (tf.nn.softmax_cross_entropy_with_logits(logits=seg_upscores,
            #                                                                       labels=pgtmask,
            #                                                                       name="seg_entropy"))
            loss_seg = tf.reduce_mean(cross_entropy_segmentation, name='seg_entropy_mean')

        # ---------------------------------------------------------------------------------
        # Coverage features
        if input_args.debug_info >= 2:
            print("+++++++++++ Coverage features ...")
        cov_feat1 = convolution_module(input_feat, 3, 3, 512, 1, 1, 'cov_feat1', group_id=3, trainable=True)
        cov_feat1 = nonlinear_module(cov_feat1, name='cov_feat1_relu', nl='relu')
        if input_args.enable_tensorboard == 'true':
            tf.summary.histogram("cov_feat1", cov_feat1)
        if input_args.debug_info >= 2:
            print("cov_feat1:")
            print(cov_feat1.get_shape())

        cov_feat2 = convolution_module(cov_feat1, 3, 3, 512, 1, 1, 'cov_feat2', group_id=3, trainable=True)
        cov_feat2 = nonlinear_module(cov_feat2, name='cov_feat2_relu', nl='relu')
        if input_args.enable_tensorboard == 'true':
            tf.summary.histogram("cov_feat2", cov_feat2)
        if input_args.debug_info >= 2:
            print("cov_feat2:")
            print(cov_feat2.get_shape())

        cov_feat3 = convolution_module(cov_feat2, 1, 1, config.TOTAL_CLASSES, 1, 1, 'cov_feat3', group_id=3,
                                       trainable=True)
        cov_feat3 = nonlinear_module(cov_feat3, name='cov_feat3_relu', nl='relu')
        if input_args.enable_tensorboard == 'true':
            tf.summary.histogram("cov_feat3", cov_feat3)
        if input_args.debug_info >= 2:
            print("cov_feat3:")
            print(cov_feat3.get_shape())

        cov_concat_feat = tf.concat(axis=3, values=[edge_influence, seg_influence, cov_feat3],
                                    name='cov_concat_feat')
        if input_args.debug_info >= 2:
            print("edge_influence:")
            print(cov_concat_feat.get_shape())

        cov_feat4 = convolution_module(cov_concat_feat, 1, 1, config.TOTAL_CLASSES, 1, 1, 'cov_feat4',
                                       group_id=3, trainable=True)
        cov_feat4 = nonlinear_module(cov_feat4, name='cov_feat4_relu', nl='sigmoid')
        if input_args.enable_tensorboard == 'true':
            tf.summary.histogram("cov_feat4", cov_feat4)
        if input_args.debug_info >= 2:
            print("cov_feat4:")
            print(cov_feat4.get_shape())

        cov_influence = convolution_module(cov_feat4, 1, 1, config.SEMANTIC_CLASSES, 1, 1, 'cov_influence', group_id=3,
                                           trainable=True)
        cov_influence = nonlinear_module(cov_influence, name='cov_influence_relu', nl='sigmoid')
        if input_args.enable_tensorboard == 'true':
            tf.summary.histogram("cov_influence", cov_influence)
        if input_args.debug_info >= 2:
            print("cov_influence:")
            print(cov_influence.get_shape())

        # cov_feat5 = convolution_module(cov_feat4, 1, 1, config.TOTAL_CLASSES, 1, 1, 'cov_feat5', group_id=3)
        coverage = cov_feat4 #nonlinear_module(cov_feat5, name='sigmoid')
        if input_args.debug_info >= 2:
            print("coverage:")
            print(coverage.get_shape())

        if input_args.execution_mode == 'train':
            # mask_coverage = tf.ones_like(pgtcoverage)
            # mask_dontcare = tf.zeros_like(pgtcoverage[:, :, :, 1])
            #
            # #filtering everything out
            # tf.multiply(pgtcoverage[:, :, :, 1], mask_coverage)
            # tf.multiply(coverage[:, :, :, 1], mask_dontcare)

            # l2 normalization
            loss_cov = tf.square((coverage - pgtcoverage))
            # loss_cov = tf.reduce_mean(loss_cov)
            loss_cov = tf.reduce_sum(loss_cov)

        # ---------------------------------------------------------------------------------
        #  BBox features
        if input_args.debug_info >= 2:
            print("+++++++++++ BBox features ...")
        bbox_feat1 = convolution_module(input_feat, 3, 3, 512, 1, 1, 'bbox_feat1', group_id=4, trainable=True)
        bbox_feat1 = nonlinear_module(bbox_feat1, name='bbox_feat1_relu', nl='relu')
        if input_args.enable_tensorboard == 'true':
            tf.summary.histogram("bbox_feat1", bbox_feat1)
        if input_args.debug_info >= 2:
            print("bbox_feat1:")
            print(bbox_feat1.get_shape())

        bbox_feat2 = convolution_module(bbox_feat1, 3, 3, 512, 1, 1, 'bbox_feat2', group_id=4, trainable=True)
        bbox_feat2 = nonlinear_module(bbox_feat2, name='bbox_feat2_relu', nl='relu')
        if input_args.enable_tensorboard == 'true':
            tf.summary.histogram("bbox_feat2", bbox_feat2)
        if input_args.debug_info >= 2:
            print("bbox_feat2:")
            print(bbox_feat2.get_shape())

        bbox_feat3 = convolution_module(bbox_feat2, 1, 1, config.BBOX_DIM, 1, 1, 'bbox_feat3', group_id=4,
                                        trainable=True)
        bbox_feat3 = nonlinear_module(bbox_feat3, name='bbox_feat3_relu', nl='relu')
        if input_args.enable_tensorboard == 'true':
            tf.summary.histogram("bbox_feat3", bbox_feat3)
        if input_args.debug_info >= 2:
            print("bbox_feat3:")
            print(bbox_feat3.get_shape())

        bbox_concat_feat = tf.concat(axis=3, values=[edge_influence, seg_influence, cov_influence, bbox_feat3],
                                     name='bbox_concat_feat')
        if input_args.debug_info >= 2:
            print("bbox_concat_feat:")
            print(bbox_concat_feat.get_shape())

        bbox_dilation1 = atrous_convolution_module(bbox_concat_feat, 3, 3, 256, 1, 'bbox_dilation1', group_id=4,
                                                   trainable=True)
        bbox_dilation1 = nonlinear_module(bbox_dilation1, name='bbox_dilation1_relu', nl='relu')
        if input_args.enable_tensorboard == 'true':
            tf.summary.histogram("bbox_dilation1", bbox_dilation1)
        if input_args.debug_info >= 2:
            print("bbox_dilation1:")
            print(bbox_dilation1.get_shape())
        bbox_dilation2 = atrous_convolution_module(bbox_dilation1, 3, 3, 128, 2, 'bbox_dilation2', group_id=4,
                                                   trainable=True)
        bbox_dilation2 = nonlinear_module(bbox_dilation2, name='bbox_dilation2_relu')
        if input_args.enable_tensorboard == 'true':
            tf.summary.histogram("bbox_dilation2", bbox_dilation2)
        if input_args.debug_info >= 2:
            print("bbox_dilation2:")
            print(bbox_dilation2.get_shape())
        bbox_dilation3 = atrous_convolution_module(bbox_dilation2, 3, 3, 64, 2, 'bbox_dilation3', group_id=4,
                                                   trainable=True)
        bbox_dilation3 = nonlinear_module(bbox_dilation3, name='bbox_dilation3_relu', nl='relu')
        if input_args.enable_tensorboard == 'true':
            tf.summary.histogram("bbox_dilation3", bbox_dilation3)
        if input_args.debug_info >= 2:
            print("bbox_dilation3:")
            print(bbox_dilation3.get_shape())
        bbox_dilation4 = atrous_convolution_module(bbox_dilation3, 3, 3, config.BBOX_DIM, 2,
                                                   'bbox_dilation4', group_id=4, trainable=True)
        bbox_dilation4 = nonlinear_module(bbox_dilation4, name='bbox_dilation4_relu', nl='relu')
        if input_args.enable_tensorboard == 'true':
            tf.summary.histogram("bbox_dilation4", bbox_dilation4)
        if input_args.debug_info >= 2:
            print("bbox_dilation4:")
            print(bbox_dilation4.get_shape())
        bbox_dilation5 = atrous_convolution_module(bbox_dilation4, 3, 3, config.BBOX_DIM, 2,
                                                   'bbox_dilation5', group_id=4, trainable=True)
        bbox_dilation5 = nonlinear_module(bbox_dilation5, name='bbox_dilation5_relu', nl='relu')
        if input_args.enable_tensorboard == 'true':
            tf.summary.histogram("bbox_dilation5", bbox_dilation5)
        if input_args.debug_info >= 2:
            print("bbox_dilation5:")
            print(bbox_dilation5.get_shape())

        bbox_regressor = convolution_module(bbox_dilation5, 1, 1, config.BBOX_DIM, 1, 1,
                                            'bbox_regressor', group_id=4, trainable=True)
        # bbox_regressor = nonlinear_module(bbox_regressor, name='bbox_regressor_relu', nl='relu')
        if input_args.enable_tensorboard == 'true':
            tf.summary.histogram("bbox_regressor", bbox_regressor)
        if input_args.debug_info >= 2:
            print("bbox_regressor:")
            print(bbox_regressor.get_shape())

        bboxes_masked = tf.multiply(bbox_regressor, pgtcovblock)
        bboxes_masked_norm = tf.multiply(bboxes_masked, pgtsizeblock)
        bboxes_obj_masked_norm = tf.multiply(bboxes_masked_norm, pgtobjblock)

        if input_args.execution_mode == 'train':
            boxA = bboxes_obj_masked_norm
            boxB = pgtbbox

            xA = tf.maximum(boxA[:, :, :, config.BBOX_DIM-4], boxB[:, :, :, config.BBOX_DIM-4])
            yA = tf.maximum(boxA[:, :, :, config.BBOX_DIM-3], boxB[:, :, :, config.BBOX_DIM-3])
            xB = tf.minimum(boxA[:, :, :, config.BBOX_DIM-2], boxB[:, :, :, config.BBOX_DIM-2])
            yB = tf.minimum(boxA[:, :, :, config.BBOX_DIM-1], boxB[:, :, :, config.BBOX_DIM-1])

            # compute the area of intersection rectangle
            interArea = tf.add(tf.subtract(xB, xA), tf.ones_like(xA)) * tf.add(tf.subtract(yB, yA), tf.ones_like(yA))

            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = tf.add(tf.subtract(boxA[:, :, :, config.BBOX_DIM-2], boxA[:, :, :, config.BBOX_DIM-4]),
                              tf.ones_like(boxA[:, :, :, config.BBOX_DIM-4])) * \
                       tf.add(tf.subtract(boxA[:, :, :, config.BBOX_DIM-1], boxA[:, :, :, config.BBOX_DIM-3]),
                              tf.ones_like(boxA[:, :, :, config.BBOX_DIM-4]))
            boxBArea = tf.add(tf.subtract(boxB[:, :, :, config.BBOX_DIM-2], boxB[:, :, :, config.BBOX_DIM-4]),
                              tf.ones_like(boxB[:, :, :, config.BBOX_DIM-4])) * \
                       tf.add(tf.subtract(boxB[:, :, :, config.BBOX_DIM-1], boxB[:, :, :, config.BBOX_DIM-3]),
                              tf.ones_like(boxB[:, :, :, config.BBOX_DIM-4]))

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            denom = tf.subtract(tf.add(boxAArea, boxBArea), interArea)
            IoU_individual = tf.divide(interArea, denom)
            IoU = tf.reduce_mean(tf.divide(interArea, denom))
            counter_mAP = tf.multiply(tf.cast(tf.greater_equal(IoU_individual, 0.85), tf.float32), tf.ones_like(IoU_individual))
            counter_mAP = tf.reduce_mean(counter_mAP)
            self.IoU = tf.subtract(tf.ones_like(IoU), IoU)
            self.mAP = tf.subtract(tf.ones_like(counter_mAP), counter_mAP)

            # L1-function for normalization
            # loss_bbox = tf.abs(bboxes_obj_masked_norm - pgtbbox)
            # loss_bbox = tf.reduce_mean(loss_bbox)
            loss_bbox = tf.square(bboxes_obj_masked_norm - pgtbbox)
            loss_bbox = tf.reduce_sum(loss_bbox)

        if input_args.execution_mode == 'train':
            if True:
                edge_weight = 1.0
                seg_weight = 1.0
                cov_weight = 500.0
                bbox_weight = 1000.0
                loss0 = edge_weight*loss_edge + seg_weight*loss_seg + cov_weight*loss_cov +bbox_weight*loss_bbox #'  '
            else:
                alpha = 0.00001
                beta = 0.00001
                loss0 = beta * loss_cov +(1 - beta) * loss_bbox # '3 * alpha * loss_edge + alpha * loss_seg +  '

        if input_args.execution_mode == 'train':
            ## Experimentation with some normalization functions
            loss1 = 0.001 * (tf.add_n(tf.get_collection('l2_1')) + tf.add_n(tf.get_collection('l2_2')) +
                               tf.add_n(tf.get_collection('l2_3')) + tf.add_n(tf.get_collection('l2_4')))

            lossall = loss0 + loss1

            global_step = variable_scope.get_variable(  # this needs to be defined for tf.contrib.layers.optimize_loss()
                "global_step", [],
                trainable=False,
                dtype=dtypes.int64,
                initializer=init_ops.constant_initializer(0, dtype=dtypes.int64))

            # self.global_step = global_step

            # train_op = tf.train.AdamOptimizer(config.LEARNING_RATE).minimize(lossall)
            train_op = tf.contrib.layers.optimize_loss(
                loss=lossall,
                global_step=global_step,
                learning_rate=config.LEARNING_RATE,
                optimizer="Adam",
                gradient_noise_scale=None,
                gradient_multipliers=None,
                clip_gradients=None,
                learning_rate_decay_fn=None,
                update_ops=None,
                variables=None,
                name=None,
                summaries=["gradients"],
                colocate_gradients_with_ops=False,
                increment_global_step=True
            )

        self.edge_upscores = edge_upscores
        self.input_feat = input_feat
        self.pgtedge = pgtedge
        self.pgtmask = pgtmask
        self.pgtbbox = pgtbbox
        self.pgtcoverage = pgtcoverage

        self.pgtcovblock = pgtcovblock
        self.pgtobjblock = pgtobjblock
        self.pgtsizeblock = pgtsizeblock

        if input_args.enable_tensorboard == 'true':
            # ===== Segmentation Mask
            # self.gt_mask_vis = tf.reduce_sum(pgtmask[:, :, :, 1:], axis=3, keep_dims=True)
            self.gt_mask_vis = tf.reduce_sum(pgtmask, axis=3, keep_dims=True)

            # self.pred_mask_vis = tf.reduce_sum(pred_seg, axis=3, keep_dims=True)
            self.pred_mask_vis = tf.cast(tf.minimum(tf.multiply(tf.cast(tf.expand_dims(pred_seg, axis=3),
                                                                        dtype=tf.float32), 32.0), 255.0),
                                         dtype=tf.uint8)
            # ===== Edges Mask
            # self.pred_edges_vis = pred_edge
            self.pred_edges_vis = tf.cast(tf.minimum(tf.multiply(tf.cast(tf.expand_dims(pred_edge, axis=3),
                                                                         dtype=tf.float32), 32.0), 255.0),
                                          dtype=tf.uint8)

            # ===== Coverage Mask
            self.pred_coverage_vis = tf.reduce_sum(coverage, axis=3, keep_dims=True)
            # self.pred_coverage_vis = tf.cast(tf.minimum(tf.multiply(tf.cast(coverage,
            #                                                              dtype=tf.float32), 32.0), 255.0),
            #                               dtype=tf.uint8)

            self.gt_coverage_vis = tf.reduce_sum(pgtcoverage, axis=3, keep_dims=True)

        if input_args.execution_mode == 'train':
            if True:
                #semmantic
                self.loss_edge = loss_edge*edge_weight
                self.loss_seg = loss_seg*seg_weight
                # BBox
                self.loss_cov = loss_cov*cov_weight
                self.loss_bbox = loss_bbox*bbox_weight
            else:
                #semmantic
                self.loss_edge = loss_edge
                self.loss_seg = loss_seg
                # BBox
                self.loss_cov = loss_cov
                self.loss_bbox = loss_bbox

            self.lossall = lossall
            self.train_op = train_op

        self.coverage = coverage
        self.bbox_regressor = bbox_regressor
        self.pred_seg = pred_seg
        self.pred_edge = pred_edge

        self.seg_upscores = seg_upscores
        self.edge_upscores = edge_upscores

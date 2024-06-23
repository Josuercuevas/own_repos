import tensorflow as tf
import numpy as np
import argparse
import os
import cv2
from PIL import Image

caffe_root = 'CAFFE_SOURCE_BUILD_PATH'  # this file is expected to be in {caffe_root}/examples

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
sys.path.append(os.path.abspath(caffe_root + 'python'))#caffe
import caffe


Squeezedet_path = 'PATH_TO_DESIRED_MODEL_TENSORFLOW_DEFINITION' # I used my tensorflow implementation of squeezeDet here.
sys.path.append(Squeezedet_path)# squeezedet
from config import *
from train import _draw_box
from nets import *

def transform_model(arguments):
    print("Getting prototxt ...")
    MODEL_PROTO_CAFFE = arguments.prototxt
    caffe.set_mode_cpu()

    print("Getting model from caffe ...")
    net = caffe.Net(MODEL_PROTO_CAFFE, caffe.TEST)
    print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))
    print("======================================================="
          "======================================================="
          "\n\n\n\n")

    print("Getting Session ...")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:

        MODEL_CLA_TF = arguments.model
        print("Getting model ... (%s)"%MODEL_CLA_TF)

        # SqueezeDet
        mc = kitti_squeezeDet_config()
        mc.BATCH_SIZE = 1
        # model parameters will be restored from checkpoint
        mc.LOAD_PRETRAINED_MODEL = False
        model = SqueezeDet(mc, -1)
        print("Getting model Saver ...")
        saver = tf.train.Saver(model.model_params)

        print("==========================================================================")
        print("Restoring model ...")
        saver.restore(session, MODEL_CLA_TF)
        print("==========================================================================")


        print("Getting Topology ...")
        vs = [v for v in tf.all_variables() if 'kernels' in v.name and 'Momentum' not in v.name]
        for v in vs:
            var_name = v.name
            print("W -->", var_name)
            var_name = var_name.split('/')
            if var_name[0]!='conv12' and var_name[0]!='conv1' :
                var_name = (var_name[0]+'_'+var_name[1])
                print("W ---->", var_name)
            else:
                var_name = var_name[0]
                print("W ------>", var_name)

            net.params[var_name][0].data[...] = np.array(v.eval(session=session)).transpose(3,2,0,1)

            print("sum_in_TF: ", np.sum(np.array(v.eval(session=session))))
            print("sum_in_CAFFE: ", np.sum(net.params[var_name][0].data))

            print(np.array(v.eval(session=session)).shape)
            print(net.params[var_name][0].data.shape)


        vs = [v for v in tf.all_variables() if 'biases' in v.name and 'Momentum' not in v.name]
        for v in vs:
            var_name = v.name
            print("B -->", var_name)
            var_name = var_name.split('/')
            if var_name[0]!='conv12' and var_name[0]!='conv1' :
                var_name = (var_name[0]+'_'+var_name[1])
                print("B ---->", var_name)
            else:
                var_name = var_name[0]
                print("B ------>", var_name)

            net.params[var_name][1].data[...] = np.array(v.eval(session=session))
            print(np.array(v.eval(session=session)).shape)
            print(net.params[var_name][0].data.shape)

        net.save(arguments.model + '.caffemodel')


        # validate
        net2 = caffe.Net(MODEL_PROTO_CAFFE, caffe.TEST)
        net2.copy_from(arguments.model + '.caffemodel')


        input_image_path = "SAMPLE_TEST_IMAGE"
        Input_image = cv2.imread(input_image_path)
        Input_image = Input_image.astype(np.float32, copy=False)
        BGR_MEANS = np.array([[[103.939, 116.779, 123.68]]])
        Input_image = cv2.resize(Input_image, (640, 480))
        Input_image = Input_image - BGR_MEANS

        det_boxes, det_probs, det_class, conv12_tf, class_feat, conf_feat = \
        session.run([model.det_boxes, model.det_probs, model.det_class, model.preds, model.pred_class_probs,
            model.pred_conf],
            feed_dict={model.image_input:[Input_image]})


        Input_image = np.array(Input_image).reshape(1, 480, 640, 3).transpose(0, 3, 1, 2) # Batch | channels | H | W
        net.forward_all(**{"data":Input_image})
        conv12_caffe = net.blobs['conv12'].data

        print("Conv12_TF is os shape: ", conv12_tf.shape)
        print("conv12_caffe shape is: ", conv12_caffe.shape)

        pred_class_probs = net.blobs['pred_class_probs'].data
        print "Class probabilities are: " + str(pred_class_probs.shape)

        # =====================================
        pred_conf = net.blobs['pred_conf'].data
        print "confidences are: " + str(pred_conf.shape)

        # =====================================
        reshape_slice_pred_box_delta = net.blobs['reshape_slice_pred_box_delta'].data
        print "Bbox deltas are: " + str(reshape_slice_pred_box_delta.shape)

        print(np.sum(class_feat), np.sum(conf_feat))
        print(np.sum(pred_class_probs), np.sum(pred_conf))

        print(class_feat[0])
        print("\n")
        print(pred_class_probs[0])

        # =====================================
        class_weight_probs = np.multiply(pred_class_probs, np.reshape(pred_conf, [1, 40*30*9, 1]))
        class_probs_conf_before_nms = np.max(class_weight_probs, axis=2)
        class_prediction_before_nms = np.argmax(class_weight_probs, axis=2)

        print(det_class[0])
        print(class_prediction_before_nms[0])

        order_TF = class_feat[0].argsort()[::-1]
        order_CAFFE = pred_class_probs[0].argsort()[::-1]

        print(order_TF)
        print(order_CAFFE)



"""
    Parsing arguments for training or testing
"""
parser = argparse.ArgumentParser(description='t-SNE implementation')
def parse_args():
    """ Parse input arguments
    """

    parser.add_argument('--prototxt', dest='prototxt',
                        help='path to prototxt to be used in conversion, default [None]',
                        default=None, type=str)
    parser.add_argument('--model', dest='model',
                        help='path for model file, default [None]',
                        default=None, type=str)
    parser.add_argument('--topology', dest='topology',
                        help='model topology to be used, default [None]',
                        default=None, type=str)
    parser.add_argument('--model_name', dest='model_name',
                        help='No need to set ...',
                        default=None, type=str)
    parser.add_argument('--mode', dest='execution_mode',
                        help='No need to set ...',
                        default=None, type=str)
    parser.add_argument('--solver', dest='solver',
                        help='No need to set ...',
                        default=None, type=str)
    parser.add_argument('--add_skyplayer', dest='add_skyplayer',
                        help='No need to set ...',
                        default=None, type=str)
    parser.add_argument('--debug', dest='debug_info',
                        help='No need to set ...',
                        default=None, type=int)

    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()

if __name__ == '__main__':
    arguments = parse_args()
    transform_model(arguments)

import numpy as np
import argparse
import os
import cv2
from os import listdir
from os.path import isfile, join
import time

caffe_root = 'CAFFE_SOURCE_BUILD_PATH_HERE'  # this file is expected to be in {caffe_root}/examples

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
sys.path.append(os.path.abspath(caffe_root + 'python'))#caffe
import caffe

print("Getting prototxt ...")
MODEL_PROTO_CAFFE = "deploy_shufflenet.prototxt"
caffe.set_mode_gpu() # gpu based implementation

print("Getting model from caffe ...")
net = caffe.Net(MODEL_PROTO_CAFFE, caffe.TEST)
net.copy_from('model_snapshots/shufflenet_beanready_caffe_trained__iter_7000.caffemodel')

path_to_scan = "samples/"
onlyfiles = [f for f in listdir(path_to_scan) if isfile(join(path_to_scan, f))]

n_runs = 50
passed = 0.0
batch_size = 1

for fid in range(len(onlyfiles)):
	input_image_path = path_to_scan + onlyfiles[fid]
	print("Processing file %s" % input_image_path)

	Input_image = cv2.imread(input_image_path)
	Input_image = Input_image.astype(np.float32, copy=False)
	BGR_MEANS = np.array([[[103.939, 116.779, 123.68]]])
	Input_image = cv2.resize(Input_image, (224, 224))
	Input_image = Input_image - BGR_MEANS
	Input_image = np.array(Input_image).reshape(1, 224, 224, 3).transpose(0, 3, 1, 2) # Batch | channels | H | W

	patch_images = []
	for pid in range(batch_size):
		patch_images.append(Input_image[0])

	Input_image = np.array(patch_images).astype(np.float32)
	print(Input_image.shape)

	total_time = 0.0
	for rep in range(n_runs):
		start_inference = time.time()

		net.forward_all(**{"data":Input_image})
		class_probs = net.blobs['class_probs'].data

		print("shape of output layer: ", class_probs.shape)
		predictions = class_probs[:, :, 0, 0]
		print("Shape of prediction extraction: ", predictions.shape)
		max_val_w = np.max(predictions)
		min_val_w = np.min(predictions)
		print("Max: %f, Min: %f" % (max_val_w, min_val_w))

		print("Class predicted were:")
		print(np.argmax(predictions, axis=1))

		end_inference = time.time()
		if rep > 0:
			passed = end_inference - start_inference
			passed *= 1000.
			total_time += passed

	print("======> Time taken for inference CAFFE: %f ms" % (total_time/(rep-1)))

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

import numpy as np

# General algorithm setting
BATCH_SIZE = 64 #for HelNet/SqueezeNet/ShuffleNet
# BATCH_SIZE = 32 #for ResNet-50

# BATCH_SIZE = 1 # just for testing

# for DSD
ENABLE_DSD_TRAINING = True
PRUNE_THRESHOLD = 0.05

# this has to be dynamic or changed accordingly
# Gray
# WIDTH_GLOBAL = 320
# HEIGHT_GLOBAL = 243
# CHANNELS_G = 1

# RGB
# WIDTH_GLOBAL = 896
# HEIGHT_GLOBAL = 592
# CHANNELS_G = 3

# # RGB
# WIDTH_GLOBAL = 1024
# HEIGHT_GLOBAL = 1024
# CHANNELS_G = 3
# # classes names
# classes_names = ['0:normal', '1:dont_care', '2:black', '3:sour', '4:shell/broken', '5:fungus', '6:insect']
# # hot encoding
# N_CLASSES = 7

# RGB
WIDTH_GLOBAL = 60
HEIGHT_GLOBAL = 60
CHANNELS_G = 3
# hot encoding
N_CLASSES = 206

# do some image augmentations if activated
ENABLE_AUGMENTATION = False

# These are settings for experimental topologies-----------------
# debug read images
DEBUG_IMAGES = False

# Enable random crop on the images
ENABLE_RANDOM_CROP = False
crop_width = 640
crop_height = 480

# New regularization layer type
ENABLE_SELU = False
# These are settings for experimental topologies-----------------

'''
    Evaluation according to the testing set input.
'''
def evaluation(predictions, labels, user_arguments):
    if user_arguments.debug_info >= 2:
        print("Evaluating testing samples against their true labels")

    correct = 0
    num_labels = N_CLASSES
    correct += np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))

    if (user_arguments.debug_info >= 2) and (user_arguments.execution_mode == 'test'):
        for sid in range(labels.shape[0]):
            if user_arguments.debug_info >= 3:
                print(predictions[sid])
                print(labels[sid])
            print(np.argmax(predictions, 1)[sid], np.argmax(labels, 1)[sid])

    class_count = np.zeros(labels.shape[1], dtype=np.int32)
    for i in range(labels.shape[0]):
        class_count[np.argmax(labels[i], 0)] += 1

    total = predictions.shape[0]

    error = 100.0 - (100 * float(correct) / float(total))

    confusions = np.zeros([num_labels, num_labels], np.float32)

    bundled = zip(np.argmax(predictions, 1), np.argmax(labels, 1))
    for predicted, actual in bundled:
        confusions[predicted, actual] += 1

    precision = 0.0
    recall = 0.0
    for i in range(labels.shape[1]):
        if class_count[i]>0:
            precision += (confusions[i, i]*1.0)/(class_count[i])
            if np.sum(confusions[i, i:labels.shape[1]]) > 0:
                recall += (confusions[i, i]*1.0)/np.sum(confusions[i,i:labels.shape[1]])

    precision /= labels.shape[0]
    precision *= 100
    recall /= labels.shape[0]
    recall *= 100

    if user_arguments.debug_info >= 2 and num_labels < 10:
        print("Confusion Matrix result from all the classes used during testing was: ")
        print(confusions)

    return error, precision, recall, confusions

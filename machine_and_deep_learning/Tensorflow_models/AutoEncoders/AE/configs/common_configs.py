"""
    Autoencoders implementation in Tensorflow

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

import numpy as np
import sys
import tensorflow as tf

# General setting
BATCH_SIZE = 1

# this has to be dynamic or changed accordingly
# # Gray
# WIDTH_GLOBAL = 320
# HEIGHT_GLOBAL = 243
# CHANNELS_G = 1
# # hot encoding
# N_CLASSES = 15

# RGB 1
# WIDTH_GLOBAL = 896
# HEIGHT_GLOBAL = 592
# CHANNELS_G = 3
## hot encoding
# N_CLASSES = 15

# # RGB 2
# WIDTH_GLOBAL = 480
# HEIGHT_GLOBAL = 360
# CHANNELS_G = 3
# # hot encoding
# N_CLASSES = 1

# RGB 2
WIDTH_GLOBAL = 1024
HEIGHT_GLOBAL = 1024
CHANNELS_G = 3
# hot encoding
N_CLASSES = 2


# Gray small
# WIDTH_GLOBAL = 32
# HEIGHT_GLOBAL = 24
# CHANNELS_G = 1
# # hot encoding
# N_CLASSES = 14

# debug read images
DEBUG_IMAGES = False

# learning rate
LEARNING_RATE = 1e-3

# allow pooling
ALLOW_POOLING = False

# Pruning threshold
PRUNE_THRESHOLD = 0.05

'''
    Evaluation according to the testing set input.
'''
def evaluation(predictions, labels, user_arguments):
    # print(np.squeeze(predictions, axis=0))
    return (1.0/np.log(np.exp(1.0)+np.mean(np.power(np.squeeze(predictions, axis=0) - labels, 2))))*100.0

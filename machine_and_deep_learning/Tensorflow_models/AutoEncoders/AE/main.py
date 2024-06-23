"""
    Autoencoders implementation in Tensorflow

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
"""

import tensorflow as tf
import sys
import time
import os
import argparse

# own modules
from model_builder import *
from database_manager import *
import configs.common_configs as config

# ========================================argument parsing =============================================

"""
    Parsing arguments for training or testing
"""
parser = argparse.ArgumentParser(description='AutoEncoders Implementation')
def parse_args():
    """ Parse input arguments
    """

    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device Id to be used for processing [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver to be used for training (moment/adam) [adam]',
                        default='adam', type=str)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs PER STAGE (D-S-D) to train [100]',
                        default=100, type=int)
    parser.add_argument('--weights', dest='pretrained_weights',
                        help='initialize with pretrained model weights [none]',
                        default=None, type=str)
    parser.add_argument('--save_model', dest='model_dir',
                        help='Directory to save the model [none]',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train/test on [dataset/]',
                        default='dataset/', type=str)
    parser.add_argument('--mode', dest='execution_mode',
                        help='mode in which the program to be executed (train/test/extraction) [train]',
                        default='train', type=str)
    parser.add_argument('--add_skip', dest='add_skyplayer',
                        help='add skip layers ResNet type (true/false) [false]',
                        default='false', type=str)
    parser.add_argument('--debug', dest='debug_info',
                        help='Debug level output (0(none)/1(warnings)/2(debug)/3(info)/4(verbose)) [4]',
                        default=4, type=int)
    parser.add_argument('--enable_tensorboard', dest='enable_tensorboard',
                        help='type of model to be used (false/true) [false]',
                        default='false', type=str)
    parser.add_argument('--extract_database', dest='extract_database',
                        help='determine if we need to build the database [false]',
                        default='false', type=str)
    parser.add_argument('--db_path', dest='location_database',
                        help='database path to be used to load images [none]',
                        default=None, type=str)
    parser.add_argument('--db_save', dest='save_database',
                        help='location where the database is gonna be saved [none]',
                        default=None, type=str)
    parser.add_argument('--augment_data', dest='augment_data',
                        help='performs data augmentation (true/false) [false]',
                        default='false', type=str)

    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()

# ==================================== argument parsing =================================================

# ============= TRAINING ==========================
def perform_training(input_args):
    if input_args.debug_info >= 1:
        print("Performing training of Fully Convolutional Auto-Encoder Net")

    # read all data to perform setting and training, but do not load it to memory we dont need to
    if input_args.extract_database == 'true':
        print("================== LOADING/Extracting TRAINING DATA ==================")
        if input_args.location_database == None or input_args.location_database == None:
            print("No database specified, please input absolute path of the database, and where to save it")
            parser.print_help()
            sys.exit(1)

        extract_database(input_args)
        sys.exit(0)

    test_x, test_y, train_x, train_y, validation_x, validation_y, train_size = read_data('dataset/Training_set.h5',
                                                                                         'dataset/Testing_set.h5',
                                                                                         'dataset/Validate_set.h5')

    data_reader = {
        'test_x': test_x,
        'test_y': test_y,
        'train_x':  train_x,
        'train_y': train_y,
        'validation_x': validation_x,
        'validation_y': validation_y,
        'train_size': train_size
    }

    if input_args.debug_info >= 4:
        print("Data reader engine is as follows: ")
        print(data_reader)

    if input_args.gpu_id >= 0:
        print("Placing device %d for training", input_args.gpu_id)
    else:
        print("WARNING: Running purely on CPU")

    # make sure we dont allocate all the memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        aenet = FC_AutoEncoder(input_args=input_args, session=session)
        if input_args.pretrained_weights is not None:
            aenet.load(sess=session, input_args=input_args)
        aenet.train(input_args=input_args, datareader=data_reader)

    if input_args.debug_info >= 1:
        print("Model Trained successfully, exiting training module")


# ============= TESTING ==========================
def perform_testing(input_args):
    if input_args.debug_info >= 1:
        print("Performing testing")

    test_x, test_y, train_x, train_y, validation_x, validation_y, train_size = read_data('dataset/Training_set.h5',
                                                                                         'dataset/Testing_set.h5',
                                                                                         'dataset/Validate_set.h5')

    if input_args.gpu_id >= 0:
        print("Placing device %d for training", input_args.gpu_id)
    else:
        print("WARNING: Running purely on CPU")

    with tf.Session(config=tf.ConfigProto()) as session:
        aenet = FC_AutoEncoder(input_args=input_args, session=session)
        if input_args.pretrained_weights is not None:
            aenet.load(sess=session, input_args=input_args)
            aenet.test(sess=session, test_x=test_x, test_y=test_y, input_args=input_args)
        else:
            print("We dont have a model weights to lead and perform prediction ...")
            sys.exit(-1)

        print("Finished testing the network ...")

# ============= TESTING ==========================
def perform_feat_extraction(input_args):
    if input_args.debug_info >= 1:
        print("Performing testing")

    test_x, test_y, train_x, train_y, validation_x, validation_y, train_size = read_data('dataset/Training_set.h5',
                                                                                         'dataset/Testing_set.h5',
                                                                                         'dataset/Validate_set.h5')

    if input_args.gpu_id >= 0:
        print("Placing device %d for training", input_args.gpu_id)
    else:
        print("WARNING: Running purely on CPU")

    with tf.Session(config=tf.ConfigProto()) as session:
        aenet = FC_AutoEncoder(input_args=input_args, session=session)
        if input_args.pretrained_weights is not None:
            aenet.load(sess=session, input_args=input_args)
            aenet.extract_features(sess=session, train_x=test_x, train_y=test_y, input_args=input_args)
        else:
            print("We dont have a model weights to lead and perform prediction ...")
            sys.exit(-1)

        print("Finished testing the network ...")


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.execution_mode == 'train':
        perform_training(args)
    elif args.execution_mode == 'test':
        perform_testing(args)
    elif args.execution_mode == 'extraction':
        perform_feat_extraction(args)
    else:
        print("No operation specified, exiting application ...")

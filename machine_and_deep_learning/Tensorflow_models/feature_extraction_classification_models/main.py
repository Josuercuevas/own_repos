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

import argparse
import os
import sys
import tensorflow as tf
import time

'''
    this has all the options related to SELU, ELU, SKIP-Layer, New dropout functions
    normalization approach for training and testing.
    Only use it if you know how to call and invoke the functions. Setting is very tricky you
    need to know what you are doing.
'''
#from BaseCode_experimental.model_builder import *

from BaseCode.model_builder import *

USE_H5PY = False
if USE_H5PY:
    from DataReader.database_manager import *
else:
    from BodyNet_module.DataReader.image_handler import *

from configs import common_configs, resnet_configs, squeeznet_configs

# ========================================argument parsing =============================================

"""
    Parsing arguments for training or testing
"""
parser = argparse.ArgumentParser(description='Train a Hierarchical model architecture')
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
                        help='number of epochs to train [300]',
                        default=300, type=int)
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
                        help='mode in which the program to be executed (train/test/extract_features/dump_dotfile/interepretability) [train]',
                        default='train', type=str)
    parser.add_argument('--add_skip', dest='add_skyplayer',
                        help='add skip layers ResNet type (true/false) [false]',
                        default='false', type=str)
    parser.add_argument('--debug', dest='debug_info',
                        help='add skip layers ResNet type (0(none)/1(warnings)/2(debug)/3(verbose)) [3]',
                        default=3, type=int)
    parser.add_argument('--model', dest='model_name',
                        help='type of model to be used (helnet/squeeznet/resnet/shufflenet/mobilenet) [resnet]',
                        default='resnet', type=str)
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
        print("Performing training of model %s"%(input_args.model_name))

    # read all data to perform setting and training, but do not load it to memory we dont need to
    if input_args.extract_database == 'true':
        print("================== LOADING/Extracting TRAINING DATA ==================")
        if input_args.location_database == None:
            print("No database specified, please input absolute path of the database, and where to save it")
            parser.print_help()
            sys.exit(1)

        extract_database(input_args)
        sys.exit(0)

    start = time.time()
    if USE_H5PY:
        test_x, test_y, train_x, train_y, validation_x, validation_y, train_size = \
            read_data(input_args.imdb_name + '/Training_set.h5',
                      input_args.imdb_name + '/Testing_set.h5',
                      input_args.imdb_name + '/Validate_set.h5')

        data_reader = {
            'test_x': test_x,
            'test_y': test_y,
            'train_x':  train_x,
            'train_y': train_y,
            'validation_x': validation_x,
            'validation_y': validation_y,
            'train_size': train_size
        }
    else:
        # ============================ TRAINING
        class_folders_paths_Train, number_of_samples_per_class_Train, samples_filenames_Train = \
            get_data_handler(datapath=input_args.imdb_name + "/train", n_classes=-1)
        train_size = 0
        for nn in range(len(number_of_samples_per_class_Train)):
            train_size += number_of_samples_per_class_Train[nn]

        print("Training dataset has %d samples at folder %s" % (train_size, input_args.imdb_name + "/train"))
        # ============================

        # ============================ TESTING
        class_folders_paths_Test, number_of_samples_per_class_Test, samples_filenames_Test = \
            get_data_handler(datapath=input_args.imdb_name + "/test", n_classes=-1)
        test_size = 0
        for nn in range(len(number_of_samples_per_class_Test)):
            test_size += number_of_samples_per_class_Test[nn]
        print("Testing dataset has %d samples at folder %s" % (test_size, input_args.imdb_name + "/test"))
        # ============================

        # ============================ VALIDATION
        class_folders_paths_Val, number_of_samples_per_class_Val, samples_filenames_Val = \
            get_data_handler(datapath=input_args.imdb_name + "/val", n_classes=-1)
        val_size = 0
        for nn in range(len(number_of_samples_per_class_Val)):
            val_size += number_of_samples_per_class_Val[nn]
        print("Validation dataset has %d samples at folder %s" % (val_size, input_args.imdb_name + "/val"))
        # ============================

        data_reader = {
            # =====================TRAINING
            'class_folders_paths_Train': class_folders_paths_Train,
            'number_of_samples_per_class_Train': number_of_samples_per_class_Train,
            'samples_filenames_Train': samples_filenames_Train,
            'train_size': train_size,

            # =====================TESTING
            'class_folders_paths_Test': class_folders_paths_Test,
            'number_of_samples_per_class_Test': number_of_samples_per_class_Test,
            'samples_filenames_Test': samples_filenames_Test,
            'test_size': test_size,

            # =====================VALIDATION
            'class_folders_paths_Val': class_folders_paths_Val,
            'number_of_samples_per_class_Val': number_of_samples_per_class_Val,
            'samples_filenames_Val': samples_filenames_Val,
            'val_size': val_size
        }

    print('Data loaded in %d ms' % ((time.time()-start)*1000))
    # time.sleep(1000000)
    # sys.exit(0)

    if input_args.debug_info >= 4:
        print("Data reader engine is as follows: ")
        print(data_reader)

    if input_args.gpu_id >= 0:
        print("Placing device %d for training", input_args.gpu_id)
    else:
        print("WARNING: Running purely on CPU")

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:

        classifier = ModelNetwork()

        if input_args.solver == 'moment':
            classifier.build(db_size=train_size, input_args=input_args)
        else:
            classifier.build(db_size=0, input_args=input_args)

        if input_args.enable_tensorboard=='true' and input_args.solver == 'moment':
            with tf.name_scope('Learning-Rate'):
                tf.summary.scalar('learning_rate', classifier.learning_rate)

        if input_args.enable_tensorboard=='true':
            tf.summary.scalar('training_accuracy', classifier.training_accuracy)

        if input_args.enable_tensorboard=='true':
            tf.summary.scalar('Loss_function', classifier.loss)

        # To be used during training
        summary_merger = tf.summary.merge_all()

        if input_args.enable_tensorboard=='true':
            # clean folder before start anything in training
            if tf.gfile.Exists('tfboard/model_training_progress'):
                tf.gfile.DeleteRecursively('tfboard/model_training_progress')
                tf.gfile.MakeDirs('tfboard/model_training_progress')

            train_writer = tf.summary.FileWriter('tfboard/model_training_progress/train', session.graph)
            train_writer.flush()

        Model_saver = tf.train.Saver()

        session.run(tf.global_variables_initializer())


        if input_args.pretrained_weights is not None:
            classifier.load(sess=session, input_args=input_args, model_saver=Model_saver)

        classifier.train(sess=session, datareader=data_reader, summary_merger=summary_merger,
                         summary_writer=train_writer, model_saver=Model_saver)

    if input_args.debug_info >= 1:
        print("Model Trained successfully, exiting training module")


# ============= TESTING ==========================
def perform_testing(input_args):
    if input_args.debug_info >= 1:
        print("Performing testing")

    if USE_H5PY:
        test_x, test_y, train_x, train_y, validation_x, validation_y, train_size = \
            read_data(input_args.imdb_name + '/Training_set.h5', input_args.imdb_name + '/Testing_set.h5',
                      input_args.imdb_name + '/Validate_set.h5')

        data_reader = {
            'test_x': test_x,
            'test_y': test_y,
            'train_x':  train_x,
            'train_y': train_y,
            'validation_x': validation_x,
            'validation_y': validation_y,
            'train_size': train_size
        }
    else:
        # ============================ TESTING
        class_folders_paths_Test, number_of_samples_per_class_Test, samples_filenames_Test = \
            get_data_handler(datapath=input_args.imdb_name + "/test", n_classes=-1)
        test_size = 0
        for nn in range(len(number_of_samples_per_class_Test)):
            test_size += number_of_samples_per_class_Test[nn]
        print("Testing dataset has %d samples at folder %s" % (test_size, input_args.imdb_name + "/test"))
        # ============================

        data_reader = {
            # =====================TESTING
            'class_folders_paths_Test': class_folders_paths_Test,
            'number_of_samples_per_class_Test': number_of_samples_per_class_Test,
            'samples_filenames_Test': samples_filenames_Test,
            'test_size': test_size,
        }

    if input_args.gpu_id >= 0:
        print("Placing device %d for training", input_args.gpu_id)
    else:
        print("WARNING: Running purely on CPU")

    with tf.Session(config=tf.ConfigProto()) as session:

        classifier = ModelNetwork()

        if input_args.solver == 'moment':
            classifier.build(db_size=train_size, input_args=input_args)
        else:
            classifier.build(db_size=0, input_args=input_args)

        if input_args.enable_tensorboard=='true':
            # clean folder before start anything in training
            if tf.gfile.Exists('tfboard/model_training_progress'):
                tf.gfile.DeleteRecursively('tfboard/model_training_progress')
                tf.gfile.MakeDirs('tfboard/model_training_progress')

            train_writer = tf.summary.FileWriter('tfboard/model_training_progress/train', session.graph)
            train_writer.flush()

        Model_saver = tf.train.Saver()
        # session.run(tf.global_variables_initializer()) # not needed in reality
        classifier.load(sess=session, input_args=input_args, model_saver=Model_saver)

        if USE_H5PY:
            classifier.test(test_x=test_x, test_y=test_y)
        else:
            classifier.test(data_reader=data_reader)

# ============= PERFORM FEATURE EXTRACTION ==========================
def feature_extraction(input_args):
    if input_args.debug_info >= 1:
        print("Performing Feature extraction for t-SNE")

    if USE_H5PY:
        test_x, test_y, train_x, train_y, validation_x, validation_y, train_size = \
            read_data(input_args.imdb_name + '/Training_set.h5', input_args.imdb_name + '/Testing_set.h5',
                      input_args.imdb_name + '/Validate_set.h5')

        data_reader = {
            'test_x': test_x,
            'test_y': test_y,
            'train_x':  train_x,
            'train_y': train_y,
            'validation_x': validation_x,
            'validation_y': validation_y,
            'train_size': train_size
        }
    else:
        # ============================ FEATURE EXTRACTION
        class_folders_paths_Test, number_of_samples_per_class_Test, samples_filenames_Test = \
            get_data_handler(datapath=input_args.imdb_name, n_classes=-1)
        test_size = 0
        for nn in range(len(number_of_samples_per_class_Test)):
            test_size += number_of_samples_per_class_Test[nn]
        print("Testing dataset has %d samples at folder %s" % (test_size, input_args.imdb_name))
        # ============================

        data_reader = {
            # ===================== FEATURE EXTRACTION
            'class_folders_paths_Test': class_folders_paths_Test,
            'number_of_samples_per_class_Test': number_of_samples_per_class_Test,
            'samples_filenames_Test': samples_filenames_Test,
            'test_size': test_size,
        }

    if input_args.gpu_id >= 0:
        print("Placing device %d for training", input_args.gpu_id)
    else:
        print("WARNING: Running purely on CPU")

    with tf.Session(config=tf.ConfigProto()) as session:
        classifier = ModelNetwork()

        if input_args.solver == 'moment':
            classifier.build(db_size=train_size, input_args=input_args)
        else:
            classifier.build(db_size=0, input_args=input_args)

        if input_args.enable_tensorboard == 'true':
            # clean folder before start anything in training
            if tf.gfile.Exists('tfboard/model_training_progress'):
                tf.gfile.DeleteRecursively('tfboard/model_training_progress')
                tf.gfile.MakeDirs('tfboard/model_training_progress')

            train_writer = tf.summary.FileWriter('tfboard/model_training_progress/train', session.graph)
            train_writer.flush()

        Model_saver = tf.train.Saver()
        # session.run(tf.global_variables_initializer()) # not needed in reality
        classifier.load(sess=session, input_args=input_args, model_saver=Model_saver)

        if USE_H5PY:
            classifier.extract_features(sess, dataset_x=train_x, dataset_y=train_y, input_args=input_args)
        else:
            classifier.extract_features(sess, datareader=data_reader, input_args=input_args)

def write_dot(input_args):
    if input_args.debug_info >= 2:
        print("Executing dot file dumping ...")

    with tf.Session(config=tf.ConfigProto()) as session:
        classifier = ModelNetwork()
        classifier.build(db_size=0, input_args=input_args)

        if input_args.enable_tensorboard=='true':
            # clean folder before start anything in training
            if tf.gfile.Exists('tfboard/model_training_progress'):
                tf.gfile.DeleteRecursively('tfboard/model_training_progress')
                tf.gfile.MakeDirs('tfboard/model_training_progress')

            train_writer = tf.summary.FileWriter('tfboard/model_training_progress/train', session.graph)
            train_writer.flush()

        Model_saver = tf.train.Saver()
        classifier.load(sess=session, input_args=input_args, model_saver=Model_saver)
        classifier.write_dotfile(sess=session, graph=session.graph, input_args=input_args)

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    assigned = False
    for id_name in range(20):
        if args.gpu_id == id_name:
            print("assigning GPU %d for processing ..."%args.gpu_id )
            os.environ['CUDA_VISIBLE_DEVICES'] = str(id_name)
            assigned = True
            break

    if not assigned:
        print("Pure CPU assigned for processing ...")
        os.environ['CUDA_VISIBLE_DEVICES'] = ' '

    if args.execution_mode == 'train':
        perform_training(args)
    elif args.execution_mode == 'test':
        perform_testing(args)
    elif args.execution_mode == 'extract_features':
        feature_extraction(args)
    elif args.execution_mode == 'dump_dotfile':
        write_dot(args)
    elif args.execution_mode == 'interepretability':
        print("NOT YET ......")
        pass
    else:
        print("No operation specified, exiting application ...")

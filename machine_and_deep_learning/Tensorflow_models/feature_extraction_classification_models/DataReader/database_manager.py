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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import sys
from PIL import Image
sys.path.append('../')
import configs.common_configs as common_configs
import image_db.create_database as db_generator


'''
    It reads all the images in the bd folder specified by the user,
    and saves the hdf5 file on disk where the user specifies it
'''
def extract_database(input_args):
    if input_args.augment_data == 'true':
        db_generator.AUGMENT_DATA = True
    else:
        db_generator.AUGMENT_DATA = False

    # in case we want to see what we have written on the folder to be used for building DB
    db_generator.DEBUG_IMAGES_WRITTEN = True

    if input_args.debug_info >= 2:
        print("Extracting database from %s, and saving it at %s"%(input_args.location_database,
                                                                  input_args.save_database))
    db_generator.perform_data_extraction(input_args.location_database, input_args.save_database)


'''
    will open the hdf5 file and return the pointer
    to the corresponding containers
'''
def read_data(train_path, test_path, validation_path):
    '''
        Load data saved in hdf5 format
        there are #_classes classes in this dataset, all labeled
        from 0 - #_classes
    '''
    trainig_file = h5py.File(train_path, 'r')
    testing_file = h5py.File(test_path, 'r')
    validate_file = h5py.File(validation_path, 'r')

    train_dataset = trainig_file["Training_images"]
    train_labels = trainig_file["Training_labels"]
    test_dataset = testing_file["Testing_images"]
    test_labels = testing_file["Testing_labels"]
    validate_dataset = validate_file["Validate_images"]
    validate_labels = validate_file["Validate_labels"]

    # dont load the data from here, do it directly reading the binary file
    train_x = train_dataset#[...]
    train_y = train_labels#[...]
    test_x = test_dataset#[...]
    test_y = test_labels#[...]
    validation_x = validate_dataset#[...]
    validation_y = validate_labels#[...]
    train_size = train_x.shape[0]

    if common_configs.DEBUG_IMAGES:
        path_dum = "debug_images/"

        print("writing Training set")
        for k in range(train_x.shape[0]):
            print("Image shape: (%d, %d, %d), label: " % \
                  (train_x.shape[1], train_x.shape[2],
                   train_x.shape[3]))
            print(train_y[k])
            print("First PEL:")
            print(train_x[k][0][0])
            # get data first
            if train_x.shape[3] == 4:
                to_save = train_x[k, :, :, :]
                temp_image = Image.fromarray(to_save, mode='RGBA')
            else:
                if train_x.shape[3] == 3:
                    to_save = train_x[k, :, :, :]
                    temp_image = Image.fromarray(to_save, mode='RGB')
                else:  # no way only Gray
                    to_save = train_x[k, :, :, 0]
                    temp_image = Image.fromarray(to_save, mode='L')

            temp_image.save(path_dum + "train/" + str(k + 1) + ".jpeg")  #

        print("writing Testing set")
        for k in range(test_x.shape[0]):
            print("Image shape: (%d, %d, %d), label:" % \
                  (test_x.shape[1], test_x.shape[2],
                   test_x.shape[3]))
            print(test_y[k][0])
            print("First PEL:")
            print(test_x[k][0][0])
            # get data first
            if test_x.shape[3] == 4:
                to_save = test_x[k, :, :, :]
                temp_image = Image.fromarray(to_save, mode='RGBA')
            else:
                if test_x.shape[3] == 3:
                    to_save = test_x[k, :, :, :]
                    temp_image = Image.fromarray(to_save, mode='RGB')
                else:  # no way only Gray
                    to_save = test_x[k, :, :, 0]
                    temp_image = Image.fromarray(to_save, mode='L')

            temp_image.save(path_dum + "test/" + str(k + 1) + ".jpeg")

        print("writing Validation set")
        for k in range(validation_x.shape[0]):
            print("Image shape: (%d, %d, %d), label:" % \
                  (validation_x.shape[1], validation_x.shape[2],
                   validation_x.shape[3]))
            print(validation_y[k])
            print("First PEL:")
            print(validation_x[k][0][0])
            # get data first
            if validation_x.shape[3] == 4:
                to_save = validation_x[k, :, :, :]
                temp_image = Image.fromarray(to_save, mode='RGBA')
            else:
                if validation_x.shape[3] == 3:
                    to_save = validation_x[k, :, :, :]
                    temp_image = Image.fromarray(to_save, mode='RGB')
                else:  # no way only Gray
                    to_save = validation_x[k, :, :, 0]
                    temp_image = Image.fromarray(to_save, mode='L')

            temp_image.save(path_dum + "validation/" + str(k + 1) + ".jpeg")

    print("Tr: %d, Tst: %d, Val: %d" % (train_x.shape[0], test_x.shape[0], validation_x.shape[0]))
    print(train_x.shape)
    print(test_x.shape)
    print(validation_x.shape)
    print(train_y.shape)
    print(test_y.shape)
    print(validation_y.shape)

    return test_x, test_y, train_x, train_y, validation_x, validation_y, train_size

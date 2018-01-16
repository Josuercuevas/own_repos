"""
    Dataset creation routine

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

import h5py
import numpy as np
from os import listdir
from os.path import isfile, join
import sys
import random
from augmentation import Perform_augmentation

# So far I am using Jpeg files, this can be changed to read RGB88
from PIL import Image

# General flags to choose which
# data goes to training, testing and validation
AUGMENT_DATA = False
DEBUG_IMAGES_WRITTEN = False
PROB_CHANGE_SIDE = 0.1
PROB_CHANGE_VALIDATION = 0.1


def perform_data_extraction(class_folder_path, save_bd_at):
    '''
        5 types of modification to every image is made
        rotation, contrast, color, sharpening, blurring
        we exit right after to check, this can be modified
    '''
    if AUGMENT_DATA:
        if Perform_augmentation(class_folder_path, save_bd_at+"/augmented"):
            print("Impossible to perform augmentation ... exiting")
            sys.exit(1)
        else:
            print("augmentation performed successfully, copy images to destination folder and train model ...")
            sys.exit(0)

    '''
        Folder where the database is stored, here we need to save
        each class in separated folders so as to create a single HDF5
        file during training.
    '''
    defined_path = class_folder_path
    channels = 0
    changed = 0
    old_width = -1
    old_height = -1

    changew_count = 0
    changeh_count = 0


    onlyfolders = [f for f in listdir(defined_path)]
    print(onlyfolders)

    onlyfiles = []
    for i in range(len(onlyfolders)):
        onlyfiles.append([])
        files_here = []
        for g in listdir(join(join(defined_path, onlyfolders[i]))):
            if isfile(join(join(defined_path, onlyfolders[i]), g)):
                files_here.append(g)
        onlyfiles[len(onlyfiles)-1].append(files_here)  # all files at once

    print("%d folders found"%len(onlyfolders))

    n_training_images = []
    n_validate_images = []
    n_testing_images = []

    for j in range(len(onlyfolders)):
        taken = int(len(onlyfiles[j][0]) * (1.0 - PROB_CHANGE_SIDE))
        valtouse = int(taken*PROB_CHANGE_VALIDATION)
        n_training_images.append(taken-valtouse)
        n_validate_images.append(valtouse)
        n_testing_images.append(len(onlyfiles[j][0])-taken)

    print("Training set is as follows:")
    print(n_training_images)
    print("Validation set is as follows:")
    print(n_validate_images)
    print("Testing set is as follows:")
    print(n_testing_images)


    #
    # Creating database of images
    #
    train_datafile = h5py.File(save_bd_at+'Training_set.h5','w')
    test_datafile = h5py.File(save_bd_at+'Testing_set.h5','w')
    validate_datafile = h5py.File(save_bd_at+'Validate_set.h5','w')
    change_class = False


    '''
        quick scan of the folder to know how much data we need to store inside the
        HDF5 binary files and how many classes we have in total
    '''
    for j in range(len(onlyfolders)):
        print("Folder %s has %d files, detailed as:"%(onlyfolders[j], len(onlyfiles[j][0])))
        for i in range(len(onlyfiles[j][0])):
            im = Image.open(join(join(defined_path, onlyfolders[j]), onlyfiles[j][0][i]), 'r')  # Can be many different formats.
            width, height = im.size
            print(onlyfiles[j][0][i])
            print("Image of size: %dx%d"%(width,height))
            print("mode: %s"%im.mode)

            if old_width != width:
                print("changing image width from %d to %d"%(old_width,width))
                old_width = width
                changew_count += 1

            if old_height != height:
                print("changing image height from %d to %d"%(old_height,height))
                old_height = height
                changeh_count += 1

            if im.mode=='L':
                if channels != 1:
                    print("changing number of channels from %d to %d"%(channels, 1))
                    changed += 1
                channels=1
            else:
                if channels != 3:
                    print("changing number of channels from %d to %d"%(channels, 3))
                    changed += 1
                channels = 3



    '''
        We will create two sets per class, one containing the images
        the other one containing the labels. However both sets will be
        contained in the same binary file, which can be quickly accessed
        during training when a single batch it is desired
    '''
    #images
    trainingset = train_datafile.create_dataset(name="Training_images",
                                                shape=[np.sum(n_training_images),
                                                       old_height, old_width, channels],
                                                dtype=h5py.h5t.NATIVE_UINT8)
    #labels
    traininglabels = train_datafile.create_dataset(name="Training_labels",
                                                shape=[np.sum(n_training_images), int(len(onlyfolders))],
                                                dtype=h5py.h5t.NATIVE_UINT8)

    # images
    testingset = test_datafile.create_dataset(name="Testing_images",
                                              shape=[np.sum(n_testing_images),
                                                     old_height, old_width, channels],
                                              dtype=h5py.h5t.NATIVE_UINT8)
    # labels
    testinglabels = test_datafile.create_dataset(name="Testing_labels",
                                                   shape=[np.sum(n_testing_images), int(len(onlyfolders))],
                                                   dtype=h5py.h5t.NATIVE_UINT8)

    #image
    validateset = validate_datafile.create_dataset(name="Validate_images",
                                              shape=[np.sum(n_validate_images),
                                                     old_height, old_width, channels],
                                              dtype=h5py.h5t.NATIVE_UINT8)
    #labels
    validatelabels = validate_datafile.create_dataset(name="Validate_labels",
                                                   shape=[np.sum(n_validate_images), int(len(onlyfolders))],
                                                   dtype=h5py.h5t.NATIVE_UINT8)

    print("Dataset to be save in hdf5 is:")
    print(trainingset.shape)
    print(traininglabels.shape)
    print(validateset.shape)
    print(validatelabels.shape)
    print(testingset.shape)
    print(testinglabels.shape)


    '''
        Now performing the real storage into the binary file to
        be used during training. The trick here is to save it in the right format so
        we dont messup with the data and the labels.

        We perform a hot encoding fot he number of classes or folders found
        so if there are 3 classes the label is as follows:
        class 1: [1 0 0]
        class 2: [0 1 0]
        class 3: [0 0 1]

        Here we handle any RGB and Gray images only, most IMPORTANT is,
        we handle one type of images only, Sames channels and sizes.
     '''
    train_idx = 0
    test_idx = 0
    validate_idx = 0
    validate_set_idx = 0
    train_set_idx = 0
    test_set_idx = 0
    for j in range(len(onlyfolders)):
        change_class=True# every time we change folder
        print("Folder %s has %d files, detailed as:"%(onlyfolders[j], len(onlyfiles[j][0])))
        for i in range(len(onlyfiles[j][0])):
            im = Image.open(join(join(defined_path, onlyfolders[j]), onlyfiles[j][0][i]), 'r')  # Can be many different formats.
            width, height = im.size
            print(onlyfiles[j][0][i])
            print("Image of size: %dx%d"%(width,height))
            print("mode: %s"%im.mode)

            if old_width != width:
                print("changing image width from %d to %d"%(old_width,width))
                old_width = width
                changew_count += 1

            if old_height != height:
                print("changing image height from %d to %d"%(old_height,height))
                old_height = height
                changeh_count += 1

            if im.mode=='L':
                if channels != 1:
                    print("changing number of channels from %d to %d"%(channels, 1))
                    changed += 1
                channels=1
            else:
                if channels != 3:
                    print("changing number of channels from %d to %d"%(channels, 3))
                    changed += 1
                channels = 3

            if changew_count>1 or changeh_count>1 or changed>1:
                print("ERROR more than one channel on images, exiting")
                break
            else:
                if change_class:
                    print("=====>>> saving Train data-set with name %s and %d samples"%\
                          (onlyfolders[j], n_training_images[j]))
                    print("=====>>> saving Test data-set with name %s and %d samples"%\
                          (onlyfolders[j], n_testing_images[j]))
                    change_class = False
                    train_idx = 0
                    test_idx = 0
                    validate_idx = 0


                print("<%d, %d>"%(j, i))
                print("reading: %s"%join(join(defined_path, onlyfolders[j]), onlyfiles[j][0][i]))
                pixel_values = list(im.getdata())
                print("size: %dx%d"%(width, height))
                print("mode: %s"%im.mode)
                pixel_values = np.array(pixel_values).reshape((height, width, channels))

                if channels>1:
                    print("first PEL: %d, class: %d"%(int(pixel_values[0][0][0]), int(j)))
                else:
                    print("first PEL: %d, class: %d" % (int(pixel_values[0][0]), int(j)))

                print("Writing image to db...")


                if (random.random() > 0.5) or (test_idx >= n_testing_images[j]):
                    if (random.random() > 0.5) or (validate_idx >= n_validate_images[j]):
                        # samples has become training
                        trainingset[train_set_idx] = pixel_values

                        for jj in range(len(onlyfolders)):
                            if jj == j:
                                traininglabels[train_set_idx, jj] = 1
                            else:
                                traininglabels[train_set_idx, jj] = 0

                        train_idx += 1
                        train_set_idx += 1
                        train_datafile.flush()
                    else:
                        # samples has become validation
                        validateset[validate_set_idx] = pixel_values

                        for jj in range(len(onlyfolders)):
                            if jj == j:
                                validatelabels[validate_set_idx, jj] = 1
                            else:
                                validatelabels[validate_set_idx, jj] = 0

                        validate_idx += 1
                        validate_set_idx += 1
                        validate_datafile.flush()
                else:
                    # samples has become testing
                    testingset[test_set_idx] = pixel_values

                    for jj in range(len(onlyfolders)):
                        if jj == j:
                            testinglabels[test_set_idx, jj] = 1
                        else:
                            testinglabels[test_set_idx, jj] = 0

                    test_idx += 1
                    test_set_idx += 1
                    test_datafile.flush()

                    if (test_idx >= n_testing_images[j]):
                        print("============= FINISHED PICKING THE DATA FOR TESTING ===========")

    print("We have a total of %d images (%d-channels)"%\
          ((train_set_idx+test_set_idx+validate_set_idx), channels))


    '''
        Random mixing the samples so in training we have variety or we may get
        stuck at a local optimum, sort of shuffling but offline, not in training time
        so we can save some cycles.
    '''
    #mixing training set
    for idx in range(train_set_idx):
        cand_to_exchange = random.randint(0, train_set_idx - 1)
        print("changing %d with %d"%(cand_to_exchange, idx))
        temp_vec = trainingset[idx]
        trainingset[idx] = trainingset[cand_to_exchange]
        trainingset[cand_to_exchange] = temp_vec
        temp_vec = traininglabels[idx]
        traininglabels[idx] = traininglabels[cand_to_exchange]
        traininglabels[cand_to_exchange] = temp_vec

    # Mixing validation set
    for idx in range(validate_set_idx):
        cand_to_exchange = random.randint(0, validate_set_idx - 1)
        print("changing %d with %d"%(cand_to_exchange, idx))
        temp_vec = validateset[idx]
        validateset[idx] = validateset[cand_to_exchange]
        validateset[cand_to_exchange] = temp_vec
        temp_vec = validatelabels[idx]
        validatelabels[idx] = validatelabels[cand_to_exchange]
        validatelabels[cand_to_exchange] = temp_vec

    # Mixing testing set
    for idx in range(test_set_idx):
        cand_to_exchange = random.randint(0, test_set_idx - 1)
        print("changing %d with %d"%(cand_to_exchange, idx))
        temp_vec = testingset[idx]
        testingset[idx] = testingset[cand_to_exchange]
        testingset[cand_to_exchange] = temp_vec
        temp_vec = testinglabels[idx]
        testinglabels[idx] = testinglabels[cand_to_exchange]
        testinglabels[cand_to_exchange] = temp_vec



    if changew_count>1 or changeh_count>1:
        print("ERROR images have different sizes, please resize and try again ...")
        sys.exit(1)

    print("%d images written, closing database ..."%(train_set_idx+test_set_idx+validate_set_idx))

    # Close the file before exiting
    train_datafile.close()
    test_datafile.close()
    validate_datafile.close()


    '''
        This part is to check we have written correctly to the DB,
        the images are loaded one by one and dump into jpeg format
        the idea is to check if what we wrote is as it supposed to be
        otherwise during training the engine will fail to converge
    '''
    if DEBUG_IMAGES_WRITTEN:
        '''
          checking if what wrote makes sense, decoding the data and dumping the images
            just for debugging
        '''
        path_dum = "dataset/images_dump/"
        trainig_file = h5py.File('dataset/Training_set.h5', 'r')
        testing_file = h5py.File('dataset/Testing_set.h5', 'r')
        validate_file = h5py.File('dataset/Validate_set.h5', 'r')

        train_dataset = trainig_file['Training_images']
        train_labels = trainig_file['Training_labels']
        test_dataset = testing_file['Testing_images']
        test_labels = testing_file['Testing_labels']
        validate_dataset = validate_file['Validate_images']
        validate_labels = validate_file['Validate_labels']

        #
        # Read data back and print it.
        #
        print("Reading data back from Database ...")
        train_data_read = train_dataset[...]
        train_label_read = train_labels[...]
        test_data_read = test_dataset[...]
        test_label_read = test_labels[...]
        validate_data_read = validate_dataset[...]
        validate_label_read = validate_labels[...]
        print("Total of [Tr: %d, Tst: %d, Val: %d] data images and [Tr: %d, Tst: %d, Val: %d] labels" % \
              (train_data_read.shape[0], test_data_read.shape[0],
               train_label_read.shape[0], test_label_read.shape[0],
               validate_data_read.shape[0], validate_label_read.shape[0]))
        print(train_data_read.shape)
        print(test_data_read.shape)

        print("writing Training set")
        for k in range(train_data_read.shape[0]):
            print("Image shape: (%d, %d, %d), label: "%\
                  (train_data_read.shape[1], train_data_read.shape[2],
                   train_data_read.shape[3]))
            print(train_label_read[k])
            print("First PEL:")
            print(train_data_read[k][0][0])
            # get data first
            if train_data_read.shape[3]==4:
                to_save = train_data_read[k,:,:,:]
                temp_image = Image.fromarray(to_save, mode='RGBA')
            else:
                if train_data_read.shape[3]==3:
                    to_save = train_data_read[k,:,:,:]
                    temp_image = Image.fromarray(to_save, mode='RGB')
                else:#no way only Gray
                    to_save = train_data_read[k, :, :, 0]
                    temp_image = Image.fromarray(to_save, mode='L')

            temp_image.save(path_dum+"train/"+str(k+1)+".jpeg")#


        print("writing Testing set")
        for k in range(test_data_read.shape[0]):
            print("Image shape: (%d, %d, %d), label:" % \
                  (test_data_read.shape[1], test_data_read.shape[2],
                   test_data_read.shape[3]))
            print(test_label_read[k][0])
            print("First PEL:")
            print(test_data_read[k][0][0])
            # get data first
            if test_data_read.shape[3]==4:
                to_save = test_data_read[k,:,:,:]
                temp_image = Image.fromarray(to_save, mode='RGBA')
            else:
                if test_data_read.shape[3]==3:
                    to_save = test_data_read[k,:,:,:]
                    temp_image = Image.fromarray(to_save, mode='RGB')
                else:#no way only Gray
                    to_save = test_data_read[k, :, :, 0]
                    temp_image = Image.fromarray(to_save, mode='L')

            temp_image.save(path_dum+"test/"+str(k+1)+".jpeg")

        print("writing Validation set")
        for k in range(validate_data_read.shape[0]):
            print("Image shape: (%d, %d, %d), label:" % \
                  (validate_data_read.shape[1], validate_data_read.shape[2],
                   validate_data_read.shape[3]))
            print(validate_label_read[k])
            print("First PEL:")
            print(validate_data_read[k][0][0])
            # get data first
            if validate_data_read.shape[3] == 4:
                to_save = validate_data_read[k, :, :, :]
                temp_image = Image.fromarray(to_save, mode='RGBA')
            else:
                if validate_data_read.shape[3] == 3:
                    to_save = validate_data_read[k, :, :, :]
                    temp_image = Image.fromarray(to_save, mode='RGB')
                else:  # no way only Gray
                    to_save = validate_data_read[k, :, :, 0]
                    temp_image = Image.fromarray(to_save, mode='L')

            temp_image.save(path_dum + "validation/" + str(k + 1) + ".jpeg")

        #
        # Close the file before exiting
        #
        trainig_file.close()
        testing_file.close()
        validate_file.close()


if __name__ == '__main__':
    perform_data_extraction("augmented", "dataset/")

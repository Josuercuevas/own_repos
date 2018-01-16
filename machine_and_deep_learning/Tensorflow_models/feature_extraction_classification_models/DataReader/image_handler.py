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

import cv2 as opencv
import numpy as np
import os
import sys
from os.path import isfile, isdir, join
from os import listdir
from PIL import Image


"""
    This function will help to get all the required information for the
    training/testing/validation dataset for each topology used in this module

    NOTE IMPORTANT:
        Training/Testing/Validation are assumed to be in separated folders !
"""
def get_data_handler(datapath=None, n_classes=-1):
    if datapath == None:
        print("There is specified path to retrieve data, quitting ...")
        return False

    if n_classes > 0:
        isthere_limit = True
    else:
        isthere_limit = False


    class_folders_paths = []
    number_of_samples_per_class = []
    samples_filenames = []

    if isthere_limit:
        # scan only up to the n_classes given by the user
        for folder_id in range(n_classes):
            path_to_scan = datapath + "/" + str(folder_id)

            onlyfiles = [f for f in listdir(path_to_scan) if isfile(join(path_to_scan, f))]

            class_folders_paths.append(path_to_scan)
            samples_filenames.append(onlyfiles)
            number_of_samples_per_class.append(len(onlyfiles))
    else:
        # use all the classes specified in this folder
        only_folders = [f for f in listdir(datapath) if isdir(join(datapath, f))]

        for folder_id in range(len(only_folders)):
            path_to_scan = datapath + "/" + str(folder_id)

            onlyfiles = [f for f in listdir(path_to_scan) if isfile(join(path_to_scan, f))]

            class_folders_paths.append(path_to_scan)
            samples_filenames.append(onlyfiles)
            number_of_samples_per_class.append(len(onlyfiles))

    """
        now we return the paths and the number of samples on each class path
        basically what the user is going to see is the folder path where all the samples
        are located. And on the other list the number of samples for each class as well as their
        filenames:

        class_folders_paths -> root folder path for each class
        number_of_samples_per_class -> sample from each class
        samples_filenames -> every sample filename located at the class_folders_paths
    """
    return class_folders_paths, number_of_samples_per_class, samples_filenames

"""
    Function in charge of retrieving the batch of samples from the specified folders and classes
    All the parameters have to be provided otherwise the function will exit with error
"""
def get_images_batch(class_folders_paths=None, number_of_samples_per_class=None,
                     samples_filenames=None, batch_size=None, use_rgb=True, normalize_samples=False):

    if class_folders_paths==None or number_of_samples_per_class==None or samples_filenames==None or batch_size==None:
        print("ERROR: function inputs are not specified, please provide all the parameters, exiting ...")
        sys.exit(-1)

    batch_rgb_pels = [] # have to the RGB pixels
    batch_classes = [] # contains the classes per sample

    # shuffle the classes
    n_classes = len(class_folders_paths)
    i = 0
    while i < batch_size:
        # shuffle the sample inside the class
        folder_id = np.random.randint(0, len(class_folders_paths))

        # we dont care about classes with samples less than 2 data points
        if number_of_samples_per_class[folder_id] <= 2:
            continue

        smp_id = np.random.randint(0, number_of_samples_per_class[folder_id])

        # load sample
        if not use_rgb:
            # we will use BGR samples
            img_pels = opencv.imread(class_folders_paths[folder_id]+'/'+samples_filenames[folder_id][smp_id])
        else:
            # we will use RGB samples
            img_pels = opencv.imread(class_folders_paths[folder_id] + '/' + samples_filenames[folder_id][smp_id])
            img_pels = opencv.cvtColor(img_pels, opencv.COLOR_BGR2RGB) # if we want to convert

        # convert to float32
        img_pels = np.array(img_pels).astype(np.float32)
        if normalize_samples:
            # normalization between [0, 1]
            img_pels = img_pels / 255.

        # now append to the batch
        batch_rgb_pels.append(img_pels)

        label = np.zeros(shape=(n_classes), dtype=np.float32)
        label[folder_id] = 1.0
        batch_classes.append(label)

        i += 1

        # print("Folder: %d, File: %d" % (folder_id, smp_id))
        # print("Filename: %s" % class_folders_paths[folder_id]+'/'+samples_filenames[folder_id][smp_id])
        # print(label[folder_id])

    return np.array(batch_rgb_pels), np.array(batch_classes)


if __name__ == "__main__":
    test_path = "CLASSES_FOLDERS_PATH"
    class_folders_paths, number_of_samples_per_class, samples_filenames = get_data_handler(datapath=test_path, n_classes=-1)
    print("%d classes found in the location specified ..." % len(samples_filenames))

    for fid in range(len(samples_filenames)):
        print("%d filenames found at location %s ..." % (len(samples_filenames[fid]), class_folders_paths[fid]))

    print(class_folders_paths)
    print(number_of_samples_per_class)

    batch_size = 5
    normalize_samples = True
    use_rgb = False
    for bid in range(2):
        batch_X, batch_Y = get_images_batch(class_folders_paths=class_folders_paths,
                                            number_of_samples_per_class=number_of_samples_per_class,
                                            samples_filenames=samples_filenames, batch_size=batch_size,
                                            use_rgb=use_rgb, normalize_samples=normalize_samples)

        print(batch_X.shape)
        print(batch_Y.shape)

        for n_id in range(batch_size):
            print(batch_Y[n_id])

            sample_dd = batch_size*bid + n_id
            img_sample = batch_X[n_id]
            if normalize_samples:
                img_sample = img_sample*255.

            img_sample = np.array(img_sample).astype(np.uint8)

            tosave = Image.fromarray(img_sample, 'RGB')
            tosave.save("/home/josue/Desktop/dump_samples/sample_%d.jpg" % sample_dd)

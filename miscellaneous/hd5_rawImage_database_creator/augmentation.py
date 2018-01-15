'''
    Author: Josue R. Cuevas
    Date: 10/10/2016

    Data augmentation code for images provided by the user, the functions
    provided here are:
    1. Rotation
    2. Sharpener
    3. Blur
    4. Brightness
    5. Contrast
    6. Color

    Copyright (C) <2018>  <Josue R. Cuevas>

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

from os import listdir
from os.path import isfile, join
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
import sys

def rotator(img, folderpath, img_name):
    '''
        Rotates the image in a fixed range [-30, 30]
    '''
    ingle_id=-30
    for angle in range(12):
        rotated = img.rotate(ingle_id)
        rotated.save(join(folderpath, "angle_"+str(ingle_id)+"_"+img_name))
        ingle_id += 5


def sharpener(img, folderpath, img_name):
    '''
        Sharpens the image in a fixed range [1, 10]
    '''
    enhancer = ImageEnhance.Sharpness(img)

    for i in range(10):
        factor = i+1
        sharpened = enhancer.enhance(factor)
        sharpened.save(join(folderpath, "sharped_"+str(factor)+"_"+img_name))

def blurrer(img, folderpath, img_name):
    '''
        Blurs the image in a fixed range [1, 5]
    '''
    factor=0
    for i in range(10):
        factor += 0.5
        blurred = img.filter(ImageFilter.GaussianBlur(radius=factor))
        blurred.save(join(folderpath, "blurred_"+str(i+1)+"_"+img_name))

def brighter(img, folderpath, img_name):
    '''
        Brightness the image in a fixed range [0, 2]
    '''
    enhancer = ImageEnhance.Brightness(img)

    factor=0
    for i in range(10):
        factor += 0.2
        bri = enhancer.enhance(factor)
        bri.save(join(folderpath, "brightness_"+str(i+1)+"_"+img_name))



def contraster(img, folderpath, img_name):
    '''
        contrast the image in a fixed range [0.5, 1.5]
    '''
    enhancer = ImageEnhance.Contrast(img)

    factor=0.5
    for i in range(10):
        factor += 0.1
        contra = enhancer.enhance(factor)
        contra.save(join(folderpath, "contrast_"+str(i+1)+"_"+img_name))

def Colorer(img, folderpath, img_name):
    '''
        Color the image in a fixed range [1, 10]
    '''
    enhancer = ImageEnhance.Color(img)

    factor=0.5
    for i in range(10):
        chroma = enhancer.enhance(i)
        chroma.save(join(folderpath, "color_"+str(i+1)+"_"+img_name))



'''
    Performing augmentation of the data to be trained in our model
    each augmentation operator is defined here
'''
def Perform_augmentation():
    defined_path = "Faces_db"

    onlyfolders = [f for f in listdir(defined_path)]
    print(onlyfolders)

    onlyfiles = []
    for i in range(len(onlyfolders)):
        onlyfiles.append([])
        files_here = []
        for g in listdir(join(join(defined_path, onlyfolders[i]))):
            if isfile(join(join(defined_path, onlyfolders[i]), g)):
                files_here.append(g)
        onlyfiles[len(onlyfiles) - 1].append(files_here)  # all files at once

    print("%d folders found" % len(onlyfolders))

    n_images = []
    for j in range(len(onlyfolders)):
        n_images.append(len(onlyfiles[j][0]))


    for j in range(len(onlyfolders)):
        change_class=True# every time we change folder
        print("Folder %s has %d files, detailed as:"%(onlyfolders[j], len(onlyfiles[j][0])))
        for i in range(len(onlyfiles[j][0])):
            im = Image.open(join(join(defined_path, onlyfolders[j]), onlyfiles[j][0][i]),
                            'r')  # Can be many different formats.
            width, height = im.size
            print(onlyfiles[j][0][i])
            print("Image of size: %dx%d" % (width, height))
            print("mode: %s" % im.mode)

            # apply some image transformations to augment the data
            print("Performing Transformations...")
            rotator(im, join("augmented/", onlyfolders[j]), onlyfiles[j][0][i])
            sharpener(im, join("augmented/", onlyfolders[j]), onlyfiles[j][0][i])
            blurrer(im, join("augmented/", onlyfolders[j]), onlyfiles[j][0][i])
            contraster(im, join("augmented/", onlyfolders[j]), onlyfiles[j][0][i])
            Colorer(im, join("augmented/", onlyfolders[j]), onlyfiles[j][0][i])

    return False

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

from scipy.misc import imread
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageEnhance, Image
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def augmentation(patch_data, counter, prefix, path_to_save, n_augs=3):

    # ============== Warping X
    def shift(x, A, w):
        return A * np.sin(2.0 * np.pi * x * w)

    for temp in range(n_augs / 2 + 1):
        temp_patch_data = np.copy(patch_data)
        A = patch_data.shape[0] / (10.0 * (temp + 1.0))
        w = (1.0 * (temp + 1)) / patch_data.shape[1]

        print(A, w)

        for i in range(patch_data.shape[1]):
            temp_patch_data[:, i] = np.roll(patch_data[:, i], int(shift(i, A, w)))

        enhanced_image = Image.fromarray(temp_patch_data)
        enhanced_image.save((path_to_save + '/' + prefix + '_' + str(counter) + '.jpg'))
        counter += 1

    # ============== Warping Y
    def shift(x, A, w):
        return A * np.sin(2.0 * np.pi * x * w)

    for temp in range(n_augs / 2 + 1):
        temp_patch_data = np.copy(patch_data)
        A = patch_data.shape[0] / (10.0 * (temp + 1.0))
        w = (1.0 * (temp + 1)) / patch_data.shape[1]

        print(A, w)

        for i in range(patch_data.shape[0]):
            temp_patch_data[i, :] = np.roll(patch_data[i, :], int(shift(i, A, w)))

        enhanced_image = Image.fromarray(temp_patch_data)
        enhanced_image.save((path_to_save + '/' + prefix + '_' + str(counter) + '.jpg'))
        counter += 1

    # ============== Sharpness
    image = Image.fromarray(np.array(patch_data).astype(np.uint8))
    enhancer = ImageEnhance.Sharpness(image)
    for i in range(n_augs):
        factor = i
        enhanced_image = enhancer.enhance(factor)
        enhanced_image.save((path_to_save + '/' + prefix + '_' + str(counter) + '.jpg'))
        counter += 1

    # ============== COLOR
    image = Image.fromarray(np.array(patch_data).astype(np.uint8))
    enhancer = ImageEnhance.Color(image)
    factor = 1.0
    for i in range(n_augs):
        factor -= 0.2
        enhanced_image = enhancer.enhance(factor)
        enhanced_image.save((path_to_save + '/' + prefix + '_' + str(counter) + '.jpg'))
        counter += 1

    # ============== Contrast
    image = Image.fromarray(np.array(patch_data).astype(np.uint8))
    enhancer = ImageEnhance.Contrast(image)
    factor = 1.0
    for i in range(n_augs):
        factor -= 0.05
        enhanced_image = enhancer.enhance(factor)
        enhanced_image.save((path_to_save + '/' + prefix + '_' + str(counter) + '.jpg'))
        counter += 1

    # ============== Brightness
    image = Image.fromarray(np.array(patch_data).astype(np.uint8))
    enhancer = ImageEnhance.Brightness(image)
    factor = 1.0
    for i in range(n_augs):
        factor -= 0.05
        enhanced_image = enhancer.enhance(factor)
        enhanced_image.save((path_to_save + '/' + prefix + '_' + str(counter) + '.jpg'))
        counter += 1

    # ============== Rotate
    image = Image.fromarray(np.array(patch_data).astype(np.uint8))
    angle = -4.0
    for i in range(n_augs*2):
        angle += 1
        enhanced_image = image.rotate(angle)
        enhanced_image.save((path_to_save + '/' + prefix + '_' + str(counter) + '.jpg'))
        counter += 1

    return counter

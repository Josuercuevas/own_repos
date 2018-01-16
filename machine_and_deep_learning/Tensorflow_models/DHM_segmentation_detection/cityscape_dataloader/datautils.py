'''
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
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

def resize_mask(blob, dest_shape=None, im_scale=None, method=None):
    assert dest_shape != None or im_scale != None
    img = blob#.transpose((1, 2, 0))
    if method is None:
        method = cv2.INTER_LINEAR
    if dest_shape is not None:
        dest_shape = dest_shape[1], dest_shape[0]
        img = cv2.resize(img, dest_shape, interpolation=method)
    else:
        img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=method)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    blob = img#.transpose((2, 0, 1))
    return blob

def show_img_mask(img, ann, ax):
    img1 = np.ones((img.shape[0], img.shape[1], 3))
    color_mask = np.random.random((1, 3)).tolist()[0]
    for i in range(3):
        img1[:, :, i] = color_mask[i]

    ax.imshow(np.dstack((img1, ann)))
    plt.show()

def show_res(img, ann, title=None):
    plt.figure()
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    ax = plt.gca()
    ax.set_autoscale_on(False)
    img1 = np.ones((img.shape[0], img.shape[1], 3))

    color_mask = np.random.random((1, 3)).tolist()[0]
    for i in range(3):
        img1[:, :, i] = color_mask[i]

    ax.imshow(np.dstack((img1, ann)))
    plt.title(title)
    plt.show()

def get_label_from_annotation(anno, lbl_class):
    valid_entries_class_labels = lbl_class
    # Stack the binary masks for each class
    labels_2d = map(lambda x: np.equal(anno, x),
                    valid_entries_class_labels)

    # Perform the merging of all of the binary masks into one matrix
    labels_2d_stacked = np.stack(labels_2d, axis=2)

    labels_2d_stacked_float = labels_2d_stacked.astype(np.float)
    return labels_2d_stacked_float

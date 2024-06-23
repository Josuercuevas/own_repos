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

#========================================================================================
# COCO
train_ann_file = "datasets/coco/annotations/instances_train2014.json"
train_dir = "datasets/coco/train2014"
train_img_name_format = "COCO_train2014_%012d.jpg"

val_ann_file = "datasets/coco/annotations/instances_val2014.json"
val_dir = "datasets/coco/val2014"
val_img_name_format = "COCO_val2014_%012d.jpg"
num_classes = 80# for COCO
class_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
              '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
              '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
              '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
              '40', '41', '42', '43', '44', '45', '46', '47', '48', '49',
              '50', '51', '52', '53', '54', '55', '56', '57', '58', '59',
              '60', '61', '62', '63', '64', '65', '66', '67', '68', '69',
              '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80']
image_size_x = 288#1280
image_size_y = 288#720


## CityScapes
# DT_Path = 'datasets/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train'
# ANN_Path = 'datasets/cityscapes/gtFine_trainvaltest/gtFine/train'
#
# val_ann_file = ""
# val_dir = ""
# val_img_name_format = ""
#
# num_classes = 8# for Cityscapes
#
# image_size_x = 512 # 2048 #
# image_size_y = 256 # 1024 #
#========================================================================================

# classes to be used during training
classes_used = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]

# mapped to this indices
classes_mapp = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]

for i in range(0, num_classes):
    classes_used.append(i+1)
    classes_mapp.append(i)
stride = 16
scale_cvg = 0.4 #0.7 #
gridbox_type = 'GRIDBOX_MIN'
coverage_type = 'RECTANGULAR'
min_cvg_len = 20
obj_norm = True
crop_bboxes = True
mean_value = [106.186019897, 111.089630127, 112.71131134]

gridbox_cvg_threshold = 0.05 #5e-20 #
gridbox_rect_thresh = 3
gridbox_rect_eps = 0.02
min_height = 22 #

if num_classes > 5:
    LEARNING_RATE = 1e-6
else:
    LEARNING_RATE = 1e-5

EDGES_CLASSES = 3
BBOX_DIM = 4
TOTAL_CLASSES = len(classes_used)
SEMANTIC_CLASSES = len(classes_used)+1
FREEZE_WEIGHTS = False
FREEZE_BODYNET = True
ENABLE_VALIDATION = False

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

# This function is to generate coverage and bounding-box labels
# Most methods translate from NVIDIA caffe/src/caffe/util/detectnet_coverage_rectangular.cpp
# created at 7 March 2017
import sys
import numpy as np
sys.path.append("../")
import config
import math

def resize_bbox_list(bboxlist, rescale_x=1, rescale_y=1):
    bboxListNew = []
    for bbox in bboxlist:
        abox = bbox
        abox[0] *= rescale_x
        abox[1] *= rescale_y
        abox[2] *= rescale_x
        abox[3] *= rescale_y
        bboxListNew.append(abox)
    return bboxListNew

def pruneBboxes(bboxlist, lbl_classes):
    FLT_EPSILON = 1.19209e-07
    bboxlist_new = []
    bboxlist_idx = []
    for idx, bbox in enumerate(bboxlist):
        if lbl_classes[idx] not in config.classes_used:
            continue
        bbox_area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        if bbox_area < FLT_EPSILON:
            continue
        bboxlist_new.append(bbox)
        bboxlist_idx.append(idx)
    return bboxlist_idx, bboxlist_new

def coverageBoundingBox(bbox):
    #find the center of the bbox as the midpoint between its top-left and bottom-right points
    center = [(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5]

    bb_width = bbox[2] - bbox[0]
    bb_height = bbox[3] - bbox[1]
    shrunkSize = [ bb_width*config.scale_cvg, bb_height*config.scale_cvg ]

    if config.gridbox_type == 'GRIDBOX_MIN':
        shrunkSize[0] =  min(bb_width, max(config.min_cvg_len, shrunkSize[0]))
        shrunkSize[1] = min(bb_height, max(config.min_cvg_len, shrunkSize[1]))
    else:
        print("not implemented yet!")
        exit(1)
    coverage = (np.subtract(center, np.multiply(shrunkSize, 0.5)), np.add(center, np.multiply(shrunkSize, 0.5)))
    return np.ravel(coverage)


def imageRectToGridRect(coverage):
    return [math.floor(1.0*coverage[0]/config.stride), math.floor(1.0*(coverage[1])/config.stride),
            math.ceil(1.0 * (coverage[2]) / config.stride), math.ceil(1.0 * (coverage[3])/config.stride)]

def bb_intersection_area(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = (xB - xA + 1) * (yB - yA + 1)
    return interArea

def objectNormValue(gridBoxArea, coverage):
    coverageArea = (coverage[2]-coverage[0]) * (coverage[3]-coverage[1])
    return 1.0*gridBoxArea / coverageArea

def generateLabel(bboxlist, lbl_classes):
    gridBoxArea = config.stride * config.stride
    FLT_EPSILON = 1.19209e-07

    grid_x = 1.0 * config.image_size_x / config.stride
    grid_y = 1.0 * config.image_size_y / config.stride
    label_coverages = np.zeros((len(config.classes_used), int(grid_x), int(grid_y)))
    label_bbox = np.zeros((4, int(grid_x), int(grid_y))) #np.zeros((4*len(config.classes_used), int(grid_x), int(grid_y)))
    #label_bbox = np.zeros((4, int(grid_x), int(grid_y)))
    label_size = np.zeros((2, int(grid_x), int(grid_y))) #np.zeros((2*len(config.classes_used), int(grid_x), int(grid_y)))
    label_obj_norm = np.zeros((1, int(grid_x), int(grid_y))) #np.zeros((len(config.classes_used), int(grid_x), int(grid_y)))
    label_foreground = np.zeros((1, int(grid_x), int(grid_y))) #np.zeros((len(config.classes_used), int(grid_x), int(grid_y)))

    # print(label_coverages.shape, label_bbox.shape, label_size.shape, label_obj_norm.shape, label_foreground.shape)

    for idx, bbox in enumerate(bboxlist):
        coverage = coverageBoundingBox(bbox)
        #print "coverage:", coverage

        g_coverage = imageRectToGridRect(coverage)
        #print "g_coverage:", g_coverage

        for g_y in range(int(g_coverage[1]), int(g_coverage[3])):
            for g_x in range(int(g_coverage[0]), int(g_coverage[2])):
                gridBox = (np.multiply([g_x, g_y], config.stride), [config.stride, config.stride])
                gridBox = np.ravel(gridBox)
                cvgArea = bb_intersection_area(coverage, gridBox) / gridBoxArea
                #print cvgArea
                if cvgArea > FLT_EPSILON:
                    if g_x < grid_x and g_y < grid_y:
                        #print lbl_classes[idx]
                        label_coverages[lbl_classes[idx], g_x, g_y] = 1.0

                        # label_topLeft_x
                        # label_bbox[0+(lbl_classes[idx]*3 + lbl_classes[idx]), g_x, g_y] = bbox[0] - gridBox[0]
                        label_bbox[0, g_x, g_y] = bbox[0] - gridBox[0]

                        # label_topLeft_y
                        # label_bbox[1+(lbl_classes[idx]*3 + lbl_classes[idx]), g_x, g_y] = bbox[1] - gridBox[1]
                        label_bbox[1, g_x, g_y] = bbox[1] - gridBox[1]

                        # label_bottomRight_x
                        # label_bbox[2+(lbl_classes[idx]*3 + lbl_classes[idx]), g_x, g_y] = bbox[2] - gridBox[0]
                        label_bbox[2] = bbox[2] - gridBox[0]

                        #label_bottomRight_y
                        # label_bbox[3+(lbl_classes[idx]*3 + lbl_classes[idx]), g_x, g_y] = bbox[3] - gridBox[1]
                        label_bbox[3] = bbox[3] - gridBox[1]

                        # Foreground
                        # label_foreground[(lbl_classes[idx]),g_x,g_y] = 1.0
                        label_foreground[0, g_x, g_y] = 1.0

                        # label_size[0+(lbl_classes[idx]*1 + lbl_classes[idx]), g_x, g_y] = 1.0 / (bbox[2]-bbox[0])
                        # label_size[1+(lbl_classes[idx]*1 + lbl_classes[idx]), g_x, g_y] = 1.0 / (bbox[3]-bbox[1])
                        label_size[0, g_x, g_y] = 1.0 / (bbox[2] - bbox[0])
                        label_size[1, g_x, g_y] = 1.0 / (bbox[3] - bbox[1])

                        minObjNorm = 1.0
                        if config.obj_norm:
                            minObjNorm = 0.0

                        dObjNormValue = objectNormValue(gridBoxArea, coverage)
                        # label_obj_norm[lbl_classes[idx], g_x, g_y] = max(minObjNorm, dObjNormValue)
                        label_obj_norm[0, g_x, g_y] = max(minObjNorm, dObjNormValue)

    return label_bbox, label_coverages, label_size, label_obj_norm, label_foreground

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

import math
import cv2 as cv
import numpy as np
import sys
import config
MAX_BOXES = 5


def gridbox_to_boxes(net_cvg, net_boxes):
    im_sz_x = config.image_size_x
    im_sz_y = config.image_size_y
    stride = config.stride

    grid_sz_x = int(im_sz_x / stride)
    grid_sz_y = int(im_sz_y / stride)

    boxes = []
    cvgs = []

    cell_width = im_sz_x / grid_sz_x
    cell_height = im_sz_y / grid_sz_y

    cvg_val = net_cvg[0:grid_sz_y, 0:grid_sz_x, 0]#net_cvg[0:grid_sz_y][0:grid_sz_x][0]

    mask = (cvg_val >= config.gridbox_cvg_threshold)
    coord = np.where(mask == 1)

    y = np.asarray(coord[0])
    x = np.asarray(coord[1])

    mx = x * cell_width
    my = y * cell_height

    net_boxes =  np.array(net_boxes)#*10000000
    #print mx, my


    x1 = (np.asarray([net_boxes[y[i],x[i],0] for i in xrange(x.size)]) + mx)
    y1 = (np.asarray([net_boxes[y[i],x[i],1] for i in xrange(x.size)]) + my)
    x2 = (np.asarray([net_boxes[y[i],x[i],2] for i in xrange(x.size)]) + mx)
    y2 = (np.asarray([net_boxes[y[i],x[i],3] for i in xrange(x.size)]) + my)

    #print net_boxes[0]
    #print [x1[0], y1[0], x2[0], y2[0]]
    #kjhkh

    boxes = np.transpose(np.vstack((x1, y1, x2, y2)))
    cvgs = np.transpose(np.vstack((x, y, np.asarray(
        [cvg_val[y[i]][x[i]] for i in xrange(x.size)]))))
    return boxes, cvgs, mask


def vote_boxes(propose_boxes, propose_cvgs, mask):
    """ Vote amongst the boxes using openCV's built-in clustering routine.
    """

    detections_per_image = []
    if not propose_boxes.any():
        return detections_per_image

    ######################################################################
    # GROUP RECTANGLES Clustering
    ######################################################################
    #print np.array(propose_boxes).tolist()
    nboxes, weights = cv.groupRectangles(
        np.array(propose_boxes).tolist(),
        config.gridbox_rect_thresh,
        config.gridbox_rect_eps)
    if len(nboxes):
        for rect, weight in zip(nboxes, weights):
            print (rect[3] - rect[1])
            if (rect[3] - rect[1]) >= config.min_height:
                confidence = math.log(weight[0])
                detection = [rect[0], rect[1], rect[2], rect[3], confidence]
                detections_per_image.append(detection)

    return detections_per_image


def cluster(net_cvg, net_boxes):
    """
    Read output of inference and turn into Bounding Boxes
    """
    batch_size = net_cvg.shape[0]
    boxes = np.zeros([batch_size, MAX_BOXES, 5])

    for i in range(batch_size):

        cur_cvg = net_cvg[i]
        cur_boxes = net_boxes[i]

        # Gather proposals that pass a threshold -
        propose_boxes, propose_cvgs, mask = gridbox_to_boxes(cur_cvg, cur_boxes)


        # Vote across the proposals to get bboxes
        # boxes_cur_image = vote_boxes(propose_boxes, propose_cvgs, mask)
        # boxes_cur_image = np.asarray(boxes_cur_image, dtype=np.float16)

        # if (boxes_cur_image.shape[0] != 0):
        #    [r, c] = boxes_cur_image.shape
        #    boxes[i, 0:r, 0:c] = boxes_cur_image

    return propose_boxes, mask

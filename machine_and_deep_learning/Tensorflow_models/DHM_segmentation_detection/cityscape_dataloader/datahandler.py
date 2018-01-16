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

import sys
sys.path.append("../")
sys.path.append('cocoapi/PythonAPI')
from pycocotools.coco import COCO
from pycocotools.coco import COCO
from pycocotools.mask import frPyObjects, decode, toBbox
import matplotlib.pyplot as plt
import config
import tensorflow as tf
import numpy as np
import random
import os
import cv2
import math
from scipy.misc import imresize
from skimage import feature
from PIL import Image
import matplotlib.patches as patches
from genlabel import getAnns
from datautils import resize_mask, show_res, get_label_from_annotation
from gridbox_extractor import resize_bbox_list, pruneBboxes, generateLabel
import time

np.set_printoptions(threshold=np.nan)

FLAGS = tf.app.flags.FLAGS

class Loader:
    def __init__(self, train=True):
        self.labels = ['unlabeled', 'ego vehicle', 'rectification border', 'out of roi', 'static', 'dynamic', 'ground',
                       'road', 'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence', 'guard rail', 'bridge',
                       'tunnel', 'pole', 'polegroup','traffic light', 'traffic sign', 'vegetation', 'terrain',
                        'sky','person', 'rider', 'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle',
                       'bicycle']

        #semantic_label = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'trafic sign',
        #                  'vegetation',
        #                  'terrain', 'sky', 'person', 'rider', 'car', 'bus', 'truck', 'on rails', 'motorcycle',
        #                  'bicycle']

        '''
            Mapping:
            0. unlabeled                        10. rail track          20. traffic sign    30. trailer
            1. ego vehicle                      11. building            21. vegetation      31. train
            2. rectification border             12. wall                22. terrain         32. motorcycle
            3. out of roi                       13. fence               23. sky             33. bicycle
            4. static                           14. guard rail          24. person          34.
            5. dynamic                          15. bridge              25. rider           35.
            6. ground                           16. tunnel              26. car
            7. road                             17. pole                27. truck
            8. sidewalk                         18. polegroup           28. bus
            9. parking                          19. traffic light       29. caravan
        '''

        if config.num_classes == 8:
            self.used_lbl = [24, 25, 26, 27, 28, 29, 32, 33]
            self.re_lbl = [1, 2, 3, 4, 5, 6, 7, 8]
        else:
            self.used_lbl = [7, 8, 11, 12, 13, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 32, 33]
            self.re_lbl = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


        if train:
            dtPath = config.DT_Path
            annPath = config.ANN_Path
            self.imgPath = []
            self.jsonPath = []

            self.instPath = []
            self.semtPath = []
            for city in os.listdir(dtPath):
                for img in os.listdir(os.path.join(dtPath, city)):
                    self.imgPath.append(os.path.join(dtPath, city, img))
                jsos = [each for each in os.listdir(os.path.join(annPath, city)) if each.endswith('.json')]
                inss = [each for each in os.listdir(os.path.join(annPath, city)) if each.endswith('instanceIds.png')]
                sems = [each for each in os.listdir(os.path.join(annPath, city)) if each.endswith('labelIds.png')]
                for jso in jsos:
                    self.jsonPath.append(os.path.join(annPath, city, jso))
                for ins in inss:
                    self.instPath.append(os.path.join(annPath, city, ins))
                for sem in sems:
                    self.semtPath.append(os.path.join(annPath, city, sem))

            print("Total train Images:", len(self.imgPath))
            self.imgPath = sorted(self.imgPath)
            print("Total train Json:", len(self.jsonPath))
            self.jsonPath = sorted(self.jsonPath)
            print("Total train Instance:", len(self.instPath))
            self.instPath = sorted(self.instPath)
            print("Total train Semantic:", len(self.semtPath))
            self.semtPath = sorted(self.semtPath)

            self.img_ids = self.imgPath


    def create_batches(self, batch_size, shuffle=True):
        batch = []

        while True:
            indices = range(len(self.imgPath))

            if shuffle:
                indices = np.random.permutation(indices)
            for index in indices:
                index += 1
                try:
                    imgpath = self.imgPath[index]
                    jsonpath = self.jsonPath[index]
                    semtPath = self.semtPath[index]
                    instPath = self.instPath[index]
                except:
                    print(index)
                    continue

                # ==========================================================================
                I = cv2.imread(imgpath)
                if len(I.shape) != 3:
                    continue
                #print(imgpath)
                inslabel, insegs, _, _ = getAnns(jsonpath, imgpath)
                ann_list = []
                rles = []
                mask_lbl = []

                for idx, seg in enumerate(insegs):
                    try:
                        rle = frPyObjects([seg], I.shape[0], I.shape[1])[0]
                        bbb = toBbox(rle)
                        #print bbb
                    except:
                        #print "continue"
                        continue
                    #rles.append(rle)
                    ann_list.append((bbb, inslabel[idx]))
                    #mask_lbl.append(inslabel[idx])

                #for idx, seg in enumerate(semsegs):
                #    try:
                #        rle = frPyObjects([seg], I.shape[0], I.shape[1])[0]
                        #bbb = toBbox(rle)
                            # print bbb
                #    except:
                            # print "continue"
                #        continue
                #    rles.append(rle)
                #    mask_lbl.append(semlabel[idx])


                S = cv2.imread(semtPath)
                if len(S.shape) != 3:
                    continue

                #S.any(not in self.used_lbl) = 0\
                S = np.array(S)

                mask = np.zeros((1024, 2048, 3))
                arr = S
                for il, l in enumerate(self.used_lbl):
                    idx = np.where(S == l)
                    mask[idx] = self.re_lbl[il]

                mask1 = mask[:, :, 0]#np.zeros((1024, 2048))
                #for ch in xrange(3):
                #    print np.max(mask[:,:,ch])
                #    mask1 += mask[:,:,ch]

                #mask = decode(rles).astype(np.float)
                S = cv2.imread(instPath)
                if len(S.shape) != 3:
                    continue

                mask2 = np.zeros((1024, 2048))
                for ch in xrange(3):
                    mask2 += S[:, :, ch]

                batch.append((I, mask1, mask2, ann_list))

                if len(batch) >= batch_size:
                    yield batch
                    batch = []
                # ==============================================================================

    def preprocess_batch(self, batch):
        imgs = []
        edges = []
        masks = []
        coverages = []
        bboxes = []
        bboxes_4 = []
        coverageblock = []
        sizeblock = []
        objblock = []
        all_used_anns = []
        classes = []

        print("Processing new batch with %d elements ..." % (len(batch)))

        for img, mask_sem, mask_ins, anns in batch:
            # handling image resizing if needed
            resized_img = imresize(img, (config.image_size_y, config.image_size_x), interp='lanczos')
            imgs.append(resized_img)

            # Handling semantic aware mask with each class on a different channel
            rescale_x = 1.0 * resized_img.shape[1] / np.array(img).shape[1]
            rescale_y = 1.0 * resized_img.shape[0] / np.array(img).shape[0]
            resized_mask_semmantic = resize_mask(mask_sem, resized_img.shape[:2])
            resized_mask_instances = resize_mask(mask_ins, resized_img.shape[:2])

            semmantic_mask = np.zeros((resized_img.shape[0], resized_img.shape[1], config.SEMANTIC_CLASSES))
            for y in range(resized_mask_semmantic.shape[0]):
                for x in range(resized_mask_semmantic.shape[1]):
                    # we put the flag on the channel that correspongs to the class
                    if resized_mask_semmantic[y, x] > 0.0:
                        semmantic_mask[y, x, int(resized_mask_semmantic[y, x])] = 1.0
            masks.append(semmantic_mask)
            # masks.append(resized_mask_semmantic)

            # print(resized_mask_semmantic.shape)
            # with open("mask_pixels.csv", "w") as text_file:
            #     for y in range(resized_mask_semmantic.shape[0]):
            #         for x in range(resized_mask_semmantic.shape[1]):
            #             text_file.write("%d,"%resized_mask_semmantic[y, x])
            #         text_file.write("\n")
            #     text_file.close()
            #
            # sys.exit(0)

            # edges only from semmantic labeling
            mask_sem = np.mean(resized_mask_semmantic, axis=2)
            edge = feature.canny(mask_sem, sigma=1.0) * 1.0
            on_edge = np.where(edge == 1)
            edge[on_edge] = 2

            # Instance edges only
            mask_instances = np.mean(resized_mask_instances, axis=2)
            edge += mask_instances
            on_edge = np.where(edge > 2)
            edge[on_edge] = 2
            # edge_obj_interior = np.array(np.array(mask)) * 1
            # edge = np.add(edge, edge_obj_interior)
            # on_edge = np.where(edge > 2)
            # edge[on_edge] = 2
            edges.append(edge)

            boxes = []
            instance_class = []
            for box, id in anns:
                # top.x
                box[0] *= rescale_x
                # top.y
                box[1] *= rescale_y
                box[2] *= rescale_x
                box[3] *= rescale_y
                # bottom.x
                box[2] += box[0]
                # bottom.y
                box[3] += box[1]

                boxes.append(box)
                instance_class.append(id)

            classes.append(instance_class)
            label_bbox, label_coverages, label_size, label_obj, label_foreground = generateLabel(boxes, instance_class)

            # label_foreground = np.expand_dims(label_foreground, axis = 0)
            # label_obj = np.expand_dims(label_obj, axis=0)

            coverage_block = np.concatenate((label_foreground, label_foreground, label_foreground, label_foreground),
                                            axis=0)
            size_block = np.concatenate((label_size, label_size), axis=0)
            obj_block = np.concatenate((label_obj, label_obj, label_obj, label_obj), axis=0)
            bbox_label_norm = np.multiply(label_bbox, size_block)
            bbox_obj_label_norm = np.multiply(bbox_label_norm, obj_block)
            # print label_coverages.shape, label_bbox.shape
            bboxes.append(bbox_obj_label_norm)
            coverageblock.append(coverage_block)
            sizeblock.append(size_block)
            objblock.append(obj_block)
            coverages.append(label_coverages)

        return np.asarray(classes), np.asarray(imgs), np.asarray(edges), np.asarray(masks), np.asarray(coverages), np.asarray(bboxes), \
               np.asarray(coverageblock), np.asarray(sizeblock), np.asarray(objblock)

    def test_coco_api(self):
        batch = self.create_batches(1, shuffle=True)
        for b in batch:
            classes, imgs, edges, masks, coverages, bboxes, covblock, sizeblock, objblock = self.preprocess_batch(b)

            print("Input:")
            print(np.array(imgs).shape)
            print("Masks:")
            print(np.array(masks).shape)
            print("Edges:")
            print(np.array(edges).shape)
            print("Maxes:")
            print(np.max(masks), np.max(edges))
            print("coverages:")
            print(coverages.shape)

            print("Flatten after average:")
            print(np.sum(masks[0], axis=2, keepdims=True).shape)


            # GRID BOX
            fig, ax = plt.subplots(1)
            ax.imshow(cv2.cvtColor(imgs[0], cv2.COLOR_BGR2RGB))
            ax.imshow(imgs[0])
            ax = plt.gca()
            ax.set_autoscale_on(False)



            for i in range(0, config.image_size_y / config.stride):
                for j in range(0, config.image_size_x / config.stride):
                    if np.sum(coverages[0][:, j, i]) != 0:
                        rect = patches.Rectangle((j * config.stride, i * config.stride), config.stride, config.stride,
                                                 linewidth=3, edgecolor='g', facecolor='none')
                    # else: # Takes too much time !!
                    #    rect = patches.Rectangle((j * config.stride, i * config.stride), config.stride, config.stride,
                    #                             linewidth=1, edgecolor='r', facecolor='none')

                        ax.add_patch(rect)


            ax.set_title('GridBox-GT')
            plt.show()



            # COVERAGE BOX
            fig, ax = plt.subplots(1)
            ax.imshow(cv2.cvtColor(imgs[0], cv2.COLOR_BGR2RGB))
            ax.imshow(imgs[0])
            ax = plt.gca()
            ax.set_autoscale_on(False)

            for i in range(0, config.image_size_y / config.stride):
                for j in range(0, config.image_size_x / config.stride):
                    if np.sum(coverages[0][:, j, i]) != 0:
                        rect = patches.Rectangle((j * config.stride, i * config.stride), config.stride, config.stride,
                                                 linewidth=2, edgecolor='g', facecolor='blue')
                        # else: # Takes too much time !!
                        #    rect = patches.Rectangle((j * config.stride, i * config.stride), config.stride, config.stride,
                        #                             linewidth=1, edgecolor='r', facecolor='none')

                        ax.add_patch(rect)

            ax.set_title('Coverage-GT')
            plt.show()

            # show_res(imgs[0], (np.sum(masks[0], axis=2, keepdims=True)), "Masks")
            show_res(imgs[0], masks[0], "Masks")
            show_res(imgs[0], edges[0], "Edges")



if __name__ == "__main__":
    loader = Loader()
    loader.test_coco_api()

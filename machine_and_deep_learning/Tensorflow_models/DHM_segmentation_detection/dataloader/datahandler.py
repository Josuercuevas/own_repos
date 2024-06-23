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

'''
    Batch Loader routines for the data and ground-truth to be used
    during training of our DHM network
'''

import sys
sys.path.append('cocoapi/PythonAPI')
from pycocotools.coco import COCO
from pycocotools.mask import frPyObjects, decode, toBbox
import matplotlib.pyplot as plt

from clustering import *
import tensorflow as tf
import config
import numpy as np
import os
import cv2
from scipy.misc import imresize
from datautils import resize_mask, show_img_mask, show_res, get_label_from_annotation
from skimage import feature
from gridbox_extractor import generateLabel
import matplotlib.patches as patches

np.set_printoptions(threshold=np.nan)


FLAGS = tf.app.flags.FLAGS

class Loader:
    def __init__(self, train=True):
        if train:
            self.image_dir = config.train_dir
            ann_file = config.train_ann_file
            self.get_image_path = self.get_train_path
        else:
            self.image_dir = config.val_dir
            ann_file = config.val_ann_file
            self.get_image_path = self.get_val_path

        self.coco = COCO(ann_file)
        cats = self.coco.loadCats(self.coco.getCatIds())
        #print cats

        names = [cat['name'] for cat in cats]
        # id is number from pycocotools
        # i is actual index used
        id2name = dict((cat["id"], cat["name"]) for cat in cats)
        self.id2i = dict((cats[i]['id'], i+1) for i in range(len(cats)))
        self.i2name = {v: id2name[k] for k, v in self.id2i.iteritems()}

        print("NUMBER OF CLASSES: %i" % len(id2name))

        self.cat_ids = self.coco.getCatIds()
        self.img_ids = self.coco.getImgIds()

        print("%i total training-imgId images" % len(self.img_ids))
        print("%i total training-catId images" % len(self.cat_ids))

        self.counter_samples = 0

    def get_train_path(self, id, name=None):
        if name == None:
            return config.train_img_name_format % id
        else:
            return name

    def get_val_path(self, id, name=None):
        if name == None:
            return config.val_img_name_format % id
        else:
            return name

    def create_batches(self, batch_size, shuffle=True):
        # 1 batch = [(image, [([x, y, w, h], id), ([x, y, w, h], id), ...]), ...]
        batch = []
        counter_samples=0

        while True:
            indices = range(len(self.img_ids))

            if shuffle:
                indices = np.random.permutation(indices)
            for index in indices:
                index += 1
                try:
                    img = self.coco.loadImgs(self.img_ids[index])[0]
                except:
                    print(index)
                    continue
                path = os.path.join(self.image_dir, self.get_image_path(id=img['id'], name=img['file_name']))
                I = cv2.imread(path).astype(np.uint8)[:, :, ::-1]
                I = np.ascontiguousarray(I)


                try:
                    if len(I.shape) != 3:
                        continue
                except:
                    print("no image exist")
                    continue

                ann_ids = self.coco.getAnnIds(imgIds=img['id'], catIds=self.cat_ids, iscrowd=None)
                anns = self.coco.loadAnns(ann_ids)
                ann_list = []

                rles = []
                for ann in anns:
                    bb = [f for f in ann["bbox"]]
                    try:
                        rle = frPyObjects(ann['segmentation'], I.shape[0], I.shape[1])[0]
                        bbb = toBbox(rle)
                    except:
                        continue
                    # make sure we dont include unknown classes
                    if self.id2i[ann["category_id"]] < 0 or self.id2i[ann["category_id"]] > config.TOTAL_CLASSES:
                        print("This class cannot be processed %d ...", self.id2i[ann["category_id"]])
                        continue

                    rles.append(rle)
                    ann_list.append((decode(rle).astype(np.float), bb, self.id2i[ann["category_id"]]))

                # print("------- RLEs-SHAPE")
                # print(len(rles))

                if len(rles) == 0:
                    print("NO RLE was extracted continue with next picture ...")
                    continue

                mask = decode(rles).astype(np.float)
                batch.append((I, mask, ann_list))

                # print("------- MASKS-SHAPE")
                # print(len(mask))
                # print(np.max(mask), np.min(mask))

                if len(batch) >= batch_size:
                    self.counter_samples += len(batch)
                    print("Getting new batch with %d elements ..." % (len(batch)))
                    yield batch
                    del mask
                    del ann_list
                    del rles
                    del batch
                    batch = []

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

        for img, mask, anns in batch:
            used_anns = []
            w = img.shape[1]
            h = img.shape[0]

            resized_img = imresize(img, (config.image_size_y, config.image_size_x), interp='lanczos')

            rescale_x = 1.0 * resized_img.shape[1] / np.array(img).shape[1]
            rescale_y = 1.0 * resized_img.shape[0] / np.array(img).shape[0]

            resized_mask_ = resize_mask(mask, resized_img.shape[:2])

            boxes = []
            # semmantic_masks = np.zeros((resized_img.shape[0], resized_img.shape[1], config.SEMANTIC_CLASSES))
            semmantic_masks = np.zeros((resized_img.shape[0], resized_img.shape[1], 1))
            instances_masks = np.zeros((resized_img.shape[0], resized_img.shape[1], resized_mask_.shape[2]))
            ids = []

            instance_id = 0
            color_offset = 255/resized_mask_.shape[2]
            for semm_mask, box, id in anns:
                # Top x
                box[0] *= rescale_x
                # Top y
                box[1] *= rescale_y
                box[2] *= rescale_x
                box[3] *= rescale_y

                # Bottom x
                box[2] += box[0]
                # Bottom y
                box[3] += box[1]

                boxes.append(box)
                ids.append(id-1)

                semm_mask = resize_mask(semm_mask, resized_img.shape[:2])
                # semm_mask = np.array(semm_mask)*id
                # m = np.array(m)

                #saving mask into instance aware tensor
                semm_mask = np.squeeze(semm_mask, axis=2)
                # print("insert in id: %d" % id)
                # print(m.shape)
                # show_res(resized_img, m, "Masks")
                # continue

                # per ID
                # semmantic_masks[:, :, 0] += semm_mask
                # index_relabeled = np.where(semmantic_masks[:, :, id] > 1)
                # semmantic_masks[index_relabeled] = 1
                temp = semmantic_masks[:, :, 0]
                pos_labels = np.where(temp == 0)
                temp[pos_labels] = np.multiply(semm_mask[pos_labels], (np.ones_like(semm_mask)*id)[pos_labels])
                semmantic_masks[:, :, 0] = temp

                # for y in range(semm_mask.shape[0]):
                #     for x in range(semm_mask.shape[1]):
                #         # we put the flag on the channel that correspongs to the class
                #         if semm_mask[y, x] > 0.0:
                #             semmantic_masks[y, x, 0] = semm_mask[y, x]*id

                # per Instance
                instances_masks[:, :, instance_id] = np.multiply(semm_mask, ((instance_id+1)*color_offset))
                instance_id += 1

                # bg = np.where(semm_mask == 0)
                # fg = np.where(semm_mask == 1)
                # semm_mask[bg] = 1
                # semm_mask[fg] = 0
                # # Label the background in the first location
                # semmantic_masks[:, :, 0] = semm_mask

            # grid box labels
            label_bbox, label_coverages, label_size, label_obj, label_foreground = generateLabel(boxes, ids)

            coverage_block = np.concatenate((label_foreground, label_foreground, label_foreground, label_foreground), axis=0)
            size_block = np.concatenate((label_size, label_size), axis=0)
            obj_block = np.concatenate((label_obj, label_obj, label_obj, label_obj), axis=0)
            bbox_label_norm = np.multiply(label_bbox, size_block)
            bbox_obj_label_norm = np.multiply(bbox_label_norm, obj_block)
            bboxes.append(bbox_obj_label_norm)
            coverageblock.append(coverage_block)
            sizeblock.append(size_block)
            objblock.append(obj_block)

            mask = np.mean(instances_masks, axis=2)

            # print("%d instances found" % (instances_masks.shape[2]))
            # show_res(resized_img, mask, "Instances-Masks")

            edge = feature.canny(mask, sigma=1.0) * 1.0
            # print(np.max(edge), np.min(edge))
            # show_res(resized_img, edge, "Canny")

            on_edge = np.where(edge == 1)
            edge[on_edge] = 2

            instances = mask
            obj_interior = np.array(np.array(instances)) * 1
            edge = np.add(edge, obj_interior)
            on_edge = np.where(edge > 2)
            edge[on_edge] = 2

            # print(np.max(edge), np.min(edge))
            # show_res(resized_img, np.multiply(edge, 20), "Edges")
            # sys.exit(0)

            # edge = get_label_from_annotation(edge, [0, 1, 2])
            # edge = get_label_from_annotation(edge, [0, 1, 2])
            # ms = get_label_from_annotation(ms, [0, 1, 2,3,4, 5])

            # ============================================================
            # Ground truth extraction from training images
            # training image
            imgs.append(resized_img)

            # Masks
            '''
                since logits need to have valid probabilities, we need to give values different than 0
            '''
            # locations = np.where(semmantic_masks <= 0.0)
            # semmantic_masks[locations] = 0.00001
            masks.append(semmantic_masks)

            # Edges
            edges.append(edge)

            # Class Id
            classes.append(ids)

            # Coverages
            coverages.append(label_coverages)

            # Bboxes
            bboxes_4.append(boxes)
            # ============================================================

        return np.asarray(classes), np.asarray(imgs), np.asarray(edges), np.asarray(masks), np.asarray(coverages), np.asarray(bboxes), \
               np.asarray(coverageblock), np.asarray(sizeblock), np.asarray(objblock), np.asarray(bboxes_4)

    def test_coco_api(self):
        loader = Loader()
        batch = loader.create_batches(1, shuffle=True)
        i = 0
        for b in batch:
            # [(image, [([x, y, w, h], id), ([x, y, w, h], id), ...]), ...]
            classes, imgs, edges, masks, coverages, bboxes, covblock, sizeblock, objblock, bboxes_4 = \
                loader.preprocess_batch(b)

            print(bboxes_4[0])

            pred_cov = coverages[0].transpose(1, 2, 0)
            pred_cov = np.expand_dims(pred_cov, axis=0)

            pred_bbox = bboxes[0].transpose(1, 2, 0)
            pred_bbox = np.expand_dims(pred_bbox, axis=0)

            fig, ax = plt.subplots(1)
            ax.imshow(imgs[0])
            ax = plt.gca()
            ax.set_autoscale_on(False)

            for i in range(0, (config.image_size_y/config.stride)):
                for j in range(0, (config.image_size_x / config.stride)):
                    if np.sum(coverages[0][:, j, i]) != 0:
                        rect = patches.Rectangle((j*config.stride, i*config.stride), config.stride, config.stride,
                                                 linewidth=3, edgecolor='g', facecolor='none')

                        # ax.add_patch(rect)

                        ax.add_artist(rect)
                        rx, ry = rect.get_xy()
                        cx = rx + rect.get_width() / 2.0
                        cy = ry + rect.get_height() / 2.0

                        ax.annotate(str(coverages[0][:, j, i]), (cx, cy), color='w', weight='bold',
                                    fontsize=6, ha='center', va='center')
                    else:
                        rect = patches.Rectangle((j * config.stride, i * config.stride), config.stride, config.stride,
                                                 linewidth=1, edgecolor='r', facecolor='none')

                        ax.add_patch(rect)

            ax.set_title('GridBox-GT')
            plt.show()

            fig, ax = plt.subplots(1)
            ax.imshow(imgs[0])
            ax = plt.gca()
            ax.set_autoscale_on(False)
            itext = 0
            print(bboxes_4[0].shape)
            print(classes)
            print(classes.shape)
            for idx, bb in enumerate(bboxes_4[0]):
                rect = patches.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], linewidth=1, edgecolor='r',
                                         facecolor='none')
                # ax.add_patch(rect)

                ax.add_artist(rect)
                rx, ry = rect.get_xy()
                cx = rx + rect.get_width() / 2.0
                cy = ry + rect.get_height() / 2.0

                ax.annotate(config.class_name[classes[0, idx]] + ": 1.0", (cx, cy), color='w', weight='bold',
                                fontsize=6, ha='center', va='center')
                itext += 1

            plt.title("BBox-GT")
            plt.show()

            # ax.set_title('GridBox/Mask/Edges-GT')
            # show_img_mask(imgs[0], masks[0], ax)
            # show_img_mask(imgs[0], edges[0], ax)
            # plt.show()

            print("%d, masks found with shape:"%(len(masks)))
            print(masks.shape)
            mask_obtained = masks[0]
            # mask_obtained[:, :, 0] = 0
            show_res(imgs[0], mask_obtained, "Masks")

            print("Edges shapes is:")
            print(edges.shape)
            show_res(imgs[0], edges[0], "Edges")

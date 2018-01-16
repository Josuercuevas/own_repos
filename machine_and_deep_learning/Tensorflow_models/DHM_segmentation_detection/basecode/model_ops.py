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

from __future__ import division
import time
from core_ops import *
import sys
from scipy.misc import imresize
import matplotlib.pyplot as plt
# from cityscape_dataloader.datautils import show_img_mask, show_res
from dataloader.datautils import show_img_mask, show_res, show_res_validation
import matplotlib.patches as patches
# from cityscape_dataloader.genlabel import *
import config
import cv2
from PIL import Image

class BaseOps(object):
    def build(self, input_args):
        if input_args.debug_info >= 1:
            print("Building class BaseOps")

    def train(self, sess, input_args, model_saver=None, datareader=None, summary_merger=None,
              summary_writer=None):
        if input_args.debug_info >= 1:
            print("Training model with configuration:\n"
                  "Learning rate: %4.12f\n"
                  "N_edges: %d\n"
                  "N_classes: %d\n"
                  "Image_transform: %dx%d\n\n"%(config.LEARNING_RATE, config.EDGES_CLASSES,
                                                config.TOTAL_CLASSES, config.image_size_y, config.image_size_x))

        # check they really exist
        if input_args.debug_info >= 1:
            print("Inside the Trainer checking visibility of all variables")
            print(self.feats.get_shape())
            print(self.pinputimage.get_shape())

        if input_args.debug_info >= 2:
            if summary_merger is not None:
                print(summary_merger)

        step = 0
        accum_retrieve_time=0
        accum_prop_time=0
        tot_steps = input_args.max_epochs*len(datareader.img_ids)
        train_batches = datareader.create_batches(input_args.batch_size, shuffle=True)
        while step <= tot_steps:
            step += 1

            batch_get = time.time()
            # ===================================================================================
            batch = train_batches.next()

            # cityscapes
            classes, imgs, edges, masks, coverages, bboxes, covblock, sizeblock, objblock, bboxes_4 = \
                datareader.preprocess_batch(batch)

            gt_edge = np.array(edges)
            gt_edge = np.expand_dims(gt_edge, axis=3)
            gt_mask = np.array(masks)
            # gt_mask = np.expand_dims(gt_mask, axis=3)
            gt_coverage = np.array(coverages).transpose(0, 3, 2, 1)
            gt_bbox = np.array(bboxes).transpose(0, 3, 2, 1)
            gt_covblock = np.array(covblock).transpose(0, 3, 2, 1)
            gt_sizeblock = np.array(sizeblock).transpose(0, 3, 2, 1)
            gt_objblock = np.array(objblock).transpose(0, 3, 2, 1)
            transformed_data = imgs  # - config.mean_value
            # ===================================================================================
            accum_retrieve_time += ((time.time() - batch_get) * 1000.0)
            print("Average time taken to retrieve the batch was: %4.4f ms" % (accum_retrieve_time / float(step)))

            batch_forwards_backwards = time.time()
            # ===================================================================================
            if input_args.enable_tensorboard == 'true':
                if summary_merger is None:
                    print("We cannot output to tensorboard if we dont have a file writer, exiting ...")
                    return

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                if (step % 10) == 0:
                    feat = sess.run([self.feats], feed_dict={self.pinputimage: transformed_data})
                    summary, lossedge, lossseg, losscov, lossbbox, lossall, _, pred_seg, pred_edge, coverage, \
                    bbox_regressor, seg_upscores, edge_upscores, IoU, MAP = sess.run(
                        [summary_merger, self.loss_edge, self.loss_seg, self.loss_cov, self.loss_bbox, self.lossall,
                         self.train_op, self.pred_seg, self.pred_edge, self.coverage, self.bbox_regressor,
                         self.seg_upscores, self.edge_upscores, self.IoU, self.mAP],
                        feed_dict={self.input_feat: feat[0],
                                   self.pgtedge: gt_edge, self.pgtmask: gt_mask, self.pgtcoverage: gt_coverage,
                                   self.pgtbbox: gt_bbox,
                                   self.pgtcovblock: gt_covblock, self.pgtsizeblock: gt_sizeblock,
                                   self.pgtobjblock: gt_objblock, self.transformed_input: transformed_data},
                        options=run_options, run_metadata=run_metadata)

                    summary_writer.add_run_metadata(run_metadata,
                                                    'step%03d' % (step))
                    summary_writer.add_summary(summary, (step))
                else:
                    feat = sess.run([self.feats], feed_dict={self.pinputimage: transformed_data})
                    summary, lossedge, lossseg, losscov, lossbbox, lossall, _, pred_seg, pred_edge, coverage, \
                    bbox_regressor, seg_upscores, edge_upscores, IoU, MAP = sess.run(
                        [summary_merger, self.loss_edge, self.loss_seg, self.loss_cov, self.loss_bbox, self.lossall,
                         self.train_op, self.pred_seg, self.pred_edge, self.coverage, self.bbox_regressor,
                         self.seg_upscores, self.edge_upscores, self.IoU, self.mAP],
                        feed_dict={self.input_feat: feat[0],
                                   self.pgtedge: gt_edge, self.pgtmask: gt_mask, self.pgtcoverage: gt_coverage,
                                   self.pgtbbox: gt_bbox,
                                   self.pgtcovblock: gt_covblock, self.pgtsizeblock: gt_sizeblock,
                                   self.pgtobjblock: gt_objblock, self.transformed_input: transformed_data},
                        options=run_options, run_metadata=run_metadata)
            else:
                feat = sess.run([self.feats], feed_dict={self.pinputimage: transformed_data})
                lossedge, lossseg, losscov, lossbbox, lossall, _, pred_seg, pred_edge, coverage, bbox_regressor \
                    , seg_upscores, edge_upscores, IoU, MAP = sess.run(
                    [self.loss_edge, self.loss_seg, self.loss_cov, self.loss_bbox, self.lossall,
                     self.train_op, self.pred_seg, self.pred_edge, self.coverage, self.bbox_regressor,
                     self.seg_upscores, self.edge_upscores, self.IoU, self.mAP],
                    feed_dict={self.input_feat: feat[0],
                               self.pgtedge: gt_edge, self.pgtmask: gt_mask, self.pgtcoverage: gt_coverage,
                               self.pgtbbox: gt_bbox,
                               self.pgtcovblock: gt_covblock, self.pgtsizeblock: gt_sizeblock,
                               self.pgtobjblock: gt_objblock, self.transformed_input: transformed_data})
            # ===================================================================================
            accum_prop_time = ((time.time() - batch_forwards_backwards) * 1000.0)
            print("time taken to perform forwards&backwards prop: %d ms" % (accum_prop_time/float(step)))
            print("Epoch %d, Batch %d (of %d), samples %d" % (int(step / len(datareader.img_ids)),
                                                              step, tot_steps, len(batch)))

            if input_args.debug_info >= 2:
                print("\nmax, min, shape for masks_pred:")
                print(np.max(pred_seg), np.min(pred_seg), pred_seg.shape)
                print("max, min, shape for edges_pred:")
                print(np.max(pred_edge), np.min(pred_edge), pred_edge.shape)
                print("max, min, shape for coverage_pred:")
                print(np.max(coverage), np.min(coverage), coverage.shape)
                print("max, min, shape for bbox_pred:")
                print(np.max(bbox_regressor), np.min(bbox_regressor), bbox_regressor.shape)
                print("\n")

                print("========================== GT and Images ==========================")
                print(gt_mask.shape, gt_edge.shape)
                print(np.max(gt_mask), np.min(gt_mask))
                print(np.max(gt_edge), np.min(gt_edge))

                print("========================== Segment and Edges ==========================")
                print(seg_upscores.shape, edge_upscores.shape)
                print(np.max(seg_upscores), np.min(seg_upscores))
                print(np.max(edge_upscores), np.min(edge_upscores))
                print("\n")



            # if lossseg > 1000.0:
            #     print("========================== GT and Images ==========================")
            #     print(gt_mask.shape, gt_edge.shape)
            #     print(np.max(gt_mask), np.min(gt_mask))
            #     print(np.max(gt_edge), np.min(gt_edge))
            #
            #     print("========================== Segment and Edges ==========================")
            #     print(seg_upscores.shape, edge_upscores.shape)
            #     print(np.max(seg_upscores), np.min(seg_upscores))
            #     print(np.max(edge_upscores), np.min(edge_upscores))
            #
            #     for tt in range(input_args.batch_size):
            #         factor = 255/gt_mask.shape[3]
            #         arr = np.sum(gt_mask[tt, :, :, 1:gt_mask.shape[3]], axis=2)*factor
            #         print(np.max(arr), np.min(arr))
            #
            #         todump = Image.fromarray(np.array(arr).astype(np.uint8))
            #         todump.save("crap_shit/segmentation_weird_"+str(step)+"_"+str(tt)+".jpg")
            #         todump = Image.fromarray(transformed_data[tt], "RGB")
            #         todump.save("crap_shit/original_image_problem_" + str(step) + "_" + str(tt) + ".jpg")
            # else:
            #     print("========================== GT and Images ==========================")
            #     print(gt_mask.shape, gt_edge.shape)
            #     print(np.max(gt_mask), np.min(gt_mask))
            #     print(np.max(gt_edge), np.min(gt_edge))
            #
            #     print("========================== Segment and Edges ==========================")
            #     print(seg_upscores.shape, edge_upscores.shape)
            #     print(np.max(seg_upscores), np.min(seg_upscores))
            #     print(np.max(edge_upscores), np.min(edge_upscores))
            #
            #     if step % 50 == 0:
            #         factor = 255 / gt_mask.shape[3]
            #         arr = np.sum(gt_mask[tt, :, :, 1:gt_mask.shape[3]], axis=2)*factor
            #         print(np.max(arr), np.min(arr))
            #
            #         todump = Image.fromarray(np.array(arr).astype(np.uint8))
            #         todump.save("crap_shit/segmentation_correct_" + str(step) + "_" + str(tt) + ".jpg")
            #         todump = Image.fromarray(transformed_data[tt], "RGB")
            #         todump.save("crap_shit/original_image_correct_" + str(step) + "_" + str(tt) + ".jpg")

            print("----> step %d\n"
                  "\tloss_edge: %.7f\n"
                  "\tloss_seg: %.7f\n"
                  "\tloss_cov: %.7f\n"
                  "\tloss_bbox: %.7f\n"
                  "\tIoU: %.7f\n"
                  "\tMAP: %.7f\n"
                  "\tloss_all: %.7f\n\n" % (step, lossedge, lossseg, losscov, lossbbox, IoU, MAP, lossall))

            if config.ENABLE_VALIDATION:
                if step % 50 == 0:
                    self.validate_model(session=sess, train_batches=train_batches, datareader=datareader)
            else:
                if step % 200 == 0:
                    self.save(sess=sess, model_saver=model_saver, global_step=step, input_args=input_args)

            if step % 10000 == 0:
                break


        print("Model has been trained, check path: %s", input_args.model_dir)

    def validate_model(self, session, train_batches, datareader):
        # validation performed with the training data just to make sure if we are predicting the BBoxes,
        # coverage, segmentations, and edges correctly, we don't care of the order, we just do it randomly
        # since we want to make sure the model accuracy is as good as it is showing on tensorboard
        for retrieve in range(20):
            batch_get = time.time()
            # ===================================================================================
            batch = train_batches.next()

            # cityscapes
            classes, imgs, edges, masks, coverages, bboxes, covblock, sizeblock, objblock, bboxes_4 = \
                datareader.preprocess_batch(batch)

            gt_edge = np.array(edges)
            gt_edge = np.expand_dims(gt_edge, axis=3)
            gt_mask = np.array(masks)
            # gt_mask = np.expand_dims(gt_mask, axis=3)
            gt_coverage = np.array(coverages).transpose(0, 3, 2, 1)
            gt_bbox = np.array(bboxes).transpose(0, 3, 2, 1)
            gt_covblock = np.array(covblock).transpose(0, 3, 2, 1)
            gt_sizeblock = np.array(sizeblock).transpose(0, 3, 2, 1)
            gt_objblock = np.array(objblock).transpose(0, 3, 2, 1)
            transformed_data = imgs  # - config.mean_value
            # ===================================================================================
            accum_retrieve_time = ((time.time() - batch_get) * 1000.0)
            print("Average time taken to retrieve the batch for validation was: %4.4f ms" % (accum_retrieve_time))

            start_time = time.time()

            feat = session.run([self.feats], feed_dict={self.pinputimage: transformed_data})
            lossedge, lossseg, losscov, lossbbox, lossall, _, pred_seg, pred_edge, pred_coverage, pred_bbox \
                , seg_upscores, edge_upscores, IoU, MAP = session.run(
                [self.loss_edge, self.loss_seg, self.loss_cov, self.loss_bbox, self.lossall,
                 self.train_op, self.pred_seg, self.pred_edge, self.coverage, self.bbox_regressor,
                 self.seg_upscores, self.edge_upscores, self.IoU, self.mAP],
                feed_dict={self.input_feat: feat[0],
                           self.pgtedge: gt_edge, self.pgtmask: gt_mask, self.pgtcoverage: gt_coverage,
                           self.pgtbbox: gt_bbox,
                           self.pgtcovblock: gt_covblock, self.pgtsizeblock: gt_sizeblock,
                           self.pgtobjblock: gt_objblock, self.transformed_input: transformed_data})

            pred_edge = np.squeeze(pred_edge, axis=[0, 1])
            pred_seg = np.squeeze(pred_seg, axis=[0, 1])

            pred_edge = pred_edge + pred_seg
            pred_seg = pred_edge + pred_seg

            elapsed_time = time.time() - start_time
            print('The elapsed time per image prediction was: %f ms' % (elapsed_time*1000.))

            print("Segmentation shape --------------------")
            print(pred_seg.shape)
            print("Max: %f, Min: %f" % (np.max(pred_seg), np.min(pred_seg)))
            show_res_validation(imgs[0], pred_seg, title="Segmentation-pred_"+str(retrieve))

            print("Edges shape --------------------")
            print(pred_edge.shape)
            print("Max: %f, Min: %f" % (np.max(pred_edge), np.min(pred_edge)))
            show_res_validation(imgs[0], pred_edge, title="Edges-pred_"+str(retrieve))

            # ======> For BBox
            fig, ax = plt.subplots(1)
            ax.imshow(imgs[0])
            ax = plt.gca()
            ax.set_autoscale_on(False)

            itext = 0
            for i in xrange(len(config.classes_used)):
                cover = pred_coverage[0, :, :, i:i + 1]
                mask = (cover > 0.4)
                coord = np.where(mask == 1)

                # print("Mask coverage --------------------")
                # print(mask.shape)
                # print("Max: %f, Min: %f" % (np.max(mask), np.min(mask)))
                # toplot = Image.fromarray((mask[:, :, 0]*255), "L")
                # toplot.save("Coverage_Map_"+str(i)+".jpeg")



                print(coord)

                y = np.asarray(coord[0])
                x = np.asarray(coord[1])
                mx = x * config.stride  # (pred_seg.shape[1]/config.stride) #
                my = y * config.stride  # (pred_seg.shape[0]/config.stride) #

                print(y, x, mx, my)
                # sys.exit(0)

                boxs = pred_bbox[0]
                # print("=============== BBOX ================")
                # print(boxs)
                # print(boxs.shape)
                # print("=====================================")

                x1 = (np.asarray([boxs[y[j], x[j], 0] for j in xrange(x.size)]) + mx - config.stride / 16)
                y1 = (np.asarray([boxs[y[j], x[j], 1] for j in xrange(x.size)]) + my - config.stride / 16)
                x2 = (np.asarray([boxs[y[j], x[j], 2] for j in xrange(x.size)]) + mx + config.stride / 16)
                y2 = (np.asarray([boxs[y[j], x[j], 3] for j in xrange(x.size)]) + my + config.stride / 16)

                boxes = np.transpose(np.vstack((x1, y1, x2, y2)))

                # print("%%%%%%%%%%%%%%%% BBOX %%%%%%%%%%%%%%%%")
                # print(boxes)
                # print(boxes.shape)
                # print("%%%%%%%%%%%%%%%% BBOX %%%%%%%%%%%%%%%%\n\n")

                detections_per_image = []
                print('********************* Clustering Boxes *********************')
                boxes1 = np.zeros([1, boxes.shape[0], 5])
                show_all = True
                if boxes.any():
                    if not show_all:
                        nboxes, weights = cv2.groupRectangles(np.array(boxes).tolist(), 0, 0.3)
                        nboxes = (self.non_max_suppression_fast(np.array(boxes), 0.1))

                        # print("//////////////////////////////////////////////////////////////////")
                        # print(nboxes, weights)
                        # print("//////////////////////////////////////////////////////////////////")

                        if len(nboxes):
                            for rect, weight in zip(nboxes, weights):
                                temp_bbox = []
                                confidence = weight[0]  # math.log()
                                detection = [rect[0], rect[1], rect[2], rect[3], confidence]
                                detections_per_image.append(detection)

                                # print (rect[2] - rect[0], rect[3] - rect[1])
                                # if (rect[3] - rect[1]) >= config.min_height:
                                #     confidence = weight[0] #5.0 #math.log()
                                #     detection = [rect[0], rect[1], rect[2], rect[3], confidence]
                                #     temp_bbox.append(detection)
                                #
                                # for tt in range(boxes.shape[0]):
                                #     if math.sqrt((boxes[tt, 0]-rect[0])*(boxes[tt, 0]-rect[0]) + (boxes[tt, 1]-rect[1])*(boxes[tt, 1]-rect[1])) <= 10.0:
                                #         detection = [boxes[tt, 0], boxes[tt, 1], boxes[tt, 2], boxes[tt, 3], confidence]
                                #         temp_bbox.append(detection)
                                #
                                # print(temp_bbox)
                                #
                                # if len(temp_bbox):
                                #     temp_bbox_xx = np.zeros([len(temp_bbox), 5])
                                #     for ff in range(len(temp_bbox)):
                                #         temp_bbox_xx[ff, 0] = temp_bbox[ff][0]
                                #         temp_bbox_xx[ff, 1] = temp_bbox[ff][1]
                                #         temp_bbox_xx[ff, 2] = temp_bbox[ff][2]
                                #         temp_bbox_xx[ff, 3] = temp_bbox[ff][3]
                                #
                                #     print([min(temp_bbox_xx[:, 0]), min(temp_bbox_xx[:, 1]),
                                #                  max(temp_bbox_xx[:, 2]), max(temp_bbox_xx[:, 3]), confidence])
                                #     detection = [min(temp_bbox_xx[:, 0]), min(temp_bbox_xx[:, 1]),
                                #                  max(temp_bbox_xx[:, 2]), max(temp_bbox_xx[:, 3]), confidence]
                                #     detections_per_image.append(detection)
                    else:
                        for jj in range(boxes.shape[0]):
                            detection = [boxes[jj, 0], boxes[jj, 1], boxes[jj, 2], boxes[jj, 3], 1.0]
                            detections_per_image.append(detection)
                print('********************* Clustering Boxes *********************\n\n')

                boxes_cur_image = np.asarray(detections_per_image, dtype=np.float32)

                # print("--------------------- Boxes left ---------------------------")
                # print(boxes_cur_image)
                # print("--------------------- Boxes left ---------------------------\n\n")

                if (boxes_cur_image.shape[0] != 0):
                    [r, c] = boxes_cur_image.shape
                    boxes1[0, 0:r, 0:c] = boxes_cur_image

                for idx, bb in enumerate(boxes1[0]):
                    if bb[4] > 0.0:  # make sure we cover something
                        print(bb[4])
                        rect = patches.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], linewidth=1,
                                                 edgecolor='r',
                                                 facecolor='none')
                        # ax.add_patch(rect)


                        ax.add_artist(rect)
                        rx, ry = rect.get_xy()
                        cx = rx + rect.get_width() / 2.0
                        cy = ry + rect.get_height() / 2.0

                        ax.annotate((config.class_name[config.classes_used[i]] + ":" + str(bb[4])), (cx, cy), color='w',
                                    weight='bold',
                                    fontsize=6, ha='center', va='center')

                        # ax.annotate(config.class_name[i] + ":" + str(bb[4]), (itext, itext * 16), color='g')
                        itext += 1

            plt.title("BBox-pred")
            plt.savefig("validation_results/BBox-pred_"+str(retrieve)+'.jpg')
            plt.close()

            # =======> for Coverage
            cover = np.mean(pred_coverage[0], axis=2)
            cover = np.expand_dims(cover, axis=2)
            plt.figure()
            plt.axis('off')
            plt.imshow(np.squeeze(cover, axis=2))
            plt.title("Coverage-pred")
            plt.savefig("validation_results/Coverage-pred_"+str(retrieve)+'.jpg')
            plt.close()






    # Malisiewicz et al.
    def non_max_suppression_fast(self, boxes, overlapThresh):
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])/1.5
            yy1 = np.maximum(y1[i], y1[idxs[:last]])/1.5
            xx2 = np.minimum(x2[i], x2[idxs[:last]])/1.5
            yy2 = np.minimum(y2[i], y2[idxs[:last]])/1.5

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        # return only the bounding boxes that were picked using the
        # integer data type
        return boxes[pick].astype("int")

    def test(self, sess, testdata, input_args, model_saver=None, save_at=None):
        if input_args.debug_info >= 1:
            print("Testing the model ...")

        # opening file with opencv and remember this is BGR data order not RGB
        image_path = testdata
        img = cv2.imread(image_path)
        img1 = imresize(img, (config.image_size_y, config.image_size_x), interp='lanczos')
        data = img1#.astype(np.float)
        transformed_data = data #- config.mean_value
        transformed_data = cv2.cvtColor(transformed_data, cv2.COLOR_BGR2RGB)

        start_time = time.time()
        feat = sess.run([self.feats], feed_dict={
            self.pinputimage: np.reshape(transformed_data, [1, config.image_size_y, config.image_size_x, 3])})

        pred_edge, pred_seg, pred_cov, pred_bbox = sess.run(
            [self.pred_edge, self.pred_seg, self.coverage, self.bbox_regressor],

            # config.SEMANTIC_CLASSES
            feed_dict={self.input_feat: feat[0],
                       self.pgtedge: np.zeros((1, config.image_size_y, config.image_size_x, 1)),
                       self.pgtmask: np.zeros((1, config.image_size_y, config.image_size_x, 1)),
                       self.pgtcoverage: np.zeros((1, int(config.image_size_x/config.stride),
                                                   int(config.image_size_y/config.stride), config.TOTAL_CLASSES))})
        pred_edge = np.squeeze(pred_edge, axis=[0, 1])
        pred_seg = np.squeeze(pred_seg, axis=[0, 1])

        pred_edge = pred_edge + pred_seg
        pred_seg = pred_edge + pred_seg

        elapsed_time = time.time() - start_time
        print('The elapsed time per image:', elapsed_time)

        # fig, ax = plt.subplots(1)
        # ax.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        # ax = plt.gca()
        # ax.set_autoscale_on(False)

        print(np.array(pred_cov).shape)
        print(np.array(pred_bbox).shape)

        # for i in xrange(len(config.classes_used)):
        #     data0 = pred_cov[:, :, :, i:i + 1]
        #     data1 = pred_bbox[:, :, :, i:i + 4]
        #     # print data0
        #     bbox, mask = cluster(data0, data1)
        #     mask = np.array(mask).astype(int)
        #     print("class: %d"%i)
        #     # print(data1)
        #     # print(data0)
        #     # print bbox
        #     # for bb in bbox[0]:
        #     #    rect = patches.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], linewidth=1, edgecolor='r',
        #     #                             facecolor='none')
        #     #    ax.add_patch(rect)
        #
        #
        #     # for idx, bb in enumerate(bboxes_4[0]):
        #     #    rect = patches.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], linewidth=1, edgecolor='r', facecolor='none')
        #     #    ax.add_patch(rect)
        #
        #     for k in range(0, int(config.image_size_y / config.stride)):
        #         for j in range(0, int(config.image_size_x / config.stride)):
        #             if np.sum(mask[0]) != 0:
        #                 rect = patches.Rectangle((k * config.stride, j * config.stride), config.stride, config.stride,
        #                                          linewidth=3, edgecolor='g',
        #                                          facecolor='none')
        #                 # else:
        #                 #    rect = patches.Rectangle((k * config.stride, j * config.stride), config.stride, config.stride,
        #                 #                             linewidth=1, edgecolor='r',
        #                 #                             facecolor='none')
        #                 ax.add_patch(rect)
        #
        # plt.title("Gridbox-pred")
        # plt.show()

        # plt.figure()
        # plt.axis('off')
        # plt.imshow(pred_seg)
        # plt.title("PureSeg-pred")
        # plt.show()

        print("Segmentation shape --------------------")
        print(pred_seg.shape)
        print("Max: %f, Min: %f"%(np.max(pred_seg), np.min(pred_seg)))
        if save_at is None:
            show_res(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), pred_seg, title="Segmentation-pred")

        print("Edges shape --------------------")
        print(pred_edge.shape)
        print("Max: %f, Min: %f" % (np.max(pred_edge), np.min(pred_edge)))
        if save_at is None:
            show_res(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), pred_edge, title="Edges-pred")

        # ======> For BBox
        fig, ax = plt.subplots(1)
        ax.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        ax = plt.gca()
        ax.set_autoscale_on(False)


        itext = 0
        for i in xrange(len(config.classes_used)):
            cover = pred_cov[0, :, :, i:i + 1]

            print("===============>>>>>>>", np.max(cover))
            # sys.exit(0)

            mask = (cover > 0.5)
            coord = np.where(mask == 1)

            # print("Mask coverage --------------------")
            # print(mask.shape)
            # print("Max: %f, Min: %f" % (np.max(mask), np.min(mask)))
            # toplot = Image.fromarray((mask[:, :, 0]*255), "L")
            # toplot.save("Coverage_Map_"+str(i)+".jpeg")



            print(coord)

            y = np.asarray(coord[0])
            x = np.asarray(coord[1])
            mx = x * config.stride # (pred_seg.shape[1]/config.stride) #
            my = y * config.stride # (pred_seg.shape[0]/config.stride) #

            print(y, x, mx, my)
            # sys.exit(0)

            boxs = pred_bbox[0]
            # print("=============== BBOX ================")
            # print(boxs)
            # print(boxs.shape)
            # print("=====================================")

            x1 = (np.asarray([boxs[y[j], x[j], 0] for j in xrange(x.size)]) + mx - config.stride/4)
            y1 = (np.asarray([boxs[y[j], x[j], 1] for j in xrange(x.size)]) + my - config.stride/4)
            x2 = (np.asarray([boxs[y[j], x[j], 2] for j in xrange(x.size)]) + mx + config.stride/4)
            y2 = (np.asarray([boxs[y[j], x[j], 3] for j in xrange(x.size)]) + my + config.stride/4)

            boxes = np.transpose(np.vstack((x1, y1, x2, y2)))

            print("%%%%%%%%%%%%%%%% BBOX %%%%%%%%%%%%%%%%")
            print(boxes)
            print(boxes.shape)
            print("%%%%%%%%%%%%%%%% BBOX %%%%%%%%%%%%%%%%\n\n")

            detections_per_image = []
            print('********************* Clustering Boxes *********************')
            boxes1 = np.zeros([1, boxes.shape[0], 5])
            show_all = False
            if boxes.any():
                if not show_all:
                    nboxes, weights = cv2.groupRectangles(np.array(boxes).tolist(), 0, 0.3)
                    nboxes = (self.non_max_suppression_fast(np.array(boxes), 0.3))

                    print("//////////////////////////////////////////////////////////////////")
                    print(nboxes, weights)
                    print("//////////////////////////////////////////////////////////////////")

                    if len(nboxes):
                        for rect, weight in zip(nboxes, weights):
                            temp_bbox = []
                            confidence = weight[0] #math.log()
                            detection = [rect[0], rect[1], rect[2], rect[3], confidence]
                            detections_per_image.append(detection)

                            # print (rect[2] - rect[0], rect[3] - rect[1])
                            # if (rect[3] - rect[1]) >= config.min_height:
                            #     confidence = weight[0] #5.0 #math.log()
                            #     detection = [rect[0], rect[1], rect[2], rect[3], confidence]
                            #     temp_bbox.append(detection)
                            #
                            # for tt in range(boxes.shape[0]):
                            #     if math.sqrt((boxes[tt, 0]-rect[0])*(boxes[tt, 0]-rect[0]) + (boxes[tt, 1]-rect[1])*(boxes[tt, 1]-rect[1])) <= 10.0:
                            #         detection = [boxes[tt, 0], boxes[tt, 1], boxes[tt, 2], boxes[tt, 3], confidence]
                            #         temp_bbox.append(detection)
                            #
                            # print(temp_bbox)
                            #
                            # if len(temp_bbox):
                            #     temp_bbox_xx = np.zeros([len(temp_bbox), 5])
                            #     for ff in range(len(temp_bbox)):
                            #         temp_bbox_xx[ff, 0] = temp_bbox[ff][0]
                            #         temp_bbox_xx[ff, 1] = temp_bbox[ff][1]
                            #         temp_bbox_xx[ff, 2] = temp_bbox[ff][2]
                            #         temp_bbox_xx[ff, 3] = temp_bbox[ff][3]
                            #
                            #     print([min(temp_bbox_xx[:, 0]), min(temp_bbox_xx[:, 1]),
                            #                  max(temp_bbox_xx[:, 2]), max(temp_bbox_xx[:, 3]), confidence])
                            #     detection = [min(temp_bbox_xx[:, 0]), min(temp_bbox_xx[:, 1]),
                            #                  max(temp_bbox_xx[:, 2]), max(temp_bbox_xx[:, 3]), confidence]
                            #     detections_per_image.append(detection)
                else:
                    for jj in range(boxes.shape[0]):
                        detection = [boxes[jj, 0], boxes[jj, 1], boxes[jj, 2], boxes[jj, 3], 1.0]
                        detections_per_image.append(detection)
            print('********************* Clustering Boxes *********************\n\n')


            boxes_cur_image = np.asarray(detections_per_image, dtype=np.float32)

            print("--------------------- Boxes left ---------------------------")
            print(boxes_cur_image)
            print("--------------------- Boxes left ---------------------------\n\n")



            if (boxes_cur_image.shape[0] != 0):
                [r, c] = boxes_cur_image.shape
                boxes1[0, 0:r, 0:c] = boxes_cur_image

            for idx, bb in enumerate(boxes1[0]):
                if bb[4] > 0.0:  # make sure we cover something
                    print("-----------", bb[4])
                    print("-----------", config.cls2clr[config.class_name[config.classes_used[i]]])
                    rect = patches.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], linewidth=1,
                                             edgecolor=config.cls2clr[config.class_name[config.classes_used[i]]],
                                             facecolor='none')
                    # ax.add_patch(rect)


                    ax.add_artist(rect)
                    rx, ry = rect.get_xy()
                    cx = rx + rect.get_width() / 2.0
                    cy = ry + rect.get_height() / 2.0

                    ax.annotate((config.class_name[config.classes_used[i]]+":"+str(bb[4])), (cx, cy), color='w', weight='bold',
                                fontsize=6, ha='center', va='center')

                    # ax.annotate(config.class_name[i] + ":" + str(bb[4]), (itext, itext * 16), color='g')
                    itext += 1

        plt.title("BBox-pred")
        if save_at is not None:
            plt.savefig(save_at+"_bboxes"+".jpg")
        else:
            plt.show()


        # =======> for Coverage
        cover = np.mean(pred_cov[0], axis=2)
        cover = np.expand_dims(cover, axis=2)
        plt.figure()
        plt.axis('off')
        plt.imshow(np.squeeze(cover, axis=2))
        plt.title("Coverage-pred")
        if save_at is not None:
            plt.savefig(save_at+"_coverage"+".jpg")
        else:
            plt.show()


    def save(self, sess, input_args, global_step, model_saver=None):
        """ Save the trained model. """
        if input_args.debug_info >= 1:
            print("Saving model to %s" %input_args.model_dir)
        model_saver.save(sess=sess, save_path=input_args.model_dir, global_step=global_step)

    def load(self, sess, input_args, model_saver=None):
        """ Load the trained model. """
        if input_args.debug_info >= 1:
            print("Loading model...")
        checkpoint = tf.train.get_checkpoint_state(input_args.pretrained_model_path)
        if checkpoint is None:
            print("Error: No saved model found. Please train first.")
            sys.exit(0)
        model_saver.restore(sess, checkpoint.model_checkpoint_path)

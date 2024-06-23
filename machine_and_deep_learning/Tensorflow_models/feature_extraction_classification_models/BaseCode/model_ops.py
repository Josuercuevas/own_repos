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

from __future__ import division
import tensorflow as tf
import numpy as np
from core_ops import *
import sys, os
from PIL import Image
import time
import h5py
import pygraphviz as pgv
from graphviz import Digraph

sys.path.append('../')
from configs import common_configs, resnet_configs, squeeznet_configs, helnet_configs, shufflenet_configs
import random
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

USE_H5PY = False
if USE_H5PY:
    from DataReader.database_manager import *
else:
    from BodyNet_module.DataReader.image_handler import *

from configs import common_configs, resnet_configs, squeeznet_configs

class BaseOps(object):
    def build(self, input_args, db_size=None):
        # storing the information
        self.input_args = input_args
        self.execution_mode = input_args.execution_mode
        self.db_size = db_size
        self.debug_level = input_args.debug_info
        self.model_name = input_args.model_name
        self.solver = input_args.solver
        self.max_epochs = input_args.max_epochs
        self.enable_tensorboard = input_args.enable_tensorboard
        self.model_dir = input_args.model_dir

        if self.debug_level >= 1:
            print("Building class BaseOps for training/testing/feature_extraction")

    """
        Training module to be used when using any input dataset, we need to specify the
        reader and summary mergers. We are assuming the data is in HD5 format
    """
    def train(self, sess=None, datareader=None, summary_merger=None, summary_writer=None, model_saver=None):
        if self.debug_level >= 1:
            print("Training model ...")

            if self.debug_level >= 2:
                if summary_merger is not None:
                    print(summary_merger)

            if self.debug_level >= 4:
                if datareader is not None:
                    print(datareader)

            if self.debug_level >= 2:
                if sess is not None:
                    print(sess)

            if self.debug_level >= 2:
                # check they really exist
                print(self.istrain)
                print(self.pinputimage)
                print(self.keepprob)
                print(self.predictions)
                print(self.loss)
                print(self.softmax_pred)

                if self.solver == 'moment':
                    print(self.learning_rate)

                print("========================== TRAINER OPTIMIZER =========================")
                print(self.optimizer)

        if datareader is None:
            print("No Data reader or fetcher, we cannot train, quitting this application")
            return

        item_dict = datareader
        item = []
        for attr_name in datareader:
            if self.debug_level >= 2:
                print("Retrieving list in dictionary:")
                print(attr_name)
            item.append(item_dict[attr_name])

        if USE_H5PY:
            train_x = item_dict['train_x']
            train_y = item_dict['train_y']
            validation_x = item_dict['validation_x']
            validation_y = item_dict['validation_y']
            test_x = item_dict['test_x']
            test_y = item_dict['test_y']
            train_size = item_dict['train_size']

            if self.debug_level >= 2:
                print("Database contains %d items, training for %d epochs" % (train_size, self.max_epochs))
        else:
            train_size = item_dict['train_size']

        # start training routines for DSD approach
        # 8888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888
        # Dense Training D
        if self.model_name == 'squeeznet':
            print("\n\n\n\n============== Training Dense Part of the model with Learning rate = %f"
                  % squeeznet_configs.ADAM_START_Lr)
        elif self.model_name == 'helnet':
            print("\n\n\n\n============== Training Dense Part of the model with Learning rate = %f"
                  % helnet_configs.ADAM_START_Lr)
        elif self.model_name == 'resnet':
            print("\n\n\n\n============== Training Dense Part of the model with Learning rate = %f"
                  % resnet_configs.ADAM_START_Lr)
        elif self.model_name == 'shufflenet':
            print("\n\n\n\n============== Training Dense Part of the model with Learning rate = %f"
                  % shufflenet_configs.ADAM_START_Lr)

        counter_int = 0
        run_id = 0  # since here e are not training DSD we dont care
        steps = int(train_size // common_configs.BATCH_SIZE)
        offset_meta = steps * self.max_epochs * run_id
        print("run_id: %d, steps: %d, offset: %d" % (run_id, steps, offset_meta))

        if USE_H5PY:
            self.train_dense_stage(train_x=train_x, stage_name=self.model_name + "_D.ckpt", train_y=train_y,
                                   train_size=train_size, steps=steps, offset_meta=offset_meta,
                                   validation_x=validation_x, validation_y=validation_y, counter_int=counter_int,
                                   reduce_lr=False, summary_merger=summary_merger, session=sess, summary_writer=summary_writer,
                                   model_saver=model_saver)
        else:
            self.train_dense_stage(train_x=None, stage_name=self.model_name + "_D.ckpt", train_y=None,
                                   train_size=None, steps=steps, offset_meta=offset_meta, validation_x=None,
                                   validation_y=None, counter_int=counter_int, reduce_lr=False,
                                   summary_merger=summary_merger, session=sess, summary_writer=summary_writer,
                                   model_saver=model_saver, datareader=item_dict)
        # 8888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888

        if common_configs.ENABLE_DSD_TRAINING:
            # 8888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888
            # Sparse Training DS
            if self.model_name == 'squeeznet':
                print("\n\n\n\n============== Training Sparse Part of the model with Learning rate = %f"
                      % squeeznet_configs.ADAM_START_Lr)
            elif self.model_name == 'helnet':
                print("\n\n\n\n============== Training Sparse Part of the model with Learning rate = %f"
                      % helnet_configs.ADAM_START_Lr)
            elif self.model_name == 'resnet':
                print("\n\n\n\n============== Training Sparse Part of the model with Learning rate = %f"
                      % resnet_configs.ADAM_START_Lr)
            elif self.model_name == 'shufflenet':
                print("\n\n\n\n============== Training Sparse Part of the model with Learning rate = %f"
                      % shufflenet_configs.ADAM_START_Lr)

            counter_int = 0
            run_id = 1  # since here e are not training DSD we dont care
            steps = int(train_size // common_configs.BATCH_SIZE)
            offset_meta = steps * self.max_epochs * run_id
            print("run_id: %d, steps: %d, offset: %d" % (run_id, steps, offset_meta))

            if USE_H5PY:
                self.train_sparse_stage(train_x=train_x, stage_name=self.model_name + "_DS.ckpt", train_y=train_y,
                                        train_size=train_size, steps=steps, offset_meta=offset_meta,
                                        validation_x=validation_x, validation_y=validation_y, counter_int=counter_int,
                                        reduce_lr=False, summary_merger=summary_merger, session=sess,
                                        summary_writer=summary_writer, model_saver=model_saver)
            else:
                self.train_sparse_stage(train_x=None, stage_name=self.model_name + "_DS.ckpt", train_y=None,
                                        train_size=None, steps=steps, offset_meta=offset_meta, validation_x=None,
                                        validation_y=None, counter_int=counter_int, reduce_lr=False,
                                        summary_merger=summary_merger, session=sess, summary_writer=summary_writer,
                                        model_saver=model_saver, datareader=item_dict)
            # 8888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888=

            # 8888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888
            # Dense Training DSD
            if self.model_name == 'squeeznet':
                print("\n\n\n\n============== Training Dense Part of the model with Learning rate = %f"
                      % (squeeznet_configs.ADAM_START_Lr / 10.0))
            elif self.model_name == 'helnet':
                print("\n\n\n\n============== Training Dense Part of the model with Learning rate = %f"
                      % (helnet_configs.ADAM_START_Lr / 10.0))
            elif self.model_name == 'resnet':
                print("\n\n\n\n============== Training Dense Part of the model with Learning rate = %f"
                      % (resnet_configs.ADAM_START_Lr / 10.0))
            elif self.model_name == 'shufflenet':
                print("\n\n\n\n============== Training Dense Part of the model with Learning rate = %f"
                      % (shufflenet_configs.ADAM_START_Lr / 10.0))

            counter_int = 0
            run_id = 2  # since here e are not training DSD we dont care
            steps = int(train_size // common_configs.BATCH_SIZE)
            offset_meta = steps * self.max_epochs * run_id
            print("run_id: %d, steps: %d, offset: %d" % (run_id, steps, offset_meta))

            if USE_H5PY:
                self.train_dense_stage(train_x=train_x, stage_name=self.model_name+"_DSD.ckpt", train_y=train_y,
                                       train_size=train_size, steps=steps, offset_meta=offset_meta,
                                       validation_x=validation_x, validation_y=validation_y, counter_int=counter_int,
                                       reduce_lr=True, summary_merger=summary_merger, session=sess,
                                       summary_writer=summary_writer, model_saver=model_saver)
            else:
                self.train_dense_stage(train_x=None, stage_name=self.model_name + "_DSD.ckpt", train_y=None,
                                       train_size=None, steps=steps, offset_meta=offset_meta, validation_x=None,
                                       validation_y=None, counter_int=counter_int, reduce_lr=True,
                                       summary_merger=summary_merger, session=sess, summary_writer=summary_writer,
                                       model_saver=model_saver, datareader=item_dict)
            # 8888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888

        return

    """
        Testing routine for the bodynet models trained in previous steps, we can use it for
        model evaluation during training if necessary
    """
    def test(self, test_x=None, test_y=None, data_reader=None, epoch=0, loss=0.0):

        if test_x is not None and test_y is not None:
            if self.debug_level >= 1:
                print("Testing the model (%s) with h5py backend..." % self.model_name)

            '''
                =============================
                Testing part to be performed in input dataset
                =============================
            '''
            # Finished a batch, now testing the validation set
            # 1.0 dropout when doing validation
            tot = 0.0
            j = 0
            while ((j + 1) * common_configs.BATCH_SIZE < test_x.shape[0]):
                test_batch_data = test_x[j * common_configs.BATCH_SIZE:(j * common_configs.BATCH_SIZE) +
                                                                      common_configs.BATCH_SIZE]
                test_batch_labels = test_y[j * common_configs.BATCH_SIZE:(j * common_configs.BATCH_SIZE) +
                                                                        common_configs.BATCH_SIZE]

                # normalizing samples
                test_batch_data = self.normalize_samples(images=test_batch_data)

                last_pred = self.softmax_pred.eval(
                    feed_dict={
                        self.pinputimage: test_batch_data,  # input image
                        self.true_labels: test_batch_labels,  # input labels
                        self.istrain: False,  # flag for training
                        self.keepprob: 1.0,  # dropout probability
                        self.reduce_lr: False
                    }
                )
                errval, accuracy1, recall, confusions = common_configs.evaluation(last_pred, test_batch_labels, self.input_args)
                tot = tot + errval
                j += 1

            if ((j * common_configs.BATCH_SIZE) < test_x.shape[0]):
                # evaluation of remaining part of the set, may not be a multiple of the batch_size
                test_batch_data = test_x[(j) * common_configs.BATCH_SIZE:]
                test_batch_labels = test_y[(j) * common_configs.BATCH_SIZE:]

                # normalizing samples
                test_batch_data = self.normalize_samples(images=test_batch_data)

                last_pred = self.softmax_pred.eval(
                    feed_dict={
                        self.pinputimage: test_batch_data,  # input image
                        self.true_labels: test_batch_labels,  # input labels
                        self.istrain: False,  # flag for training
                        self.keepprob: 1.0,  # dropout probability
                        self.reduce_lr: False
                    }
                )
                errval, accuracy1, recall, confusions = common_configs.evaluation(last_pred, test_batch_labels, self.input_args)
                j += 1

            # for visualization purposes
            tot = tot + errval
            print('Testing Acc (epoch-%d): %.5f, Loss: %.5f\n' % (epoch, (tot / j), loss))
        elif data_reader is not None:
            if self.debug_level >= 1:
                print("Testing the model (%s) with own data-handler backend..." % self.model_name)

            class_folders_paths_Test = data_reader['class_folders_paths_Test']
            number_of_samples_per_class_Test = data_reader['number_of_samples_per_class_Test']
            samples_filenames_Test = data_reader['samples_filenames_Test']
            test_size = data_reader['test_size']

            batches = int(test_size / common_configs.BATCH_SIZE)
            tot = 0.0
            j = 0
            for bath_id in range(batches):
                batch_X, batch_Y = get_images_batch(class_folders_paths=class_folders_paths_Test,
                                                    number_of_samples_per_class=number_of_samples_per_class_Test,
                                                    samples_filenames=samples_filenames_Test,
                                                    batch_size=common_configs.BATCH_SIZE, use_rgb=True,
                                                    normalize_samples=True)
                last_pred = self.softmax_pred.eval(
                    feed_dict={
                        self.pinputimage: batch_X,  # input image
                        self.true_labels: batch_Y,  # input labels
                        self.istrain: False,  # flag for training
                        self.keepprob: 1.0,  # dropout probability
                        self.reduce_lr: False
                    }
                )
                errval, accuracy1, recall, confusions = common_configs.evaluation(last_pred, batch_Y,
                                                                                  self.input_args)
                j += 1

                # for visualization purposes
                tot = tot + errval
            print('Testing Acc (epoch-%d): %.5f, Loss: %.5f\n' % (epoch, (tot / j), loss))
        else:
            print("ERROR: No dataset for testing has been provided, exiting ...")

        return

    """
        Feature extraction for visualization or other in depth model analysis
    """
    def extract_features(self, sess, input_args, datareader=None, dataset_x=None, dataset_y=None):
        if self.debug_level >= 1:
            print("Extracting features from dataset given by the user ... NOT YET")

        if data_reader is not None:
            print("Reading directly from Image files ...")

        return

    """
        Extracts the dot file in order to visualize the network and the operations in it
    """
    def write_dotfile(self, sess, graph, input_args):
        if input_args.debug_info >= 2:
            print("We will write a dot and png files to visualize the network ...")

        # dot file writer
        flowchart = pgv.AGraph()

        layers_name = []
        for node in graph.as_graph_def().node:
            if node.name.split('/')[-1] == 'AvgPool' or node.name.split('/')[-1] == 'Softmax' \
                    or node.name.split('/')[-1] == 'weights' or node.name.split('/')[-1] == 'biases' \
                    or node.name.split('/')[-1] == 'gamma' or node.name.split('/')[-1] == 'beta' \
                    or 'add' in node.name.split('/')[-1] or node.name.split('/')[-1] == 'moving_mean' \
                    or node.name.split('/')[-1] == 'moving_variance' or node.name.split('/')[-1] == 'input_image' \
                    or 'Relu' in node.name.split('/')[-1] or '_residual_layer_withConv' in node.name \
                    or '_residual_layer_noConv' in node.name or 'Pool' in node.name or 'pool' in node.name:

                if len(layers_name) > 0:
                    plyname = node.name.split('/')[0]
                    if plyname not in layers_name:
                        layers_name.append(plyname)
                else:
                    layers_name.append(node.name.split('/')[0])

        for nodeid in range(len(layers_name)):
            print("This node is [%d]: %s " % (nodeid, layers_name[nodeid]))

        for nodeid in range(len(layers_name) - 1):
            if input_args.model_name == 'resnet':
                # here we need to link the nodes correctly
                nodes_with_skips = [12, 44, 86, 148]

                if nodeid in nodes_with_skips:
                    flowchart.add_edge(layers_name[nodeid], layers_name[nodeid + 3])
                    flowchart.add_edge(layers_name[nodeid - 8], layers_name[nodeid + 1])
                    flowchart.add_edge(layers_name[nodeid + 1], layers_name[nodeid + 2])
                    flowchart.add_edge(layers_name[nodeid + 2], layers_name[nodeid + 3])
                else:
                    flowchart.add_edge(layers_name[nodeid], layers_name[nodeid + 1])
            elif input_args.model_name == 'squeeznet':
                fire_modules_nodes = [5, 12, 21, 28, 37, 44, 52, 59, 67, 74]
                single_connections_skip3 = [7, 14, 23, 30, 39, 46, 54, 61, 69, 76]

                if nodeid in fire_modules_nodes:
                    flowchart.add_edge(layers_name[nodeid], layers_name[nodeid + 1])
                    flowchart.add_edge(layers_name[nodeid], layers_name[nodeid + 3])
                elif nodeid in single_connections_skip3:
                    flowchart.add_edge(layers_name[nodeid], layers_name[nodeid + 3])
                else:
                    flowchart.add_edge(layers_name[nodeid], layers_name[nodeid + 1])
            elif input_args.model_name == 'helnet':
                flowchart.add_edge(layers_name[nodeid], layers_name[nodeid + 1])

        print(flowchart.string())  # print to screen
        print("Wrote bodynet_" + input_args.model_name + ".dot")
        flowchart.write("bodynet_" + input_args.model_name + ".dot")  # write to simple.dot

        towriteflowchart = pgv.AGraph('bodynet_' + input_args.model_name + '.dot')  # create a new graph from file
        towriteflowchart.layout()  # layout with default (neato)


        # dot = Digraph()
        # dot.format = 'png'
        #
        # for n in graph.as_graph_def().node:
        #     dot.node(n.name, label=n.name)
        #
        #     for i in n.input:
        #         dot.edge(i, n.name)
        #
        # dot.render("bodynet_" + input_args.model_name)


        os.system("dot -Tpng bodynet_" + input_args.model_name + ".dot -o bodynet_" + input_args.model_name + ".png")
        os.system("rm bodynet_" + input_args.model_name + ".dot")
        print("Wrote bodynet_" + input_args.model_name + ".png")

        return

    """
        Loading model from a particular folder or checkpoint
    """
    def load(self, sess, input_args, model_saver=None):
        """ Load the trained model. """
        if os.path.isdir(input_args.pretrained_weights):
            if self.debug_level >= 1:
                print("Loading PRETRAINED model from folder: %s" % input_args.pretrained_weights)
            checkpoint = tf.train.get_checkpoint_state(input_args.pretrained_weights)
            if checkpoint is None:
                print("Error: No saved model found. Please train first.")
                sys.exit(0)
            model_saver.restore(sess, checkpoint.model_checkpoint_path)
        else:
            if self.debug_level >= 1:
                print("Loading PRETRAINED model from file: %s" % input_args.pretrained_weights)
            model_saver.restore(sess, input_args.pretrained_weights)

        return



# ========================================================== DSD stages
    def normalize_samples(self, images):
        # we assume images are 8-bits RGB or BGR we dont care
        images = np.array(images).astype(np.float32) / 255.0
        return images

    def augment_samples(self, images, labels):
        # perform some data augmentations
        return images, labels

    def train_dense_stage(self, train_x, train_y, train_size, steps, offset_meta, validation_x, validation_y, stage_name,
                          counter_int, reduce_lr, summary_merger, session, summary_writer, model_saver, datareader=None):
        if reduce_lr:
            epochs = int(self.max_epochs / 10 + 1)
            print("changing epochs to ... %d", epochs)
        else:
            epochs = self.max_epochs

        if not USE_H5PY:
            # =========================== TRAINING
            class_folders_paths_Train = datareader['class_folders_paths_Train']
            number_of_samples_per_class_Train = datareader['number_of_samples_per_class_Train']
            samples_filenames_Train = datareader['samples_filenames_Train']
            train_size = datareader['train_size']
            # ===========================

            # =========================== VALIDATION
            class_folders_paths_Val = datareader['class_folders_paths_Val']
            number_of_samples_per_class_Val = datareader['number_of_samples_per_class_Val']
            samples_filenames_Val = datareader['samples_filenames_Val']
            val_size = datareader['val_size']
            # ===========================

        # # ===================== remove this in real production
        # if offset_meta == 0:
        #     epochs = 1 # just test
        # # ===================== remove this in real production


        for i in range(epochs):
            # permute the list of batches
            shuffled_step = range(steps)  # np.random.permutation(steps)#
            for step in shuffled_step:
                if USE_H5PY:
                    offset = (step * common_configs.BATCH_SIZE) % (train_size - common_configs.BATCH_SIZE)
                    batch_data = train_x[offset:(offset + common_configs.BATCH_SIZE), :, :, :]
                    batch_labels = train_y[offset:(offset + common_configs.BATCH_SIZE)]

                    # normalizing samples
                    batch_data = self.normalize_samples(images=batch_data)
                else:
                    offset = (step * common_configs.BATCH_SIZE) % (train_size - common_configs.BATCH_SIZE)
                    batch_data, batch_labels = get_images_batch(class_folders_paths=class_folders_paths_Train,
                                                                number_of_samples_per_class=number_of_samples_per_class_Train,
                                                                samples_filenames=samples_filenames_Train,
                                                                batch_size=common_configs.BATCH_SIZE, use_rgb=True,
                                                                normalize_samples=True)

                if common_configs.ENABLE_AUGMENTATION:
                    self.augment_samples(images=batch_data, labels=batch_labels)

                # 0.5 dropout probability in deep layers
                feed_dict1 = {
                    self.pinputimage: batch_data,  # input image
                    self.true_labels: batch_labels, # input labels
                    self.istrain: True, # flag for training
                    self.keepprob: 0.5, # dropout probability
                    self.reduce_lr: reduce_lr
                }

                if self.debug_level >= 3:
                    print("Feed Dictionary <Epoch: %d, Step: %d, Data_taken: (%d, %d), Max: %f, Min: %f>"
                          % (i, step, offset, (offset + common_configs.BATCH_SIZE), np.max(batch_data),
                             np.min(batch_data)))
                    print('Images with shape: ', batch_data.shape)
                    print('Labels with shape: ', batch_labels.shape)

                    if self.debug_level >= 4:
                        print(feed_dict1)

                if self.enable_tensorboard == 'true':
                    if summary_merger is None:
                        print("We cannot output to tensorboard if we dont have a file writer, exiting ...")
                        return

                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                    if (counter_int % 2000) == 0:
                        if self.solver != 'adam':
                            summary, _, l, predictions, train_acc = session.run(
                                [summary_merger, self.optimizer, self.loss, self.softmax_pred, self.training_accuracy],
                                feed_dict=feed_dict1, options=run_options, run_metadata=run_metadata)

                            summary_writer.add_run_metadata(run_metadata, 'step%03d' % (steps * i + step + offset_meta))
                            summary_writer.add_summary(summary, (steps * i + step + offset_meta))
                            summary_writer.flush()
                        else:
                            summary, _, l, predictions, train_acc = session.run(
                                [summary_merger, self.optimizer, self.loss, self.softmax_pred, self.training_accuracy],
                                feed_dict=feed_dict1, options=run_options, run_metadata=run_metadata)

                            summary_writer.add_run_metadata(run_metadata, 'step%03d' % (steps * i + step + offset_meta))
                            summary_writer.add_summary(summary, (steps * i + step + offset_meta))
                            summary_writer.flush()
                    else:
                        if self.solver != 'adam':
                            summary, _, l, lr, predictions, train_acc = session.run(
                                [summary_merger, self.optimizer, self.loss, self.softmax_pred, self.training_accuracy],
                                feed_dict=feed_dict1)
                        else:
                            summary, _, l, predictions, train_acc = session.run(
                                [summary_merger, self.optimizer, self.loss, self.softmax_pred, self.training_accuracy],
                                feed_dict=feed_dict1)
                else:
                    if self.solver != 'adam':
                        _, l, lr, predictions, train_acc = session.run(
                            [self.optimizer, self.loss, self.softmax_pred, self.training_accuracy],
                            feed_dict=feed_dict1)
                    else:
                        _, l, predictions, train_acc = session.run(
                            [self.optimizer, self.loss, self.softmax_pred,  self.training_accuracy],
                            feed_dict=feed_dict1)

                assert not np.isnan(l), 'Model diverged with loss = NaN'

                # keep track of everything
                counter_int = counter_int + 10

                if self.solver != 'adam':
                    print("Epoch-%d, iteration-%d of %d: Mini-batch loss: %.5f Acc: %.5f Lr: %.5f"
                          % (i, offset, train_size, l, train_acc, lr))
                else:
                    print("Epoch-%d, iteration-%d of %d: Mini-batch loss: %.5f Acc: %.5f"
                          % (i, offset, train_size, l, train_acc))

                if self.debug_level >= 4:
                    print("=================================================")
                    print(predictions)
                    print(batch_data)
                    print("=================================================")
                if self.debug_level >= 2:
                    print("=================================================")
                    error, accuracy1, recall, confusions = common_configs.evaluation(predictions, batch_labels,
                                                                                     self.input_args)
                    print("Accuracy is: %f" % (accuracy1))
                    print("=================================================")

                # saving checkpoint for references
                if (counter_int % 5000) == 0:
                    print("Saving checkpoint")
                    checkpoint_path = self.model_dir + stage_name
                    filename = model_saver.save(sess=session, save_path=checkpoint_path, global_step=counter_int)
                    # break

            '''
                =============================
                Validation part after a single epoch
                =============================
            '''
            # Finished a batch, now testing the validation set
            # 1.0 dropout when doing validation
            if USE_H5PY:
                tot = 0
                j = 0
                while ((j + 1) * common_configs.BATCH_SIZE < validation_x.shape[0]):
                    val_batch_data = validation_x[j * common_configs.BATCH_SIZE:(j * common_configs.BATCH_SIZE)
                                                                                 + common_configs.BATCH_SIZE]
                    val_batch_labels = validation_y[j * common_configs.BATCH_SIZE:(j * common_configs.BATCH_SIZE)
                                                                                + common_configs.BATCH_SIZE]

                    # normalizing samples
                    val_batch_data = self.normalize_samples(images=val_batch_data)

                    last_pred = self.softmax_pred.eval(
                        feed_dict={
                            self.pinputimage: val_batch_data,  # input image
                            self.true_labels: val_batch_labels,  # input labels
                            self.istrain: False,  # flag for training
                            self.keepprob: 1.0,  # dropout probability
                            self.reduce_lr: reduce_lr
                        }
                    )
                    errval, accuracy1, recall, confusions = common_configs.evaluation(last_pred, val_batch_labels, self.input_args)
                    tot = tot + errval
                    j += 1

                if ((j * common_configs.BATCH_SIZE) < validation_x.shape[0]):
                    # evaluation of remaining part of the set, may not be a multiple of the batch_size
                    val_batch_data = validation_x[(j) * common_configs.BATCH_SIZE:]
                    val_batch_labels = validation_y[(j) * common_configs.BATCH_SIZE:]

                    # normalizing samples
                    val_batch_data = self.normalize_samples(images=val_batch_data)

                    last_pred = self.softmax_pred.eval(
                        feed_dict={
                            self.pinputimage: val_batch_data,  # input image
                            self.true_labels: val_batch_labels,  # input labels
                            self.istrain: False,  # flag for training
                            self.keepprob: 1.0,  # dropout probability
                            self.reduce_lr: reduce_lr
                        }
                    )
                    errval, accuracy1, recall, confusions = common_configs.evaluation(last_pred, val_batch_labels, self.input_args)
                    j += 1

                # for visualization purposes
                tot = tot + errval
                print('Validation Acc (epoch-%d): %.5f, Loss: %.5f\n' % (i, (tot / j), l))
            else:
                if self.debug_level >= 1:
                    print("Validating the model (%s) with own data-handler backend..." % self.model_name)

                batches = int(val_size / common_configs.BATCH_SIZE)
                tot = 0.0
                j = 0
                for bath_id in range(batches):
                    batch_X, batch_Y = get_images_batch(class_folders_paths=class_folders_paths_Val,
                                                        number_of_samples_per_class=number_of_samples_per_class_Val,
                                                        samples_filenames=samples_filenames_Val,
                                                        batch_size=common_configs.BATCH_SIZE, use_rgb=True,
                                                        normalize_samples=True)
                    last_pred = self.softmax_pred.eval(
                        feed_dict={
                            self.pinputimage: batch_X,  # input image
                            self.true_labels: batch_Y,  # input labels
                            self.istrain: False,  # flag for training
                            self.keepprob: 1.0,  # dropout probability
                            self.reduce_lr: False
                        }
                    )
                    errval, accuracy1, recall, confusions = common_configs.evaluation(last_pred, batch_Y,
                                                                                      self.input_args)
                    j += 1

                    # for visualization purposes
                    tot = tot + errval
                print('Validation Acc (epoch-%d): %.5f, Loss: %.5f\n' % (i, (tot / j), l))

            print("Model saved in file: %s" % filename)

    def mask_weight(self, var, thres):
        return tf.to_float(tf.greater_equal(tf.abs(var), tf.ones_like(var) * thres))

    def train_sparse_stage(self, train_x, train_y, train_size, steps, offset_meta, validation_x, validation_y,
                           stage_name, counter_int, reduce_lr, summary_merger, session, summary_writer, model_saver,
                           datareader=None):
        # ===================== Prepare everything =========================
        vars = {v.name: v for v in tf.trainable_variables()}
        h5f = h5py.File('maskpruned.h5', 'w')

        for var in vars:
            _, variance = tf.nn.moments(tf.reshape(vars[var], [-1]), [0])
            threshold = np.sqrt(variance.eval())
            temp = self.mask_weight(vars[var], common_configs.PRUNE_THRESHOLD)
            h5f.create_dataset(var, data=temp.eval())

        h5f = h5py.File('maskpruned.h5', 'r')
        domask = [vars[var].assign(tf.multiply(vars[var], np.array(h5f[var]))) for var in vars]
        # ===================== Prepare everything =========================

        if not USE_H5PY:
            # =========================== TRAINING
            class_folders_paths_Train = datareader['class_folders_paths_Train']
            number_of_samples_per_class_Train = datareader['number_of_samples_per_class_Train']
            samples_filenames_Train = datareader['samples_filenames_Train']
            train_size = datareader['train_size']
            # ===========================

            # =========================== VALIDATION
            class_folders_paths_Val = datareader['class_folders_paths_Val']
            number_of_samples_per_class_Val = datareader['number_of_samples_per_class_Val']
            samples_filenames_Val = datareader['samples_filenames_Val']
            val_size = datareader['val_size']
            # ===========================


        # # ===================== remove this in real production
        # epochs = 1 # just test
        # # ===================== remove this in real production

        for i in range(epochs):
            # permute the list of batches
            shuffled_step = range(steps)  # np.random.permutation(steps)#
            for step in shuffled_step:
                if USE_H5PY:
                    offset = (step * common_configs.BATCH_SIZE) % (train_size - common_configs.BATCH_SIZE)
                    batch_data = train_x[offset:(offset + common_configs.BATCH_SIZE), :, :, :]
                    batch_labels = train_y[offset:(offset + common_configs.BATCH_SIZE)]

                    # normalizing samples
                    batch_data = self.normalize_samples(images=batch_data)
                else:
                    offset = (step * common_configs.BATCH_SIZE) % (train_size - common_configs.BATCH_SIZE)
                    batch_data, batch_labels = get_images_batch(class_folders_paths=class_folders_paths_Train,
                                                                number_of_samples_per_class=number_of_samples_per_class_Train,
                                                                samples_filenames=samples_filenames_Train,
                                                                batch_size=common_configs.BATCH_SIZE, use_rgb=True,
                                                                normalize_samples=True)

                if common_configs.ENABLE_AUGMENTATION:
                    self.augment_samples(images=batch_data, labels=batch_labels)

                # 0.5 dropout probability in deep layers
                feed_dict1 = {
                    self.pinputimage: batch_data,  # input image
                    self.true_labels: batch_labels,  # input labels
                    self.istrain: True,  # flag for training
                    self.keepprob: 0.5,  # dropout probability
                    self.reduce_lr: reduce_lr
                }

                if self.debug_level >= 3:
                    print("Feed Dictionary <Epoch: %d, Step: %d, Data_taken: (%d, %d), Max: %f, Min: %f>"
                          % (i, step, offset, (offset + common_configs.BATCH_SIZE), np.max(batch_data),
                             np.min(batch_data)))
                    print('Images with shape: ', batch_data.shape)
                    print('Labels with shape: ', batch_labels.shape)

                    if self.debug_level >= 4:
                        print(feed_dict1)

                if self.enable_tensorboard == 'true':
                    if summary_merger is None:
                        print("We cannot output to tensorboard if we dont have a file writer, exiting ...")
                        return

                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                    if (counter_int % 2000) == 0:
                        if self.solver != 'adam':
                            summary, _, l, predictions, train_acc = session.run(
                                [summary_merger, self.optimizer, self.loss, self.softmax_pred, self.training_accuracy],
                                feed_dict=feed_dict1, options=run_options, run_metadata=run_metadata)

                            summary_writer.add_run_metadata(run_metadata, 'step%03d' % (steps * i + step + offset_meta))
                            summary_writer.add_summary(summary, (steps * i + step + offset_meta))
                            summary_writer.flush()
                        else:
                            summary, _, l, predictions, train_acc = session.run(
                                [summary_merger, self.optimizer, self.loss, self.softmax_pred, self.training_accuracy],
                                feed_dict=feed_dict1, options=run_options, run_metadata=run_metadata)

                            summary_writer.add_run_metadata(run_metadata, 'step%03d' % (steps * i + step + offset_meta))
                            summary_writer.add_summary(summary, (steps * i + step + offset_meta))
                            summary_writer.flush()
                    else:
                        if self.solver != 'adam':
                            summary, _, l, lr, predictions, train_acc = session.run(
                                [summary_merger, self.optimizer, self.loss, self.softmax_pred, self.training_accuracy],
                                feed_dict=feed_dict1)
                        else:
                            summary, _, l, predictions, train_acc = session.run(
                                [summary_merger, self.optimizer, self.loss, self.softmax_pred, self.training_accuracy],
                                feed_dict=feed_dict1)
                else:
                    if self.solver != 'adam':
                        _, l, lr, predictions, train_acc = session.run(
                            [self.optimizer, self.loss, self.softmax_pred, self.training_accuracy],
                            feed_dict=feed_dict1)
                    else:
                        _, l, predictions, train_acc = session.run(
                            [self.optimizer, self.loss, self.softmax_pred, self.training_accuracy],
                            feed_dict=feed_dict1)

                assert not np.isnan(l), 'Model diverged with loss = NaN'

                # ======================================= PERFORM PRUNING HERE
                print("Pruning Weights in the Network for sparse training ...")
                session.run(domask)
                # ======================================= PERFORM PRUNING HERE

                # keep track of everything
                counter_int = counter_int + 10

                if self.solver != 'adam':
                    print("Epoch-%d, iteration-%d of %d: Mini-batch loss: %.5f Acc: %.5f Lr: %.5f"
                          % (i, offset, train_size, l, train_acc, lr))
                else:
                    print("Epoch-%d, iteration-%d of %d: Mini-batch loss: %.5f Acc: %.5f"
                          % (i, offset, train_size, l, train_acc))

                if self.debug_level >= 4:
                    print("=================================================")
                    print(predictions)
                    print(batch_data)
                    print("=================================================")
                if self.debug_level >= 2:
                    print("=================================================")
                    error, accuracy1, recall, confusions = common_configs.evaluation(predictions, batch_labels, self.input_args)
                    print("Accuracy is: %f" % (accuracy1))
                    print("=================================================")

                # saving checkpoint for references
                if (counter_int % 5000) == 0:
                    print("Saving checkpoint")
                    checkpoint_path = self.model_dir + stage_name
                    filename = model_saver.save(sess=session, save_path=checkpoint_path, global_step=counter_int)
                    # break

            '''
                =============================
                Validation part after a single epoch
                =============================
            '''
            # Finished a batch, now testing the validation set
            # 1.0 dropout when doing validation
            if USE_H5PY:
                tot = 0
                j = 0
                while ((j + 1) * common_configs.BATCH_SIZE < validation_x.shape[0]):
                    val_batch_data = validation_x[j * common_configs.BATCH_SIZE:(j * common_configs.BATCH_SIZE)
                                                                                 + common_configs.BATCH_SIZE]
                    val_batch_labels = validation_y[j * common_configs.BATCH_SIZE:(j * common_configs.BATCH_SIZE)
                                                                                + common_configs.BATCH_SIZE]

                    # normalizing samples
                    val_batch_data = self.normalize_samples(images=val_batch_data)

                    last_pred = self.softmax_pred.eval(
                        feed_dict={
                            self.pinputimage: val_batch_data,  # input image
                            self.true_labels: val_batch_labels,  # input labels
                            self.istrain: False,  # flag for training
                            self.keepprob: 1.0,  # dropout probability
                            self.reduce_lr: reduce_lr
                        }
                    )
                    errval, accuracy1, recall, confusions = common_configs.evaluation(last_pred, val_batch_labels, self.input_args)
                    tot = tot + errval
                    j += 1

                if ((j * common_configs.BATCH_SIZE) < validation_x.shape[0]):
                    # evaluation of remaining part of the set, may not be a multiple of the batch_size
                    val_batch_data = validation_x[(j) * common_configs.BATCH_SIZE:]
                    val_batch_labels = validation_y[(j) * common_configs.BATCH_SIZE:]

                    # normalizing samples
                    val_batch_data = self.normalize_samples(images=val_batch_data)

                    last_pred = self.softmax_pred.eval(
                        feed_dict={
                            self.pinputimage: val_batch_data,  # input image
                            self.true_labels: val_batch_labels,  # input labels
                            self.istrain: False,  # flag for training
                            self.keepprob: 1.0,  # dropout probability
                            self.reduce_lr: reduce_lr
                        }
                    )
                    errval, accuracy1, recall, confusions = common_configs.evaluation(last_pred, val_batch_labels, self.input_args)
                    j += 1

                # for visualization purposes
                tot = tot + errval
                print('Validation Acc (epoch-%d): %.5f, Loss: %.5f\n' % (i, (tot / j), l))
            else:
                if self.debug_level >= 1:
                    print("Validating the model (%s) with own data-handler backend..." % self.model_name)

                batches = int(val_size / common_configs.BATCH_SIZE)
                tot = 0.0
                j = 0
                for bath_id in range(batches):
                    batch_X, batch_Y = get_images_batch(class_folders_paths=class_folders_paths_Val,
                                                        number_of_samples_per_class=number_of_samples_per_class_Val,
                                                        samples_filenames=samples_filenames_Val,
                                                        batch_size=common_configs.BATCH_SIZE, use_rgb=True,
                                                        normalize_samples=True)
                    last_pred = self.softmax_pred.eval(
                        feed_dict={
                            self.pinputimage: batch_X,  # input image
                            self.true_labels: batch_Y,  # input labels
                            self.istrain: False,  # flag for training
                            self.keepprob: 1.0,  # dropout probability
                            self.reduce_lr: False
                        }
                    )
                    errval, accuracy1, recall, confusions = common_configs.evaluation(last_pred, batch_Y,
                                                                                      self.input_args)
                    j += 1

                    # for visualization purposes
                    tot = tot + errval
                print('Validation Acc (epoch-%d): %.5f, Loss: %.5f\n' % (i, (tot / j), l))

            print("Model saved in file: %s" % filename)

        return 0
    # ==================================================================================

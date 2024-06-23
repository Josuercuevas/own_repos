"""
    Autoencoders implementation in Tensorflow

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
"""

import os
import sys
import random
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tflearn.layers.conv import upsample_2d, max_pool_2d, conv_2d_transpose, conv_2d
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.core import reshape
import time
import tarfile
import numpy as np
import cv2
import configs.common_configs as config
import h5py
from PIL import Image

class FC_AutoEncoder(object):
    def __init__(self, input_args, session=None):
        if input_args.execution_mode == 'train':
            self.build_trainer(input_args)
        else:
            self.build_inference(input_args)

        self.session = session
        self.saver = tf.train.Saver()  # max_to_keep=5, keep_checkpoint_every_n_hours=1

        if input_args.enable_tensorboard=='true' and input_args.execution_mode == 'train':
            print("Constructing Summarizer ...")
            tf.summary.scalar('Training_accuracy', self.training_accuracy)
            tf.summary.scalar('Loss_RMSE', self.loss)
            if input_args.debug_info >= 3:# highly verbose and images to be dumped
                tf.summary.image('Original_Input', self.ground_truth)
                tf.summary.image('Reconstruction', self.upscore_layer)
                tf.summary.image('Error_reconstruction', self.error_rec)
            self.summary_merger = tf.summary.merge_all()

        if input_args.enable_tensorboard == 'true':
            print("Saving model graph ...")
            # clean folder before start anything in training
            if tf.gfile.Exists('tfboard/model_training_progress'):
                tf.gfile.DeleteRecursively('tfboard/model_training_progress')
                tf.gfile.MakeDirs('tfboard/model_training_progress')

            train_writer = tf.summary.FileWriter('tfboard/model_training_progress/train', self.session.graph)

            train_writer.flush()
            self.summary_writer = train_writer

        self.session.run(tf.global_variables_initializer())
        self.checkpoint_dir = input_args.model_dir


    def train_dense_stage(self, input_args, train_x, train_y, train_size, steps, offset_meta,
                          validation_x, stage_name, counter_int, reduce_lr):
        epochs = input_args.max_epochs

        for i in range(epochs):
            # permute the list of batches
            shuffled_step = range(steps)  # np.random.permutation(steps)#
            for step in shuffled_step:
                offset = (step * config.BATCH_SIZE) % (train_size - config.BATCH_SIZE)
                batch_data = train_x[offset:(offset + config.BATCH_SIZE), :, :, :]
                batch_labels = train_y[offset:(offset + config.BATCH_SIZE)]

                # 0.5 dropout probability in deep layers
                feed_dict1 = {
                    self.pinputimage: batch_data,#input image
                    self.ground_truth: batch_data,#input image
                    self.reduce_lr: reduce_lr
                }

                if input_args.debug_info >= 2:
                    print("Feed Dictionary <Epoch: %d, Step: %d, Data_taken: (%d, %d)>"
                          %(i, step, offset, (offset + config.BATCH_SIZE)))
                    if input_args.debug_info >= 4:
                        print(feed_dict1)

                if input_args.enable_tensorboard == 'true':
                    if self.summary_merger is None:
                        print("We cannot output to tensorboard if we dont have a file writer, exiting ...")
                        return

                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                    if (counter_int%50) == 0:
                        if input_args.solver != 'adam':
                            summary, _, l, predictions, train_acc = self.session.run(
                                [self.summary_merger, self.optimizer, self.loss, self.upscore_layer,
                                 self.training_accuracy],
                                feed_dict=feed_dict1, options=run_options, run_metadata=run_metadata)

                            self.summary_writer.add_run_metadata(run_metadata,
                                                          'step%03d' % (steps * i + step + offset_meta))
                            self.summary_writer.add_summary(summary, (steps * i + step + offset_meta))
                            self.summary_writer.flush()
                        else:
                            summary, _, l, predictions, train_acc = self.session.run(
                                [self.summary_merger, self.optimizer, self.loss, self.upscore_layer,
                                 self.training_accuracy],
                                feed_dict=feed_dict1, options=run_options, run_metadata=run_metadata)

                            self.summary_writer.add_run_metadata(run_metadata,
                                                          'step%03d' % (steps * i + step + offset_meta))
                            self.summary_writer.add_summary(summary, (steps * i + step + offset_meta))
                            self.summary_writer.flush()
                    else:
                        if input_args.solver != 'adam':
                            summary, _, l, lr, predictions, train_acc = self.session.run(
                                [self.summary_merger, self.optimizer, self.loss, self.upscore_layer,
                                 self.training_accuracy], feed_dict=feed_dict1)
                        else:
                            summary, _, l, predictions, train_acc = self.session.run(
                                [self.summary_merger, self.optimizer, self.loss, self.upscore_layer,
                                 self.training_accuracy], feed_dict=feed_dict1)
                else:
                    if input_args.solver != 'adam':
                        _, l, lr, predictions, train_acc = self.session.run([self.optimizer,
                                                                             self.loss, self.upscore_layer,
                                                                             self.training_accuracy],
                                                                            feed_dict=feed_dict1)
                    else:
                        _, l, predictions, train_acc = self.session.run([self.optimizer,
                                                                         self.loss, self.upscore_layer,
                                                                         self.training_accuracy],
                                                                        feed_dict=feed_dict1)

                assert not np.isnan(l), 'Model diverged with loss = NaN'

                # keep track of everything
                counter_int = counter_int + 10

                if input_args.solver != 'adam':
                    print("Epoch-%d, iteration-%d of %d: Mini-batch loss: %.5f Acc: %.5f Lr: %.5f"
                               % (i, offset, train_size, l, train_acc, lr) )
                else:
                    print("Epoch-%d, iteration-%d of %d: Mini-batch loss: %.5f Acc: %.5f"
                               % (i, offset, train_size, l, train_acc) )

                if input_args.debug_info >= 4:
                    print("=================================================")
                    print(predictions)
                    print(batch_data)
                    print("=================================================")
                if input_args.debug_info >= 2:
                    print("=================================================")
                    accuracy1 = config.evaluation(predictions, batch_data, input_args)
                    print("Accuracy is: %f"%(accuracy1))
                    print("=================================================")

                # saving checkpoint for references
                if (counter_int%1000) == 0:
                    print("Saving checkpoint")
                    checkpoint_path = input_args.model_dir + stage_name
                    filename = self.saver.save(sess=self.session, save_path=checkpoint_path,
                                               global_step=counter_int/10)
                    # break

            '''
                =============================
                Validation part after a single epoch
                =============================
            '''
            #Finished a batch, now testing the validation set
            # 1.0 dropout when doing validation
            tot = 0
            j = 0
            while ((j + 1) * config.BATCH_SIZE < validation_x.shape[0]):
                last_pred = self.upscore_layer.eval(
                        feed_dict={
                            self.pinputimage: validation_x[j * config.BATCH_SIZE:(j * config.BATCH_SIZE)
                                                                                 + config.BATCH_SIZE],
                            self.ground_truth: validation_x[j * config.BATCH_SIZE:(j * config.BATCH_SIZE)
                                                                                  + config.BATCH_SIZE],
                            self.reduce_lr: reduce_lr
                        }
                    )
                errval = config.evaluation(last_pred, validation_x[j * config.BATCH_SIZE:(j * config.BATCH_SIZE) +
                                                                                         config.BATCH_SIZE], input_args)
                tot = tot + errval
                j += 1

            if ((j * config.BATCH_SIZE) < validation_x.shape[0]):
                # evaluation of remaining part of the set, may not be a multiple of the batch_size
                last_pred = self.upscore_layer.eval(
                        feed_dict={
                            self.pinputimage: validation_x[(j) * config.BATCH_SIZE:],
                            self.ground_truth: validation_x[(j) * config.BATCH_SIZE:],
                            self.reduce_lr: reduce_lr
                        }
                    )
                errval = config.evaluation(last_pred, validation_x[(j) * config.BATCH_SIZE:], input_args)
                j += 1

            # for visualization purposes
            tot = tot + errval
            print('Validation Acc (epoch-%d): %.5f, Loss: %.5f\n'% (i, (tot / j), l) )


            print("Model saved in file: %s" % filename)


    def mask_weight(self, var, thres):
        return tf.to_float(tf.greater_equal(tf.abs(var), tf.ones_like(var) * thres))

    def train_sparse_stage(self, input_args, train_x, train_y, train_size, steps, offset_meta,
                          validation_x, stage_name, counter_int, reduce_lr):
        # ===================== Prepare everything =========================
        vars = {v.name: v for v in tf.trainable_variables()}
        h5f = h5py.File('maskpruned.h5', 'w')

        for var in vars:
            _, variance = tf.nn.moments(tf.reshape(vars[var], [-1]), [0])
            threshold = np.sqrt(variance.eval())
            temp = self.mask_weight(vars[var], config.PRUNE_THRESHOLD)
            h5f.create_dataset(var, data=temp.eval())

        h5f = h5py.File('maskpruned.h5', 'r')
        domask = [vars[var].assign(tf.multiply(vars[var], np.array(h5f[var]))) for var in vars]
        # ===================== Prepare everything =========================


        for i in range(input_args.max_epochs):
            # permute the list of batches
            shuffled_step = range(steps)  # np.random.permutation(steps)#
            for step in shuffled_step:
                offset = (step * config.BATCH_SIZE) % (train_size - config.BATCH_SIZE)
                batch_data = train_x[offset:(offset + config.BATCH_SIZE), :, :, :]
                batch_labels = train_y[offset:(offset + config.BATCH_SIZE)]

                # 0.5 dropout probability in deep layers
                feed_dict1 = {
                    self.pinputimage: batch_data,#input image
                    self.ground_truth: batch_data,#input image
                    self.reduce_lr: reduce_lr
                }

                if input_args.debug_info >= 2:
                    print("Feed Dictionary <Epoch: %d, Step: %d, Data_taken: (%d, %d)>"
                          %(i, step, offset, (offset + config.BATCH_SIZE)))
                    if input_args.debug_info >= 4:
                        print(feed_dict1)

                if input_args.enable_tensorboard == 'true':
                    if self.summary_merger is None:
                        print("We cannot output to tensorboard if we dont have a file writer, exiting ...")
                        return

                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                    if (counter_int%50) == 0:
                        if input_args.solver != 'adam':
                            summary, _, l, predictions, train_acc = self.session.run(
                                [self.summary_merger, self.optimizer, self.loss, self.upscore_layer,
                                 self.training_accuracy],
                                feed_dict=feed_dict1, options=run_options, run_metadata=run_metadata)

                            self.summary_writer.add_run_metadata(run_metadata,
                                                          'step%03d' % (steps * i + step + offset_meta))
                            self.summary_writer.add_summary(summary, (steps * i + step + offset_meta))
                            self.summary_writer.flush()
                        else:
                            summary, _, l, predictions, train_acc = self.session.run(
                                [self.summary_merger, self.optimizer, self.loss, self.upscore_layer,
                                 self.training_accuracy],
                                feed_dict=feed_dict1, options=run_options, run_metadata=run_metadata)

                            self.summary_writer.add_run_metadata(run_metadata,
                                                          'step%03d' % (steps * i + step + offset_meta))
                            self.summary_writer.add_summary(summary, (steps * i + step + offset_meta))
                            self.summary_writer.flush()
                    else:
                        if input_args.solver != 'adam':
                            summary, _, l, lr, predictions, train_acc = self.session.run(
                                [self.summary_merger, self.optimizer, self.loss, self.upscore_layer,
                                 self.training_accuracy], feed_dict=feed_dict1)
                        else:
                            summary, _, l, predictions, train_acc = self.session.run(
                                [self.summary_merger, self.optimizer, self.loss, self.upscore_layer,
                                 self.training_accuracy], feed_dict=feed_dict1)
                else:
                    if input_args.solver != 'adam':
                        _, l, lr, predictions, train_acc = self.session.run([self.optimizer,
                                                                             self.loss, self.upscore_layer,
                                                                             self.training_accuracy],
                                                                            feed_dict=feed_dict1)
                    else:
                        _, l, predictions, train_acc = self.session.run([self.optimizer,
                                                                         self.loss, self.upscore_layer,
                                                                         self.training_accuracy],
                                                                        feed_dict=feed_dict1)

                assert not np.isnan(l), 'Model diverged with loss = NaN'

                # ======================================= PERFORM PRUNING HERE
                print("Pruning Weights in the Network for sparse training ...")
                self.session.run(domask)
                # ======================================= PERFORM PRUNING HERE

                # keep track of everything
                counter_int = counter_int + 10

                if input_args.solver != 'adam':
                    print("Epoch-%d, iteration-%d of %d: Mini-batch loss: %.5f Acc: %.5f Lr: %.5f"
                               % (i, offset, train_size, l, train_acc, lr) )
                else:
                    print("Epoch-%d, iteration-%d of %d: Mini-batch loss: %.5f Acc: %.5f"
                               % (i, offset, train_size, l, train_acc) )

                if input_args.debug_info >= 4:
                    print("=================================================")
                    print(predictions)
                    print(batch_data)
                    print("=================================================")
                if input_args.debug_info >= 2:
                    print("=================================================")
                    accuracy1 = config.evaluation(predictions, batch_data, input_args)
                    print("Accuracy is: %f"%(accuracy1))
                    print("=================================================")

                # saving checkpoint for references
                if (counter_int%1000) == 0:
                    print("Saving checkpoint")
                    checkpoint_path = input_args.model_dir + stage_name
                    filename = self.saver.save(sess=self.session, save_path=checkpoint_path,
                                               global_step=counter_int/10)
                    # break

            '''
                =============================
                Validation part after a single epoch
                =============================
            '''
            #Finished a batch, now testing the validation set
            # 1.0 dropout when doing validation
            tot = 0
            j = 0
            while ((j + 1) * config.BATCH_SIZE < validation_x.shape[0]):
                last_pred = self.upscore_layer.eval(
                        feed_dict={
                            self.pinputimage: validation_x[j * config.BATCH_SIZE:(j * config.BATCH_SIZE)
                                                                                 + config.BATCH_SIZE],
                            self.ground_truth: validation_x[j * config.BATCH_SIZE:(j * config.BATCH_SIZE)
                                                                                  + config.BATCH_SIZE],
                            self.reduce_lr: reduce_lr
                        }
                    )
                errval = config.evaluation(last_pred, validation_x[j * config.BATCH_SIZE:(j * config.BATCH_SIZE) +
                                                                                         config.BATCH_SIZE], input_args)
                tot = tot + errval
                j += 1

            if ((j * config.BATCH_SIZE) < validation_x.shape[0]):
                # evaluation of remaining part of the set, may not be a multiple of the batch_size
                last_pred = self.upscore_layer.eval(
                        feed_dict={
                            self.pinputimage: validation_x[(j) * config.BATCH_SIZE:],
                            self.ground_truth: validation_x[(j) * config.BATCH_SIZE:],
                            self.reduce_lr: reduce_lr
                        }
                    )
                errval = config.evaluation(last_pred, validation_x[(j) * config.BATCH_SIZE:], input_args)
                j += 1

            # for visualization purposes
            tot = tot + errval
            print('Validation Acc (epoch-%d): %.5f, Loss: %.5f\n'% (i, (tot / j), l) )


            print("Model saved in file: %s" % filename)




        return 0

    def train(self, input_args, datareader=None):
        """
            trainer module
            :param input_args: user options
            :param datareader: database retriever or reader
            :return: training result
        """
        if input_args.debug_info >= 1:
            print("Training model ...")

            if input_args.debug_info >= 2:
                if input_args.enable_tensorboard == 'true':
                    if self.summary_merger is not None:
                        print(self.summary_merger)

            if input_args.debug_info >= 4:
                if datareader is not None:
                    print(datareader)

            if input_args.debug_info >= 2:
                if self.session is not None:
                    print(self.session)

            if input_args.debug_info >= 2:
                # check they really exist
                print("========================== TRAINER INPUT =========================")
                print(self.pinputimage)
                print(self.ground_truth)
                print("========================== TRAINER OUTPUT =========================")
                print(self.upscore_layer)
                print("========================== TRAINER MEASURES =========================")
                print(self.training_accuracy)
                print(self.loss)
                print("========================== trainer optimizer =========================")
                print(self.optimizer)

        if datareader is None:
            print("No Data reader or fetcher, we cannot train, quitting this application")
            return

        item_dict = datareader
        item = []
        for attr_name in datareader:
            if input_args.debug_info >= 2:
                print("Retrieving list in dictionary:")
                print(attr_name)
            item.append(item_dict[attr_name])

        train_x = item_dict['train_x']
        train_y = item_dict['train_y']
        validation_x = item_dict['validation_x']
        validation_y = item_dict['validation_y']
        test_x = item_dict['test_x']
        test_y = item_dict['test_y']
        train_size = item_dict['train_size']

        if input_args.debug_info >= 2:
            print("Database contains %d items"%train_size)

        if input_args.debug_info >= 4 and input_args.enable_tensorboard == 'true':
            # The embedding variable, which needs to be stored
            # Note this must a Variable not a Tensor!
            print("======= SAMPLES ARE =======================")
            print(train_x.shape)
            embedding_var = tf.Variable(train_x, name='Embedding_train_samples')
            self.session.run(embedding_var.initializer)
            summary_writer_emb = tf.summary.FileWriter('tfboard/model_training_progress/train')
            config_emb = projector.ProjectorConfig()
            embedding = config_emb.embeddings.add()
            embedding.tensor_name = embedding_var.name

            # Comment out if you don't have metadata
            embedding.metadata_path = os.path.join('tfboard/model_training_progress/train', 'metadata.tsv')

            # # Comment out if you don't want sprites
            # embedding.sprite.image_path = os.path.join(LOG_DIR, 'sprite.png')
            # embedding.sprite.single_image_dim.extend([imgs.shape[1], imgs.shape[1]])

            projector.visualize_embeddings(summary_writer_emb, config_emb)
            saver_emb = tf.train.Saver([embedding_var])
            saver_emb.save(self.session,
                           os.path.join('tfboard/model_training_progress/train', 'model.ckpt'), 1)

            names = ['name1', 'name2', 'name3', 'name4', 'name5', 'name6', 'name7', 'name8',
                     'name9', 'name10', 'name11', 'name12', 'name13', 'name14', 'name15']
            metadata_file = open(os.path.join('tfboard/model_training_progress/train', 'metadata.tsv'), 'w')
            metadata_file.write('Name\tClass\n')
            for dd in range(train_y.shape[0]):
                metadata_file.write('%d\t%s\n' % (dd, names[np.argmax(train_y[dd])]))
            metadata_file.close()


        # 8888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888
        # Dense Training D
        print("\n\n\n\n============== Training Dense Part of the model with Learning rate = %f"%config.LEARNING_RATE)
        counter_int = 0
        run_id = 0  # since here e are not training DSD we dont care
        steps = int(train_size // config.BATCH_SIZE)
        offset_meta = steps * input_args.max_epochs * run_id
        self.train_dense_stage(input_args=input_args, train_x=train_x, stage_name="autoencoders_D.ckpt",
                               train_y=train_y, train_size=train_size, steps=steps, offset_meta=offset_meta,
                               validation_x=validation_x, counter_int=counter_int, reduce_lr=False)
        # 8888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888

        # 8888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888
        # Sparse Training DS
        print("\n\n\n\n============== Training Dense Sparse Part of the model with Learning rate = %f"%config.LEARNING_RATE)
        counter_int = 0
        run_id = 1  # since here e are not training DSD we dont care
        steps = int(train_size // config.BATCH_SIZE)
        offset_meta = steps * input_args.max_epochs * run_id
        self.train_sparse_stage(input_args=input_args, train_x=train_x, stage_name="autoencoders_DS.ckpt",
                                train_y=train_y, train_size=train_size, steps=steps, offset_meta=offset_meta,
                                validation_x=validation_x, counter_int=counter_int, reduce_lr=False)
        # 8888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888=

        # 8888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888
        # Dense Training DSD
        print("\n\n\n\n============== Training Dense Sparse Dense Part of the model with Learning rate = %f"%
              (config.LEARNING_RATE / 10.0))
        counter_int = 0
        run_id = 2  # since here e are not training DSD we dont care
        steps = int(train_size // config.BATCH_SIZE)
        offset_meta = steps * input_args.max_epochs * run_id
        self.train_dense_stage(input_args=input_args, train_x=train_x, stage_name="autoencoders_DSD.ckpt",
                               train_y=train_y, train_size=train_size, steps=steps, offset_meta=offset_meta,
                               validation_x=validation_x, counter_int=counter_int, reduce_lr=True)
        # 8888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888

    def test(self, sess, test_x, test_y, input_args, epoch=0, loss=0.0):
        """
            tester module
            :param input_args: user options
            :param datareader: database retriever or reader
            :return: testing result
        """
        if input_args.debug_info >= 1:
            print("Testing the model...")

        # 1.0 dropout when doing validation
        tot = 0
        j = 0
        while ((j + 1) * config.BATCH_SIZE < test_x.shape[0]):
            errval = config.evaluation(
                self.softmax_pred.eval(
                    feed_dict={
                        self.pinputimage: test_x[j * config.BATCH_SIZE:(j * config.BATCH_SIZE)
                                                                                     + config.BATCH_SIZE],
                        self.true_labels: test_y[j * config.BATCH_SIZE:(j * config.BATCH_SIZE)
                                                                                     + config.BATCH_SIZE],
                        self.keepprob: 1.0,
                        self.istrain: False
                    }
                ),
                test_y[j * config.BATCH_SIZE:(j * config.BATCH_SIZE) +
                                             config.BATCH_SIZE], input_args)[0]
            tot = tot + errval
            j += 1

        if ((j * config.BATCH_SIZE) < test_x.shape[0]):
            # evaluation of remaining part of the set, may not be a multiple of the batch_size
            errval = config.evaluation(
                self.softmax_pred.eval(
                    feed_dict={
                        self.pinputimage: test_x[(j) * config.BATCH_SIZE:],
                        self.true_labels: test_y[(j) * config.BATCH_SIZE:],
                        self.keepprob: 1.0,
                        self.istrain: False
                    }
                ), test_y[(j) * config.BATCH_SIZE:], input_args)[0]
            j += 1

        # for visualization purposes
        tot = tot + errval
        print('Testing set Acc (epoch-%d): %.5f, Loss: %.5f\n' % (epoch, (100 - tot / j), loss))



    # it will extract only the features that can be used later
    def extract_features(self, sess, train_x, train_y, input_args, epoch=0, loss=0.0):
        if input_args.debug_info >= 1:
            print("Extracting features from the model...")

        op = sess.graph.get_operations()
        for m in op:
            print(m.values())

        # Layer to be used
        flattened_tensor = sess.graph.get_tensor_by_name('BatchNormalization_encoder3:0')

        # features_extracted = []
        # images_entered = []

        if tf.gfile.Exists('features_extracted'):
            tf.gfile.DeleteRecursively('features_extracted')
            tf.gfile.MakeDirs('features_extracted')
        else:
            tf.gfile.MakeDirs('features_extracted')

        folder_path = 'features_extracted/'
        f_handle = file('features_extracted/features.npy', 'a')
        label_handle = open('features_extracted/labels.txt', 'w')

        for image_id in range(train_x.shape[0]):#range(10):#
            image_to_pass = train_x[image_id:image_id+1]
            features = sess.run(flattened_tensor,
                                feed_dict={
                                    self.pinputimage: image_to_pass,
                                    self.true_labels: train_y[image_id:image_id+1],
                                    self.keepprob: 1.0,
                                    self.istrain: False
                                }
                                )
            im = Image.fromarray(image_to_pass[0])
            im.save(folder_path+'image_'+str(image_id)+'.jpg')
            np.save(f_handle, features)
            label_handle.write("%s\n"%str(np.argmax(train_y[image_id])))
            label_handle.flush()

            print(image_to_pass.shape)
            print(features.shape)

        f_handle.close()
        label_handle.close()


    def build_inference(self, input_args):
        """
            Inferer engine to be used for for prediction of samples
            :param input_args: user options
            :return: nothing
        """
        # Input part to be used for training network
        if input_args.debug_info >= 1:
            print("Building inferer engine for the Network")

        pinputimage = tf.placeholder(tf.float32, shape=[None, config.HEIGHT_GLOBAL,
                                                        config.WIDTH_GLOBAL,
                                                        config.CHANNELS_G], name="input_image")
        self.pinputimage = pinputimage

        # determine if we are training or testing
        istrain = tf.placeholder(tf.bool, name="istrain")
        self.istrain = istrain

        # for dropout in case we are traaining or not
        keepprob = tf.placeholder(tf.float32, name="keep_probabilty")
        self.keepprob = keepprob

        true_labels = tf.placeholder(tf.float32, shape=[None, config.N_CLASSES], name="ground_truth")
        self.true_labels = true_labels

        if input_args.debug_info >= 2:
            print("Input Image shape:")
            print(self.pinputimage.get_shape())

        # =================================================================
        # =================================================================
        # =================================================================
        # ===================== ENCODER PART =====================
        # =================================================================
        # =================================================================
        # =================================================================

        input_channels = pinputimage.get_shape().as_list()[-1]
        conv_input = self.conv_layer(x=pinputimage, W_shape=[3, 3, input_channels, 32],
                                     b_shape=32, name='conv_input', stride=1)
        if input_args.debug_info >= 2:
            print("conv_input shape:")
            print(conv_input.get_shape())

        # ------------------------ ENCODER 1
        encoder_1 = self.conv_layer(conv_input, W_shape=[3, 3, 32, 32], b_shape=32,
                                    name='encoder_1', stride=1)
        # add a batch normalization
        encoder_1 = batch_normalization(encoder_1, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002,
                                        trainable=True, restore=True, reuse=False, scope=None,
                                        name='BatchNormalization_encoder1')
        if input_args.debug_info >= 2:
            print("encoder_1 shape:")
            print(encoder_1.get_shape())

        if config.ALLOW_POOLING:
            encoder_1, encoder_1_argmax = self.maxpool_layer(encoder_1, name='pooling_encoder1')
            # encoder_1 = max_pool_2d(encoder_1, [1, 2, 2, 1], [1, 2, 2, 1],
            #                         name='pooling_encoder1', padding='same')

        # ------------------------ ENCODER 2
        encoder_2 = self.conv_layer(encoder_1, W_shape=[3, 3, 32, 64], b_shape=64,
                                    name='encoder_2', stride=1)
        # add a batch normalization
        encoder_2 = batch_normalization(encoder_2, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002,
                                        trainable=True, restore=True, reuse=False, scope=None,
                                        name='BatchNormalization_encoder2')
        if input_args.debug_info >= 2:
            print("encoder_2 shape:")
            print(encoder_2.get_shape())

        if config.ALLOW_POOLING:
            encoder_2, encoder_2_argmax = self.maxpool_layer(encoder_2, name='pooling_encoder2')
            # encoder_2 = max_pool_2d(encoder_2, [1, 2, 2, 1], [1, 2, 2, 1],
            #                         name='pooling_encoder2', padding='same')

        # -------------------------- ENCODER 3
        encoder_3 = self.conv_layer(encoder_2, W_shape=[3, 3, 64, config.N_CLASSES], b_shape=config.N_CLASSES,
                                    name='encoder_3', stride=1)
        # add a batch normalization
        encoder_3 = batch_normalization(encoder_3, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002,
                                        trainable=True, restore=True, reuse=False, scope=None,
                                        name='BatchNormalization_encoder3')

        if input_args.debug_info >= 2:
            print("encoder_3 shape:")
            print(encoder_3.get_shape())

        drop_layer = self.dropout_module(encoder_3, keepprob, istrain)
        if input_args.debug_info >= 2:
            print("Dropout layer")
            print(drop_layer.get_shape())

        # reshaping to match the classes
        predictions = tf.reshape(drop_layer, [-1, config.N_CLASSES])
        if input_args.debug_info >= 2:
            print("Model reshaped and new resulted in:")
            print(predictions.get_shape())

        self.predictions = predictions

        with tf.name_scope('output'):
            # Performing softmax to determine class prediction
            softmax_pred = tf.nn.softmax(self.predictions)

        self.softmax_pred = softmax_pred

    def build_trainer(self, input_args):
        """
            Trainer engine to be used for optimization
            :param input_args: user options
            :return: nothing
        """
        # Input part to be used for training network
        if input_args.debug_info >= 1:
            print("Building trainer engine for the Network")

        pinputimage = tf.placeholder(tf.float32, shape=[None, config.HEIGHT_GLOBAL,
                                                        config.WIDTH_GLOBAL,
                                                        config.CHANNELS_G], name="input_image")
        ground_truth = tf.placeholder(tf.float32, shape=[None, config.HEIGHT_GLOBAL, config.WIDTH_GLOBAL,
                                                       config.CHANNELS_G])

        reduce_lr = tf.placeholder(tf.bool, name="istrain")

        self.pinputimage = pinputimage
        self.ground_truth = ground_truth

        if input_args.debug_info >= 2:
            print("Input Image shape:")
            print(self.pinputimage.get_shape())
            print("Ground Truth Image shape:")
            print(self.ground_truth.get_shape())

        # =================================================================
        # =================================================================
        # =================================================================
        # ===================== ENCODER PART =====================
        # =================================================================
        # =================================================================
        # =================================================================

        input_channels = pinputimage.get_shape().as_list()[-1]
        conv_input = self.conv_layer(x=pinputimage, W_shape=[3, 3, input_channels, 32],
                                     b_shape=32, name='conv_input', stride=1)
        if input_args.debug_info >= 2:
            print("conv_input shape:")
            print(conv_input.get_shape())

        # ------------------------ ENCODER 1
        encoder_1 = self.conv_layer(conv_input, W_shape=[3, 3, 32, 32], b_shape=32,
                                 name='encoder_1', stride=1)
        # add a batch normalization
        encoder_1 = batch_normalization(encoder_1, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002,
                                        trainable=True, restore=True, reuse=False, scope=None,
                                        name='BatchNormalization_encoder1')
        if input_args.debug_info >= 2:
            print("encoder_1 shape:")
            print(encoder_1.get_shape())

        if config.ALLOW_POOLING:
            encoder_1, encoder_1_argmax = self.maxpool_layer(encoder_1, name='pooling_encoder1')
            # encoder_1 = max_pool_2d(encoder_1, [1, 2, 2, 1], [1, 2, 2, 1],
            #                         name='pooling_encoder1', padding='same')

        # ------------------------ ENCODER 2
        encoder_2 = self.conv_layer(encoder_1, W_shape=[3, 3, 32, 64], b_shape=64,
                                 name='encoder_2', stride=1)
        # add a batch normalization
        encoder_2 = batch_normalization(encoder_2, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002,
                                        trainable=True, restore=True, reuse=False, scope=None,
                                        name='BatchNormalization_encoder2')
        if input_args.debug_info >= 2:
            print("encoder_2 shape:")
            print(encoder_2.get_shape())

        if config.ALLOW_POOLING:
            encoder_2, encoder_2_argmax = self.maxpool_layer(encoder_2, name='pooling_encoder2')
            # encoder_2 = max_pool_2d(encoder_2, [1, 2, 2, 1], [1, 2, 2, 1],
            #                         name='pooling_encoder2', padding='same')

        # -------------------------- ENCODER 3
        encoder_3 = self.conv_layer(encoder_2, W_shape=[3, 3, 64, config.N_CLASSES], b_shape=config.N_CLASSES,
                                name='encoder_3', stride=1)
        # add a batch normalization
        encoder_3 = batch_normalization(encoder_3, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002,
                                        trainable=True, restore=True, reuse=False, scope=None,
                                        name='BatchNormalization_encoder3')
        if input_args.debug_info >= 2:
            print("encoder_3 shape:")
            print(encoder_3.get_shape())


        # =================================================================
        # =================================================================
        # =================================================================
        # ===================== DECODER PART =====================
        # =================================================================
        # =================================================================
        # =================================================================

        # ------------------------------- DECODER 1
        decoder_1 = self.deconv_layer(encoder_3, W_shape=[3, 3, 64, config.N_CLASSES], b_shape=64,
                                    name='decoder_1', stride=1)
        # add a batch normalization
        decoder_1 = batch_normalization(decoder_1, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002,
                                        trainable=True, restore=True, reuse=False, scope=None,
                                        name='BatchNormalization_decoder1')
        if input_args.debug_info >= 2:
            print("decoder_1 shape:")
            print(decoder_1.get_shape())

        # --------------------------------- DECODER 2
        decoder_2 = self.deconv_layer(decoder_1, W_shape=[3, 3, 32, 64], b_shape=32,
                                name='decoder_2', stride=1)
        # add a batch normalization
        decoder_2 = batch_normalization(decoder_2, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002,
                                        trainable=True, restore=True, reuse=False, scope=None,
                                        name='BatchNormalization_decoder2')
        if config.ALLOW_POOLING:
            decoder_2 = self.maxunpool_layer(decoder_2, self.unravel_argmax(encoder_1_argmax,
                                                                            tf.to_int64(tf.shape(encoder_1))))
            decoder_2 = tf.to_float(decoder_2)
            # decoder_2 = upsample_2d(decoder_2, [1, 2, 2, 1], name='unpooling_decoder3')

        if input_args.debug_info >= 2:
            print("decoder_2 shape:")
            print(decoder_2.get_shape())

        # --------------------------------- DECODER 3
        decoder_3 = self.deconv_layer(decoder_2, W_shape=[3, 3, 32, 32], b_shape=32,
                                      name='decoder_3', stride=1)
        # add a batch normalization
        decoder_3 = batch_normalization(decoder_3, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002,
                                        trainable=True, restore=True, reuse=False, scope=None,
                                        name='BatchNormalization_decoder3')
        if config.ALLOW_POOLING:
            decoder_3 = self.maxunpool_layer(decoder_3, self.unravel_argmax(encoder_1_argmax,
                                                                            tf.to_int64(tf.shape(encoder_1))))
            decoder_3 = tf.to_float(decoder_3)
            # decoder_3 = upsample_2d(decoder_3, [1, 2, 2, 1], name='unpooling_decoder3')

        if input_args.debug_info >= 2:
            print("decoder_3 shape:")
            print(decoder_3.get_shape())

        # ------------------------------------------ PREDICTIONS
        with tf.name_scope('Reconstructed_Images'):
            upscore_layer = self.deconv_layer(decoder_3, W_shape=[1, 1, input_channels, 32], b_shape=input_channels,
                                              name='upscore_layer', stride=1)
            # add a batch normalization
            upscore_layer = batch_normalization(upscore_layer, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002,
                                            trainable=True, restore=True, reuse=False, scope=None,
                                            name='BatchNormalization_upscore_layer')
            self.upscore_layer = upscore_layer
        if input_args.debug_info >= 2:
            print("upscore_layer shape:")
            print(upscore_layer.get_shape())

        # =================================== OPTIMIZATION AND LOSSES
        # reconstruction error
        with tf.name_scope('Reconstruction_error'):
            error_rec = tf.pow(self.upscore_layer - self.ground_truth, 2, name='SE')
            self.error_rec = tf.minimum(error_rec*255.0, 255.0)

        with tf.name_scope('Loss_RMSE'):
            weight_decay = 0.00005
            loss = tf.log(tf.exp(1.0) + tf.reduce_mean(tf.pow(self.upscore_layer - self.ground_truth, 2)) +
                          weight_decay * (tf.add_n(tf.get_collection('l2_0'))), name='reg_log_RMSE')

        with tf.name_scope('Training_accuracy'):
            training_accuracy = (1.0 / loss) * 100.0

        # Indicators
        self.training_accuracy = training_accuracy
        self.loss = loss
        self.reduce_lr = reduce_lr

        self.optimizer = tf.cond(reduce_lr, lambda: tf.train.AdamOptimizer(config.LEARNING_RATE).minimize(self.loss),
                                 lambda: tf.train.AdamOptimizer(config.LEARNING_RATE/10).minimize(self.loss))

    def weight_variable(self, shape, name='weights', group_id=0):
        """
            get weight variables
            :param shape: shape of the tensor to be returned
            :return: weight variable tensor
        """
        # initial = tf.truncated_normal(shape, stddev=0.1, name=name)
        # return tf.Variable(initial)
        initializer = tf.truncated_normal_initializer(stddev=0.1)
        var = tf.get_variable(name=name, shape=shape, initializer=initializer)
        tf.add_to_collection('l2_'+str(group_id), tf.nn.l2_loss(var))
        return var

    def bias_variable(self, shape, name='bias'):
        """
            get biases tensor
            :param shape: shape of the tensor to be returned
            :return: biases variable tensor
        """
        # initial = tf.constant(0.1, shape=shape, name=name)
        # return tf.Variable(initial)
        init = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=init)

    def conv_layer(self, x, W_shape, b_shape, name, stride=1, padding='SAME', re_use=None):
        """
            Convolutional layer for feature extraction
            :param x: input map
            :param W_shape: weights tensor
            :param b_shape: biases tensor
            :param name: name of the scope to be used
            :param padding: available padding
            :param re_use: if we reuse these variables
            :return: Feature Map for extraction
        """
        with tf.variable_scope(name, reuse=re_use) as scope:
            W = self.weight_variable(W_shape)
            b = self.bias_variable([b_shape])
            return tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding) + b)

    def leaky_relu(self, x, leak=0.2, name="lrelu"):
        """Leaky rectifier.
            Parameters
            ----------
            x : Tensor
                The tensor to apply the nonlinearity to.
            leak : float, optional
                Leakage parameter.
            name : str, optional
                Variable scope to use.
            Returns
            -------
            x : Tensor
                Output of the nonlinearity.
        """
        with tf.variable_scope(name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)

    def maxpool_layer(self, x, name='maxpool', kernel_size=2, stride=2, padding='SAME'):
        return tf.nn.max_pool_with_argmax(x, ksize=[1, kernel_size, kernel_size, 1], name=name,
                                          strides=[1, stride, stride, 1], padding=padding)

    def unravel_argmax(self, argmax, shape):
        output_list = []
        output_list.append(argmax // (shape[2] * shape[3]))
        output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
        return tf.stack(output_list)

    def maxunpool_layer(self, x, argmax):
        x_shape = tf.shape(x)
        output = tf.zeros([x_shape[1] * 2, x_shape[2] * 2, x_shape[3]])

        height = tf.shape(output)[0]
        width = tf.shape(output)[1]
        channels = tf.shape(output)[2]

        t1 = tf.to_int64(tf.range(channels))
        t1 = tf.tile(t1, [(width // 2) * (height // 2)])
        t1 = tf.reshape(t1, [-1, channels])
        t1 = tf.transpose(t1, perm=[1, 0])
        t1 = tf.reshape(t1, [channels, height // 2, width // 2, 1])

        t2 = tf.squeeze(argmax)
        t2 = tf.stack((t2[0], t2[1]), axis=0)
        t2 = tf.transpose(t2, perm=[3, 1, 2, 0])

        t = tf.concat(3, [t2, t1])
        indices = tf.reshape(t, [(height // 2) * (width // 2) * channels, 3])

        x1 = tf.squeeze(x)
        x1 = tf.reshape(x1, [-1, channels])
        x1 = tf.transpose(x1, perm=[1, 0])
        values = tf.reshape(x1, [-1])

        delta = tf.SparseTensor(indices, values, tf.to_int64(tf.shape(output)))
        return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_reorder(delta)), 0)

    def deconv_layer(self, x, W_shape, b_shape, name, stride=1, padding='SAME', re_use=None):
        """
            Deconvolutional layer for image reconstruction
            :param x: input map
            :param W_shape: weights tensor
            :param b_shape: biases tensor
            :param name: name of the scope to be used
            :param padding: available padding
            :param re_use: if we reuse these variables
            :return: Feature Map for reconstruction
        """
        with tf.variable_scope(name, reuse=re_use) as scope:
            W = self.weight_variable(W_shape)
            b = self.bias_variable([b_shape])
            x_shape = tf.shape(x)
            out_shape = tf.stack([x_shape[0], x_shape[1]*stride, x_shape[2]*stride, W_shape[2]])
            return self.leaky_relu(
                tf.add(
                    tf.nn.conv2d_transpose(
                        x, W, out_shape, [1, stride, stride, 1], padding=padding
                    ),
                b))
            # return conv_2d_transpose(incoming=x, nb_filter=b_shape, padding='same', activation='relu',
            #                          filter_size=[W_shape[0], W_shape[1]], output_shape=[out_shape[0], out_shape[1]],
            #                          strides=stride)

    def dropout_module(self, x, keep_prob, is_train):
        """ Apply a dropout layer. """
        return tf.cond(is_train, lambda: tf.nn.dropout(x, keep_prob), lambda: x)

    def load(self, sess, input_args):
        """ Load the trained model. """
        if input_args.debug_info >= 1:
            print("Loading model...")
        checkpoint = tf.train.get_checkpoint_state(input_args.pretrained_weights)
        if checkpoint is None:
            print("Error: No saved model found. Please train first.")
            sys.exit(0)
        self.saver.restore(sess, checkpoint.model_checkpoint_path)

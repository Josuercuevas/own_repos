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

import argparse
from basecode.model_builder import *
# from cityscape_dataloader.datahandler import *
from dataloader.datahandler import *
import tensorflow as tf

# ========================================argument parsing =============================================
"""
    Parsing arguments for training or testing
"""
def parse_args():
    """ Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Hierarchical model architecture')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver to be used for training (moment/adam) [adam]',
                        default='adam', type=str)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train [300]',
                        default=300, type=int)
    parser.add_argument('--batch_size', dest='batch_size',
                        help='number of samples to be processed by batch [4]',
                        default=4, type=int)
    parser.add_argument('--weights', dest='pretrained_weights',
                        help='initialize with pretrained model weights [none]',
                        default=None, type=str)
    parser.add_argument('--pretreined_path', dest='pretrained_model_path',
                        help='initialize with pretrained model weights [none]',
                        default=None, type=str)
    parser.add_argument('--save_model', dest='model_dir',
                        help='Directory to save the model [none]',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train/test on [datasets/]',
                        default='datasets/', type=str)
    parser.add_argument('--mode', dest='execution_mode',
                        help='mode in which the program to be executed (train/test/test_cocoapi) [train]',
                        default='train', type=str)
    parser.add_argument('--add_skip', dest='add_skyplayer',
                        help='add skip layers ResNet type (true/false) [false]',
                        default='false', type=str)
    parser.add_argument('--debug', dest='debug_info',
                        help='add skip layers ResNet type (0(none)/1(warnings)/2(debug)/3(verbose)) [3]',
                        default=3, type=int)
    parser.add_argument('--model', dest='model_name',
                        help='type of model to be used (helnet/squeeznet) [squeeznet]',
                        default='squeeznet', type=str)
    parser.add_argument('--n_classes', dest='number_classes',
                        help='number of classes to be contained in the dataset used for training, default 80 (COCO)',
                        default=80, type=int)
    parser.add_argument('--enable_tensorboard', dest='enable_tensorboard',
                        help='type of model to be used (false/true) [false]',
                        default='false', type=str)
    parser.add_argument('--resume_train', dest='resume_training',
                        help='resume training from last checkpoint (false/true) [false]',
                        default='false', type=str)

    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()

# ==================================== argument parsing =================================================


# ================================== coco api test ===================================================

"""
    is gonna test the database parsing for the model to be
    trained, is with the objective to check if the masks
    edges and classes are correct in the databse
"""
def perform_test_cocoapi(input_args):
    if input_args.debug_info >= 1:
        print("Testing COCO api for parsing the corresponding data from coco or cityscapes ...")

    data_loader = Loader()
    data_loader.test_coco_api()

# =================================== coco api test ==================================================

# ============================== model training invoker =======================================================
"""
    Here we will perform training of the model architecture to be
    used in our project
"""
def perform_training(input_args):
    if input_args.debug_info >= 1:
        print("Training module invoked...")

    with tf.Session() as session:
        if input_args.debug_info >= 1:
            print("Building architecture, and preparing data fetcher ...")

        data_loader = Loader(True)
        model = ModelNetwork()
        model.build(input_args=input_args)

        if input_args.enable_tensorboard=='true':
            tf.summary.scalar('Total_Loss_function', model.lossall)
            tf.summary.scalar('BBox_Loss_function', model.loss_bbox)
            tf.summary.scalar('Coverage_Loss_function', model.loss_cov)
            tf.summary.scalar('Edges_Loss_function', model.loss_edge)
            tf.summary.scalar('Segmentation_Loss_function', model.loss_seg)
            tf.summary.scalar('Intersection_over_Union', model.IoU)
            tf.summary.scalar('MAP', model.mAP)
            if input_args.debug_info >= 3:  # highly verbose and images to be dumped
                tf.summary.image('Input_image', model.transformed_input)
                tf.summary.image('GT_Mask', model.gt_mask_vis)
                tf.summary.image('Pred_Mask', model.pred_mask_vis)
                tf.summary.image('GT_Edges', model.pgtedge)
                tf.summary.image('Pred_Edges', model.pred_edges_vis)
                tf.summary.image('GT_coverage', model.gt_coverage_vis)
                tf.summary.image('Pred_coverage', model.pred_coverage_vis)
            tf.summary.histogram(name='Layer_Gradients', values=model.train_op)

        # To be used during training
        summary_merger = tf.summary.merge_all()

        Model_saver = tf.train.Saver()

        if input_args.debug_info >= 1:
            print("Setting parameters ...")
        # setting for training
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        # loading pretrained model
        if input_args.debug_info >= 1:
            print("Loading pretrained model body...")

        if input_args.resume_training=='true':
            print("Resuming training from last check point ....")
            model.load(sess=session, input_args=input_args, model_saver=Model_saver)
        else:
            model.load_pretrained(model_path=input_args.pretrained_weights, session=session, input_args=input_args)

        if input_args.debug_info >= 1:
            print("Saving graph for visualization...")
            graph = session.graph
            with graph.as_default():
                Train_writer = tf.summary.FileWriter(logdir='tfboard/model_training_progress', graph=graph)
                Train_writer.flush()

        model.train(sess=session, input_args=input_args, model_saver=Model_saver, datareader=data_loader,
                    summary_merger=summary_merger, summary_writer=Train_writer)

# =================================== model training invoker ==================================================


# ============================== using model to test input images =================================================
"""
    This is the testing process to be used in our model architecture
"""
def perform_testing(input_args):
    if input_args.debug_info >= 1:
        print("Testing module invoked ...")

    with tf.Session() as session:
        if input_args.debug_info >= 1:
            print("Building architecture, and preparing data fetcher ...")

        model = ModelNetwork()
        model.build(input_args=input_args)
        Model_saver = tf.train.Saver(max_to_keep=2)

        if input_args.debug_info >= 1:
            print("Setting parameters ...")
        # setting for training
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        # model.load_pretrained(model_path=input_args.pretrained_weights, session=session, input_args=input_args)
        model.load(sess=session, input_args=input_args, model_saver=Model_saver)
        if input_args.debug_info >= 1:
            print("Saving graph for visualization...")
            graph = session.graph
            with graph.as_default():
                Train_writer = tf.summary.FileWriter(logdir='tfboard/model_training_progress', graph=graph)
                Train_writer.flush()


        pathdata = "datasets/sample_tests"
        listfile = os.listdir(pathdata)
        i = 0
        while True:
            testdata = os.path.join(pathdata, listfile[i])
            model.test(session, testdata, input_args=input_args, model_saver=Model_saver,
                       save_at="datasets/sample_results/"+str(i))
            i += 1

# ================================ using model to test input images ==============================================

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    assigned = False
    for id_name in range(20):
        if args.gpu_id == id_name:
            print("assigning GPU %d for processing ..."%args.gpu_id)
            os.environ['CUDA_VISIBLE_DEVICES'] = str(id_name)
            assigned = True
            break

    if not assigned:
        print("Pure CPU assigned for processing ...")
        os.environ['CUDA_VISIBLE_DEVICES'] = ' '

    if args.execution_mode == 'train':
        perform_training(args)
    elif args.execution_mode == 'test':
        perform_testing(args)
    elif args.execution_mode == 'test_cocoapi':
        perform_test_cocoapi(args)
    else:
        print("No operation specified, exiting application ...")

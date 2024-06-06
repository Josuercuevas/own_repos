from configs.conf import (LOGFILES, DEBUG_LEVEL, LOGE, LOGI, RES_PATH)
import argparse
from time import time
import logging
from logging.handlers import RotatingFileHandler
from inference.sample_diffusion import SampleDiffusionUnconditional
from unittests.build_ldm import BuildTestLDM
from unittests.build_ae import BuildTestAE
from trainer.trainer_main import TrainerEngine


def parse_arguments():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
        
    parser = argparse.ArgumentParser(
        prog="Main script for training and testing LDM models",
        description="This script can be used to train and test LDM models. It can also be used to run unit-tests for the LDM model.")
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "--debug_level",
        dest="debug_level",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="Debug level [0: error, 1: warning, 2:info, 3: debug]",
        required=False,
    )
    parser.add_argument(
        "--operation",
        dest="operation",
        type=str,
        default="sample",
        choices=["train", "test_unconditional"],
        help="train/test model, sample an image(s), or generates a diffusion sequence [train, test_unconditional]",
        required=False,
    )
    parser.add_argument(
        "--logfile",
        dest="logfile",
        type=str,
        default=None,
        help="Logfile to be generated when executing the script, default outputs everything to stdout/stderr",
        required=False,
    )
    parser.add_argument(
        "--logdir",
        dest="logdir",
        type=str,
        default='logs',
        help="Path to resource files (models, images, etc...)",
        required=False,
    )
    parser.add_argument(
        "--run_unittests",
        dest="run_unittests",
        action="store_true",
        help="will execute the unit tests if provided in cmd",
        required=False,
    )
    parser.add_argument(
        "-ns",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=50000
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "-v",
        "--vanilla_sample",
        default=False,
        action='store_true',
        help="vanilla sampling (default option is DDIM sampling)?",
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the batch size to use when generating the samples or to train the model",
        default=10
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    return parser

def run_unittests(dst_path=RES_PATH, run_ae=False, run_diffusion=False):
    if run_ae:
        tester = BuildTestAE(modelcheckpoint=dst_path)
        tester.check_model()
    
    if run_diffusion:
        tester = BuildTestLDM(modelcheckpoint=dst_path)
        tester.check_model()

if __name__ == '__main__':
    parser = parse_arguments()
    args, unknown = parser.parse_known_args()

    starttime = time()

    # configure resource folder
    if args.logdir == "":
        resource_folder_path = RES_PATH
    else:
        resource_folder_path = args.logdir

    # config logger
    loglevel = DEBUG_LEVEL[args.debug_level]

    if args.logfile is not None:
        logging.basicConfig(format="%(asctime)s %(levelname)s PID_%(process)d %(message)s", 
                            level=loglevel, 
                            handlers=[RotatingFileHandler(args.logfile, maxBytes=2048, backupCount=5)]
                            )
    else:
        logging.basicConfig(stream=LOGFILES, level=loglevel, 
                            format="%(asctime)s %(levelname)s PID_%(process)d %(message)s")

    LOGI(f"Script has been configured as follows: \n{args}")

    # in case you want to run unit-tests
    if args.run_unittests:
        run_unittests(dst_path=resource_folder_path, run_ae=True, run_diffusion=True)
    
    success = True
    try:
        # here we perform any operation requested by the user
        if args.operation == "train":
            trainer = TrainerEngine(arg_parser=parser)
            trainer.run()
        elif args.operation == "test_unconditional":
            inferencer = SampleDiffusionUnconditional(arg_parser=parser)
            inferencer.get_samples()
        else:
            LOGE(f"{args.operation} operation is not supported")
    except Exception as e:
        LOGE(f"Exception occurred:\n{e}")
        success = False

    LOGI(f"Finished operation {args.operation} in {((time()-starttime))/60.0} mins with success={success}")
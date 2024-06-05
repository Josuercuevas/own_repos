from configs.conf import (LOGFILES, DEBUG_LEVEL, LOGE, LOGI, RES_PATH)
import argparse
from time import time
import logging
from logging.handlers import RotatingFileHandler
from unittests.sample_diffusion import SampleDiffusionUnconditional

def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="LDM model training & inference engines",
        description="Trains and Generates samples from trained model"
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
        choices=["train", "test"],
        help="train/test model, sample an image(s), or generates a diffusion sequence [train, test, sample, genseq]",
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
        default=None,
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
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-n",
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
    
    return parser

def run_unittests(dst_path=RES_PATH, run_ae=False, run_diffusion=False):
    # unet model
    raise NotImplementedError

if __name__ == '__main__':
    parser = parse_arguments()
    args = parser.parse_args()

    starttime = time()

    # configure resource folder
    if args.logdir is None:
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
    
    try:
        # here we perform any operation requested by the user
        if args.operation == "train":
            raise NotImplementedError(f"Training is not implemented yet")
        elif args.operation == "test":
            tester = SampleDiffusionUnconditional(arg_parser=parser)
            success = tester.get_samples()
        elif args.operation == "sample":
            raise NotImplementedError(f"Sampling is not implemented yet")
        elif args.operation == "genseq":
            raise NotImplementedError(f"Sequence generation is not implemented yet")
        else:
            LOGE(f"{args.operation} operation is not supported")
    except Exception as e:
        LOGE(f"Exception occurred:\n{e}")
        success = False

    LOGI(f"Finished operation {args.operation} in {((time()-starttime))/60.0} mins with success={success}")
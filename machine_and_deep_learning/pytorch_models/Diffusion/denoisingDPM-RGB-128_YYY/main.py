from unittests import (test_unet, test_diffusion)
from trainer.train_model import DiffusionModelTrainer
from sampler.sample_images import DiffusionModelSampler
from sampler.sample_sequence import DiffusionModelSequencer
from tester.test_model import DiffusionModelTester
import logging
from logging.handlers import RotatingFileHandler
from utils.configs import (LOGFILES, DEBUG_LEVEL, 
                           LOGE, LOGI, RES_PATH)
import argparse
from time import time

def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="DDPM model training & inference engines",
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
        choices=["train", "test", "sample", "genseq"],
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
        "--path2interfiles",
        dest="path2interfiles",
        type=str,
        default=None,
        help="Path to resource files (models, images, etc...)",
        required=True,
    )
    parser.add_argument(
        "--run_unittests",
        dest="run_unittests",
        action="store_true",
        help="will execute the unit tests if provided in cmd",
        required=False,
    )
    
    return parser.parse_args()

def run_unittests(dst_path=RES_PATH, run_ae=False, run_diffusion=False):
    # unet model
    if run_ae:
        resp = test_unet.TestAutoEncoderUnet(dst_path=dst_path)
        if not resp.success:
            LOGE(f"Unet based AutoEncoder FAILED!")
        else:
            LOGI(f"Unet based AutoEncoder PASSED!")
        
        LOGI("\n\n ============= NEW TEST ============= \n\n")
    
    if run_diffusion:
        resp = test_diffusion.TestGaussianDiffusion(dst_path=dst_path)
        if not resp.success:
            LOGE(f"Diffusion Model FAILED!")
        else:
            LOGI(f"Diffusion Model PASSED!")

if __name__ == '__main__':
    args = parse_arguments()

    starttime = time()

    # configure resource folder
    if args.path2interfiles is None:
        resource_folder_path = RES_PATH
    else:
        resource_folder_path = args.path2interfiles

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
    
    # here we perform any operation requested by the user
    if args.operation == "train":
        trainer = DiffusionModelTrainer()
        if trainer.success:
            trainer.fit()
    elif args.operation == "test":
        tester = DiffusionModelTester()
        if tester.success:
            tester.test_model_loss()
    elif args.operation == "sample":
        sample_generator = DiffusionModelSampler()
        if sample_generator.success:
            sample_generator.get_samples(smp_per_class=100)
    elif args.operation == "genseq":
        denoise_sequencer = DiffusionModelSequencer()
        if denoise_sequencer.success:
            denoise_sequencer.get_denoising_sequence()
    else:
        LOGE(f"{args.operation} operation is not supported")

    LOGI(f"Finished main script, time was: {((time()-starttime))/60.0} mins")
    
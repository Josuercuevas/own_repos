import logging
from sys import stdout
from torch.nn import functional as torchF

# general configs
LOGD = logging.debug
LOGI = logging.info
LOGW = logging.warning
LOGE = logging.error
DEBUG_LEVEL = {0: logging.ERROR, 1: logging.WARNING, 2: logging.INFO, 3: logging.DEBUG}

# by default logs are sent to stdout
LOGFILES = stdout

# supported activations
ACTIVATION_FUNCTIONS = {
    "relu": torchF.relu,
    "mish": torchF.mish,
    "silu": torchF.silu
}

# supported activations
SUPPORTED_LOSSES = {
    "l1_mae": torchF.l1_loss,
    "l2_mse": torchF.mse_loss
}

# supported noise schedules
SUPPORTED_NOISE_SCHEDULES = ["cosine", "linear"]

# default resources path
RES_PATH = 'resources'

PARAM_SETTING = {
    "learning_rate": 2e-4, # from paper (https://arxiv.org/pdf/2006.11239.pdf) for images with resolution below 256x256
    "batch_size": 64,  # depends on memory
    "iterations": 800000, # in the paper they trained for 800k iterations, for 1000 steps
    "checkpoint_rate": 500, # save checkpoint every 1000 iterations
    "model_snapshot": "models/latest-model.pth", # latest checkpoint snapshot
    "optimizer_snapshot": "optimizers/latest-optimizer.pth", # latest optimizer snapshot
    "iteration_snapshot": "optimizers/latest-iteration.txt", # latest iteration checkpoint
    "image_datapath": "datasets/cifar10_128", # path to images dataset
    "linear_schedule_low": 1e-4, # lower bound of Beta, appendix B in the paper
    "linear_schedule_high": 0.02, # upper bound of Beta, appendix B in the paper
    "beta_schedule": "linear", # default in paper https://arxiv.org/pdf/2006.11239.pdf
    "loss_type": "l2_mse",
    "use_labels": True, # if we want to use label conditioning   ##############################################################################################
    "initial_channels": 128,  ### make sure about this part, 256? maybe
    "channel_mults": (1, 2, 2, 2), # design of PixelCNN++ (https://github.com/openai/pixel-cnn)
    "num_timesteps": 1000, # appendix B in paper https://arxiv.org/pdf/2006.11239.pdf
    "num_res_blocks": 2, # appendix B in paper https://arxiv.org/pdf/2006.11239.pdf
    "time_emb_dim": 512, # appendix B in paper https://arxiv.org/pdf/2006.11239.pdf
    "norm": "group_norm",
    "dropout": 0.1, # appendix B in paper https://arxiv.org/pdf/2006.11239.pdf
    "activation": "silu", # appendix B in paper https://arxiv.org/pdf/2006.11239.pdf
    "attention_res": (1,), # appendix B in paper https://arxiv.org/pdf/2006.11239.pdf
    "ema_decay": 0.9999, # appendix B in paper https://arxiv.org/pdf/2006.11239.pdf
    "ema_update_rate": 1, # appendix B in paper https://arxiv.org/pdf/2006.11239.pdf
    "ema_start": 2000, # appendix B in paper https://arxiv.org/pdf/2006.11239.pdf
    "num_groups": 32, # appendix in paper https://arxiv.org/pdf/2006.11239.pdf
    "num_classes": 2, # for label conditioning  ############################################################################################################
    "img_channels": 64,   ####################################################################################################################################
    "img_size": (16, 16), # H and W of the image to be processed ############################################################################################
    "image_pad": 0,
    "dataparallelism": True, # to process on multiple devices
    "num_workers": 8, # number of workers for data loading
}
from os.path import exists
from utils.configs import (LOGE, LOGI, LOGW, RES_PATH,
                           PARAM_SETTING, ACTIVATION_FUNCTIONS,
                           SUPPORTED_LOSSES, SUPPORTED_NOISE_SCHEDULES)
from .diffusion import GaussianDiffusion
from .autoencoder import AutoEncoderUnet
from utils.misc import (get_noise_schedule)
import torch

class ModelCreator:
    def __init__(self):
        self.success = False
        self.configs = PARAM_SETTING
        self.activations = ACTIVATION_FUNCTIONS
        self.noises = SUPPORTED_NOISE_SCHEDULES
        self.losses = SUPPORTED_LOSSES
        self.resources = RES_PATH
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.unet_model = None
        self.diffusion_model = None
        self.optimizer = None
        self.dataparallel = False
        self.starting_iteration = 0
        self.best_train_loss = 1e12

        LOGW(f"Initializing the training engine with configs:\n{self.configs}, {self.device}")

        try:
            self.__create_ae_unet()
            self.__create_diffusion()
            self.__init_model_optimizer()
            self.success = True
        except Exception as e:
            LOGE(f"Error while initializing model training engine: {e}")
    
    def __create_ae_unet(self):
        LOGI("Creating AutoEncoder Unet model")

        self.unet_model = AutoEncoderUnet(
            img_channels=self.configs["img_channels"],
            initial_channels=self.configs["initial_channels"],
            channel_mults=self.configs["channel_mults"],
            num_res_blocks=self.configs["num_res_blocks"],
            time_emb_dim=self.configs["time_emb_dim"],
            norm=self.configs["norm"],
            dropout=self.configs["dropout"],
            activation=self.activations[self.configs["activation"]],
            attention_res=self.configs["attention_res"],
            num_classes=self.configs["num_classes"],
            num_groups=self.configs["num_groups"],
            image_pad=self.configs["image_pad"]
        )
    
    def __create_diffusion(self):
        LOGI("Creating diffusion model")

        if self.configs["beta_schedule"] not in self.noises:
            raise ValueError(f"Unsupported noise schedule: {self.configs['beta_schedule']}")

        # how betas are scheduled during training for DDPM
        if self.configs["beta_schedule"] == "cosine":
            # paper https://arxiv.org/pdf/2102.09672.pdf, to make samples
            # in late steps more relevant.
            sfactor = 0.008
            betas = get_noise_schedule(Tstep=self.configs["num_timesteps"], schedule=self.configs["beta_schedule"], 
                                        low=None, high=None, s_factor=sfactor)
        else:
            # scheduled used in the paper, appendix B (https://arxiv.org/pdf/2006.11239.pdf)
            low_val = (self.configs["linear_schedule_low"] * 1000) / self.configs["num_timesteps"]
            high_val = (self.configs["linear_schedule_high"] * 1000) / self.configs["num_timesteps"]
            betas = get_noise_schedule(Tstep=self.configs["num_timesteps"], schedule=self.configs["beta_schedule"], 
                                        low=low_val, high=high_val, s_factor=None)
        
        # now building the diffusion process.
        self.diffusion_model = GaussianDiffusion(
            model=self.unet_model, # unet used to reconstruct the noise added to the image
            img_size=self.configs["img_size"], # size of the input image
            img_channels=self.configs["img_channels"], # channels of the input image
            num_classes=self.configs["num_classes"], # number of classes to be used
            betas=betas, # noise scheduler
            loss_type=self.configs["loss_type"], # loss function
            ema_decay=self.configs["ema_decay"], # ema decay factor
            ema_start=self.configs["ema_start"], # point in which decay starts
            ema_update_rate=self.configs["ema_update_rate"] # update rate for ema
        )
    
    def __init_model_optimizer(self):
        LOGI("Initializing model and optimizer")
        # reload pretrained model if available and specified
        pretrained_model = f"{self.resources}/{self.configs['model_snapshot']}"
        if exists(pretrained_model):
            LOGI(f"There is a pretrained model in path: {pretrained_model}")
            self.diffusion_model.load_state_dict(torch.load(pretrained_model))
            LOGI("Done Loading Pretrained Model!")
        
        latest_iteration = f"{self.resources}/{self.configs['iteration_snapshot']}"
        if exists(latest_iteration):
            with open(latest_iteration, "rb") as iterfile:
                self.starting_iteration = int(iterfile.readline())
                self.best_train_loss = float(iterfile.readline())
                LOGI(f"We will restart treaining from iteration {self.starting_iteration} (best loss: {self.best_train_loss})")
        
        if self.configs["dataparallelism"] and (torch.cuda.device_count() > 1):
            # we can use multiple devices
            LOGI(f"Data parallelism available and activated")
            self.diffusion_model = torch.nn.DataParallel(self.diffusion_model)
            self.dataparallel = True
        
        # move the model to available accelerator
        self.diffusion_model.to(self.device)

        # optimizer to be used, defined in paper https://arxiv.org/pdf/2006.11239.pdf
        # this lead to best performance
        if self.dataparallel:
            self.optimizer = torch.optim.Adam(self.diffusion_model.module.parameters(), lr=self.configs["learning_rate"])
        else:
            self.optimizer = torch.optim.Adam(self.diffusion_model.parameters(), lr=self.configs["learning_rate"])
        
        # load optimizer settings of the model above if availabe and specified
        optimizer_snapshot = f"{self.resources}/{self.configs['optimizer_snapshot']}"
        if exists(optimizer_snapshot):
            LOGI(f"There is a pretrained optimizer in path: {optimizer_snapshot}")
            self.optimizer.load_state_dict(torch.load(optimizer_snapshot))
            LOGI(f"Done Loading Pretrained Optimizer, and looks like this: \n{self.optimizer}")
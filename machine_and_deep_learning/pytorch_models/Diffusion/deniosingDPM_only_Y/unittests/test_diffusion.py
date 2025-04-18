from models.autoencoder import AutoEncoderUnet
from models.diffusion import GaussianDiffusion
from utils.configs import (LOGE, LOGW, LOGI, 
                           ACTIVATION_FUNCTIONS)
from utils.misc import get_noise_schedule
import torch
from torchviz import make_dot

class TestGaussianDiffusion:
    """
    Unit test to determine if the GaussianDiffusion model can be created
    successfully and some dummy data are forward passed
    """
    def __init__(self, dst_path):
        self.img_channels=3
        self.initial_channels=128
        self.channel_mults=(1, 2, 2, 2)
        self.num_res_blocks = 2
        self.time_emb_dim=128 * 4
        self.time_emb_scale = 1
        self.norm="group_norm"
        self.dropout=0.1
        self.activation="silu"
        self.attention_res=(1,)
        self.num_classes=10
        self.num_groups = 32
        self.image_pad=0
        self.num_timesteps = 1000
        self.dst_path = dst_path
        self.img_size = (64, 64)
        self.loss_type = "l2_mse"
        self.ema_decay = 0.9999
        self.ema_update_rate = 1
        self.ema_start = 2000
        self.beta_schedule = "linear"
        self.schedule_low=1e-4
        self.schedule_high=0.02
        self.success = False
        self.activations = ACTIVATION_FUNCTIONS

        self.__run()
    
    def __run(self):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        dummy_input = torch.randn((1,3,64,64), device=device)
        LOGI(dummy_input)
        yclass = torch.randint(1, self.num_classes, (1,), device=device)
        LOGI(yclass)

        LOGW("Creating Diffusion Model")
        try:
            model = AutoEncoderUnet(
                img_channels=self.img_channels,
                initial_channels=self.initial_channels,
                channel_mults=self.channel_mults,
                num_res_blocks=self.num_res_blocks,
                time_emb_dim=self.time_emb_dim,
                norm=self.norm,
                dropout=self.dropout,
                activation=self.activations[self.activation],
                attention_res=self.attention_res,
                num_classes=self.num_classes,
                num_groups=self.num_groups,
                image_pad=self.image_pad
            )

            # how betas are scheduled during training for DDPM
            if self.beta_schedule == "cosine":
                # paper https://arxiv.org/pdf/2102.09672.pdf, to make samples
                # in late steps more relevant.
                sfactor = 0.008
                betas = get_noise_schedule(Tstep=self.num_timesteps, schedule=self.beta_schedule, 
                                            low=None, high=None, s_factor=sfactor)
            else:
                # scheduled used in the paper, appendix B (https://arxiv.org/pdf/2006.11239.pdf)
                low_val = self.schedule_low * 1000 / self.num_timesteps
                high_val = self.schedule_high * 1000 / self.num_timesteps
                betas = get_noise_schedule(Tstep=self.num_timesteps, schedule=self.beta_schedule, 
                                            low=low_val, high=high_val, s_factor=None)

            # now building the diffusion process.
            diffusion_model = GaussianDiffusion(
                model=model, # unet used to reconstruct the noise added to the image
                img_size=self.img_size, # size of the input image
                img_channels=self.img_channels, # channels of the input image
                num_classes=self.num_classes, # number of classes to be used
                betas=betas, # noise scheduler
                loss_type=self.loss_type, # loss function
                ema_decay=self.ema_decay, # ema decay factor
                ema_start=self.ema_start, # point in which decay starts
                ema_update_rate=self.ema_update_rate # update rate for ema
            )
        except Exception as e:
            LOGE(f"Test Diffusion Model creation did not pass, error: {str(e)}")
        
        LOGI(f"The model looks like this: {diffusion_model}")

        LOGW("Forward Pass")
        try:
            loss_value = diffusion_model(x=dummy_input, yclass=yclass)
        except Exception as e:
            LOGE(f"Test Diffusion Model inference did not pass, error: {str(e)}")

        LOGW(f"Loss from the current model is: \n{loss_value}")

        LOGW("Creating image")
        make_dot(loss_value, params=dict(list(model.named_parameters()))).render(f"{self.dst_path}/doc/diffusion_visualizer.dot", format="png")

        self.success = True
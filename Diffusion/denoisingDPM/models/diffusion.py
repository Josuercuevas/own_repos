import numpy as np
import torch
from torch import nn
from functools import partial
from copy import deepcopy
from utils.exp_moving_average import ExpMovingAverage
from utils.misc import (extract_noise_factor, get_loss_function)
from utils.configs import (SUPPORTED_LOSSES, LOGD)

class GaussianDiffusion(nn.Module):
    """
    Builds Gaussian Diffusion Model. Forwarding through the module returns the tensor we need to denoise
    to reverse the diffusion process. After the denoising is done the module return a scalar loss tensor.

    Input:
        x: tensor of shape (B, img_channels, H, W)
        yclass (optional): tensor of shape (B)
        model: model which estimates diffusion noise, usually Unet
        img_size: image size tuple (H, W)
        img_channels: number of image channels
        betas: diffusion betas to be used for noise/denoise
        loss_type: loss function to be used in the diffusion process, "l1_mae" or "l2_mse"
        ema_decay (optional): model weights exponential moving average decay
        ema_start (optional): number of steps to warmup before implementing EMA
        ema_update_rate (optional): number of steps before each update to EMA
    Output:
        scalar loss tensor after deffusion/denoising
    """
    def __init__(self, model, img_size, img_channels, num_classes, betas, loss_type="l2_mse", ema_decay=0.9999,
                 ema_start=5000, ema_update_rate=1):
        LOGD("Initializing Diffusion Model Class")

        super().__init__()

        # AutoEncoder model, usually unet
        self.model = model
        # As this is not going to change in the same way as the main model
        # this is the one to be used for inference
        self.ema_model = deepcopy(model)

        # Setting up EMA
        self.ema = ExpMovingAverage(ema_decay)
        self.ema_decay = ema_decay
        self.ema_start = ema_start
        self.ema_update_rate = ema_update_rate
        self.step = 0 # initialization

        self.img_size = img_size
        self.img_channels = img_channels
        self.num_classes = num_classes

        if loss_type not in SUPPORTED_LOSSES:
            raise ValueError(f"{loss_type} loss function is not supported")

        self.loss_type = loss_type
        # steps to be used for training the model
        self.num_timesteps = len(betas)

        # section 2 above the equation 4 in https://arxiv.org/pdf/2006.11239.pdf
        alphas = 1.0 - betas # from diffusion models this is how is computed
        alphas_bar = np.cumprod(alphas) # this is the cummulative product for alpha

        # for convinience as we need to do this a lot,
        # converts input to torch tensor of float32
        to_torch = partial(torch.tensor, dtype=torch.float32)

        LOGD("Creating register buffers for constant values")
        # these are parameters that are not going to change, but we need them during training
        # by default it is persistent which makes them available in the state_dict for the model/module
        self.register_buffer("betas", to_torch(betas)) # variance schedule
        self.register_buffer("alphas", to_torch(alphas)) # noise weights
        self.register_buffer("alphas_bar", to_torch(alphas_bar)) # cummulative noise weights

        # refer to equation 4 in https://arxiv.org/pdf/2006.11239.pdf
        self.register_buffer("sqrt_alphas_bar", to_torch(np.sqrt(alphas_bar)))
        self.register_buffer("sqrt_one_minus_alphas_bar", to_torch(np.sqrt(1 - alphas_bar)))

        # for reversing the noising process, predicting X(t-1)
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))
        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_bar)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))
    
    def update_ema(self):
        # updating the EMA for the model every ema_update_rate
        self.step += 1
        if self.step % self.ema_update_rate == 0:
            if self.step < self.ema_start:
                # if we are still in the warmup phase, we use the model as is
                self.ema_model.load_state_dict(self.model.state_dict())
            else:
                # if the warmup phase is done, we update EMA
                self.ema.update_model_average(self.ema_model, self.model)
    
    @torch.no_grad()
    def remove_noise(self, x, tstep, yclass, use_ema=True):
        LOGD(f"Removing noise from timestep {tstep}")

        # removing the predicted noise from X(t) to get X(t-1), Algorithm 2 (sampling)
        # from paper https://arxiv.org/pdf/2006.11239.pdf
        if use_ema:
            # this model was the one trained with exponential moving average
            pred_noise = self.ema_model(x, tstep, yclass)
        else:
            # model trained without EMA, basically what we get at each iteration
            pred_noise = self.model(x, tstep, yclass)
        
        # here is where we denoise the image, take a look of equation 11 in 
        # https://arxiv.org/pdf/2006.11239.pdf
        scale_factor = extract_noise_factor(self.remove_noise_coeff, tstep, x.shape)
        denoised = x - (scale_factor * pred_noise)
        scale_factor = extract_noise_factor(self.reciprocal_sqrt_alphas, tstep, x.shape)
        denoised_scaled = denoised * scale_factor
        
        return denoised_scaled

    # this method samples and generates the final image
    # but also can generate a sequence for visualization
    @torch.no_grad()
    def sample(self, batch_size, device, yclass=None, use_ema=True, gen_seq=False):
        LOGD("Generating some samples/sequences for visualization purposes")

        if yclass is not None and batch_size != len(yclass):
            raise ValueError("Batch size and length of yclass are different")

        # This is the starting point in the denoising process, X(T)
        LOGD("Getting noise samples")
        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)

        if gen_seq:
            # we want to keep track of all the steps for later visualization
            diffusion_sequence = [x.cpu().detach()]
        
        # we run through all the steps, which is specified during training
        # in reverse order starting from T-1 -> -1
        LOGD("Generating Image/Sequence")
        for t in range(self.num_timesteps - 1, -1, -1):
            # same step for all the images to be generated in the batch/samples
            t_batch = torch.tensor([t], device=device).repeat(batch_size)

            # running denoising process, basically predicting the noise and removing it
            # from the image generated above
            x = self.remove_noise(x, t_batch, yclass, use_ema)

            # noise added after denoising, based on paper algorithm 2 (inference)
            # sigma(t) * Z -> https://arxiv.org/pdf/2006.11239.pdf
            if t > 0:
                # this is going to be reused again until we reached t = 0
                x += extract_noise_factor(self.sigma, t_batch, x.shape) * torch.randn_like(x)
            
            if gen_seq:
                diffusion_sequence.append(x.cpu().detach())
        
        if gen_seq:
            return diffusion_sequence
        else:
            return x.cpu().detach()
    
    def add_noise_x(self, x, tstep, noise):
        LOGD(f"Adding noise to timestep {tstep}")

        # based on the paper this is how we add noise to the model, basically keeping "sqrt_alphas_bar"
        # from the original image, and adding "sqrt_one_minus_alphas_bar" noise to it.
        # check algorigm 1 (training) from https://arxiv.org/pdf/2006.11239.pdf
        noisy_x = extract_noise_factor(self.sqrt_alphas_bar, tstep, x.shape) * x
        noisy_x += extract_noise_factor(self.sqrt_one_minus_alphas_bar, tstep, x.shape) * noise
        return noisy_x

    def compute_losses(self, x, tstep, yclass):
        LOGD(f"computing loss for steps: {tstep}")

        # this is the heart of the diffusion process, this is the noise to be reconstructed at each step
        noise = torch.randn_like(x)

        # this is where noise is added to the input
        noisy_x = self.add_noise_x(x, tstep, noise)

        # this is the reconstructed noise from Unet
        pred_noise = self.model(noisy_x, tstep, yclass)

        # this is where noise reconstruction is optimized
        resulting_loss = get_loss_function(loss_type=self.loss_type, estimated_noise=pred_noise,
                                           gt_noise=noise)
        
        return resulting_loss

    def forward(self, x, yclass=None):
        b, c, h, w = x.shape
        device = x.device

        # to make sure we don't screw up image input sizes.
        if h != self.img_size[0]:
            raise ValueError(f"image height ({h}) is different from diffusion params {self.img_size[0]})")
        if w != self.img_size[1]:
            raise ValueError(f"image width ({w}) is different from diffusion params {self.img_size[1]})")
        
        # this is to generate random steps from [0, 1000]-steps 
        # for 800k iterations which is the value used in the model
        # appending B in https://arxiv.org/pdf/2006.11239.pdf
        tstep = torch.randint(0, self.num_timesteps, (b,), device=device)

        return self.compute_losses(x, tstep, yclass)
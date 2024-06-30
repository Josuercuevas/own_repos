from torch import nn
import torchvision
from .configs import (SUPPORTED_LOSSES, LOGE, LOGI,
                     SUPPORTED_NOISE_SCHEDULES)
import numpy as np

def get_norm(norm, num_channels, num_groups):
    """
    Just a function to determine which normalization is to be used
    """
    if norm == "group_norm":
        return nn.GroupNorm(num_groups, num_channels)
    elif norm == "batch_norm":
        return nn.BatchNorm2d(num_channels)
    elif norm == "intance_norm":
        return nn.InstanceNorm2d(num_channels, affine=True)
    elif norm is None:
        return nn.Identity()
    else:
        raise ValueError(f"Normalization Type ({norm}) is not supported")

def get_loss_function(loss_type, estimated_noise, gt_noise):
    # this is where noise reconstruction is optimized
    loss_func = SUPPORTED_LOSSES[loss_type]
    if loss_type == "l1_mae":
        return loss_func(estimated_noise, gt_noise)
    elif loss_type == "l2_mse":
        return loss_func(estimated_noise, gt_noise)
    else:
        LOGE(f"{loss_type} loss function is not supported!")
    
    return None


def extract_noise_factor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    # the output is a noise tensor of shape [b, 1, 1, 1],
    # as each sample in the batch will have specific noise to use
    # depending on the t-step used
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# Generates the noise schedule
def get_noise_schedule(Tstep, low, high, schedule="linear", s_factor=0.008):
    LOGI(f"Generating Noise schedule in the form of a {schedule} function")
    if schedule not in SUPPORTED_NOISE_SCHEDULES:
        LOGE(f"Noise schedule {schedule} is not supported")
        return None
    
    if schedule == "cosine":
        # figure 3 and 4 in paper https://arxiv.org/pdf/2102.09672.pdf
        # adds noise more slowly so last steps are still relevant for training
        def get_factor(t, T):
            return np.cos((((t / T) + s_factor) / (1 + s_factor)) * (np.pi / 2)) ** 2
        
        alphas = []
        f0 = get_factor(0, Tstep)

        for t in range(Tstep + 1):
            alphas.append(get_factor(t, Tstep) / f0)
        
        betas = []

        for t in range(1, Tstep + 1):
            # make sure we dont go over 0.999 otherwise training becomes unstable
            betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))
        
        return np.array(betas)
    elif schedule == "linear":
        # noise addition to the samples may make some late samples in the training
        # less relevant as they don't really contribute that much
        return np.linspace(low, high, Tstep)

class RescalePixels(object):
    def __call__(self, sample):
        return 2 * sample - 1
        
def get_transform(img_size):
    # scales the pixels form [-1, +1] which is best for training stability of
    # AutoEncoder models
    return torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]#,
         # torchvision.transforms.Resize(size=img_size, antialias=True),
        #  RescalePixels()]
    )

def infinite_loader(dataloader):
    """
    https://github.com/lucidrains/denoising-diffusion-pytorch/
    """
    while True:
        for samples in dataloader:
            yield samples

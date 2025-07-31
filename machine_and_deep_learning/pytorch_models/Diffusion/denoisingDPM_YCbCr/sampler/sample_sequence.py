from models.initializer import ModelCreator
from utils.configs import (LOGE, LOGI, LOGD)
from PIL import Image
import torch
import numpy as np
import time


MAX_SAMPLES_ROW = None

class DiffusionModelSequencer:
    def __init__(self):
        self.success = False
        try:
            model_generated = ModelCreator()
            if not model_generated.success:
                raise Exception("Error initializing model, aborting training")
            
            # getting all the information from the model generated
            self.diffusion_model = model_generated.diffusion_model
            self.dataparallel = model_generated.dataparallel
            self.configs = model_generated.configs
            self.resources = model_generated.resources
            self.device = model_generated.device

            # mark as successful initialization
            self.success = True
        except Exception as e:
            LOGE(f"Error while initializing model training engine: {e}")
    
    def get_denoising_sequence(self):
        LOGI("Generating denoising sequence")
        if self.configs["use_labels"]:
            MAX_SAMPLES_ROW = self.configs["num_classes"]
            if self.dataparallel:
                samples = self.diffusion_model.module.sample(batch_size=MAX_SAMPLES_ROW,
                                                                device=self.device,
                                                                use_ema=True,
                                                                gen_seq=True,
                                                                yclass=torch.arange(start=0, 
                                                                                    end=self.configs["num_classes"], step=1,
                                                                                    device=self.device)
                                                            )
            else:
                samples = self.diffusion_model.sample(batch_size=MAX_SAMPLES_ROW,
                                                        device=self.device,
                                                        use_ema=True,
                                                        gen_seq=True,
                                                        yclass=torch.arange(start=0, 
                                                                            end=self.configs["num_classes"], step=1,
                                                                            device=self.device)
                                                    )
        else:
            MAX_SAMPLES_ROW = 10
            if self.dataparallel:
                samples = self.diffusion_model.module.sample(batch_size=MAX_SAMPLES_ROW,
                                                                device=self.device, use_ema=True,
                                                                gen_seq=True)
            else:
                samples = self.diffusion_model.sample(batch_size=MAX_SAMPLES_ROW, device=self.device,
                                                        use_ema=True, gen_seq=True)
        
        LOGI("combining images to GIF")
        mosaic_image = []
        for idx2, row_data_sample in enumerate(samples):
            if idx2 % 10 != 0:
                continue

            # samples generated, and undo the normalization from [-1, 1] so now it will be from [0, 1]
            # then channels are moved at the end for reconstruction/visualization
            h, w = self.configs["img_size"]
            new_im = Image.new('RGB', (w*MAX_SAMPLES_ROW, h))
            if idx2 == 0:
                LOGI(f"Horizontal concatenated image is of size: {new_im.size}")
            row_data_sample = ((row_data_sample + 1) / 2).clip(0, 1).permute(0, 2, 3, 1).numpy()
            if idx2 == 0:
                LOGI(f"Set of images to be dumped are of shape: {row_data_sample.shape}")
            x_offset = 0
            for idx in range(row_data_sample.shape[0]):
                im = Image.fromarray(((row_data_sample[idx]*255).astype(np.uint8)).clip(0, 255))
                if idx2 == 0:
                    LOGD(f"Image converted is of shape: {im.size}")
                new_im.paste(im, (x_offset,0))
                x_offset += im.size[0]
            mosaic_image.append(new_im)
        mosaic_image[0].save(f"{self.resources}/doc/denoising_sequence_{time.time()}.gif", save_all=True,
                             append_images=mosaic_image[1:], fps=33, loop=0)
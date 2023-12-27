from models.initializer import ModelCreator
from utils.configs import (LOGE, LOGI, LOGD)
from PIL import Image
import torch
import numpy as np


class DiffusionModelSampler:
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
    
    def get_samples(self, smp_per_class=1):
        LOGI("Generating some samples")
        if self.configs["use_labels"]:
            n_classes_2use = self.configs["num_classes"]
            total_images = n_classes_2use * smp_per_class
            if total_images > self.configs["num_classes"]:
                start_smp = 0
                end_smp = self.configs["num_classes"]
            else:
                start_smp = 0
                end_smp = total_images
        else:
            end_smp = 0
            total_images = self.configs["num_classes"]
        
        mosaic_image = []
        with torch.no_grad():
            if self.dataparallel:
                self.diffusion_model.module.eval() # for model only
            self.diffusion_model.eval() # for dataparallel module

            while end_smp <= total_images:
                LOGI(f"Generating {end_smp-start_smp}-samples with classes from [{0}, {n_classes_2use})")

                if self.configs["use_labels"]:
                    if self.dataparallel:
                        samples = self.diffusion_model.module.sample(batch_size=(end_smp-start_smp),
                                                                        device=self.device,
                                                                        use_ema=True,
                                                                        gen_seq=False,
                                                                        yclass=torch.arange(start=0, 
                                                                                            end=n_classes_2use, step=1,
                                                                                            device=self.device)
                                                                    )
                    else:
                        samples = self.diffusion_model.sample(batch_size=(end_smp-start_smp),
                                                                device=self.device,
                                                                use_ema=True,
                                                                gen_seq=False,
                                                                yclass=torch.arange(start=0, 
                                                                                    end=n_classes_2use, step=1,
                                                                                    device=self.device)
                                                            )
                else:
                    if self.dataparallel:
                        samples = self.diffusion_model.module.sample(batch_size=(end_smp-start_smp),
                                                                        device=self.device, use_ema=True,
                                                                        gen_seq=False)
                    else:
                        samples = self.diffusion_model.sample(batch_size=(end_smp-start_smp), device=self.device,
                                                                use_ema=True, gen_seq=False)
            
                LOGI("Dumping some images for later visualization")
                # samples generated, and undo the normalization from [-1, 1] so now it will be from [0, 1]
                # then channels are moved at the end for reconstruction/visualization
                h, w = self.configs["img_size"]
                new_im = Image.new('RGB', (w*(end_smp-start_smp), h))
                LOGI(f"Horizontal concatenated image is of size: {new_im.size}")
                samples = ((samples + 1) / 2).clip(0, 1).permute(0, 2, 3, 1).numpy()
                LOGI(f"Set of images to be dumped are of shape: {samples.shape}")
                x_offset = 0
                for idx in range(samples.shape[0]):
                    im = Image.fromarray(((samples[idx]*255).astype(np.uint8)).clip(0, 255))
                    LOGD(f"Image converted is of shape: {im.size}")
                    new_im.paste(im, (x_offset,0))
                    x_offset += im.size[0]
                mosaic_image.append(new_im)

                # shift things for next iteration
                start_smp = end_smp
                end_smp += self.configs["num_classes"]
        
        w, h = mosaic_image[0].size
        new_im = Image.new('RGB', (w, h*len(mosaic_image)))
        LOGI(f"Vertically concatenated image is of size: {new_im.size}")
        y_offset = 0
        for img_row in mosaic_image:
            new_im.paste(img_row, (0,y_offset))
            y_offset += im.size[1]
        
        new_im.save(f"{self.resources}/doc/generated_samples.png")
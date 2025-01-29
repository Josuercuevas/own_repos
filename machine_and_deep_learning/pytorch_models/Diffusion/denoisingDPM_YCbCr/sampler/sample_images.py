from models.initializer import ModelCreator
from utils.configs import (LOGE, LOGI, LOGD)
from PIL import Image
import torch
import numpy as np
import cv2
from sampler.dct_to_rgb_converter import to_img


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
            end_smp = 24
            start_smp = 0
            # total_images = self.configs["num_classes"]
            total_images = 24
        
        mosaic_image = []
        img_counter = 0
        with torch.no_grad():
            if self.dataparallel:
                self.diffusion_model.module.eval() # for model only
            self.diffusion_model.eval() # for dataparallel module

            while end_smp <= total_images:
                print("while loop....")
                # LOGI(f"Generating {end_smp-start_smp}-samples with classes from [{0}, {n_classes_2use})")

                if self.configs["use_labels"]:
                    print("why using labels??")
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
                samples = samples.numpy()

                '''
                The code below (lines 96 - 121) was modified to handle the DCT outputs
                '''
                # Save samples as numpy array (individually)              
                print("saving generated images....", samples.shape)
                np.save(f"{self.resources}/doc/generated_images_{img_counter}.npy", samples, allow_pickle=True)
                img_counter += 1
                
                # Convert to RGB image and accumulate into mosaic list
                for idx in range(samples.shape[0]):
                    im = to_img(samples[idx])
                    mosaic_image.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                
                # shift things for next iteration
                start_smp = end_smp
                end_smp += self.configs["num_classes"]

        print("DONE!")

        # Convert the mosaic list into a horizontal mosaic array
        concatenated_image = np.concatenate(mosaic_image, axis=1) 

        # Convert array to image
        image_to_save = Image.fromarray(concatenated_image)

        # Save the image
        image_to_save.save(f"{self.resources}/doc/generated_samples.png")

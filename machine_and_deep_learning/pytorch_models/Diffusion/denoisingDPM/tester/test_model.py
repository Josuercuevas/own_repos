from utils.misc import (get_transform, infinite_loader)
from models.initializer import ModelCreator
from utils.configs import (LOGE, LOGI, LOGW, LOGD)
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from time import time

DEBUG_IMAGES = False


class DiffusionModelTester:
    def __init__(self):
        self.success = False
        try:
            model_generated = ModelCreator()
            if not model_generated.success:
                raise Exception("Error initializing model, aborting training")
            
            # getting all the information from the model generated
            self.best_train_loss = model_generated.best_train_loss
            self.starting_iteration = model_generated.starting_iteration
            self.diffusion_model = model_generated.diffusion_model
            self.dataparallel = model_generated.dataparallel
            self.configs = model_generated.configs
            self.losses = model_generated.losses
            self.resources = model_generated.resources
            self.device = model_generated.device
            self.optimizer = model_generated.optimizer

            # taking case of the dataloader
            self.__init_dataloader()

            # mark as successful initialization
            self.success = True
        except Exception as e:
            LOGE(f"Error while initializing model training engine: {e}")

    def __init_dataloader(self):
        LOGI("Initializing dataloaders")
        # iterator for testing set, remember things are normalized from [0, 1] when using torchvision
        test_dataset = datasets.ImageFolder(
            root=f"{self.resources}/{self.configs['image_datapath']}/test/",
            transform=get_transform(img_size=self.configs["img_size"]), # normalize things between [-1, 1]
        )

        if DEBUG_IMAGES:
            # debug images to make sure we are dealing with proper values
            for x, y in test_dataset:
                LOGI(f"Testing Image shape: {x.shape}, Labels are: {y}")
                LOGI(f"Testing Image pixels values range [{torch.min(x)}, {torch.max(x)}]")
                x = ((x + 1) / 2).clip(0, 1).permute(1, 2, 0).numpy()
                x = Image.fromarray(((x*255).astype(np.uint8)).clip(0, 255))
                x.show()
                break

        # no need to cycle it as is done once, during testing
        self.test_loader = DataLoader(test_dataset, batch_size=self.configs["batch_size"], drop_last=True,
                                num_workers=self.configs["num_workers"])
    
    def test_model_loss(self):
        LOGI("Starting Diffusion Model Testing")

        try:
            # do some test and check performance
            LOGI("Running the model in testing mode")
            test_loss = 0
            start_time = time()
            with torch.no_grad():
                if self.dataparallel:
                    self.diffusion_model.module.eval() # for model only
                self.diffusion_model.eval() # for dataparallel module

                LOGI("Forward passing through diffusion process")
                for x, y in self.test_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)

                    if self.configs["use_labels"]:
                        output_loss = self.diffusion_model(x, y)
                    else:
                        output_loss = self.diffusion_model(x)
                    
                    if self.dataparallel:
                        output_loss = output_loss.mean()

                    test_loss += output_loss.item()
            
            test_loss /= len(self.test_loader)
            iter_time_sec = float(time() - start_time)
            LOGW(f"TestLoss-{test_loss} (completed in {iter_time_sec} seconds)")
        except KeyboardInterrupt:
            LOGW("Keyboard interrupt, stopping training!")
        
        LOGI("Model Testing has completed successfully")
        return True
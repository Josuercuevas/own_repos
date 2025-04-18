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
MAX_BATCH_VISUALIZE = 10

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/training_0105")

class DiffusionModelTrainer:
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
        # iterator for training set, remember things are normalized from [0, 1] when using torchvision
        train_dataset = datasets.ImageFolder(
            root=f"{self.resources}/{self.configs['image_datapath']}/train/",
            transform=get_transform(img_size=self.configs["img_size"]), # normalize things between [-1, 1]
        )

        if DEBUG_IMAGES:
            # debug images to make sure we are dealing with proper values
            for x, y in train_dataset:
                LOGI(f"Training Image shape: {x.shape}, Labels are: {y}")
                LOGI(f"Training Image pixels values range [{torch.min(x)}, {torch.max(x)}]")
                x = ((x + 1) / 2).clip(0, 1).permute(1, 2, 0).numpy()
                x = Image.fromarray(((x*255).astype(np.uint8)).clip(0, 255))
                x.show()
                break

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

        # function that loads the dataset for training, and rollsback if needed
        # for diffusion models we don't train epochs but steps!
        self.train_loader = infinite_loader(DataLoader(train_dataset, batch_size=self.configs["batch_size"],
                                                shuffle=True, drop_last=True, 
                                                num_workers=self.configs["num_workers"])
                                        )

        # no need to cycle it as is done once, during testing
        self.test_loader = DataLoader(test_dataset, batch_size=self.configs["batch_size"], drop_last=True,
                                num_workers=self.configs["num_workers"])

    def fit(self):
        LOGI("Starting Diffusion Model Training")

        try:
            train_loss = 0.0
            start_time = time()
            for iteration in range(self.starting_iteration+1, self.configs["iterations"]+1):
                if iteration % self.configs["checkpoint_rate"] == 0:
                    LOGI(f"Running Iteration-{iteration}")
                
                if self.dataparallel:
                    self.diffusion_model.module.train() # for model only
                self.diffusion_model.train() # for dataparallel module

                # gets the images normalized between [-1, 1] and the label for each image
                if iteration % self.configs["checkpoint_rate"] == 0:
                    LOGI("getting new data samples")

                LOGD(f"Getting samples for iteration-{iteration}")
                x, y = next(self.train_loader)
                x = x.to(self.device).float()
                y = y.to(self.device)
                if iteration == 1:
                    if self.configs["use_labels"]:
                        writer.add_graph(self.diffusion_model, (x, y))
                    else:
                        writer.add_graph(self.diffusion_model, x)
                    writer.close()
                LOGD(f"shape = {x.shape}, min = {torch.min(x)}, max = {torch.max(x)}")

                # send to accelator if configured
                if iteration % self.configs["checkpoint_rate"] == 0:
                    LOGI(f"Sending samples to {self.device}")
                    LOGI(f'\nx size = {x.size()}, xmin {torch.min(x)}, xmax {torch.max(x)}')
                    LOGI(f'y size = {x.size()}, ymin {torch.min(y)}, ymax {torch.max(y)}')

                # if labels are to used, then we feed them to create the embeddings 
                # and train the biases
                if iteration % self.configs["checkpoint_rate"] == 0:
                    LOGI("Forward passing through diffusion process")

                if self.configs["use_labels"]:
                    LOGD(x.shape, y.shape)
                    LOGD(type(x))
                    LOGD(y)
                    output_loss = self.diffusion_model(x, y)
                else:
                    output_loss = self.diffusion_model(x)
                
                if self.dataparallel:
                    output_loss = output_loss.mean()
                
                # compute loss during training
                if iteration % self.configs["checkpoint_rate"] == 0:
                    LOGI(f"output loss looks like: {output_loss}")
                train_loss += output_loss.item()

                if iteration % self.configs["checkpoint_rate"] == 0:
                    LOGI("Performing Gradient optimization")

                # make sure everytihing is cleaned up
                self.optimizer.zero_grad()
                # backpropagate the loss to the model
                output_loss.backward()
                # update the model weights
                self.optimizer.step()

                writer.add_scalar('Training Loss', output_loss.item(), iteration)

                # now we update things for EMA
                if iteration % self.configs["checkpoint_rate"] == 0:
                    LOGI("Updating EMA weights")

                if self.dataparallel:
                    self.diffusion_model.module.update_ema()
                else:
                    self.diffusion_model.update_ema()

                # now check if we need to save things up
                if iteration % self.configs["checkpoint_rate"] == 0:
                    # do some test and check performance
                    LOGI("Running the model in testing mode")
                    test_loss = 0
                    test_counter = 0
                    with torch.no_grad():
                        if self.dataparallel:
                            self.diffusion_model.module.eval() # for model only
                        self.diffusion_model.eval() # for dataparallel module

                        LOGI("Forward passing through diffusion process")
                        for x, y in self.test_loader:
                            x = x.to(self.device).float()
                            y = y.to(self.device)

                            if self.configs["use_labels"]:
                                output_loss = self.diffusion_model(x, y)
                            else:
                                output_loss = self.diffusion_model(x)
                            
                            if self.dataparallel:
                                output_loss = output_loss.mean()

                            test_loss += output_loss.item()
                            test_counter += 1
                    
                    # same idea as in training, but here we Sample, so we need to provide info on the labels
                    # basically generate 1 sample per class, example the input y = [0, 1, ..., 9]
                    LOGI("Generating some samples")
                    if self.configs["use_labels"]:
                        n_classes_2use = self.configs["num_classes"]
                        if n_classes_2use > MAX_BATCH_VISUALIZE:
                            n_classes_2use = MAX_BATCH_VISUALIZE
                            # random pick MAX_BATCH_VISUALIZE classes
                            start_class = np.randint(0, self.configs["num_classes"])
                            if (start_class+n_classes_2use) >= self.configs["num_classes"]:
                                start_class = self.configs["num_classes"] - n_classes_2use
                            
                            end_class = start_class + n_classes_2use
                        else:
                            start_class = 0
                            end_class = n_classes_2use
                        
                        if self.dataparallel:
                            samples = self.diffusion_model.module.sample(batch_size=n_classes_2use,
                                                                            device=self.device,
                                                                            use_ema=True,
                                                                            gen_seq=False,
                                                                            yclass=torch.arange(start=start_class, 
                                                                                                end=end_class, step=1,
                                                                                                device=self.device)
                                                                        )
                        else:
                            samples = self.diffusion_model.sample(batch_size=n_classes_2use,
                                                                    device=self.device,
                                                                    use_ema=True,
                                                                    gen_seq=False,
                                                                    yclass=torch.arange(start=start_class, 
                                                                                        end=end_class, step=1,
                                                                                        device=self.device)
                                                                )
                    else:
                        # generate MAX_BATCH_VISUALIZE samples
                        if self.dataparallel:
                            samples = self.diffusion_model.module.sample(batch_size=MAX_BATCH_VISUALIZE,
                                                                            device=self.device, use_ema=True,
                                                                            gen_seq=False)
                        else:
                            samples = self.diffusion_model.sample(batch_size=MAX_BATCH_VISUALIZE, device=self.device,
                                                                    use_ema=True, gen_seq=False)

                    # keeping in perspective
                    test_loss /= len(self.test_loader)
                    train_loss /= self.configs["checkpoint_rate"]
                    iter_time_sec = (float(time() - start_time) / self.configs["checkpoint_rate"])
                    LOGW(f"Iteration-{iteration}: TestLoss-{test_loss}, TrainLoss-{train_loss}, sec/iter: {iter_time_sec}")

                    LOGI("Dumping some images for later visualization")
                    # samples generated, and undo the normalization from [-1, 1] so now it will be from [0, 1]
                    # then channels are moved at the end for reconstruction/visualization
                    h, w = self.configs["img_size"]
                    new_im = Image.new('RGB', (w*self.configs["num_classes"], h))
                    samples = ((samples + 1) / 2).clip(0, 1).permute(0, 2, 3, 1).numpy()
                    LOGI(f"Set of images to be dumped are of shape: {samples.shape}")
                    x_offset = 0
                    for idx in range(self.configs["num_classes"]):
                        im = Image.fromarray(((samples[idx]*255).astype(np.uint8)).clip(0, 255))
                        LOGD(f"Image converted is of shape: {im.size}")
                        new_im.paste(im, (x_offset,0))
                        x_offset += im.size[0]
                    new_im.save(f"{self.resources}/sample_images/train/sample_iter_{iteration}.png")

                    update_snapsnot = False
                    if self.best_train_loss > train_loss:
                        self.best_train_loss = train_loss
                        update_snapsnot = True

                    # restart things for next logging iteration
                    train_loss = 0
                    start_time = time()

                    LOGI(f"saving checkpoints (updating best checkpoint: {update_snapsnot})")
                    # current models at given iteration
                    tmp_modelname = f"{self.resources}/models/iteration-{iteration}-model.pth"
                    tmp_optimizername = f"{self.resources}/optimizers/iteration-{iteration}-optimizer.pth"
                    if self.dataparallel:
                        torch.save(self.diffusion_model.module.state_dict(), tmp_modelname)
                    else:
                        torch.save(self.diffusion_model.state_dict(), tmp_modelname)
                    torch.save(self.optimizer.state_dict(), tmp_optimizername)

                    if update_snapsnot:
                        # this is for the snapshot
                        final_modelname = f"{self.resources}/models/latest-model.pth"
                        final_optimizername = f"{self.resources}/optimizers/latest-optimizer.pth"
                        if self.dataparallel:
                            torch.save(self.diffusion_model.module.state_dict(), final_modelname)
                        else:
                            torch.save(self.diffusion_model.state_dict(), final_modelname)
                        torch.save(self.optimizer.state_dict(), final_optimizername)
                    
                    # we make sure to keep track which iteration we are at!
                    latest_iteration = f"{self.resources}/{self.configs['iteration_snapshot']}"
                    with open(latest_iteration, "w") as file2write:
                        file2write.write(f"{iteration}\n{self.best_train_loss}\n")
        except KeyboardInterrupt:
            LOGW("Keyboard interrupt, stopping training!")

        LOGI("Model Training has completed successfully")
        return True
from utils.misc import (get_transform, infinite_loader)
from models.initializer import ModelCreator
from utils.configs import (LOGE, LOGI, LOGW, LOGD)
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from time import time
import torchvision
import os
from trainer.dct2rgb import to_img
from scipy.ndimage import zoom
import cv2
import sys

# tensorboard - added on 2024.07.10
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/training_0802")

DEBUG_IMAGES = False
MAX_BATCH_VISUALIZE = 10

def reshape(arr):
    # print('reshaping with array shape = ', arr.shape)
    FLATTEN_KEY = np.arange(64).reshape((8, 8))
    arr = np.asarray(arr)
    if arr.shape != (64,):
        print("array shape --> ", arr.shape)
        raise ValueError('Array needs to be macroblock of shape 8x8')
    rows = cols = 8
    macroblock = np.empty(shape=(rows, cols))
    for j in range(rows):
        for i in range(cols):
            idx = FLATTEN_KEY[j, i]
            macroblock[j, i] = arr[idx]
    # print('reshape ok')
    return macroblock


def dct_2_rgb(dct_y, dct_cb, dct_cr):
    # print(f'Y = {dct_y.shape}, Cb = {dct_cb.shape}, Cr = {dct_cr.shape}')
    dct_y = dct_y.transpose((1, 2, 0))
    dct_cb = dct_cb.transpose((1, 2, 0))
    dct_cr = dct_cr.transpose((1, 2, 0))
    # print(f'Y = {dct_y.shape}, Cb = {dct_cb.shape}, Cr = {dct_cr.shape}')


    rows, cols, _ = dct_y.shape
    imgY_rec = np.ones(shape=(8*rows, 8*cols))
    for j in range(rows):
        for i in range(cols):
            spectrogram = reshape(dct_y[j, i])
            macroblock = cv2.idct(spectrogram) + 128
            imgY_rec[8 * j: 8 * (j + 1), 8 * i: 8 * (i + 1)] = macroblock
    # print('wuuut')
    imgY_rec[imgY_rec < 0] = 0
    imgY_rec[imgY_rec > 255] = 255
    imgY_rec = np.uint8(imgY_rec)
    # print("all ok 1")

    # Cb-Channel
    rows, cols, _ = dct_cb.shape
    # print('dct_cb shape = ', dct_y.shape)
    imgCb_rec = np.ones(shape=(8*rows, 8*cols))
    for j in range(rows):
        for i in range(cols):
            spectrogram = reshape(dct_cb[j, i])
            macroblock = cv2.idct(spectrogram) + 128
            imgCb_rec[8 * j: 8 * (j + 1), 8 * i: 8 * (i + 1)] = macroblock
    imgCb_rec[imgCb_rec < 0] = 0
    imgCb_rec[imgCb_rec > 255] = 255
    imgCb_rec = np.uint8(imgCb_rec)
    # print("all ok 2")

    # Cr-Channel
    rows, cols, _ = dct_cr.shape
    imgCr_rec = np.ones(shape=(8*rows, 8*cols))
    for j in range(rows):
        for i in range(cols):
            spectrogram = reshape(dct_cr[j, i])
            macroblock = cv2.idct(spectrogram) + 128
            imgCr_rec[8 * j: 8 * (j + 1), 8 * i: 8 * (i + 1)] = macroblock
    imgCr_rec[imgCr_rec < 0] = 0
    imgCr_rec[imgCr_rec > 255] = 255
    imgCr_rec = np.uint8(imgCr_rec)

    # print("all ok 3")

    img_rec = np.dstack((imgY_rec, imgCr_rec, imgCb_rec))
    img_rec = np.uint8(img_rec)
    img_rec = cv2.cvtColor(img_rec, cv2.COLOR_YCrCb2BGR)
    img_rec[img_rec < 0] = 0
    img_rec[img_rec > 255] = 255
    # print("all ok 4")

    # Visualization
    print(img_rec.shape)
    print(img_rec.min(), img_rec.max())
    return img_rec


def denormalize(Y_norm, minY, maxY):
    return ((Y_norm + 1) * (maxY - minY) / 2) + minY


def downsample(x):
    downsample_factors = (1, 0.5, 0.5)
    # Apply zoom to downsample
    Cr_downsampled = zoom(x, downsample_factors, order=3)
    # If you used transpose after upsampling, apply the same transpose to restore the original order
    # Cr_downsampled = Cr_downsampled.transpose(1, 2, 0)
    return Cr_downsampled


def DCT_to_RGB(x):
    minY = -1024
    minCb = -993
    minCr = -891
    maxY = 1016
    maxCb = 980
    maxCr = 1034
    x = x.numpy()

    # first, denormalize
    Y_norm = x[:64, :, :]
    Cb_norm = x[64:128, :, :]
    Cr_norm = x[128:, :, :]
    Y = denormalize(Y_norm, minY, maxY)
    Cb = denormalize(Cb_norm, minCb, maxCb)
    Cr = denormalize(Cr_norm, minCr, maxCr)

    # now, downsample
    # Cb = downsample(Cb)
    # Cr = downsample(Cr)

    # DCT to RGB
    image = dct_2_rgb(Y,Cb,Cr)

    # cv2.imshow("Reconstructed Image", image)
    # cv2.waitKey()

    return image




'''
Our custom dataloader.
Our data was prepared as follows:

1. Find a CIFAR10 dataset that was upsampled from (32 x 32 x 3) --> (128 x 128 x 3)      ----> because 32 x 32 x 3 becomes 4 x 4 x 192when converted to DCT
2. Extract all images for cats and airplanes 
3. Convert these images to DCT   --> Y = (16 x 16 x 64),  Cb and Cr = (8 x 8 x 64)
4. Upsample the Cb and Cr components from (8 x 8 x 64) --> (16 x 16 x 64)
5. Concatenate all arrays into one --> (16 x 16 x 192) 
6. Transpose --> (192 x 16 x 16)
7. Save as arrays into PC
''' 
class CustomDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None, files_subfolders=True):
        self.img_labels = [x for x in os.listdir(img_dir)]   # LIST OF THE NAMES OF ALL CLASSES/ARRAYS
        self.img_dir = img_dir
        self.files_subfolders = files_subfolders # to mark that files in separated subfolders
        self.transform = transform,    # NOT DOING ANYTHING
        self.target_transform = target_transform  # NOT DOING ANYTHING

    def __len__(self):
        return len(self.img_labels)


    def __getitem__(self, idx):

        # Get the path for the array
        if self.files_subfolders:
            folder_class = f"class{idx}"
            files_paths = [x for x in os.listdir(os.path.join(self.img_dir, folder_class))]
            n_files = len(files_paths)
            idx_file = np.random.randint(0, n_files)
            LOGD(f"{self.img_dir}, {self.img_labels}, {idx}, {folder_class}, {n_files} files inside, {files_paths[idx_file]}")
            img_path = os.path.join(os.path.join(self.img_dir, folder_class), files_paths[idx_file])
        else:
            img_path = os.path.join(self.img_dir, self.img_labels[idx])
        
        LOGW(f"file to load is: {img_path}")
        sys.exit(0)
        
        # Load the concatenated array
        image = np.load(img_path, allow_pickle=True)

        # Extract the Y, Cb, and Cr components
        Y = image[:64, :, :]
        Cb = image[64:128, :, :]
        Cr = image[128:, :, :]

        # Values for normalization
        minY = -1024
        minCb = -993
        minCr = -891
        maxY = 1016
        maxCb = 980
        maxCr = 1034

        # Normalize between -1 and 1
        Y_norm = (2*(Y-minY))/(maxY-minY) - 1
        Cb_norm = (2*(Cb-minCb))/(maxCb-minCb) - 1
        Cr_norm = (2*(Cr-minCr))/(maxCr-minCr) - 1

        if not self.files_subfolders:
            # Dictionary for labels
            classes_dictionary = {
                "airplanes":0,
                "cats":1,
            }
            # Obtain the label for the given image
            animal = self.img_labels[idx].split("_")[0]
            label = classes_dictionary[animal]
        else:
            label = idx

        # Concatenate normalized arrays into one
        image = np.concatenate((Y_norm, Cb_norm, Cr_norm), axis=0)

        # Convert the array to a tensor (for training)
        image = torch.from_numpy(image).clip(-1, 1)

        # Return the image and label (as tensors)
        return image, torch.tensor(label)
        




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
        train_dataset = CustomDataset(img_dir=f"{self.resources}/{self.configs['image_datapath']}/train/")
        if DEBUG_IMAGES:
            # debug images to make sure we are dealing with proper values
            max_print = 4
            for x, y in train_dataset:
                if max_print > 0:
                    LOGI(f"Training Image shape: {x.shape}, Labels are: {y}")
                    LOGI(f"Training Image pixels values range [{torch.min(x)}, {torch.max(x)}]")
                    _ = DCT_to_RGB(x)
                else:
                    break
                max_print -= 1

        # iterator for testing set, remember things are normalized from [0, 1] when using torchvision
        test_dataset = CustomDataset(img_dir=f"{self.resources}/{self.configs['image_datapath']}/test/")
        if DEBUG_IMAGES:
            # debug images to make sure we are dealing with proper values
            max_print = 4
            for x, y in test_dataset:
                if max_print > 0:
                    LOGI(f"Testing Image shape: {x.shape}, Labels are: {y}")
                    LOGI(f"Testing Image pixels values range [{torch.min(x)}, {torch.max(x)}]")
                    _ = DCT_to_RGB(x)
                else:
                    break
                max_print -= 1
        
        # function that loads the dataset for training, and rollsback if needed
        # for diffusion models we don't train epochs but steps!
        self.train_loader = infinite_loader(DataLoader(train_dataset, batch_size=self.configs["batch_size"],
                                                       shuffle=True, drop_last=True, num_workers=self.configs["num_workers"]))

        # no need to cycle it as is done once, during testing
        self.test_loader = DataLoader(test_dataset, shuffle=True, batch_size=self.configs["batch_size"], drop_last=True,
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

                
                LOGW(f"Getting samples for iteration-{iteration}")
                x, y = next(self.train_loader)
                if iteration == 1:
                    writer.add_graph(self.diffusion_model, x)
                    writer.close()
                LOGW(f"shape = {x.shape}, min = {torch.min(x)}, max = {torch.max(x)}")
                
                # CHECKING THAT THE DATA THAT GOES INTO THE MODEL IS CORRECT
                # print("converting X to array")
                # x_array = x.cpu().detach()

                # print("visualizing it, this shape btw --> ", x_array.shape)
                # to_img(x_array)

                # # send to accelator if configured
                # if iteration % self.configs["checkpoint_rate"] == 0:
                #     LOGI(f"Sending samples to {self.device}")

                x = x.to(self.device).float()
                y = y.to(self.device)

                # print(f'\nx size = {x.size()}, xmin {torch.min(x)}, xmax {torch.max(x)}')
                # print(f'y size = {x.size()}, ymin {torch.min(y)}, ymax {torch.max(y)}')

                # if labels are to used, then we feed them to create the embeddings 
                # and train the biases
                if iteration % self.configs["checkpoint_rate"] == 0:
                    LOGI("Forward passing through diffusion process")

                if self.configs["use_labels"]:
                    # print(x.shape, y.shape)
                    # print(type(x))
                    # print(y)
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
                            start_class = np.random.randint(0, self.configs["num_classes"])
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
                    test_loss /= test_counter
                    train_loss /= self.configs["checkpoint_rate"]
                    iter_time_sec = (float(time() - start_time) / self.configs["checkpoint_rate"])
                    LOGW(f"Iteration-{iteration}: TestLoss-{test_loss}, TrainLoss-{train_loss}, sec/iter: {iter_time_sec}")
                    
                    LOGW("creating a sample generated at this time")
                    dct_example = samples[0]
                    img_ = DCT_to_RGB(dct_example)
                    cv2.imwrite(f'resources/sample_images/train/img_at_iter#{iteration}.png', img_)

                    # LOGI("Dumping some images for later visualization")
                    # # samples generated, and undo the normalization from [-1, 1] so now it will be from [0, 1]
                    # # then channels are moved at the end for reconstruction/visualization
                    # h, w = self.configs["img_size"]
                    # new_im = Image.new('RGB', (w*self.configs["num_classes"], h))
                    # samples = ((samples + 1) / 2).clip(0, 1).permute(0, 2, 3, 1).numpy()
                    # LOGI(f"Set of images to be dumped are of shape: {samples.shape}")
                    # x_offset = 0
                    # for idx in range(self.configs["num_classes"]):
                    #     im = Image.fromarray(((samples[idx]*255).astype(np.uint8)).clip(0, 255))
                    #     LOGD(f"Image converted is of shape: {im.size}")
                    #     new_im.paste(im, (x_offset,0))
                    #     x_offset += im.size[0]
                    # new_im.save(f"{self.resources}/sample_images/train/sample_iter_{iteration}.png")

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
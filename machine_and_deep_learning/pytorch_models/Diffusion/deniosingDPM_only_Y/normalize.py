import torch, os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm


std = [5.1367e+02, 4.6343e+01, 2.1453e+01, 1.3208e+01, 8.7820e+00, 5.5851e+00,
        3.1441e+00, 1.4912e+00, 4.3850e+01, 2.1830e+01, 1.4961e+01, 1.0452e+01,
        7.3803e+00, 4.5937e+00, 2.5974e+00, 1.3274e+00, 1.9911e+01, 1.4302e+01,
        1.1208e+01, 8.4798e+00, 6.0947e+00, 3.8615e+00, 2.0962e+00, 1.1486e+00,
        1.2193e+01, 9.7168e+00, 8.1500e+00, 6.4329e+00, 4.5759e+00, 2.6238e+00,
        1.5405e+00, 8.8679e-01, 8.1195e+00, 6.7715e+00, 5.6743e+00, 4.3351e+00,
        3.0638e+00, 1.6327e+00, 9.2992e-01, 5.6582e-01, 5.2514e+00, 4.3453e+00,
        3.5041e+00, 2.7121e+00, 1.8296e+00, 1.0602e+00, 5.5554e-01, 3.3295e-01,
        2.8343e+00, 2.2484e+00, 1.7636e+00, 1.3134e+00, 8.6360e-01, 4.9518e-01,
        2.8539e-01, 1.8230e-01, 1.2356e+00, 8.7754e-01, 7.1191e-01, 5.5629e-01,
        3.5623e-01, 2.7399e-01, 1.5646e-01, 9.6213e-02, 1.1073e+02, 9.4412e+00,
        3.7267e+00, 1.7729e+00, 6.2737e-01, 3.0021e-01, 1.6179e-01, 1.0221e-01,
        8.7445e+00, 3.2063e+00, 2.1366e+00, 9.4218e-01, 3.6211e-01, 1.7970e-01,
        1.0040e-01, 6.6890e-02, 3.5285e+00, 2.0772e+00, 1.2729e+00, 5.2202e-01,
        2.7313e-01, 1.3564e-01, 7.7369e-02, 6.0080e-02, 1.7092e+00, 9.1385e-01,
        5.1828e-01, 2.9689e-01, 1.6961e-01, 9.1410e-02, 5.3844e-02, 5.8131e-02,
        6.6917e-01, 3.8600e-01, 2.8771e-01, 1.7264e-01, 1.1093e-01, 6.5619e-02,
        2.9161e-02, 3.2019e-02, 3.5295e-01, 2.1293e-01, 1.6649e-01, 1.1655e-01,
        8.0124e-02, 4.4198e-02, 1.9934e-02, 2.3075e-02, 2.1348e-01, 1.2776e-01,
        1.1720e-01, 7.3025e-02, 4.2211e-02, 2.9024e-02, 1.5441e-02, 2.7040e-02,
        1.4283e-01, 9.3490e-02, 7.4448e-02, 5.6354e-02, 3.2876e-02, 1.9934e-02,
        1.8486e-02, 2.3586e-02, 1.3038e+02, 1.1018e+01, 4.1919e+00, 2.0257e+00,
        7.6157e-01, 4.0913e-01, 2.4375e-01, 1.8395e-01, 1.0416e+01, 3.6208e+00,
        2.3742e+00, 1.0891e+00, 4.5823e-01, 2.3934e-01, 1.5293e-01, 1.1642e-01,
        4.1452e+00, 2.3333e+00, 1.4601e+00, 6.1904e-01, 3.3400e-01, 1.8789e-01,
        1.2633e-01, 1.1030e-01, 2.0538e+00, 1.0733e+00, 6.0818e-01, 3.6323e-01,
        2.2245e-01, 1.3089e-01, 9.7526e-02, 9.6901e-02, 8.1253e-01, 4.6039e-01,
        3.3767e-01, 2.3079e-01, 1.5312e-01, 9.0488e-02, 7.1061e-02, 6.4223e-02,
        4.0445e-01, 2.4484e-01, 1.9677e-01, 1.4060e-01, 1.0146e-01, 6.8973e-02,
        4.7508e-02, 4.2567e-02, 2.5135e-01, 1.5377e-01, 1.2550e-01, 9.7248e-02,
        8.3418e-02, 6.2148e-02, 5.1227e-02, 4.1719e-02, 1.8844e-01, 1.1952e-01,
        9.3906e-02, 6.9797e-02, 4.7592e-02, 3.0882e-02, 2.9567e-02, 1.5441e-02]


mean = [-1.1177e+02,  7.1595e-03,  1.6841e-02,  1.1061e-02,  3.5439e-03,
         2.0797e-03, -3.5928e-03,  1.3560e-04,  1.7445e-01, -2.6986e-03,
        -9.0576e-04,  6.9967e-04, -7.6622e-04, -4.4250e-05, -6.5125e-04,
         1.1792e-04,  6.2918e-03,  2.7812e-03,  1.3818e-03, -9.3457e-04,
         1.3090e-03,  7.3456e-04,  8.2530e-04, -1.3102e-04,  1.8251e-02,
         1.3643e-03, -5.6329e-04,  2.2369e-04, -3.4595e-04, -1.0205e-04,
         9.1146e-05,  8.3252e-05,  4.5903e-03,  6.3374e-03,  2.7836e-03,
         1.3017e-03,  5.2502e-04,  1.9023e-05,  1.4494e-04,  1.5710e-04,
         2.4324e-03, -1.9409e-05,  7.9069e-04, -4.5833e-04, -1.3763e-04,
         1.9149e-04,  4.0588e-05,  1.2166e-05, -2.6499e-03, -5.3385e-04,
         8.1329e-04,  2.8377e-04,  4.7607e-05, -3.7231e-06, -5.1270e-05,
         5.6030e-05,  1.0840e-04,  6.9255e-05, -2.0313e-04,  2.9907e-06,
         7.6335e-05, -1.4242e-05,  1.1637e-05,  6.1035e-06, -7.9089e+01,
        -1.1266e-01,  7.4395e-03, -8.3633e-03,  4.5166e-04, -6.9173e-05,
        -8.1380e-06, -1.7904e-04,  2.9369e-02,  5.3353e-04,  2.6724e-03,
         3.1152e-04,  2.6855e-04,  4.8828e-05, -1.6276e-05,  4.8828e-05,
         5.2441e-03, -7.5643e-04,  1.3604e-03,  2.7669e-04,  1.8311e-04,
        -4.4759e-05, -4.0690e-05, -1.2207e-05, -3.7227e-03, -8.5937e-04,
         2.6855e-04,  1.5869e-04, -6.9173e-05,  1.6276e-05,  0.0000e+00,
        -2.0345e-05,  2.5635e-04, -8.5449e-05,  2.4007e-04, -3.2552e-05,
         6.9173e-05, -2.0345e-05, -1.2207e-05, -8.1380e-06, -2.7669e-04,
        -4.0690e-06, -8.1380e-05, -8.1380e-06,  3.6621e-05, -2.0345e-05,
        -4.0690e-06,  4.0690e-06,  8.9518e-05, -4.0690e-06, -1.2207e-05,
        -2.8483e-05, -1.6276e-05, -8.1380e-06, -4.0690e-06,  2.4414e-05,
        -1.6276e-05,  2.8483e-05, -2.8483e-05, -4.0690e-06,  4.0690e-06,
         1.2207e-05, -8.1380e-06,  4.0690e-06,  1.0503e+02, -1.0790e-01,
         6.4629e-03, -1.1979e-02, -1.3875e-03, -3.7435e-04, -1.5055e-04,
        -1.4648e-04, -2.1426e-01, -1.7725e-04, -2.4481e-03, -4.0820e-04,
        -3.4993e-04,  4.0690e-06,  1.2207e-04,  0.0000e+00, -3.2236e-03,
        -6.0938e-04, -1.6178e-04,  2.9704e-04, -4.0690e-05,  5.6966e-05,
         8.1380e-06,  2.8483e-05, -7.2812e-03, -4.6729e-04, -5.2897e-04,
        -1.5462e-04,  6.9173e-05,  6.9173e-05, -4.0690e-06,  6.1035e-05,
        -4.8421e-04,  9.7656e-05, -4.0690e-05,  7.3242e-05, -8.1380e-05,
         4.8828e-05,  1.2207e-05, -4.8828e-05,  3.6621e-05,  1.5462e-04,
        -1.7497e-04,  1.2207e-05,  8.1380e-06,  2.0345e-05, -2.8483e-05,
        -2.4414e-05, -1.1800e-04,  4.0690e-06,  3.2552e-05, -4.4759e-05,
         1.2207e-05, -8.1380e-06,  3.2552e-05, -2.0345e-05,  4.4759e-05,
        -1.1800e-04, -1.2207e-05,  2.0345e-05,  8.1380e-06, -1.6276e-05,
         1.2207e-05,  4.0690e-06]



class CustomDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_labels = [x for x in os.listdir(img_dir)]   # LIST OF THE NAMES OF ALL ARRAYS
        self.img_dir = img_dir
        self.transform = transform,    # NOT DOING ANYTHING
        self.target_transform = target_transform  # NOT DOING ANYTHING
        self.mean = [0]*192
        self.std = [1]*192

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Get the path for a random array
        img_path = os.path.join(self.img_dir, self.img_labels[idx])   # C:\Users\vminanda\Desktop\josue code - diffussion\own_repos-master\own_repos-master\machine_and_deep_learning\pytorch_models\Diffusion\denoisingDPM\resources\datasets\concatenated_outputs\train\00000-20240603T070734Z-001_00007.npy, C:\Users\vminanda\Desktop\josue code - diffussion\own_repos-master\own_repos-master\machine_and_deep_learning\pytorch_models\Diffusion\denoisingDPM\resources\datasets\concatenated_outputs\train\00000-20240603T070734Z-001_00152.npy 
        
        # Load the array with numpy
        image = np.load(img_path, allow_pickle=True)
        image = torch.from_numpy(image)
        # for i in range(image.shape[0]):
        #     image[i, :, :] = (image[i, :, :] - self.mean[i]) / self.std[i]
        return image



class TestCustomDataset(Dataset):
    def __init__(self, img_dir, mean, std, transform=None, target_transform=None):
        self.img_labels = [x for x in os.listdir(img_dir)]   # LIST OF THE NAMES OF ALL ARRAYS
        self.img_dir = img_dir
        self.transform = transform,    # NOT DOING ANYTHING
        self.target_transform = target_transform  # NOT DOING ANYTHING
        self.mean = [0]*192
        self.std = [1]*192
        self.real_mean = torch.from_numpy(np.array(mean))
        self.real_std = torch.from_numpy(np.array(std))
        print(f"real mean shape = {self.real_mean.shape}, std shape = {self.real_std.shape}")

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Get the path for a random array
        img_path = os.path.join(self.img_dir, self.img_labels[idx])   # C:\Users\vminanda\Desktop\josue code - diffussion\own_repos-master\own_repos-master\machine_and_deep_learning\pytorch_models\Diffusion\denoisingDPM\resources\datasets\concatenated_outputs\train\00000-20240603T070734Z-001_00007.npy, C:\Users\vminanda\Desktop\josue code - diffussion\own_repos-master\own_repos-master\machine_and_deep_learning\pytorch_models\Diffusion\denoisingDPM\resources\datasets\concatenated_outputs\train\00000-20240603T070734Z-001_00152.npy 
        
        # Load the array with numpy
        image = np.load(img_path, allow_pickle=True)
        image = torch.from_numpy(image)

        print(image.shape)

        print(f"orig max = {torch.max(image)}, min = {torch.min(image)}")

        for i in range(image.shape[0]):
            image[i, :, :] = (image[i, :, :] - self.real_mean[i]) / self.real_std[i]
        
        print(f"norm max = {torch.max(image)}, min = {torch.min(image)}")

        print(".....denormalizing now")

        for i in range(image.shape[0]):
            image[i, :, :] = (image[i, :, :] * self.real_std[i]) + self.real_mean[i]

        print(f"back to original max = {torch.max(image)}, min = {torch.min(image)}")

        return image



# NORMALIZE
def normalize():
    # dataset
    image_dataset = CustomDataset(img_dir=f"D:/own_repos/machine_and_deep_learning/pytorch_models/Diffusion/denoisingDPM/resources/datasets/concatenated_outputs/test")

    # data loader
    image_loader = DataLoader(
        image_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # placeholders
    psum = torch.tensor([0.0]*192)
    psum_sq = torch.tensor([0.0]*192)

    # loop through images
    for inputs in tqdm(image_loader):
        psum += inputs.sum(axis=[0, 2, 3])
        psum_sq += (inputs**2).sum(axis=[0, 2, 3])

    # pixel count
    count = 3000 * 128 * 128

    # mean and std
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean**2)
    total_std = torch.sqrt(total_var)

    print(f"total_mean = {total_mean}, \n total_std = {total_std}")




def test_normalization():
    # dataset
    image_dataset = TestCustomDataset(img_dir=f"D:/own_repos/machine_and_deep_learning/pytorch_models/Diffusion/denoisingDPM/resources/datasets/concatenated_outputs/test",
                                      mean=mean, 
                                      std=std)

    # data loader
    image_loader = DataLoader(
        image_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # loop through images
    for inputs in tqdm(image_loader):
        print("OK")



if __name__ == "__main__": 
    test_normalization()




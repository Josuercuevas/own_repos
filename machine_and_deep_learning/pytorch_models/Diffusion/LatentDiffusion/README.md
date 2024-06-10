# Latent Diffusion Models (LDM)

Replicating the results from [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/pdf/2112.10752). Currently only CelebA-HQ model and dataset has been tested although the idea of this repo is to replicate all the models with the datasets in the original paper.

## Installing dependencies

just run the script **install_dependecies.sh** from your terminal and it should install all the dependencies, if newer versions of each package are needed please watch out for possible incompatibilities.

## Prepare training/testing/validation data

**CelebA-HQ**: please follow the instruction at **[data/download-celeba-hq/](data/download-celeba-hq/)**

**FFHQ**: TBC.

**CIN**: TBC.

**LSUN**: TBC.

**Text2Img**: TBC.

## Training your own models

### Pre-trained models

If you don't want to train the first stages (AutoEncoder) please download the pre-trained weights by running the scripts at **inference/download_pretrained_first_stages.sh**. The weights will be downloaded and extracted in the subfolder **models/first_stage_models/MODEL_NAME**

### Training commands for **AutoEncoder** - First Stage

```shell
# assume we want to train VQ-VAE-256 8x8x64
reset && clear && CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --base configs/autoencoder/autoencoder_kl_8x8x64.yaml --operation "train" --gpus 0,1,2,3 --debug_level 2 
```

### Training commands for **Latent Diffusion Model (LDM)** - Second Stage

You need first stage **AutoEncoder** model in order to proceed with this stage. You can either train it by yourself or download it as described in the previous section.

To train the LDM please use the following command:

```shell
# start training from scratch if no folder present, assume we would like to train celebA-HQ-LDM-VQ-4
reset && clear && CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --base configs/latent-diffusion/celebahq-ldm-vq-4.yaml --operation "train" --gpus 0,1,2,3 --debug_level 2 
```

```shell
# to resume training with latest model and step, assume we are training celebA-HQ-LDM-VQ-4
reset && clear && CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --base configs/latent-diffusion/celebahq-ldm-vq-4.yaml --operation "train" --gpus 0,1,2,3 --resume logs/LOGDIR2USE/checkpoints/last.ckpt --debug_level 2
```

## Testing Commands

### Open source pre-trained models

If you don't want to train the models by yourself, please use the scripts in **inference/download_pretrained_ldms_models.sh** to download the pre-trained models. The weights will be downloaded and extracted in the subfolder **models/ldm/MODEL_NAME**, then just follow the instructions below.


### Unconditional LDM

Our own in-house pretrained model for celebA-HD can be found in [this link](https://drive.google.com/drive/folders/12V2EokNvumOLTC9OAIUbPjiaQNE9v92D?usp=sharing). The next step is to run the following command to generate samples:

```shell
# Make sure configuration is in the same folder as the model checkpoint, assume celeb256 model:
cp models/ldm/celeba256/config.yaml PATH2CHECKPOINT/

# Run the following cmd-line assuming you don't want to run DDIM(faster) for sampling:
reset && clear && CUDA_VISIBLE_DEVICES=0 python main.py -r PATH2CHECKPOINT/model.ckpt --debug_level 2 --operation "test_unconditional" -ns 2 --batchsize 2 --vanilla_sample
```

```shell
# Make sure configuration is in the same folder as the model checkpoint, assume celeb256 model:
cp models/ldm/celeba256/config.yaml PATH2CHECKPOINT/

# Run the following cmd-line assuming using DDIM for sampling:
reset && clear && CUDA_VISIBLE_DEVICES=0 python main.py -r PATH2CHECKPOINT/model.ckpt --debug_level 2 --operation "test_unconditional" -ns 2 --batchsize 2 -custom_steps 200 -eta 0.5
```

# Comments

- The code is heavily based on the original [open-sourced code](https://github.com/CompVis/latent-diffusion?tab=readme-ov-file)

- Data generation code is havily based on [open-sourced code](https://github.com/suvojit-0x55aa/celebA-HQ-dataset-download)

# Contributors

1. Josue Rodolfo Cuevas Juarez
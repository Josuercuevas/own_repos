# Latent Diffusion Models (LDM)

Replicating the results from [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/pdf/2112.10752)

## Installing dependencies

just run the script **install_dependecies.sh** from your terminal and it should install all the dependencies.

## Prepare training/testing/validation data

**CelebA-HQ**: please follow the instruction at **[data/download-celeba-hq/](data/download-celeba-hq/)**

## Training models

### Pre-trained models

If you don't want to train the first stages (encoder) please download the pre-trained weights by running the scripts at **inference/download_pretrained_first_stages.sh**. The weights will be downloaded and extracted in the subfolder **models/first_stage_models/MODEL_NAME**

### Training commands for First Stage AutoEncoder

```shell
# assume we want to train VQ-VAE-256 8x8x64
reset && clear && CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --base configs/autoencoder/autoencoder_kl_8x8x64.yaml --operation "train" --gpus 0,1,2,3 --debug_level 2 
```

### Training commands for LDM

```shell
# start training from scratch if no folder present
reset && clear && CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --base configs/latent-diffusion/celebahq-ldm-vq-4.yaml --operation "train" --gpus 0,1,2,3 --debug_level 2 
```

```shell
# resume training with latest model and step
reset && clear && CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --base configs/latent-diffusion/celebahq-ldm-vq-4.yaml --operation "train" --gpus 0,1,2,3 --resume logs/LOGDIR2USE/checkpoints/last.ckpt --debug_level 2 
```

## Testing Commands

### Pre-trained models

If you don't want to train the models by yourself, please use the scripts in **inference/download_pretrained_ldms_models.sh** to download the pre-trained models. The weights will be downloaded and extracted in the subfolder **models/ldm/MODEL_NAME**

### Unconditional LDM

```shell
# make sure configuration is in the same folder as the model checkpoint, assume celeb256 model:
cp models/ldm/celeba256/config.yaml PATH2CHECKPOINT/

# Run the following cmd-line assuming you don't want to run DDIM for sampling:
reset && clear && CUDA_VISIBLE_DEVICES=0 python main.py -r PATH2CHECKPOINT/model.ckpt --debug_level 2 --operation "test_unconditional" -ns 2 --batch_size 2 --vanilla_sample
```
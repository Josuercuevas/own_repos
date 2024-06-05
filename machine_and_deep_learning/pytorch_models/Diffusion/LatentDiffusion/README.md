# Latent Diffusion Models (LDM)

Replicating the results from [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/pdf/2112.10752)

## Installing dependencies

just run the script **install_dependecies.sh** from your terminal and it should install all the dependencies.

## Training models

### Pre-trained models

If you don't want to train the first stages (encoder) please download the pre-trained weights by running the scripts at **unittests/download_pretrained_first_stages.sh**. The weights will be downloaded and extracted in the subfolder **models/first_stage_models/MODEL_NAME**

### TBC ...

## Testing Commands

### Pre-trained models

If you don't want to train the models by yourself, please use the scripts in **unittests/download_pretrained_ldms_models.sh** to download the pre-trained models. The weights will be downloaded and extracted in the subfolder **models/ldm/MODEL_NAME**

### Unconditional LDM

```shell
# make sure configuration is in the same folder as the model checkpoint, assume celeb256 model:
cp models/ldm/celeba256/config.yaml PATH2CHECKPOINT/

# Run the following cmd-line assuming you don't want to run DDIM for sampling:
reset && clear && CUDA_VISIBLE_DEVICES=0 python main.py -r PATH2CHECKPOINT/model.ckpt --debug_level 2 --operation 'test' -n 2 --batch_size 2 --vanilla_sample
```

docker run --privileged -it -d -v $(pwd):/workspace --name istr_mask --gpus=all --shm-size=64gb istr_mask/dev:v1.0.0

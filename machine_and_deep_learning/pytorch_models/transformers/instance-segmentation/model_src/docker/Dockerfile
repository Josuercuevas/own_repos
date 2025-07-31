FROM nvidia/cuda:12.0.1-devel-ubuntu22.04

RUN apt-get update
RUN apt-get install -y build-essential apt-utils

RUN apt-get install -y cmake git libgtk2.0-dev pkg-config libavcodec-dev \
    libavformat-dev libswscale-dev git-lfs
RUN  apt-get update && apt-get install -y python3 python3-pip \
    python3-dev libtbb2 libtbb-dev libjpeg-dev libatlas-base-dev \
    gfortran webp zlib1g-dev

RUN pip3 install numpy opencv-python matplotlib scikit-learn torch torchvision torchaudio torch-dct timm shapely

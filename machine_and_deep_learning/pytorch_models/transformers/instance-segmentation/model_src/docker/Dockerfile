FROM nvidia/cuda:12.0.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential  && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake pkg-config libgtk2.0-dev libavcodec-dev libavformat-dev \
    libswscale-dev git-lfs && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y python3 python3-pip \
    python3-dev libtbb2 libtbb-dev libjpeg-dev libatlas-base-dev \
    gfortran webp zlib1g-dev

RUN pip3 install numpy opencv-python matplotlib scikit-learn 
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
RUN pip3 install torch-dct timm shapely


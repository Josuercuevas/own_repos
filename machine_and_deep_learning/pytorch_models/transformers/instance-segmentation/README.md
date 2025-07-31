# ISTR: End-to-End Instance Segmentation via Transformers

This is a code implementation for [**this paper**](https://arxiv.org/abs/2105.00637).

## Installation
The codes are built on top of [Detectron2](https://github.com/facebookresearch/detectron2), [SparseR-CNN](https://github.com/PeizeSun/SparseR-CNN), and [AdelaiDet](https://github.com/aim-uofa/AdelaiDet).

#### Requirements
- Python=3.8
- PyTorch=1.6.0, torchvision=0.7.0, cudatoolkit=10.1
- OpenCV for visualization

#### Steps
1. Install the repository (we recommend to use [Anaconda](https://www.anaconda.com/) for installation.)
```
conda create -n ISTR python=3.8 -y
conda activate ISTR
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
or (conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch)
pip install opencv-python
pip install scipy
pip install shapely
git clone --recurse-submodules git@github.com:Josuercuevas/own_repos.git
cd own_repos/machine_and_deep_learning/pytorch_models/transformers/instance-segmentation/model_src/
python setup.py build develop
```

#### Dataset
For the dataset please refer to [this code](data_preparation/), as for the input data to the model please refer to [this diagram](input-format.jpg)

2. Train ISTR (e.g., with ResNet50 backbone)
```shell
CUDA_VISIBLE_DEVICES=0 python3 predictor/ISTR/train_net.py --num-gpus 1 --config-file predictor/ISTR/configs/ISTR-AE-R50-3x.yaml
```

3. Evaluate ISTR (e.g., with ResNet50 backbone)
```shell
CUDA_VISIBLE_DEVICES=1 python3 predictor/ISTR/train_net.py --num-gpus 1 --config-file predictor/ISTR/configs/ISTR-AE-R50-3x.yaml --eval-only MODEL.WEIGHTS ./output/model_final.pth
```

4. Visualize the detection and segmentation results (e.g., with ResNet50 backbone)
```shell
# as a single file
CUDA_VISIBLE_DEVICES=1 python3 demo/demo.py --config-file predictor/ISTR/configs/ISTR-AE-R50-3x.yaml --input input1.jpg --output ./output --confidence-threshold 0.4 --opts MODEL.WEIGHTS ./output/model_final.pth

# a list of files
reset && clear && CUDA_VISIBLE_DEVICES=1 python3 demo/demo.py --config-file predictor/ISTR/configs/ISTR-AE-R50-3x.yaml --input datasets/coco/test2017/*1233.jpg --confidence-threshold 0.4 --opts MODEL.WEIGHTS output/model_final.pth
```
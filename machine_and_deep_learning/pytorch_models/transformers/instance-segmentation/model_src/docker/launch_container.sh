docker run --privileged -it -d -v $(pwd):/workspace -v /opt/research/datasets/coco_datasets/coco_2017:/workspace/datasets --name trans_mask --gpus=all --shm-size=64gb trans_mask/dev:v1.0.0

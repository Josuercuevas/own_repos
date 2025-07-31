#!/bin/bash

S3_BUCKET="s3://jebi-ai-deploy/packages"
# if libraries folder not found then create it
if [ ! -d "packages" ]; then
	mkdir packages
fi  
echo "Downloading libraries"
if [ ! -f "packages/opencv4.5.5.tgz" ]; then
		echo "Downloading opencv"
		aws s3 cp $S3_BUCKET/opencv/opencv4.5.5.tgz packages/opencv4.5.5.tgz
else
		echo "opencv already exists"
fi
if [ ! -f "packages/cudnn_8.5.0.96_cuda114_x64.tgz" ]; then
		echo "Downloading cudnn"
		aws s3 cp $S3_BUCKET/cudnn/cudnn_8.5.0.96_cuda114_x64.tgz packages/cudnn_8.5.0.96_cuda114_x64.tgz
else
		echo "cudnn already exists"
fi
if [ ! -f "packages/tensorrt_8.5.3.1_cuda114_x64.tgz" ]; then
		echo "Downloading TensorRT"
		aws s3 cp $S3_BUCKET/tensorrt/tensorrt_8.5.3.1_cuda114_x64.tgz packages/tensorrt_8.5.3.1_cuda114_x64.tgz
else
		echo "TensorRT already exists"
fi
if [ ! -f "packages/ZEDx64v414.tar.gz" ]; then
		echo "Downloading ZED"
		aws s3 cp $S3_BUCKET/zed/ZEDx64v414.tar.gz packages/ZEDx64v414.tar.gz
else
		echo "ZED already exists"
fi
 
echo "Downloading libraries completed. Building image"

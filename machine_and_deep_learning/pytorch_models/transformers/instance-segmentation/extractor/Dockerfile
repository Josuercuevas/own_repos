# Start from the official nvidia cuda 11.4 image
FROM nvidia/cuda:11.4.3-devel-ubuntu20.04
# Set the working directory in the container to /prime-workspace

# Switch to non-root user
WORKDIR /prime-workspace

# Add the current directory contents into the container at /jari-workspace
ADD packages /prime-workspace/

# Make port 80 available to the world outside this container
EXPOSE 80
ENV GPUARCH=86
ENV CUDA_VERSION=11.4
#set location to america lima
ENV TZ=America/Lima
#set locale
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV CUDA=/usr/local/cuda
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
ENV TRT_LIB_PATH=/usr/local/cuda/lib64
ENV TRT_INCLUDE_PATH=/usr/local/cuda/include
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
	wget git pkg-config \
	python3 python3-pip \
	libgl1-mesa-glx \ 
	ccache libjpeg-dev \
	libpng-dev libtiff-dev \
	libavcodec-dev libavformat-dev \ 
	libavresample-dev libswscale-dev \
	libv4l-dev libxvidcore-dev ffmpeg \
	libssl-dev zstd

#Installing CUDNN
RUN tar -xzf cudnn_8.5.0.96_cuda114_x64.tgz && \
	cp cudnn/include/cudnn*.h /usr/local/cuda/include && \
	cp -P cudnn/lib/libcudnn* /usr/local/cuda/lib64 && \
	chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn* && \
	rm -r cudnn
# Installing TensorRT
ENV TENSORRT_VERSION="8.5.3.1"
RUN tar -xzf tensorrt_${TENSORRT_VERSION}_cuda114_x64.tgz && \
	python3 -m pip install tensorrt/python/tensorrt-${TENSORRT_VERSION}-cp38-none-linux_x86_64.whl && \
	python3 -m pip install tensorrt/uff/uff-0.6.9-py2.py3-none-any.whl && \
	python3 -m pip install tensorrt/graphsurgeon/graphsurgeon-0.4.6-py2.py3-none-any.whl && \
	python3 -m pip install tensorrt/onnx_graphsurgeon/onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl && \
	cp tensorrt/include/* /usr/local/cuda/include && \
	cp -P tensorrt/lib/lib* /usr/local/cuda/lib64 && \
	cp tensorrt/bin/trtexec /usr/local/bin && \
	rm -r tensorrt
# We need to add some plugin to the origina TensorRT for our applications
RUN wget https://github.com/Kitware/CMake/releases/download/v3.13.5/cmake-3.13.5.tar.gz && \
	tar -xf cmake-3.13.5.tar.gz && cd cmake-3.13.5 && ./configure && make -j4 && make install && \
	cd .. && rm -rf cmake-3.13.5.tar.gz cmake-3.13.5 
RUN git clone -b 8.5.3 https://github.com/nvidia/TensorRT && cd TensorRT && \
	git submodule update --init --recursive && \
	mkdir -p build && cd build && \
	cmake .. -DGPU_ARCHS=${GPUARCH} -DCUDA_VERSION=${CUDA_VERSION} -DTRT_LIB_DIR=/usr/local/cuda/lib64/ -DCMAKE_C_COMPILER=/usr/bin/gcc -DTRT_BIN_DIR=/home/zeduserTensorRT/build/out && \
	make nvinfer_plugin -j4 && \
	rm /usr/local/cuda/lib64/libnvinfer_plugin.* && \
	cp -P libnvinfer_plugin.*  /usr/local/cuda/lib64/ && \
	ldconfig && \
	cd /prime-workspace && rm -rf TensorRT
RUN python3 -m pip install tqdm
RUN tar -xf ZEDx64v414.tar.gz && cd ZED414 && bash linux_install_release.sh --skip_tools --skip_od_module --runtime_only --silent && \
	python3 -m pip install --ignore-installed /prime-workspace/ZED414/pyzed-4.1-cp38-cp38-linux_x86_64.whl && cd /prime-workspace && rm -rf ZED414
RUN tar -xzf opencv4.5.5.tgz && cd opencv-4.5.5 && mkdir -p build && cd build && \
	cmake ../ -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local \
	-DBUILD_EXAMPLES=false -DBUILD_TESTS=false -DWITH_VTK=false -DBUILD_JAVA=false \
	-DWITH_LIBREALSENSE=false -DBUILD_JASPER=false -DBUILD_opencv_dnn=true \
	-DBUILD_TESTS=false -DBUILD_PERF_TESTS=false -DWITH_V4L=Yes -DWITH_LIBV4L=yes \
	-DBUILD_opencv_python2=false -DBUILD_opencv_python3=true -DWITH_QT=false \
	-DPYTHON3_PACKAGES_PATH=/usr/local/lib/python3.8/dist-packages && make -j4 && make install && \
	cd /prime-workspace && rm -rf opencv-4.5.5
	

# own_repos

Ideas developed or integrated with other publicly available projects, this repository is detailed as follows:

1. **[machine_and_deep_learning](./machine_and_deep_learning)**: Experiments conducted on machine and deep learning algorithms, where 3 main frameworks were used: [Caffe](http://caffe.berkeleyvision.org/), [Tensorflow](https://www.tensorflow.org/), [pytorch](https://pytorch.org/)

	a. **[pytorch_models/Diffusion/](./machine_and_deep_learning/pytorch_models/Diffusion)**: own pytorch implementation of models such as [Denoising Diffusion Probabilistic Model (DDPM)](https://arxiv.org/pdf/2006.11239.pdf), [Latent Diffusion Models](https://arxiv.org/pdf/2112.10752) with class conditioning and multi-gpu support.

	b. **[FarePredictor](./machine_and_deep_learning/FarePredictor)**: Exprimenting with Machine Learning models for predicting Taxi ride fares.

	c. **[Tensorflow_models/Autoencoders](./machine_and_deep_learning/Tensorflow_models/Autoencoders)**: Own tensroflow implementation of [Denoising AutoEncoders (DAE)](https://paperswithcode.com/method/denoising-autoencoder), and [AutoEncoders (AE)](http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/)

	d. **[Tensorflow_models/DHM_segmentation_detection](./machine_and_deep_learning/Tensorflow_models/DHM_segmentation_detection)**: An attempt to replicate the results from [Deep Hierarchical Models for Joint Object Detection](http://on-demand.gputechconf.com/gtc/2017/presentation/s7347-joe-chen-a-deep-hierarchial-model-for-joint-object-detection.pdf), which could be consider as an early attempt into what is know today as [Multi-Task Learning (MTL)](https://en.wikipedia.org/wiki/Multi-task_learning)

	e. **[Tensorflow_models/feature_extraction_classification_models](./machine_and_deep_learning/Tensorflow_models/feature_extraction_classification_models)**: different tensorflow implementations for [SqueezeNet](https://arxiv.org/abs/1602.07360), [Resnet](https://arxiv.org/abs/1512.03385), [ShuffleNet](https://arxiv.org/abs/1707.01083), and [MobileNet](https://arxiv.org/abs/1704.04861).

	f. **[caffe_models/ShuffleNet](./machine_and_deep_learning/caffe_models/ShuffleNet)**: Experimentation with shufflenet topology

	g. **[Tensorflow2Caffe_converter](./machine_and_deep_learning/Tensorflow2Caffe_converter)**: Model converter from Tensorflow to Caffe.


2. **[computer_vision_img_vid](./computer_vision_img_vid)**: Different computer vision algorithms implemented on CPU and GPU for image and video in raw and compressed domain **(H.264 standard)**, the folder is structured as follows:

	a. **[canny_edge_detection](./computer_vision_img_vid/canny_edge_detection)**: Canny edge detection, fully implemented on GPU

	b. **[colormap_extractor](./computer_vision_img_vid/colormap_extractor)**: Color mapping extraction from RGB images

	c. **[data_augmentation](./computer_vision_img_vid/data_augmentation)**: Python implemented data augmentation for input images

	d. **[gstreamer](./computer_vision_img_vid/gstreamer)**: further divided as follows:

		i. **gst_imgproc**: blob and skin detector on gstreamer, also moment normalization and color-retinex implementations on gstreamer.

		ii. **gst_rgb2gray**: RGB2GRAY implementation on gstreamer

		iii. **gst_rgbmapping**: Color mapping implementation for gstreamer

	e. **[hough_transform_lines_circles](./computer_vision_img_vid/hough_transform_lines_circles)**: Line and circle extraction using hough transform, fully implemented on GPU

	f. **[LBP_extract_module](./computer_vision_img_vid/LBP_extract_module)**: LBP feature extraction from sample images

	g. **[spatio_temporal_saliency_maps](./computer_vision_img_vid/spatio_temporal_saliency_maps)**: Static and Dynamic saliency mapping extraction from video/images

	h. **[ToneMapping](./computer_vision_img_vid/ToneMapping)**: Color enhancement using Tone mapping algorithm

3. **[metaheuristic_algorithms](./metaheuristic_algorithms)**: Implementation of several meta-heuristic algorithms including one developed during my master and PhD. degree. The folder is structured as follows:

	a. **[VOA](./metaheuristic_algorithms/VOA)**: Virus Optimization Algorithm proposed for the first time in 2009 but accepted until 2014.

	b. **TBD**: more to be added in the future ...

1. **[miscellaneous](./miscellaneous)**: Subfolder containing different ideas tested over the past years, and do not have any specific field of application. This subfolder has the following structure:

	a. **[Bbox_filter](./miscellaneous/Bbox_filter)**: Bounding box filter for object detection algorithms (python)

	b. **[bitstream_analizer_openh264based](./miscellaneous/bitstream_analizer_openh264based)**: h264-bitstream saliency map extractor (C/C++)

	c. **[ffmpeg_video_handler_c](./miscellaneous/ffmpeg_video_handler_c)**: FFMPEG library based video/camera frame extractor (C/C++)

	d. **[fft_conformance](./miscellaneous/fft_conformance)**: FFT conformance test to determine performance and accuracy (C/C++)

	e. **[gif_generator](./miscellaneous/gif_generator)**: GIF generator application (C/C++)

	f. **[h264_decoder_module_python](./miscellaneous/h264_decoder_module_python)**: FFMPEG video decoder (python)

	g. **[hd5_rawImage_database_creator](./miscellaneous/hd5_rawImage_database_creator)**: HDF5 database image file generator (python)

	h. **[test_gstreamer_thread_priorities](./miscellaneous/test_gstreamer_thread_priorities)**: Gstreamer thread priority test (C/C++)

	i. **[wavpack_gstreamer](./miscellaneous/wavpack_gstreamer)**: Gstreamer Wavpack plugin encoder and file writer with Metadata Tags (C/C++)


5. **[caffe_own](./caffe_own)**: [Caffe](http://caffe.berkeleyvision.org/) repository with some modifications to support models and layers proposed over the past 5 years, for example: MobileNet, ShuffleNet, SSD, MaskRCNN, GAN, etc.

### Contributors

Josue R. Cuevas

josuercuevas@gmail.com

# own_repos

Ideas developed or integrated with other publicly available projects, this repository is detailed as follows:

1. **miscellaneous**: Subfolder containing different ideas tested over the past years, and do not have any specific field of application. This subfolder has the following structure:

	a. **Bbox_filter**: Bounding box filter for object detection algorithms (python)

	b. **bitstream_analizer_openh264based**: h264-bitstream saliency map extractor (C/C++)

	c. **ffmpeg_video_handler_c**: FFMPEG library based video/camera frame extractor (C/C++)

	d. **fft_conformance**: FFT conformance test to determine performance and accuracy (C/C++)

	e. **gif_generator**: GIF generator application (C/C++)

	f. **h264_decoder_module_python**: FFMPEG video decoder (python)

	g. **hd5_rawImage_database_creator**: HDF5 database image file generator (python)

	h. **test_gstreamer_thread_priorities**: Gstreamer thread priority test (C/C++)

	i. **wavpack_gstreamer**: Gstreamer Wavpack plugin encoder and file writer with Metadata Tags (C/C++)

2. **computer_vision_img_vid**: Different computer vision algorithms implemented on CPU and GPU for image and video in raw and compressed domain **(H.264 standard)**, the folder is structured as follows:

	a. **canny_edge_detection**: Canny edge detection, fully implemented on GPU

	b. **colormap_extractor**: Color mapping extraction from RGB images

	c. **data_augmentation**: Python implemented data augmentation for input images

	d. **gstreamer**: further divided as follows:

		i. **gst_imgproc**: blob and skin detector on gstreamer, also moment normalization and color-retinex implementations on gstreamer.

		ii. **gst_rgb2gray**: RGB2GRAY implementation on gstreamer

		iii. **gst_rgbmapping**: Color mapping implementation for gstreamer

	e. **hough_transform_lines_circles**: Line and circle extraction using hough transform, fully implemented on GPU

	f. **LBP_extract_module**: LBP feature extraction from sample images

	g. **spatio_temporal_saliency_maps**: Static and Dynamic saliency mapping extraction from video/images

	h. **ToneMapping**: Color enhancement using Tone mapping algorithm

3. **metaheuristic_algorithms**: Implementation of several meta-heuristic algorithms including one developed during my master and PhD. degree. The folder is structured as follows:

	a. **VOA**: Virus Optimization Algorithm proposed for the first time in 2009 but accepted until 2014.

	b. **TBD**: more to be added in the future ...

4. **machine_and_deep_learning**: Experiments conducted on machine and deep learning algorithms, where two main frameworks were used: [Caffe](http://caffe.berkeleyvision.org/) and [Tensorflow](https://www.tensorflow.org/).

5. **caffe_own**: [Caffe](http://caffe.berkeleyvision.org/) repository with some modifications to support models and layers proposed over the past 4 years, for example: MobileNet, ShuffleNet, SSD, MaskRCNN, GAN, etc.


### Contributors

Josue R. Cuevas

josuercuevas@gmail.com

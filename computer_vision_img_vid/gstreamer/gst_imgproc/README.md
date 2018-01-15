# Image processing samples in Gstreamer

Multiple Gstreamer based image processing algorithms implementation, they are detailed as follow:

1. Frame Enhancement

2. Skin detection (with texture extraction)

3. Blob detection from skin patches extracted

4. Moment Normalization for blobs extracted

### Frame_enhancement:

Based in the Retinex Algorithm developed by [Edwin H. Land] for color consistency [http://en.wikipedia.org/wiki/Color_constancy], this element is based in the IPOL journal source code which depends on "libfftw3". The idea is to implement Retinex for a better and more robust skin detection, since for now we do this based on Thresholds for RGB or YUV frames. You can test this element by entering the following pipeline in your terminal:

>gst-launch-1.0 v4l2src ! videoconvert ! colorretinex ! autovideosink	(for live webcam input)

>gst-launch-1.0 filesrc location=source_file_path.mp4 ! qtdemux  ! h264parse ! avdec_h264 ! colorretinex ! autovideosink	(for recoded video input)

> IMPORTANT:
> 1. A learning mechanism for skin colors is under consideration in the future.
> 2. The threshold value for the retinex algorithm should be integer [0 ... 255], with a default value of 4.
> 3. This element is particularly slow, beware of take this into consideration if you want to do further processing in your pipeline

### Skin_detector:

RGB or YUV threshold based element, which determines patches of skin from an input frame, the only controllable parameter that could be input by the user is called "normalize" which by default is "0". The "normalize" parameter is used in case the user wants to visualize a black and white output frame when sinking the result coming from the Skin_detector, the reason of having this option is because the original result is a binary image (0-1) where the skin detection is marked by "1" while "0" means no skin was detected. The "normalize" will transform "1" to "255" while "0" remains the same. To launch the element and visualize its result just input the following pipeline into the terminal:

>gst-launch-1.0 v4l2src ! videoconvert ! skindetector normalize=1 ! autovideosink	(for live webcam input)

>gst-launch-1.0 filesrc location=source_file_path.mp4 ! qtdemux  ! h264parse ! avdec_h264 ! skindetector normalize=1 ! autovideosink	(for recoded video input)

### Blob_detector:

From the patches of skin detected (binary image), this element will extract the blobs in the image, therefore the output is a gray-level frame. Each gray level represent an independent blob, the Blob_detector uses an approach developed by [Kampei Baba], the idea is as follows:

>1. starting with processed_label = 2, go to step 2.
>2. Get bounding rectangle containing all skin-labeled (value of 1) pixels in the frame.
>3. If the bounding rectangle containing all skin-labeled (value of 1) pixels in the frame, is smaller than the allowed blob size go to Step 9.
>4. Scan the first column or row of the bounding rectangle found in Step 2 and when a pixel with value of "1" is found, means that this pixels has not been labeled as "processed", therefore start blob detection using Flood_fill algorithm or a fast version of it, labeling the pixels of the blob with the value in "processed_label".
>5. Estimate the size of the blob detected by the Flood_fill algorithm.
>6. If the derived blob has a size smaller or larger than the allowed one, reject it, otherwise store it.
>7. Increase the value of "processed_label" by 1.
>8. Go to step 2
>9. Exit the Blob_detector and output results

>####IMPORTANT NOTE: in our case we keep track of (xmin, xmax, ymin, ymax) inside the Fill_flood implementation, therefore saving computation time in estimating the blob dimensions or Bounding Box of the blob.

The below pipeline could be used if you'd like to test the Blob_detector element in Gstreamer, the "normalize=1" helps to visualize the gray label frame with values between 0-255, otherwise there is no control on the number contained in the output frame:

>gst-launch-1.0 v4l2src ! videoconvert ! skindetector ! blobdetector normalize=1 ! autovideosink	(for live webcam input)

>gst-launch-1.0 filesrc location=source_file_path.mp4 ! qtdemux  ! h264parse ! avdec_h264 ! skindetector ! blobdetector normalize=1 ! autovideosink	(for recoded video input)

> IMPORTANT: The blobs information (x, y, width, height) detected in this element are stored as metadata information on the output frame, which is later to be used in case normalization is performed. The definition of the meta information and commonly used structures can be found inside the "common" folder.

### Frame_normalization:

Based on Pei and Lin normalization [http://ntur.lib.ntu.edu.tw/bitstream/246246/142437/1/25.pdf] using image's moments (up to third moments), the idea is to normalize the blobs (or patches of skin) detected, making the result invariant to rotation, skewness, scale, and translation. Therefore, Easing further detections or comparison fo the blobs with predefined models. The algorithm is as follows:

>1. Compute Area, mean vector, covariance matrix, third-oder central moments of the incoming patch.
>2. Estimate eigenvalues and vectors from the covariance matrix.
>3. Compute the scale matrix "A".
>4. Compute the third-order central moments of the compacted image (predefined equations in the article using A matrix)
>5. Calculate tensor values from the third-oder central moments of the compacted image
>6. Estimate the rotation angle of the compacted image (to make the resulting normalization invariant to rotation)
>7. Normalize the Original image using the previously extracted information.

The below pipeline could be used if you'd like to test the Frame_normalization element in Gstreamer, the "normalize=1" helps to visualize the black/white frame, since the output normalization is a binary patch, additionally the patch has 2 properties called padx and pady, to define a PADDING in x,y in order to avoid indexing outside the patch when normalizing the image. The default values for padx=512 and pady=512:

>gst-launch-1.0 v4l2src ! videoconvert ! skindetector ! blobdetector ! momentnormalization normalize=1  padx=256 pady=256 ! autovideosink	(for live webcam input)

>gst-launch-1.0 filesrc location=source_file_path.mp4 ! qtdemux  ! h264parse ! avdec_h264 ! skindetector  padx=256 pady=256 ! blobdetector ! momentnormalization normalize=1 ! autovideosink	(for recoded video input)

> IMPORTANT: no pixel interpolation is performed yet, therefore some black pixels may appear betten white pixels in the resulting normalized frame. This is going to be fixed in future releases.

### Supported features

>- RGB colorspace frames
>- YUV colorspace frames
>- YUV, RGB and HSV colorspace for skin detection
>- No third library dependencies, except libfftw3 for the Frame_enhancement (colorretinex) element

### TODO's in the already developed elements

>- Different retinex threshold values for RGB, YUV colospaces, and channels (if possible)
>- Learning mechanism for skin colors
>- Pixel interpolation for the normalization element
>- Interlance frames support. (only planar for now)

### Full Pipelines examples

>gst-launch-1.0 v4l2src ! videoconvert ! colorretinex thres=2 ! skindetector ! blobdetector ! momentnormalization normalize=1 padx=256 pady=256 ! autovideosink	(for live webcam input)

>gst-launch-1.0 filesrc location=source_file_path.mp4 ! qtdemux  ! h264parse ! avdec_h264 ! colorretinex thres=2 ! skindetector ! blobdetector ! momentnormalization normalize=1  padx=256 pady=256 ! autovideosink	(for recoded video input in mp4)


## FIXES

1. Gst-object style coding of each element, in order to avoid memory overhead, therefore speeding up the processing of the incoming frames (Further improved)

2. Removal of memory allocations and copies for incoming and ongoing VideoFrames in the pipeline (further improved)

3. Gstbuffer metadata handling through video metadata functions, for query and extraction. Taking care of accessing the right metadata API after registration downstream.

4. Bug-fixes:
>- Color retinex: RGB frames with packed and planar format support (previously only planar format was supported)  (FIXED)
>- Skin detection: RGB frames with packed and planar format support (previously only planar format was supported)  (FIXED)
>- Blob detection: wrong indexing was found, which made the code crashed (FIXED)
>- Moment normalization: there is an indexing problem when normalizing the detected blobs,  (FIXED)


## Contributor

Josue R. Cuevas

josuercuevas@gmail.com

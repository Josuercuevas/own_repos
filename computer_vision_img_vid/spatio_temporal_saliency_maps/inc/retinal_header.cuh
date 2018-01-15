#include "cuda_includes.h"
#include "defines_values.h"

/*
	This function is in charge of managing all the process in order to enhance the visual quality of the frame 
	therefore it should get easier for the API to calculate the saliency maps, the summary is:
	
	Input:
		1.	The frames to be processed, defined when the user called the API
		2.	Width of the frame
		3.	Height of the frame
		4.	The number of frames or depth of the array for frames
		5.	The number of channels (1 for gray, 3 for RGB)
	Output:
		1.	The filtered data after implementing retinal filtering
*/
int retinal_main(unsigned char *frames[], unsigned char *filtered[], const int width, 
	const int height, const int n_frames, const int channels, const int pix_dist);



/*
	Function to compress the pixel values per block, or in simple words estimate the average
	intensity of a block, to later calculate the averge of the whole image
*/
extern __global__ void pixel_compression(unsigned char *intensities, float* average, float* roots, const int size, 
	const int channels, const int n_frames, const int pix_dist);



/*
	Will estimate the value need to perform the retina filtering
*/
extern __global__ void estimator(float* taos, float *average, const int maximum, const int minimum, 
	const int n_frames, const int ave_size);


/*
	Final step before the estimation of the correction values for the retinal filter enhancement
	of the frames
*/
extern __global__ void pixel_correction(unsigned char* intensities, unsigned char *RGB, int* Ds,int maximum, 
	int minimum, int width,	int height, const int channels, const int n_frames, const int pix_dist);
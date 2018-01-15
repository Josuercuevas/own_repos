#include "cuda_includes.h"
#include "defines_values.h"
#include <cmath>

/*
	Function in charge of managing all the interations for the estimation of the static
	saliency maps, using the thread and device predefined after calling it, its input arguments are:
	1. Frames to be processed by the function
	2. Cointainer to store the maps
	3. Width of each frame
	4. Height of each frame
	5. Number of frames to be processed
	6. The device to be used to process these frames (in case that we have multiple devices)

*/
int static_map_main(unsigned char *frames[], unsigned char *static_maps[],const int width, 
	const int height,const int n_maps, const int GPU_id, const int orientations, const int py_levels);

/*
	Function in charge of generating the gabor filters dynamically at running time, just once!
*/
void generate_gabor_filter(float *&filter, int &x_size, int &y_size, float sigma, float theta, float lambda, 
	float psi, float gamma);

/*
	Function in charge of generating the gaussian filters dynamically at running time, just once!
*/
void generate_gaussian_filter(float *&filter, int &x_size, int &y_size, float sigma, const int scale);


/*
	Function in charge of building the gaussian pyramids after obtaining the gaussian filters
*/
void gauss_pyramid(unsigned char *frames[], const int width, const int height, const int n_frames, 
	float *filter,	const int f_xsize, const int f_ysize);

/*
	kernel in charge of building the pyramid
*/
extern __global__ void pyramid_kernel(unsigned char *frames, const int n_frames, const int width, const int height, 
	const int filter_xsize,	const int filter_ysize);


/*
	function in charge of using gabor banks
*/
void gabor_banks(unsigned char *frames[], float *d_accumulator, const int n_frames, const int width, 
	const int height, float *filter, const int f_xsize, const int f_ysize);

/*
	kernel in charge of performing gabor convolution
*/
extern __global__ void gabor_conv(float *accumulator, const int n_frames, const int width, const int height, 
	const int filter_xsize, const int filter_ysize);

/*
	normalizing the values of the map in order to how in a 8-bit image
*/
extern __global__ void normalization_maps(float *values, unsigned char *normalized, const int width, const int n_pixels, 
	float *maximum, float *minimum, const int n_maps);
extern __global__ void find_min_val(const float* values, float* d_min, const int n_pixels, const int n_maps);
extern __global__ void find_max_val(const float* values, float* d_max, const int n_pixels, const int n_maps);
extern __global__ void smooth(float *values, float *filter,	const int width, const int height, 
	const int filter_xsize,	const int filter_ysize, const int n_maps);
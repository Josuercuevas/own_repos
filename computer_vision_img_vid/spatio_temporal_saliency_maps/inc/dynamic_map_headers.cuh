#include "cuda_includes.h"
#include "defines_values.h"
//#include <CImg.h>
//
//using namespace cimg_library;

/*
	Function in charge of managing all the interations for the estimation of the dynamic
	saliency map, using the thread and device predefined after calling it, its input arguments are:
	1. Frames to be processed by the function
	2. Cointainer to store the map
	3. Width of each frame
	4. Height of each frame
	5. Number of frames to be processed
	6. The device to be used to process these frames (in case that we have multiple devices)

*/
int dynamic_map_main(unsigned char *frames[], unsigned char *dynamic_map[],const int width, 
	const int height,const int n_frames, const int GPU_id, float *ages);

/*
	Function in charge of calculating the gradients for every component in the saliency map, the ouput are 
	3 vectors constaining the gradients for the x and y and t components
*/
extern __global__ void Gradients(float *frames, float *Gx, float *Gy, float *Gt, 
	const int width, const int height, const int n_frames, float *ages, const int scale);

/*
	velocities estimation for the x and y components, taking into account the frame sequence or
	time in this case.
*/
extern __global__ void Velocities(float *Gx, float *Gy, float *Gt, float *Vx, float *Vy, const int width, 
	const int height, const int n_frames, const int iterations, const float alpha);

/*
	function in charge of estimating the velocity magnitud x and y
*/
extern __global__ void Magnitudes(float *magnitude, float *Vx, float *Vy, const int width, const int height,
	const int maps_per_frame);

/*
	function in charge of finding a maximum value for normalization of the magnitudes for the 
	velocities, which are floats. Therefore a normal atomic function will not do the job, 
	so we use reduction
*/
extern __global__ void find_max(const float* values, float* d_max, const int n_pixels, const int maps_per_frame);

/*
	normalization of the magnitudes
*/
extern __global__ void normalization(float *values, unsigned char* normalized, const int width,
	const int n_pixels, float *maximum, const int maps_per_frame);
extern __global__ void smooth(float *values, const int width, const int height, const int maps_per_frame);
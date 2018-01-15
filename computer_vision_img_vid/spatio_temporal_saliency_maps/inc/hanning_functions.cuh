#include "cuda_includes.h"
#include "defines_values.h"

/*
	Main function of the hanning filtering for noise reduction, controls all the interations
	between GPU and CPU
*/
int hanny_main(unsigned char *frames[], const int width, const int height, const int n_frames, 
	const int GPU_id);

/*
	denoising kernel is in charge of averaging the pixel (X,Y) of a given frame using all its
	neighbors in a window of size = winsize*winsize
*/
extern __global__ void denoising(unsigned char *frame, unsigned char *denoised_frame, const int width, 
	const int pixels, const int n_frames);

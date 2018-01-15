#include "cuda_includes.h"
#include "defines_values.h"

int compensation_main(unsigned char *frames[], const int width, const int height, const int n_frames, 
	const int channels, bool first_frame);

__global__ void motion_estimation_kernel(unsigned char *frames, const int width, const int height, 
	const int n_frames, const int channels, const bool first_frame);
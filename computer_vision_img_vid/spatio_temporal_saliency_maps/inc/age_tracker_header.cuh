#include "cuda_includes.h"
#include "defines_values.h"

/*
	main function ocoontrolling the tracking of the pixels age
*/
int fade_tracker_main(unsigned char *maps[], const int width, const int height, const int n_maps, float *ages);

extern __global__ void fade_implementation(unsigned char *maps, float *ages, const int n_pixels, const int n_maps); 

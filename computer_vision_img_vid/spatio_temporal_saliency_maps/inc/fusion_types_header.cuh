#include "cuda_includes.h"
#include "defines_values.h"

/*
	function in charge of fusing the maps using their average
*/
int average_fusion_main(unsigned char *maps[], unsigned char *before_fusion[],	const int width, 
		const int height, const int n_maps);

/*
	kernel that averages both maps, static and dynamic
*/
extern __global__ void average_kernel(unsigned char *map, const int width, const int height, const int n_maps);


/*
	function in charge of fusing the maps using their weighted average
*/
int w_average_fusion_main(unsigned char *maps[], unsigned char *before_fusion[],	const int width, 
		const int height, const int n_maps);

/*
	kernel that averages both maps, static and dynamic
*/
extern __global__ void w_average_kernel(unsigned char *map, const int width, const int height, const int n_maps
	, const float weight);


/*
	function in charge of fusing the maps using their normalized sum
*/
int n_sum_fusion_main(unsigned char *maps[], unsigned char *before_fusion[],	const int width, 
		const int height, const int n_maps);

/*
	kernel that averages both maps, static and dynamic
*/
extern __global__ void sum_kernel(float *map, const int width, const int height, const int n_maps);
extern __global__ void max_kernel(float* d_min, const int n_pixels, const int n_maps);
extern __global__ void min_kernel(float* d_min, const int n_pixels, const int n_maps);
extern __global__ void normalization_kernel(unsigned char *map, const int width, const int height, 
	const int n_maps, float *d_max, float *d_min);
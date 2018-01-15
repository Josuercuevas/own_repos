#include "cuda_includes.h"
#include "defines_values.h"

int alpha_blending_main(unsigned char *blend_frames[], unsigned char *frames[], unsigned char *maps[], 
	const int width, const int height, const int n_maps, const int channels, const int pix_dist);

__device__ inline void dilute(unsigned char fore_pix, unsigned char back_pix, unsigned char &blended_pix);

__global__ void blending_frames(unsigned char *maps,unsigned char *blend_frames, const int n_pixels, const int n_maps, 
	const int channels, const int pix_dist);
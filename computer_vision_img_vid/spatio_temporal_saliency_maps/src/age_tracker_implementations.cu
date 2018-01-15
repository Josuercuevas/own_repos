#include "cpp_includes.h"
#include "math_functions.h"
#include "age_tracker_header.cuh"

int fade_tracker_main(unsigned char *maps[], const int width, const int height, const int n_maps, float *ages){
	const int n_pixels = width*height, BLOCKS = (THREADS+n_pixels-1)/THREADS;
	float *d_ages;
	unsigned char *d_frames;
	cudaMalloc((void**)&d_frames,sizeof(unsigned char)*width*height*n_maps);
	cudaMalloc((void**)&d_ages,sizeof(float)*width*height);

	cudaMemcpyAsync(d_ages,ages,sizeof(float)*width*height,cudaMemcpyHostToDevice);
	for(int frame=0;frame<n_maps;frame++)
		cudaMemcpyAsync(d_frames+frame*n_pixels,maps[frame],sizeof(unsigned char)*width*height,cudaMemcpyHostToDevice);

	fade_implementation<<<BLOCKS,THREADS>>>(d_frames,d_ages,n_pixels,n_maps);

	for(int frame=0;frame<n_maps;frame++)
		cudaMemcpyAsync(maps[frame],d_frames+frame*n_pixels,sizeof(unsigned char)*width*height,cudaMemcpyDeviceToHost);

	cudaFree(d_ages);
	cudaFree(d_frames);

	return 0;//exits without a problem
}


//FUNCTION TO ESTIMATE THE AGE OF EVERY PIXEL IN THE SALIENCY MAP
__global__ void fade_implementation(unsigned char *maps, float *ages, const int n_pixels, const int n_maps){
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

	if(idx>=n_pixels) return;

	for(int frame=0;frame<n_maps;frame++){
		float current_brightness = ((float)maps[idx + frame*n_pixels]) - ages[idx];//determines the brightness after checking the pixel's age
		if(current_brightness > 0)
			maps[idx + frame*n_pixels] = (unsigned char)((int)current_brightness);
		else
			maps[idx + frame*n_pixels] = (unsigned char)(0);
	}
}
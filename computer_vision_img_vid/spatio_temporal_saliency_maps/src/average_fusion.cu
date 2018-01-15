#include "cpp_includes.h"
#include "fusion_types_header.cuh"
#include "math_functions.h"


texture<unsigned char,1,cudaReadModeElementType> tex_dynamic;
texture<unsigned char,1,cudaReadModeElementType> tex_static;

int average_fusion_main(unsigned char *maps[], unsigned char *before_fusion[],	const int width, const int height, const int n_maps){
	/* 
		the variable "before_fusion" brings the maps together ready for fusion
	*/
	const int n_pixels = width*height, BLOCKS = (n_pixels+THREADS-1) / THREADS;
	unsigned char *d_static_maps, *d_dynamic_maps, *d_combined;
	cudaMalloc((void**)&d_static_maps,sizeof(unsigned char)*n_pixels*n_maps);
	cudaMalloc((void**)&d_dynamic_maps,sizeof(unsigned char)*n_pixels*n_maps);
	cudaMalloc((void**)&d_combined,sizeof(unsigned char)*n_pixels*n_maps);
    for(int frame=0;frame<n_maps;frame++){
		cudaMemcpyAsync(d_static_maps+(frame*n_pixels),before_fusion[frame],sizeof(unsigned char)*n_pixels,cudaMemcpyHostToDevice);
		cudaMemcpyAsync(d_dynamic_maps+(frame*n_pixels), before_fusion[n_maps + frame],sizeof(unsigned char)*n_pixels,cudaMemcpyHostToDevice);
	}

	cudaBindTexture(NULL,tex_static,d_static_maps,sizeof(unsigned char)*n_pixels*n_maps);
	cudaBindTexture(NULL,tex_dynamic,d_dynamic_maps,sizeof(unsigned char)*n_pixels*n_maps);

	average_kernel<<<BLOCKS,THREADS>>>(d_combined,width,height,n_maps);

	for(int frame=0;frame<n_maps;frame++)
		cudaMemcpy(maps[frame], d_combined+(frame*n_pixels),sizeof(unsigned char)*n_pixels,cudaMemcpyDeviceToHost);

	cudaFree(d_static_maps);
	cudaFree(d_dynamic_maps);
	cudaFree(d_combined);

	cudaUnbindTexture(tex_static);
	cudaUnbindTexture(tex_dynamic);
	
	return 0;//exists without a problem
}




__global__ void average_kernel(unsigned char *map, const int width, const int height, const int n_maps){
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int n_pixels = width*height;

	if(idx>=n_pixels) return;//inactive threads

	float static_map;
	float dynamic_map;

	for(int frame=0;frame<n_maps;frame++){//combining static and dynamic maps for each frame
		static_map = (float)tex1Dfetch(tex_static,idx + frame*n_pixels);
		dynamic_map = (float)tex1Dfetch(tex_dynamic,idx + frame*n_pixels);
		map[idx + frame*n_pixels] = (unsigned char)(0.5*(static_map + dynamic_map));
	}
}
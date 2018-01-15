#include "cpp_includes.h"
#include "fusion_types_header.cuh"
#include "math_functions.h"

texture<unsigned char,1,cudaReadModeElementType> tex_dynamic;
texture<unsigned char,1,cudaReadModeElementType> tex_static;
texture<float,1,cudaReadModeElementType> tex_summation;

int n_sum_fusion_main(unsigned char *maps[], unsigned char *before_fusion[],	const int width, const int height, const int n_maps){
	/* 
		the variable "before_fusion" brings the maps together ready for fusion
	*/
	const int n_pixels = width*height, BLOCKS = (n_pixels+THREADS-1) / THREADS;
	unsigned char *d_static_maps, *d_dynamic_maps, *d_maps;
	float *d_min, *d_max, *d_combined;

	cudaMalloc((void**)&d_max,sizeof(float)*n_maps);
	cudaMalloc((void**)&d_min,sizeof(float)*n_maps);
	cudaMalloc((void**)&d_static_maps,sizeof(unsigned char)*n_pixels*n_maps);
	cudaMalloc((void**)&d_dynamic_maps,sizeof(unsigned char)*n_pixels*n_maps);
	cudaMalloc((void**)&d_combined,sizeof(float)*n_pixels*n_maps);
	cudaMalloc((void**)&d_maps,sizeof(float)*n_pixels*n_maps);

    for(int frame=0;frame<n_maps;frame++){
		cudaMemcpyAsync(d_static_maps+(frame*n_pixels),before_fusion[frame],sizeof(unsigned char)*n_pixels,cudaMemcpyHostToDevice);
		cudaMemcpyAsync(d_dynamic_maps+(frame*n_pixels), before_fusion[n_maps+frame],sizeof(unsigned char)*n_pixels,cudaMemcpyHostToDevice);
	}

	cudaBindTexture(NULL,tex_static,d_static_maps,sizeof(unsigned char)*n_pixels*n_maps);
	cudaBindTexture(NULL,tex_dynamic,d_dynamic_maps,sizeof(unsigned char)*n_pixels*n_maps);

	sum_kernel<<<BLOCKS,THREADS>>>(d_combined,width,height,n_maps);//calculates the sums

	cudaBindTexture(NULL,tex_summation,d_combined,sizeof(float)*n_pixels*n_maps);//binds the summations to texture for a faster access

	max_kernel<<<BLOCKS,THREADS>>>(d_max,width*height,n_maps);
	min_kernel<<<BLOCKS,THREADS>>>(d_min,width*height,n_maps);
	normalization_kernel<<<BLOCKS,THREADS>>>(d_maps,width,height,n_maps,d_max,d_min);


	for(int frame=0;frame<n_maps;frame++)
		cudaMemcpy(maps[frame], d_maps+(frame*n_pixels),sizeof(unsigned char)*n_pixels,cudaMemcpyDeviceToHost);

	cudaFree(d_max);
	cudaFree(d_min);
	cudaFree(d_static_maps);
	cudaFree(d_dynamic_maps);
	cudaFree(d_combined);
	cudaFree(d_maps);

	cudaUnbindTexture(tex_static);
	cudaUnbindTexture(tex_dynamic);
	cudaUnbindTexture(tex_summation);

	return 0;//exists without a problem
}

__global__ void normalization_kernel(unsigned char *map, const int width, const int height, const int n_maps, float *d_max, float *d_min){
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int n_pixels = width*height;

	if(idx>=n_pixels) return;//inactive threads

	for(int frame=0;frame<n_maps;frame++){//combining static and dynamic maps for each frame
		map[idx + frame*n_pixels] = (unsigned char)((tex1Dfetch(tex_summation,idx + frame*n_pixels)-d_min[frame])/(d_max[frame]-d_min[frame])*255.0);
	}
}

__global__ void sum_kernel(float *map, const int width, const int height, const int n_maps){
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int n_pixels = width*height;

	if(idx>=n_pixels) return;//inactive threads

	for(int frame=0;frame<n_maps;frame++){//combining static and dynamic maps for each frame
		map[idx + frame*n_pixels] = ((float)tex1Dfetch(tex_static,idx + frame*n_pixels) + (float)tex1Dfetch(tex_dynamic,idx + frame*n_pixels));
	}
}

__device__ float Maxf_val(float* address, float val){
    int *address_as_int =(int*)address;
    int old = *address_as_int, assumed;
    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,__float_as_int(val));
        }
    return __int_as_float(old);
}


__global__ void max_kernel(float* d_max, const int n_pixels, const int n_maps){
	__shared__ float shared[THREADS];
	for(int frame=0;frame<n_maps;frame++){
		int tid = threadIdx.x;
		int gid = (blockDim.x * blockIdx.x) + tid;
		shared[tid] = 0.0f;

		if (gid < n_pixels)
			shared[tid] = tex1Dfetch(tex_summation,gid + frame*n_pixels);
		__syncthreads();

		for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
		{
			if (tid < s && gid < n_pixels)
				shared[tid] = max(shared[tid], shared[tid + s]);
			__syncthreads();
		}
		if (tid == 0)
			float a = Maxf_val(&d_max[frame], shared[0]);
	}
}


__device__ float Minf_val(float* address, float val){
    int *address_as_int =(int*)address;
    int old = *address_as_int, assumed;
    while (val < __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,__float_as_int(val));
        }
    return __int_as_float(old);
}


__global__ void min_kernel(float* d_min, const int n_pixels, const int n_maps){
	__shared__ float shared[THREADS];

	for(int frame=0;frame<n_maps;frame++){
		int tid = threadIdx.x;
		int gid = (blockDim.x * blockIdx.x) + tid;
		shared[tid] = 0.0f;

		if (gid < n_pixels)
			shared[tid] = tex1Dfetch(tex_summation,gid + frame*n_pixels);
		__syncthreads();

		for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
		{
			if (tid < s && gid < n_pixels)
				shared[tid] = min(shared[tid], shared[tid + s]);
			__syncthreads();
		}
		if (tid == 0)
			float a = Minf_val(&d_min[frame], shared[0]);
	}
}
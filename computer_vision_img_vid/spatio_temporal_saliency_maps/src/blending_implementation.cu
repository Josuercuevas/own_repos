#include "cpp_includes.h"
#include "math_functions.h"
#include "blending_header.cuh"

texture<unsigned char,1,cudaReadModeElementType> Map_intensities;
texture<unsigned char,1,cudaReadModeElementType> RGB_intensities;


int alpha_blending_main(unsigned char *blend_frames[], unsigned char *frames[], unsigned char *maps[], const int width, const int height, const int n_maps, const int channels, const int pix_dist){
	
	unsigned char *d_frames, *d_maps;//pointers to the original frames and the maps
	unsigned char *d_blended;//pointer to the blended frames
	const int n_pixels = width*height;
	const int BLOCKS = (n_pixels+THREADS-1)/THREADS;
	size_t frame_size = sizeof(unsigned char)*n_pixels*channels;
	size_t sequence_size = frame_size*n_maps;

	cudaMalloc((void**)&d_frames,sequence_size);//Since is RGB
	cudaMalloc((void**)&d_maps,sizeof(unsigned char)*n_pixels*n_maps);
	cudaMalloc((void**)&d_blended,sequence_size);

	for(int frame=0;frame<n_maps;frame++){//copying the frames and maps to the pointer on device to later bind them to texture
		cudaMemcpyAsync(d_frames+(frame*n_pixels*channels),frames[frame],frame_size,cudaMemcpyHostToDevice);//copy the original frames
		cudaMemcpyAsync(d_maps+(frame*n_pixels),maps[frame],sizeof(unsigned char)*n_pixels,cudaMemcpyHostToDevice);//copies the maps
	}

	cudaBindTexture(NULL,RGB_intensities,d_frames,sequence_size);//Binding original frames which are RGB
	cudaBindTexture(NULL,Map_intensities,d_maps,sizeof(unsigned char)*n_pixels*n_maps);//binding maps


	blending_frames<<<BLOCKS,THREADS>>>(d_maps,d_blended,n_pixels,n_maps,channels,pix_dist);

	for(int frame=0;frame<n_maps;frame++){//copying the blended frames back to the host
		cudaMemcpyAsync(blend_frames[frame],d_blended+(frame*n_pixels*channels),frame_size,cudaMemcpyDeviceToHost);
	}

	cudaFree(d_frames);
	cudaFree(d_maps);
	cudaFree(d_blended);
	cudaUnbindTexture(RGB_intensities);
	cudaUnbindTexture(Map_intensities);

	return 0;//exits without a problem
}


__device__ inline void dilute(unsigned char fore_pix, unsigned char back_pix, unsigned char &blended_pix){
	//doing the blending by diluting the RGB image and enhancing the blending with the maps
	blended_pix = (unsigned char)(int)(alpha_blend*(float)back_pix + (1.0-alpha_blend)*(float)fore_pix);
}

__global__ void blending_frames(unsigned char *maps,unsigned char *blend_frames, const int n_pixels, const int n_maps, const int channels, const int pix_dist){
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
	
	if(idx>=n_pixels) return;

	unsigned char original_frame; //For RGB if we have 3 channels
	unsigned char mapped_frame; //for the mapped values
	unsigned char diluted_pixel; //value that contains the diluted values of the pixels
	const int n_channels = channels;

	for(int frame=0;frame<n_maps;frame++){
		mapped_frame = tex1Dfetch(Map_intensities,idx + frame*n_pixels);
		for(int cha=0;cha<n_channels;cha++){
			if(pix_dist == 1){//interlace
				original_frame = tex1Dfetch(RGB_intensities,idx*n_channels + cha + frame*n_pixels*n_channels);//RGB frames in the sequence
				dilute(mapped_frame,original_frame,diluted_pixel);
				blend_frames[idx*n_channels + cha + frame*n_pixels*n_channels] = diluted_pixel;
			}
			else{//planar
				original_frame = tex1Dfetch(RGB_intensities,idx + cha*n_pixels + frame*n_pixels*n_channels);//RGB frames in the sequence
				dilute(mapped_frame,original_frame,diluted_pixel);
				blend_frames[idx + cha*n_pixels + frame*n_pixels*n_channels] = diluted_pixel;
			}
		}
	}
}
#include "cuda_includes.h"
#include "cpp_includes.h"
#include "math_functions.h"
#include "hanning_functions.cuh"


int hanny_main(unsigned char *frames[], const int width, const int height, const int n_frames, const int GPU_id){
	int n_gpus;

	//printf("%s\n",cudaGetErrorString(cudaGetLastError()));//in case we want to see that every part of the cuda implementation is right

	cudaGetDeviceCount(&n_gpus);
	if(n_gpus>1)
		//we use the thread id coming from the main program to identify which device to use
		cudaSetDevice(GPU_id);
	else if (n_gpus==1)
		//if no multi_gpu is supported we just use the default device
		cudaSetDevice(0);
	else{//is unlikely to get here since we will detect this problem during the frame enhancement process
		printf("There is no GPU support for your implementation!! ... \n");
		system("pause");
		return 1;//exits with an error code
	}
		
	//number of pixels to be processed per frame
	const int n_pixels = width*height;	
	//size in bytes of one single frame to be processed
	size_t memory_bytes = sizeof(unsigned char)*n_pixels;
	//the pointers in charge of containing the frames
	unsigned char *d_frames, *d_processed;
	//for cuda implementation
	const int BLOCKS = ((n_pixels+THREADS-1)/THREADS);

	//allocating the values per device at each frame
	cudaMalloc((void**)&d_frames,memory_bytes*n_frames);//will have a copy of the frames
	cudaMalloc((void**)&d_processed,memory_bytes*n_frames);//will contain the denoised values
	//printf("%s\n",cudaGetErrorString(cudaGetLastError()));//in case we want to see that every part of the cuda implementation is right

	//copy the frames on host as a 1D array
	for(int frame=0;frame<n_frames;frame++)
		cudaMemcpy(d_frames+n_pixels,frames[frame],memory_bytes,cudaMemcpyHostToDevice);	
	//printf("%s\n",cudaGetErrorString(cudaGetLastError()));//in case we want to see that every part of the cuda implementation is right

	//call the kernel to process all the frames at the same time in the GPU assigned to do so
	denoising<<<BLOCKS,THREADS>>>(d_frames,d_processed,width,n_pixels,n_frames);
	//printf("%s\n",cudaGetErrorString(cudaGetLastError()));//in case we want to see that every part of the cuda implementation is right

	//gets back the denoised frames
	for(int frame=0;frame<n_frames;frame++)
		cudaMemcpyAsync(frames[frame],d_processed+n_pixels,memory_bytes,cudaMemcpyDeviceToHost);
	//printf("%s\n",cudaGetErrorString(cudaGetLastError()));//in case we want to see that every part of the cuda implementation is right

	cudaFree(d_processed);
	cudaFree(d_frames);
	//printf("%s\n",cudaGetErrorString(cudaGetLastError()));//in case we want to see that every part of the cuda implementation is right
	//system("pause");

	return 0;//exits without problems
}


__global__ void denoising(unsigned char *frames, unsigned char *denoised_frame, const int width, const int pixels, const int n_frames){
	unsigned int pix_id = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int tid = threadIdx.x;//a block of "THREADS" threads (specified variable at the defines_values.h file)

	if(pix_id>=pixels) return;

	const int winsize = 3, right = winsize/2, left = winsize/2, shift = winsize/2;//widnow size to do the denoising
	int x = pix_id%width, y = pix_id/width;
	int position;//tells me where in the frame the thread is located
	__shared__ int intensities[winsize][left+THREADS+right];

	for(int frame=0;frame<n_frames;frame++){

		//copy the pixel values to shared memory because we do have multiple accesses for them
		if(x>=shift && x<=(width-shift) && y>=shift && y<((pixels/width)-shift)){//just checking we are inside the frame
			if(tid>0 && tid<(blockDim.x-1)){
				//copies only up and down
				for(int i=0;i<winsize;i++){
					position = pix_id + ((i-(shift))*width);
					intensities[i][tid+left] = (int)frames[position + frame*pixels];//copies only up and down of the current pixels
				}
			}
			else{//this part is the boundary of the window therefore we need to copy up and down as well as left for tid=0 and right for tid=255
				if(tid==0){//means tid=255, therefore, copy left
					for(int j=0;j<=left;j++)
					{
						for(int i=0;i<winsize;i++)
						{
							position = pix_id + ((i-(shift))*width) - j;
							intensities[i][left-j] = (int)frames[position + frame*pixels];
						}
					}
				}
				else if(tid==(blockDim.x-1)){//means tid=255, therefore, copy right                                                                                                                                                                                                                                                                                                                                                                                                                s only up and down of the pixels to the right of the 255-shift position
					for(int j=0;j<=right;j++)
					{
						for(int i=0;i<winsize;i++)
						{
							position = pix_id + ((i-(shift))*width) + j;
							intensities[i][(tid+left)+j] = (int)frames[position + frame*pixels];
						}
					}
				}
			}
		}
		__syncthreads();

		//starting the hanning denoising for every pixel in the frame
		float pix_denoised = 0;
		if(x>shift && x<(width-shift) && y>shift && y<((pixels/width)-shift)){//just checking we are inside the allowable borders
			pix_denoised += (intensities[shift][(tid+left)]*0.5);//center
			pix_denoised += (intensities[shift+1][(tid+left)-shift]*0.25);//left
			pix_denoised += (intensities[shift+1][(tid+left)+shift]*0.25);//right

			//if we take the average of the whole window, we want to have a squared window and not linear as the above one
			//float hanning_weigth_center = 2.0*(1.0/(winsize*winsize));//the weight at the center (double the weight)
			//float hanning_weigth_neighbors = (1.0-hanning_weigth_center)/(winsize*winsize-1);//the weights for neightbors

			//for(int i=-shift;i<=shift;i++){//denoising the pixel		
			//	for(int j=-shift;j<=shift;j++){ 
			//		/******************************performing convolution******************************************/
			//		if(i==0 && j==0)//at the center of the window
			//			pix_denoised += (intensities[shift+i][(tid+left)+j]*hanning_weigth_center);
			//		else//not the center
			//			pix_denoised += (intensities[shift+i][(tid+left)+j]*hanning_weigth_neighbors);
			//	}
			//}
		}
		__syncthreads();

		if(x>shift && x<(width-shift) && y>shift && y<((pixels/width)-shift)){//just checking we are inside the frame
			int pix_val = (int)pix_denoised;;//since is an integer not a float
			if(pix_val<=255)//just to make sure we dont surpass the allowable values ... this if() statement is just for safety ... most likely nevers takes the ELSE path
				denoised_frame[pix_id + frame*pixels] = (unsigned char)pix_val;
			else
				denoised_frame[pix_id + frame*pixels] = (unsigned char)255;
		}
		else{//we use the original value of the frame
			denoised_frame[pix_id + frame*pixels] = (unsigned char)frames[pix_id + frame*pixels];
		}
	}
}

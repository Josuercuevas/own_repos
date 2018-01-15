#include "cpp_includes.h"
#include "math_functions.h"
#include "static_map_header.cuh"

texture<float,1,cudaReadModeElementType> pyramid_filter;
texture<unsigned char,1,cudaReadModeElementType> intensities;

/******************************************************** GAUSSIAN PYRAMIDS *****************************************************************/
void gauss_pyramid(unsigned char *frames[], const int width, const int height, const int n_frames, float *filter, const int f_xsize, const int f_ysize){
	//calling kernels in charge of building pyramid
	const int n_pixels = width*height;
	const int BLOCKS = (n_pixels+THREADS-1)/THREADS;
	size_t frame_size = sizeof(unsigned char)*n_pixels, filter_size = sizeof(float)*f_xsize*f_ysize;
	unsigned char *d_frames;
	float *d_filter;
	cudaMalloc((void**)&d_frames,frame_size*n_frames);//allocating the necessary memory on device for the processing of each frame (all the frames at the same time)
	cudaMalloc((void**)&d_filter,filter_size);
	cudaMemcpyAsync(d_filter,filter,filter_size,cudaMemcpyHostToDevice);
	for(int frame=0;frame<n_frames;frame++)
		cudaMemcpyAsync(d_frames+(frame*n_pixels),frames[frame],frame_size,cudaMemcpyHostToDevice);//copy all the frames at the same time
	
	//call kernel for one pyramid in all frames
	cudaBindTexture(NULL,intensities,d_frames,frame_size*n_frames);
	cudaBindTexture(NULL,pyramid_filter,d_filter,filter_size);
	pyramid_kernel<<<BLOCKS,THREADS>>>(d_frames, n_frames,width,height,f_xsize,f_ysize);

	for(int frame=0;frame<n_frames;frame++)
		cudaMemcpyAsync(frames[frame],d_frames+(frame*n_pixels),frame_size,cudaMemcpyDeviceToHost);//copy all the frames at the same time
	//freeing memory
	cudaUnbindTexture(intensities);
	cudaUnbindTexture(pyramid_filter);
	cudaFree(d_frames);
	cudaFree(d_filter);
}


__global__ void pyramid_kernel(unsigned char *frames, const int n_frames, const int width, const int height, const int filter_xsize, const int filter_ysize){
	unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x, tid = threadIdx.x;
	int n_pixels = width*height;
	

	if(idx>=n_pixels) return;//inactive threads

	const int f_xsize = filter_xsize, f_ysize = filter_ysize;
	const int filter_y = f_ysize/2, filter_x = f_xsize/2, shift = f_ysize/2;//to help me control the window for the filter
	int x=idx%width, y=idx/width;
	int position;//determines the position of the pixel
	__shared__ float s_filter[Py_FILTER_SIZE*Py_FILTER_SIZE];

	if(tid<f_ysize){//each thread copies a row
		for(int i=0;i<f_xsize;i++)
			s_filter[tid*f_xsize + i] = tex1Dfetch(pyramid_filter,tid*f_xsize + i);
	}
	__syncthreads();


	//CONSTRUCTING PYRAMID
	//we are allowed to contruct the pyramid for all frames, if we are inside the pictures, dont surpass the borders and the windows dont overlap
	for(int frame=0;frame<n_frames;frame++){//using the same filter for all the frames
		float pix_val = 0;
		
		//using the filter
		if(x>=filter_x && x<(width-filter_x) && y>=filter_y && y<(height-filter_y) && x%f_xsize==0 && y%f_ysize==0){// 
			int pos_y=0;
			for(int i=-filter_y; i<=filter_y; i++){//height of the filter
				int pos_x = 0;
				for(int j=-filter_x; j<=filter_x; j++){//width of the filter
					position = idx + i*width + j;//position of the pixel used in the window for building the pyramid
					pix_val += s_filter[pos_y*f_xsize + pos_x]*(float)tex1Dfetch(intensities,position + frame*n_pixels);//inte[filter_y+i][tid+filter_x+j];//
					pos_x++;
				}
				pos_y++;
			}
		}
		__syncthreads();

		if(x>=filter_x && x<(width-filter_x) && y>=filter_y && y<(height-filter_y) && x%f_xsize==0 && y%f_ysize==0){//
			if(pix_val>=0 && pix_val<=255)
				for(int i=-filter_y; i<=filter_y; i++){//height of the filter
					for(int j=-filter_x; j<=filter_x; j++){//width of the filter
						position = idx + i*width + j;//position of the pixel used in the window for building the pyramid						
							frames[position + frame*n_pixels] = (unsigned char)ceil(pix_val);
					}
				}
		}

		__syncthreads();
	}
}

/***************************************************** END Gaussian Pyramids*****************************************************************/



/************************* FUNCTION IN CHARGE OF GETTING THE GAUSSSIAN FILTER DYNAMICALLY *******************************/
void generate_gaussian_filter(float *&filter, int &x_size, int &y_size, float sigma, const int scale){
	
	x_size = 2*scale + 1;//calcualte the size of the window, for example,if scale=1 then is a 3x3 windows
	y_size = x_size;
	filter = new float[x_size*y_size];//size of the filter and memory allocation
	int fy=0;
	float accum= 0.0;
	for(int y=-scale;y<=scale;y++){//generating the values
		int fx=0;
		for(int x=-scale;x<=scale;x++){
			filter[fy*x_size + fx] = (1.0/(2.0*pi_val*sigma*sigma))*exp(-0.5*(((x*x)+(y*y))/(2*sigma*sigma)));
			accum += filter[fy*x_size + fx];
			fx++;
		}
		fy++;
	}
	/*
		Note: the values just generated have to be normalized because their weight is larger than 1.0
		therefore the best way to normilize them is by dividing the each member by the sum of the whole whindow
	*/
	fy = 0;
	for(int y=-scale;y<=scale;y++){//normalizing values
		int fx=0;
		for(int x=-scale;x<=scale;x++){
			filter[fy*x_size + fx] /= accum;
			///printf("%4.10f\t",filter[fy*x_size + fx]);
			fx++;
		}
		//printf("\n");
		fy++;
	}
	//printf("\n\n");
}
/************************* FUNCTION IN CHARGE OF GETTING THE GAUSSSIAN FILTER DYNAMICALLY *******************************/
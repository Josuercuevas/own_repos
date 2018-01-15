#include "cpp_includes.h"
#include "math_functions.h"
#include "static_map_header.cuh"

texture<float,1,cudaReadModeElementType> gabor_filter;
texture<unsigned char,1,cudaReadModeElementType> intensities;

/****************************Gabor banks***********************************************/
void gabor_banks(unsigned char *frames[], float *d_accumulator, const int width, const int height, const int n_frames, float *filter,	const int f_xsize, const int f_ysize){
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
	cudaBindTexture(NULL,gabor_filter,d_filter,filter_size);
	
	//now convolve the image here for each scale in the different orientations	
	gabor_conv<<<BLOCKS,THREADS>>>(d_accumulator,n_frames,width,height,f_xsize,f_ysize);
	//std::cout<<cudaGetErrorString(cudaGetLastError())<<std::endl;
	
	//freeing memory
	cudaUnbindTexture(intensities);
	cudaUnbindTexture(gabor_filter);
	cudaFree(d_frames);
	cudaFree(d_filter);
}

__global__ void gabor_conv(float *accumulator, const int n_frames, const int width, const int height, const int filter_xsize, const int filter_ysize){
	unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x, tid = threadIdx.x;
	int n_pixels = width*height;
	

	if(idx>=n_pixels) return;//inactive threads

	const int f_xsize = filter_xsize, f_ysize = filter_ysize;
	const int filter_y = f_ysize/2, filter_x = f_xsize/2;//to help me control the window for the filter
	int x=idx%width, y=idx/width;
	int position;//determines the position of the pixel
	__shared__ float s_filter[Gb_FILTER_SIZE*Gb_FILTER_SIZE];//gabor filters

	if(tid<f_ysize){//each thread which returns true is allowed to copy a row of the filter
		for(int i=0;i<f_xsize;i++){
			s_filter[tid*f_xsize + i] = tex1Dfetch(gabor_filter,tid*f_xsize + i);
		}
	}
	__syncthreads();

	//using gabor banks
	//we are allowed to contruct the pyramid for all frames, if we are inside the pictures, dont surpass the borders and the windows dont overlap
	for(int frame=0;frame<n_frames;frame++){//using the same filter for all the frames	
		
		float pix_val = 0;

		if(x>=filter_x && x<(width-filter_x) && y>=filter_y && y<(height-filter_y)){												
			//using the filter
			int pos_y=0;
			for(int i=-filter_y; i<=filter_y; i++){//height of the filter
				int pos_x = 0;
				for(int j=-filter_x; j<=filter_x; j++){//width of the filter
					position = idx + i*width + j;//position of the pixel used in the window for implementing gabor
					pix_val += s_filter[pos_y*f_xsize + pos_x]*(float)tex1Dfetch(intensities,position + frame*n_pixels);//
					pos_x++;
				}
				pos_y++;
			}
		}
		__syncthreads();

		if(x>=filter_x && x<(width-filter_x) && y>=filter_y && y<(height-filter_y)){
			if(pix_val>0)
				accumulator[idx + frame*n_pixels] += pix_val;
		}
		__syncthreads();
	}
}
/************************* End gabor banks ******************************************/






/************************* FUNCTION IN CHARGE OF GETTING THE GABOR FILTER DYNAMICALLY *******************************/
void generate_gabor_filter(float *&filter, int &x_size, int &y_size, float sigma, float theta, float lambda, 
	float psi, float gamma){
	float sigma_x = sigma;
	float sigma_y = sigma/gamma;
 
	//filter dimensionality
	float nstds = 3;
	float temp = max(abs(nstds*sigma_x*cos(theta)),abs(nstds*sigma_y*sin(theta)));
	int xmax = ceil(max(1.0,temp));//# of rows
	temp = max(abs(nstds*sigma_x*sin(theta)),abs(nstds*sigma_y*cos(theta)));
	int ymax = ceil(max(1.0,temp));//# of columns
	int xmin = -xmax, ymin = -ymax;

	x_size = (xmax-xmin)+1; y_size = (ymax-ymin)+1;// columns, rows
	filter = new float[x_size*y_size];//size of the filter
	//looping for each coefficient in the filter
	int fy=0; 
	for(int y=ymin;y<=ymax;y++){//row
		int fx=0; 
		for(int x=xmin;x<=xmax;x++)//colum
		{
			//rotation of every x and y
			float x_theta=x*cos(theta)+y*sin(theta);
			float y_theta=-x*sin(theta)+y*cos(theta);

			//calculating the value at that location in the filter
			filter[fy*x_size + fx] = exp(-0.5*(((x_theta*x_theta)/(sigma_x*sigma_x)) + 
				((y_theta*y_theta)/(sigma_y*sigma_y)))) * cos(2*pi_val/lambda*x_theta+psi);
			fx++;
			//printf("%4.10f\t",filter[fy*x_size + fx]);
		}
		//printf("\n");
		fy++;
	}
	//printf("\n\n");
}
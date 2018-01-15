#include "cpp_includes.h"
#include "math_functions.h"
#include "static_map_header.cuh"

texture<float,1,cudaReadModeElementType> pyramid_filter;
texture<float,1,cudaReadModeElementType> gabor_filter;
texture<unsigned char,1,cudaReadModeElementType> intensities;


int static_map_main(unsigned char *frames[], unsigned char *static_maps[],const int width, const int height,const int n_maps, const int GPU_id, const int orientations, const int py_levels){
	
	//printf("%s\n",cudaGetErrorString(cudaGetLastError()));//in case we want to see that every part of the cuda implementation is right
	

	int n_pixels = width*height;
	//seting the device to process this map
	int n_gpus;
	cudaGetDeviceCount(&n_gpus);
	if(n_gpus>1)
		//setting the gpu to process the map if we have more than 1
		cudaSetDevice(GPU_id);
	else if(n_gpus==1)
		//default gpu in case we have only one
		cudaSetDevice(0);
	else{
		//in case we dont detect any gpu, however is unlikely to detect it here, because the retinal filtering
		//process would have detected first
		printf("No GPU support for your rig!! ...\n");
		system("pause");
		return 1;//exits with error code
	}

	//building the filters for gabor convolution
	float *d_accumulator;
	float **gabor_filter;//to store the garbor filters
	unsigned char **temp_frames;
	float **gauss_filters;//to store the filters for gaussian pyramids
	int *gabor_xsize, *gabor_ysize;//size of the filter in gabor
	int *gauss_xsize, *gauss_ysize;//size of the filter in gaussian
	float *d_max, *d_min;
	unsigned char *d_normalized;
	const int BLOCKS = (n_pixels+THREADS-1)/THREADS;
	float *d_filter;

	cudaMalloc((void**)&d_max,sizeof(float)*n_maps);
	cudaMalloc((void**)&d_min,sizeof(float)*n_maps);
	cudaMalloc((void**)&d_normalized,sizeof(unsigned char)*n_pixels*n_maps);	
	cudaMalloc((void**)&d_accumulator,sizeof(float)*n_pixels*n_maps);
	cudaMemset(d_max,0,sizeof(float)*n_maps);
	cudaMemset(d_min,10E100,sizeof(float)*n_maps);
	cudaMemset(d_accumulator,0,sizeof(float)*n_pixels*n_maps);

	gauss_xsize = (int*)malloc(sizeof(int)*py_levels);//size of the filter
	gauss_ysize = (int*)malloc(sizeof(int)*py_levels);//size of the filter
	gauss_filters = new float*[py_levels];//for the gaussian filters

	//temporal frames, because we want to make every scale and orientation from the original frame
	//is not a recursive process!!
	temp_frames = new unsigned char*[n_maps];
	for(int frame=0;frame<n_maps;frame++){//copy frames
		temp_frames[frame] = new unsigned char[n_pixels];
		memcpy(temp_frames[frame],frames[frame],sizeof(unsigned char)*n_pixels);//temporal copy of frames to avoid changing the real frames
	}

	for(int scales=0;scales<py_levels;scales++)//scales to be used for gabor
	{	
		//generate gaussian filter for pyramid, this is on the cpu
		generate_gaussian_filter(gauss_filters[scales],gauss_xsize[scales],gauss_ysize[scales],0.5,scales);
		//performs the operation for gaussian pyramid samplings
		gauss_pyramid(temp_frames,width,height,n_maps,gauss_filters[scales],gauss_xsize[scales],gauss_ysize[scales]);//building gaussian pyramids per frame
		//printf("%s\n",cudaGetErrorString(cudaGetLastError()));//in case we want to see that every part of the cuda implementation is right

		////visualization before mapping
		//for(int frame=0;frame<n_frames;frame++){//copy original frames
		//	Mat visu = cv::Mat(height,width,CV_8UC1);
		//	memcpy(visu.data,temp_frames[frame],sizeof(unsigned char)*n_pixels);
		//	char image_address[256];
		//	char num[10];
		//	char format[10] = ".bmp";
		//	strcpy(image_address, "../Release/pyramid");//folder address for storing pictures taken by the camera
		//	sprintf(num, "%06i", frame*py_levels + scales);
		//	strcat(image_address, num);   
		//	strcat(image_address, format);
		//	imwrite(image_address,visu);
		//}


		if(scales==0)
		{
			//declares memory for the filters in gabor, the same size and coefficients no matter the scale
			//therefore is done ONCE
			gabor_xsize = (int*)malloc(sizeof(int)*orientations);
			gabor_ysize = (int*)malloc(sizeof(int)*orientations);
			gabor_filter = new float*[orientations];
		}

		for(int ori=0;ori<orientations;ori++)//orientations to be used for gabor filtering
		{			
			if(scales==0)//declares memory for the filters in gabor, the same size and coefficients no matter the scale
			{
				/*
					generating gabor filters for each orientation, which are the same no matter which scale or frame we are using
					only one time for the scales to be used, since the coefficients are always the same
				*/
				float theta = ((180.0/orientations)*ori*pi_val)/180;//convert to radians since the sine and cosine only accept radian values
				generate_gabor_filter(gabor_filter[ori],gabor_xsize[ori],gabor_ysize[ori],1.7,theta,2.0,0.0,0.8);//generating gabor banks for convolutions
			}

			//now convolve the image here for each scale in the different orientations
			//time_t start = clock(), end;
			gabor_banks(temp_frames, d_accumulator,width,height,n_maps,gabor_filter[ori],gabor_xsize[ori],gabor_ysize[ori]);
			//printf("%s\n",cudaGetErrorString(cudaGetLastError()));//in case we want to see that every part of the cuda implementation is right
			//end = clock();
			//printf("The time elapsed for the gabor bank: %f ms.\n",((float)(end-start)/CLOCKS_PER_SEC)*1000);
			
		}
	}

	////visualization gabor mapping
	//for(int frame=0;frame<n_frames;frame++){//copy original frames
	//	Mat visu1 = cv::Mat(height,width,CV_8UC1);
	//	Mat visu2 = cv::Mat(height,width,CV_8UC1);
	//	memcpy(visu1.data,temp_frames[frame],sizeof(unsigned char)*n_pixels);
	//	cudaMemcpy(visu2.data,d_accumulator + frame*n_pixels,sizeof(unsigned char)*n_pixels,cudaMemcpyDeviceToHost);

	//	char image_address[256];
	//	char num[10];
	//	char format[10] = ".bmp";
	//	strcpy(image_address, "../Release/frame");//folder address for storing pictures taken by the camera
	//	sprintf(num, "%06i", frame);
	//	strcat(image_address, num);   
	//	strcat(image_address, format);
	//	imwrite(image_address,visu1);

	//	char image_address2[256];
	//	strcpy(image_address, "../Release/accum");//folder address for storing pictures taken by the camera
	//	strcat(image_address, num);   
	//	strcat(image_address, format);
	//	imwrite(image_address,visu2);
	//}


	cudaMalloc((void**)&d_filter,sizeof(float)*gauss_xsize[1]*gauss_ysize[1]);
	cudaMemcpy(d_filter,gauss_filters[1],sizeof(float)*gauss_xsize[1]*gauss_ysize[1],cudaMemcpyHostToDevice);
	//Data smoothing 
	smooth<<<BLOCKS,THREADS>>>(d_accumulator,d_filter,width,height,gauss_xsize[1],gauss_ysize[1],n_maps);
	//printf("%s\n",cudaGetErrorString(cudaGetLastError()));//in case we want to see that every part of the cuda implementation is right

	//Maximum value in the accumulator, per frame
	find_max_val<<<BLOCKS,THREADS>>>(d_accumulator,d_max,n_pixels,n_maps);
	//printf("%s\n",cudaGetErrorString(cudaGetLastError()));//in case we want to see that every part of the cuda implementation is right

	//Minimum value in the accumulator, per frame
	find_min_val<<<BLOCKS,THREADS>>>(d_accumulator,d_min,n_pixels,n_maps);
	//printf("%s\n",cudaGetErrorString(cudaGetLastError()));//in case we want to see that every part of the cuda implementation is right

	//Normalization of the summantion, per frame
	normalization_maps<<<BLOCKS,THREADS>>>(d_accumulator,d_normalized,width,n_pixels,d_max,d_min,n_maps);
	//printf("%s\n",cudaGetErrorString(cudaGetLastError()));//in case we want to see that every part of the cuda implementation is right
	//system("pause");

	for(int frame=0;frame<n_maps;frame++){//copy frames, after normalization
		cudaMemcpyAsync(static_maps[frame],d_normalized+(frame*n_pixels),sizeof(unsigned char)*n_pixels,cudaMemcpyDeviceToHost);
		free(temp_frames[frame]);//freeing the temporal frames
	}

	for(int levels=0;levels<py_levels;levels++)//freeing the gaussian filters
		free(gauss_filters[levels]);

	for(int ori=0;ori<orientations;ori++)//freeing the gabor filters
		free(gabor_filter[ori]);

	delete(temp_frames);
	delete(gauss_filters);
	delete(gabor_filter);
	free(gauss_xsize);
	free(gauss_ysize);
	free(gabor_xsize);
	free(gabor_ysize);

	//Freeing gpu
	cudaFree(d_filter);
	cudaFree(d_accumulator);
	cudaFree(d_max);
	cudaFree(d_min);
	cudaFree(d_normalized);


	//printf("%s\n",cudaGetErrorString(cudaGetLastError()));//in case we want to see that every part of the cuda implementation is right
	//system("pause");
	return 0;//exits without problems
}





/*****************************NORMALIZATION PROCESS********************************************/
__global__ void normalization_maps(float *values, unsigned char* normalized, const int width, const int n_pixels, float *maximum, float *minimum, const int n_maps){
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
	
	if(idx>=n_pixels) return;


	for(int frame=0;frame<n_maps;frame++){
		/*if(idx==0)
			printf("boundaries: <%f, %f>\n",maximum[frame], minimum[frame]);*/
		normalized[idx + frame*n_pixels] = (unsigned char)(((values[idx + frame*n_pixels]-minimum[frame])/(maximum[frame]-minimum[frame]))*255.0);
		/*if(normalized[idx + frame*n_pixels]<0 || normalized[idx + frame*n_pixels]>255)
			printf("oops: %i",(int)normalized[idx + frame*n_pixels]);*/
	}
}


__device__ float atomicMinf_val(float* address, float val){
    int *address_as_int =(int*)address;
    int old = *address_as_int, assumed;
    while (val < __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,__float_as_int(val));
        }
    return __int_as_float(old);
}


__global__ void find_min_val(const float* values, float* d_min, const int n_pixels, const int n_maps){
	__shared__ float shared[THREADS];

	for(int frame=0;frame<n_maps;frame++){
		int tid = threadIdx.x;
		int gid = (blockDim.x * blockIdx.x) + tid;
		shared[tid] = 0.0f;

		if (gid < n_pixels)
			shared[tid] = values[gid + frame*n_pixels];
		__syncthreads();

		for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
		{
			if (tid < s && gid < n_pixels)
				shared[tid] = min(shared[tid], shared[tid + s]);
			__syncthreads();
		}
		if (tid == 0)
			float a = atomicMinf_val(&d_min[frame], shared[0]);
	}
}



__device__ float atomicMaxf_val(float* address, float val){
    int *address_as_int =(int*)address;
    int old = *address_as_int, assumed;
    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,__float_as_int(val));
        }
    return __int_as_float(old);
}


__global__ void find_max_val(const float* values, float* d_max, const int n_pixels, const int n_maps){
	__shared__ float shared[THREADS];
	for(int frame=0;frame<n_maps;frame++){
		int tid = threadIdx.x;
		int gid = (blockDim.x * blockIdx.x) + tid;
		shared[tid] = 0.0f;

		if (gid < n_pixels)
			shared[tid] = values[gid + frame*n_pixels];
		__syncthreads();

		for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
		{
			if (tid < s && gid < n_pixels)
				shared[tid] = max(shared[tid], shared[tid + s]);
			__syncthreads();
		}
		if (tid == 0)
			float a = atomicMaxf_val(&d_max[frame], shared[0]);
	}
}

__global__ void smooth(float *values, float *filter, const int width, const int height, const int filter_xsize, const int filter_ysize, const int n_maps){

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
			s_filter[tid*f_xsize + i] = filter[tid*f_xsize + i];
	}



	//smoothing
	//we are allowed to contruct the pyramid for all frames, if we are inside the pictures, dont surpass the borders and the windows dont overlap
	if(x>=filter_x && x<(width-filter_x) && y>=filter_y && y<(height-filter_y) && x%f_xsize==0 && y%f_ysize==0){
		for(int frame=0;frame<n_maps;frame++){//using the same filter for all the frames
			float pix_val = 0;
								
			//using the filter
			int pos_y=0;
			for(int i=-filter_y; i<=filter_y; i++){//height of the filter
				int pos_x = 0;
				for(int j=-filter_x; j<=filter_x; j++){//width of the filter
					position = idx + i*width + j;//position of the pixel used in the window for building the pyramid
					pix_val += s_filter[pos_y*f_xsize + pos_x]*values[position + frame*n_pixels];
					pos_x++;
				}
				pos_y++;
			}

			//assigning the new value to the pixels in the window, and make sure we dont have overlapping windows
			pos_y=0;
			for(int i=-filter_y; i<=filter_y; i++){//height of the filter
				int pos_x = 0;
				for(int j=-filter_x; j<=filter_x; j++){//width of the filter
					position = idx + i*width + j;//position of the pixel used in the window for building the pyramid
					values[position + frame*n_pixels] = ceil(pix_val);
					pos_x++;
				}
				pos_y++;
			}
		}
	}
}
/*****************************NORMALIZATION PROCESS********************************************/
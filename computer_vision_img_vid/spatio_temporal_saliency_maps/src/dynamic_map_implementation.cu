#include "cuda_includes.h"
#include "cpp_includes.h"
#include "math_functions.h"
#include "dynamic_map_headers.cuh"
#include "static_map_header.cuh"


//#include "opencv2\opencv.hpp"
//using namespace cv;

using namespace std;


texture<float,1,cudaReadModeElementType> accum_text;

int dynamic_map_main(unsigned char *frames[], unsigned char *dynamic_map[],const int width, const int height,const int n_frames, const int GPU_id, float *ages){
	
	//printf("Error---> %s\n",cudaGetErrorString(cudaGetLastError()));
	int n_gpus;
	cudaGetDeviceCount(&n_gpus);
	if(n_gpus>1)
		cudaSetDevice(GPU_id);
	else if(n_gpus==1)
		cudaSetDevice(0);
	else{
		printf("No GPU device found!! ...\n");//unlikely to get here but if any problem we have this line
		system("pause");
		return 1;
	}

	//declaring variables to process this map
	const float alpha = 10.0;//weight for the velocity vector calculation
	const float iterations = 10;//iterations to converge to a good estimation for the velocity in x and y
	const int n_pixels = width*height;
	const int BLOCKS = (n_pixels+THREADS-1)/THREADS;
	size_t frame_size = n_pixels*sizeof(unsigned char);
	unsigned char *d_frames;
	float *d_maps, *d_vx, *d_vy, *d_ages;//the d_map is basically an image with a velocity components for x and y
	float *d_Gx, *d_Gy, *d_Gt, *d_max;//gradients for each component
	
	//building the filters for gabor convolution
	float *d_accumulator;
	const int py_levels = 1, orientations = 4;
	float **gabor_filter;//to store the garbor filters
	unsigned char **temp_frames;
	float **gauss_filters;//to store the filters for gaussian pyramids
	int *gabor_xsize, *gabor_ysize;//size of the filter in gabor
	int *gauss_xsize, *gauss_ysize;//size of the filter in gaussian
	unsigned char *dyna;

	cudaMalloc((void**)&d_Gx,n_pixels*sizeof(float)*(n_frames-1));//constains the gradient in the x-component
	cudaMalloc((void**)&d_Gy,n_pixels*sizeof(float)*(n_frames-1));//constains the gradient in the y-component
	cudaMalloc((void**)&d_Gt,n_pixels*sizeof(float)*(n_frames-1));//constains the gradient in the t-component
	cudaMalloc((void**)&d_frames,frame_size*n_frames);//temporal copy of the frames for gpu
	cudaMalloc((void**)&d_maps,n_pixels*sizeof(float)*n_frames);//dynamic map on gpu

	cudaMalloc((void**)&d_vx,n_pixels*sizeof(float));//constains the velocity in the x-component
	cudaMalloc((void**)&d_vy,n_pixels*sizeof(float));//constains the velocity in the y component
	cudaMalloc((void**)&d_max,sizeof(float));//have the maximum for the velocity magnitudes	
	cudaMalloc((void**)&d_ages,n_pixels*sizeof(float));//constains the ages in the pixels		
	cudaMalloc((void**)&d_accumulator,sizeof(float)*n_pixels);//accumulates the magnitudes for the velocities in all frames
	cudaMalloc((void**)&dyna,n_pixels*sizeof(unsigned char));//dyamic map after normalization

	cudaMemset(d_accumulator,0,sizeof(float)*n_pixels);
	cudaMemset(d_max,0,sizeof(float));

	cudaMemcpyAsync(d_ages,ages,n_pixels*sizeof(float),cudaMemcpyHostToDevice);//tracking the ages acording to intensity variation

	gauss_xsize = (int*)malloc(sizeof(int)*py_levels);//size of the filter
	gauss_ysize = (int*)malloc(sizeof(int)*py_levels);//size of the filter
	gauss_filters = new float*[py_levels];//for the gaussian filters

	//temporal frames
	temp_frames = new unsigned char*[n_frames];
	for(int frame=0;frame<n_frames;frame++){//copy frames
		temp_frames[frame] = new unsigned char[n_pixels];
		memcpy(temp_frames[frame],frames[frame],sizeof(unsigned char)*n_pixels);
	}	

	for(int scales=0;scales<py_levels;scales++)//
	{		
		//generates the gaussian filter
		generate_gaussian_filter(gauss_filters[scales],gauss_xsize[scales],gauss_ysize[scales],0.5,scales);//generate gaussian filters for convolutions
		//gaussian pyramids if needed
		gauss_pyramid(temp_frames,width,height,n_frames,gauss_filters[scales],gauss_xsize[scales],gauss_ysize[scales]);//building gaussian pyramids per frame

		if(scales==0)//declares memory for the filters in gabor, the same size and coefficients no matter the scale
		{
			gabor_xsize = (int*)malloc(sizeof(int)*orientations);
			gabor_ysize = (int*)malloc(sizeof(int)*orientations);
			gabor_filter = new float*[orientations];
		}

		for(int ori=0;ori<orientations;ori++)
		{			
			if(scales==0)//declares memory for the filters in gabor, the same size and coefficients no matter the scale
			{
				/*
					generating gabor filters for each orientation, which are the same no matter which scale or frame we are using
					only one time for the scales to be used, since the coefficients are always the same
				*/
				float theta = ((180.0/orientations)*ori*pi_val)/180;//convert to radians since the sine and cosine only accept radian values
				generate_gabor_filter(gabor_filter[ori],gabor_xsize[ori],gabor_ysize[ori],1.0,theta,1.5,0.0,0.5);//generating gabor banks for convolutions
			}

			//resetting counters per orientation to avoid overflow in the values
			cudaMemset(d_maps,0,n_pixels*sizeof(float)*n_frames);
			cudaMemset(d_vx,0,n_pixels*sizeof(float));
			cudaMemset(d_vy,0,n_pixels*sizeof(float));
			cudaMemset(d_Gx,0,n_pixels*sizeof(float)*(n_frames-1));
			cudaMemset(d_Gy,0,n_pixels*sizeof(float)*(n_frames-1));
			cudaMemset(d_Gy,0,n_pixels*sizeof(float)*(n_frames-1));	

			//now convolve the image here for each scale in the different orientations
			//time_t start = clock(), end;
			gabor_banks(temp_frames, d_maps,width,height,n_frames,gabor_filter[ori],gabor_xsize[ori],gabor_ysize[ori]);
			//end = clock();
			//printf("The time elapsed for the gabor bank: %f ms.\n",((float)(end-start)/CLOCKS_PER_SEC)*1000);

			//estimating gradients
			Gradients<<<BLOCKS,THREADS>>>(d_maps,d_Gx,d_Gy,d_Gt,width,height,n_frames,d_ages,scales+ori);

			//estimating velocity components
			Velocities<<<BLOCKS,THREADS>>>(d_Gx,d_Gy,d_Gt,d_vx,d_vy,width,height, n_frames, iterations,alpha);


			/*just for visualization*/
			/*CImg<float> accumulators(width,height);
			cudaMemcpyAsync(accumulators._data,d_maps,sizeof(float)*width*height,cudaMemcpyDeviceToHost);
			char f_name[256];
			char num[10];
			char format[10] = ".bmp";
			strcpy(f_name, "../test_videos/frames/frame");
			sprintf(num, "%06i", scales+ori);
			strcat(f_name, num);   
			strcat(f_name, format);
			accumulators.save(f_name);*/


			//estimating the magnitude value
			Magnitudes<<<BLOCKS,THREADS>>>(d_accumulator,d_vx,d_vy,width,height,1);
		}
	}
	//system("pause");

	//smoothing
	cudaBindTexture(NULL,accum_text,d_accumulator,sizeof(float)*n_pixels);
	smooth<<<BLOCKS,THREADS>>>(d_accumulator,width,height,1);

	//finding the maximum magnitude for normalization
	find_max<<<BLOCKS,THREADS>>>(d_accumulator,d_max,n_pixels,1);

	//normalization of the dynamic map
	normalization<<<BLOCKS,THREADS>>>(d_accumulator,dyna,width,n_pixels,d_max,1);

	//copying the map from device to host
	cudaMemcpyAsync(dynamic_map[0],dyna,n_pixels*sizeof(unsigned char),cudaMemcpyDeviceToHost);//copying the dynamic maps
	cudaMemcpyAsync(ages,d_ages,n_pixels*sizeof(float),cudaMemcpyDeviceToHost);//copying the ages of the pixels
	for(int frame=0;frame<n_frames;frame++){//copying the memory to the host
		free(temp_frames[frame]);//freeing the temporal frames
	}
	
		
	//freeing memory from host, and device
	for(int levels=0;levels<py_levels;levels++)//freeing the gaussian filters
		free(gauss_filters[levels]);

	for(int ori=0;ori<orientations;ori++)//freeing the gabor filters
		free(gabor_filter[ori]);

	delete(temp_frames);
	delete(gauss_filters);
	delete(gabor_filter);

	cudaFree(dyna);
	cudaFree(d_ages);
	cudaFree(d_max);
	cudaFree(d_Gx);
	cudaFree(d_Gy);
	cudaFree(d_Gt);
	cudaFree(d_frames);
	cudaFree(d_vx);
	cudaFree(d_vy);
	cudaFree(d_accumulator);
	cudaFree(d_maps);
	cudaUnbindTexture(accum_text);
	//printf("Error---> %s\n",cudaGetErrorString(cudaGetLastError()));
	//system("pause");
	return 0;//exits without problem
}


__global__ void smooth(float *values, const int width, const int height, const int maps_per_frame){
	int n_pixels = width*height;
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int x=idx%width, y=idx/width;
	int tid = threadIdx.x;
	
	if (idx>=n_pixels) return;

	float filter[3*3];
	filter[0]=0.0625; filter[1]=0.125; filter[2]=0.0625; 
	filter[3]=0.125; filter[4]=0.25; filter[5]=0.125; 
	filter[6]=0.0625; filter[7]=0.125; filter[8]=0.0625;

	for(int frame=0;frame<maps_per_frame;frame++){
		if(x>0 && y>0 && x<(width-1) && y<(height-1)){//take care of the borders
			float brightness=0;
			for(int i=-1;i<=1;i++)//3x3 window
			{
				for(int j=-1;j<=1;j++)//3x3 window
				{
					brightness += filter[(i+1)*3+(j+1)]*tex1Dfetch(accum_text,idx + ((i*width) + j) + frame*n_pixels);
				}
			}
			values[idx + frame*n_pixels] = (unsigned char)brightness;
		}else{
			values[idx + frame*n_pixels] = 0;
		}
	}
}




__global__ void normalization(float *values, unsigned char* normalized, const int width, const int n_pixels, float *maximum, const int maps_per_frame){
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
	
	if(idx>=n_pixels) return;

	for(int frame=0;frame<maps_per_frame;frame++){
		float max_val = maximum[frame];
		if(max_val>0){
			float ratio = (values[idx + frame*n_pixels]/(maximum[frame]*0.1));
			float temp = (ratio*255.0f);
			if(temp>=0){
				normalized[idx + frame*n_pixels] = (unsigned char)min(254,(int)temp);
			}
			else
				normalized[idx + frame*n_pixels] = 0;
		}
		else
			normalized[idx + frame*n_pixels] = 0;
	}

}


__device__ float atomicMaxf(float* address, float val){
    int *address_as_int =(int*)address;
    int old = *address_as_int, assumed;
    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,__float_as_int(val));
        }
    return __int_as_float(old);
}


__global__ void find_max(const float* values, float* d_max, const int n_pixels, const int maps_per_frame){
   __shared__ float shared[THREADS];
	int tid = threadIdx.x;
	int gid = (blockDim.x * blockIdx.x) + tid;

	for(int frame=0;frame<maps_per_frame;frame++){
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
		  float a = atomicMaxf(&d_max[frame], shared[0]);
	}
}



__global__ void Magnitudes(float *magnitude, float *Vx, float *Vy, const int width, const int height, const int maps_per_frame){
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int tid = threadIdx.x;
	int n_pixels = height*width;

	if(idx>=n_pixels) return;

	for(int frame=0;frame<maps_per_frame;frame++){
		float vx = Vx[idx + frame*n_pixels], vy = Vy[idx + frame*n_pixels];
		float temp = sqrtf(((vx*vx)+(vy*vy)));;

		if(temp<0 || isinf(temp) || isnan(temp)){
			temp=0;
		}
		magnitude[idx + frame*n_pixels] += temp;		
	}
}

__global__ void Velocities(float *Gx, float *Gy, float *Gt, float *Vx, float *Vy, const int width, const int height, const int n_frames, const int iterations, const float alpha){

	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int tid = threadIdx.x;
	const int n_pixels = width*height;

	if (idx>=n_pixels) return;

	const int left = 1, right = 1, shift = 1;
	__shared__ float _Vx[3][right+THREADS+left];
	__shared__ float _Vy[3][right+THREADS+left];
	int x=idx%width, y=idx/width;

	_Vx[0][tid] = 0;
	_Vx[1][tid] = 0;
	_Vx[2][tid] = 0;
	_Vy[0][tid] = 0;
	_Vy[1][tid] = 0;
	_Vy[2][tid] = 0;
	if(tid==blockDim.x-1){
		_Vx[0][tid+right] = 0;
		_Vx[1][tid+right] = 0;
		_Vx[2][tid+right] = 0;
		_Vy[0][tid+right] = 0;
		_Vy[1][tid+right] = 0;
		_Vy[2][tid+right] = 0;
		_Vx[0][tid+2*right] = 0;
		_Vx[1][tid+2*right] = 0;
		_Vx[2][tid+2*right] = 0;
		_Vy[0][tid+2*right] = 0;
		_Vy[1][tid+2*right] = 0;
		_Vy[2][tid+2*right] = 0;
	}
	__syncthreads();

	
	for(int frame=0;frame<(n_frames-1);frame++){
		float gradient_x = Gx[idx+frame*n_pixels], gradient_y = Gy[idx+frame*n_pixels], gradient_t = Gt[idx+frame*n_pixels];
		float alpha_constant = alpha;
		if(x>=left && x<(width-(2*left)) && y>=left && y<(height-(2*left))){
			for(int ite=0;ite<iterations;ite++){
				//only for the pixels withing the borders of the frame
				if(tid>=0){//we dont count the pixels ouside the boundaries and the window					
					//average of the x component
					float Avg_x = (_Vx[1][tid+left+1] + _Vx[1][tid+left-1] + _Vx[0][tid+left] + _Vx[2][tid+left])/(6.0f) + 
						(_Vx[0][tid+left-1] + _Vx[2][tid+left-1] + _Vx[0][tid+left+1] + _Vx[2][tid+left+1])/(12.0f);
					//average of the y component
					float Avg_y = (_Vy[1][tid+left+1] + _Vy[1][tid+left-1] + _Vy[0][tid+left] + _Vy[2][tid+left])/(6.0f) + 
						(_Vy[0][tid+left-1] + _Vy[2][tid+left-1] + _Vy[0][tid+left+1] + _Vy[2][tid+left+1])/(12.0f);
					//step size for the iterations
					float step_size = ((gradient_x*Avg_x+gradient_y*Avg_y+gradient_t)/
						(alpha_constant*alpha_constant + gradient_x*gradient_x + gradient_y*gradient_y));
					//updating velocity value at the center of the window of 3x3
					float temp1 = Avg_x - gradient_x*step_size;
					float temp2 = Avg_y - gradient_y*step_size;

					if(!isinf(temp1) && !isnan(temp1) && !isinf(temp2) && !isnan(temp2)){
						_Vx[1][tid+shift] += temp1;
						_Vy[1][tid+shift] += temp2;
					}
					else{
						_Vx[1][tid+shift] += 0;
						_Vy[1][tid+shift] += 0;
					}

				}
			}
		}else{//outside allowable boundaries
			_Vx[1][tid+shift] = 0;
			_Vy[1][tid+shift] = 0;
		}
		__syncthreads();

		//updating the dynamic map
		if(!isinf(_Vx[1][tid+shift]) && !isnan(_Vx[1][tid+shift]) && !isinf(_Vy[1][tid+shift]) && !isnan(_Vy[1][tid+shift])){
			Vx[idx] += (_Vx[1][tid+shift]);
			Vy[idx] += (_Vy[1][tid+shift]);
		}else{
			Vx[idx] = 0;
			Vy[idx] = 0;
		}
		__syncthreads();
	}
}



__global__ void Gradients(float *frames, float *Gx, float *Gy, float *Gt, const int width, const int height, const int n_frames, float *ages, const int scale){
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int tid = threadIdx.x;
	const int n_pixels = width*height;

	if (idx>=n_pixels) return;

	const int shift = 1;
	__shared__ float Inte_current[2][THREADS+shift];//intensities for every two frames (current frame)
	__shared__ float Inte_next[2][THREADS+shift];//intensities for every two frames (next frame)
	int x = idx%width, y=idx/width;
	int counter = 0;


	float accum_grad = 0.0f;

	for(int frame=0;frame<(n_frames-1);frame++){//calculation done every two frames, so we increment the frame number by 2
		/*storing the values of the intensity for each pixel in each frame into shared memory fetched from texture*/
		if(tid<blockDim.x && (x<(width-1) && y<(height-1))){//we are not at the boundaries of the threadblock, we copy only up and down			
			int position = idx;//current pixel
			Inte_current[0][tid] = frames[0*n_pixels + position];//current frame
			Inte_next[0][tid] = frames[(1+frame)*n_pixels + position];//next frame
			position = idx + width;//next pixel in y-direction (down)
			Inte_current[1][tid] = frames[0*n_pixels + position];//current frame
			Inte_next[1][tid] = frames[(1+frame)*n_pixels + position];//next frame
		}
		if(tid==(blockDim.x-1) && (x<(width-1) && y<(height-1))){//in this part we need an extra copy for the gradient calculation at tid == 255, or boundary of the threadblock
			//up and down of the tid==256, which cannot be accessed previously
			int position = idx+1;//next pixel in x-direction
			Inte_current[0][tid+1] = frames[0*n_pixels + position];//current frame
			Inte_next[0][tid+1] = frames[(1+frame)*n_pixels + position];//next frame
			position = idx+width+1;//next pixel in y-direction
			Inte_current[1][tid+1] = frames[0*n_pixels + position];//current frame
			Inte_next[1][tid+1] = frames[(1+frame)*n_pixels + position];//next frame
		}
		__syncthreads();//synchronization for the copy of the frames' intensity values

		if(x<(width-1) && y<(height-1))//boudary of the image
		{
			if(tid<blockDim.x){//taking care of the threadblock boundary
				float temp1 = (((Inte_current[0][tid+1]-Inte_current[0][tid]) + (Inte_current[1][tid+1]-Inte_current[1][tid])) + 
					((Inte_next[0][tid+1]-Inte_next[0][tid]) + (Inte_next[1][tid+1]-Inte_next[1][tid])))/4;//Gradient in x for current and next frame

				float temp2 = (((Inte_current[0][tid]-Inte_current[1][tid]) + (Inte_current[0][tid+1]-Inte_current[1][tid+1])) + 
					((Inte_next[0][tid]-Inte_next[1][tid]) + (Inte_next[0][tid+1]-Inte_next[1][tid+1])))/4;//Gradient in y for current and next frame

				float temp3 = (((Inte_next[0][tid]-Inte_current[0][tid]) + (Inte_next[0][tid+1]-Inte_next[0][tid+1])) + 
					((Inte_next[1][tid]-Inte_current[1][tid]) + (Inte_next[1][tid+1]-Inte_current[1][tid+1])))/4;//Gradient in t for current and next frame
				
				if(fabs(Inte_current[0][tid]-Inte_next[0][tid])>60.0*((n_frames-1)/2.0)){//Gradient's threshold
					/*recover the pixel's position since u predicted only for the future*/
					//int pos = 

					Gt[idx + frame*n_pixels] = temp3;
					Gx[idx + frame*n_pixels] = temp1;
					Gy[idx + frame*n_pixels] = temp2;
					accum_grad += temp3;
				}else{
					Gx[idx + frame*n_pixels] = 0;
					Gy[idx + frame*n_pixels] = 0;
					Gt[idx + frame*n_pixels] = 0;
				}
				if(fabs(Inte_current[0][tid]-Inte_next[0][tid])<30)//allows us to see if we have to consider this pixels as the same or new ones (aging)
					counter++;
			}
		}else{//don't consider the edges of the picture
			Gx[idx + frame*n_pixels] = 0;
			Gy[idx + frame*n_pixels] = 0;
			Gt[idx + frame*n_pixels] = 0;
		}
		__syncthreads();//synchronization after every frame
	}


	if((((n_frames-1)*accum_grad))<((n_frames-1)*30))//surpressing gradients
	{
		for(int frame=0;frame<(n_frames-1);frame++){
			Gt[idx + frame*n_pixels] = 0;
			Gx[idx + frame*n_pixels] = 0;
			Gy[idx + frame*n_pixels] = 0;
		}
	}
	__syncthreads();

	if(counter==(n_frames-1) && scale==0)//that means that the frames didn't change in this sequence
		ages[idx] += 0.1;//reduces the brightness if the pixel ages
	else if(scale==0)
		ages[idx] = 0.0;//pays 100% of attention since the frames may have changed in the sequence
}
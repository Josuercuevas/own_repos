#include "cuda_includes.h"
#include "cpp_includes.h"
#include "retinal_header.cuh"
#include "math_functions.h"
#include <tchar.h>
#include "atlstr.h"
#include "CIMG_disp.h"

#ifdef _DEBUG
bool _trace(TCHAR *format, ...)
{
   TCHAR buffer[1000];

   va_list argptr;
   va_start(argptr, format);
   wvsprintf(buffer, format, argptr);
   va_end(argptr);

   OutputDebugString(buffer);

   return true;
}
#endif



using namespace std;

int retinal_main(unsigned char *frames[], unsigned char *filtered[], const int width, const int height, const int n_frames, const int channels, const int pix_dist){
	
	//USES_CONVERSION;
	//TCHAR message1[1000];
	//TCHAR message2[1000];
	//TCHAR message3[1000];
	//printf("%s\n",cudaGetErrorString(cudaGetLastError()));//in case we want to see that every part of the cuda implementation is right

	int n_GPU;
	//OutputDebugString(_T("========>Retinal enhancer ....!!!!!\n"));
	//gets the number of GPUs in the rig
	if(cudaGetDeviceCount(&n_GPU) != cudaSuccess){
		printf("Problem obtaining the number of devices in your computer...!\n");
		system("pause");
		return 1;//exits since no device id was obtained
	}

	const int n_pixels = (width*height);	
	const int BLOCKS = (n_pixels+THREADS-1)/THREADS;//Grid dimension
	size_t frame_size = sizeof(unsigned char)*n_pixels*channels, size_partials = sizeof(float)*BLOCKS;	
	
	/*calculation of the theoretical K for the tone mapping approach*/
	float *partials, *d_partials;//partial values from the reduction in gpu
	float *d_roots, *roots;//roots for the retinal enhancement
	unsigned char *RGB_intensities;//RGB values of the frames
	unsigned char *temp_filtered;
	int *D;//calculate the compressed values and rounds them to the nearest intenger
	int *d_Ds;//values for the device
	float *d_taos, *taos;//values of the parameters for the retinal enhancement

	/*Allocating the memory on the device*/
	cudaMalloc((void**)&RGB_intensities,frame_size*n_frames);//since is RGB for the RGB frame
	cudaMalloc((void**)&temp_filtered,(frame_size/channels)*n_frames);//since is RGB for the filtered frames converted to 1D
	cudaMalloc((void**)&partials,size_partials*n_frames);//partial values after reduction
	cudaMalloc((void**)&taos,sizeof(float)*THREADS*n_frames);//parameters per thread
	cudaMalloc((void**)&D,sizeof(int)*THREADS*n_frames);//correction of pixel intensity
	cudaMalloc((void**)&d_roots,sizeof(float)*THREADS*n_frames);//correction of pixel intensity
	
	/*_tcscpy(message1, A2T(cudaGetErrorString(cudaGetLastError())));
	_trace(message1);
	OutputDebugString(_T("\n"));*/
	//printf("%s\n",cudaGetErrorString(cudaGetLastError()));//in case we want to see that every part of the cuda implementation is right

	d_partials = (float*)malloc(size_partials*n_frames);
	d_taos = (float*)malloc(sizeof(float)*THREADS*n_frames);
	d_Ds = (int*)malloc(sizeof(int)*THREADS*n_frames);

	//constant values no matter the frame (but be careful if the height and width of the frames changes)
	int n=0;
	roots = (float*)malloc(sizeof(float)*THREADS*n_frames);
	for(int i=0;i<(THREADS*n_frames);i++){
		roots[i] =powf((float)n+1,(float)1/n_pixels);//
		n++;
		if(n>n_pixels)
			n=0;
	}
	cudaMemcpyAsync(d_roots,roots,sizeof(float)*THREADS*n_frames,cudaMemcpyHostToDevice);
	/*_tcscpy(message1, A2T(cudaGetErrorString(cudaGetLastError())));
	_trace(message1);
	OutputDebugString(_T("\n"));*/

	//copying roots ended
	for(int frame=0;frame<n_frames;frame++)//combining the frames into a single 1D array
	{
		cudaMemcpyAsync(RGB_intensities+(n_pixels*channels*frame),frames[frame],frame_size,cudaMemcpyHostToDevice);//copying the frame for the GPU
	}
	/*_tcscpy(message1, A2T(cudaGetErrorString(cudaGetLastError())));
	_trace(message1);
	OutputDebugString(_T("\n"));*/
	
	//printf("%s\n",cudaGetErrorString(cudaGetLastError()));//in case we want to see that every part of the cuda implementation is right

	/*calling the kernel for computation of average intensity per block*/	
	pixel_compression<<<BLOCKS,THREADS>>>(RGB_intensities, partials, d_roots, n_pixels,channels, n_frames, pix_dist);
	/*_tcscpy(message1, A2T(cudaGetErrorString(cudaGetLastError())));
	_trace(message1);
	OutputDebugString(_T("\n"));*/
	
	//printf("compression: %s\n",cudaGetErrorString(cudaGetLastError()));//in case we want to see that every part of the cuda implementation is right

	cudaMemcpyAsync(d_partials,partials,size_partials*n_frames,cudaMemcpyDeviceToHost);//copying
	/*_tcscpy(message1, A2T(cudaGetErrorString(cudaGetLastError())));
	_trace(message1);
	OutputDebugString(_T("\n"));*/
	
	//printf("%s\n",cudaGetErrorString(cudaGetLastError()));//in case we want to see that every part of the cuda implementation is right

	/*partial products for the array containing the blocks' products, per frame*/
	for(int frame=0;frame<n_frames;frame++){
		for(int i=0;i<BLOCKS;i++)//calculating the real mean of the whole frame iterating for the blocks-mean
		{
			d_partials[0 + frame*BLOCKS] *= d_partials[i + frame*BLOCKS];
		}
	}

	cudaMemcpyAsync(partials,d_partials,size_partials*n_frames,cudaMemcpyHostToDevice);//copying
	/*_tcscpy(message1, A2T(cudaGetErrorString(cudaGetLastError())));
	_trace(message1);
	OutputDebugString(_T("\n"));*/
	
	//printf("%s\n",cudaGetErrorString(cudaGetLastError()));//in case we want to see that every part of the cuda implementation is right

	estimator<<<1,THREADS>>>(taos, partials, 255, 1,n_frames, BLOCKS);//does an estimation of the taos   
	/*_tcscpy(message1, A2T(cudaGetErrorString(cudaGetLastError())));
	_trace(message1);
	OutputDebugString(_T("\n"));*/
	
	//printf("%s\n",cudaGetErrorString(cudaGetLastError()));//in case we want to see that every part of the cuda implementation is right

	cudaMemcpyAsync(d_taos,taos,sizeof(float)*THREADS*n_frames,cudaMemcpyDeviceToHost);//copying the tao values from gpu
	/*_tcscpy(message1, A2T(cudaGetErrorString(cudaGetLastError())));
	_trace(message1);
	OutputDebugString(_T("\n"));*/
	
	//printf("COPY TAOS: %s\n",cudaGetErrorString(cudaGetLastError()));//in case we want to see that every part of the cuda implementation is right

	int *R_tao;
	R_tao = (int*)malloc(sizeof(int)*n_frames);
	for(int frame=0;frame<n_frames;frame++){
		R_tao[frame] = (int)ceil(d_taos[0+frame*THREADS]);//tao found is rounded up since the RGB values are integers
		//calculation of Ds, and is 255 because that is the maximum intensity allowed per frame per channel
		for(int j=0;j<256;j++)
		{
			float temp =log(((float)j+R_tao[frame])/(1.0+R_tao[frame]))/log((255.0+R_tao[frame])/(1.0+R_tao[frame]));
			if(temp>=0 && temp<256)
				d_Ds[j+frame*THREADS]=(int)ceil(255*temp);
			else
				d_Ds[j+frame*THREADS] = j;
		}
		//printf("%i\n",R_tao[frame]);
	}

	cudaMemcpyAsync(D,d_Ds,sizeof(float)*THREADS*n_frames,cudaMemcpyHostToDevice);//copying	
	/*_tcscpy(message1, A2T(cudaGetErrorString(cudaGetLastError())));
	_trace(message1);
	OutputDebugString(_T("\n"));*/
	
	//printf("TAOS: %s\n",cudaGetErrorString(cudaGetLastError()));//in case we want to see that every part of the cuda implementation is right

	/*calling kernel for adjusting the intensity levels for each pixel on the channel*/
	pixel_correction<<<BLOCKS,THREADS>>>(temp_filtered, RGB_intensities, D,255,1,width,height,channels,n_frames,pix_dist);
	/*_tcscpy(message1, A2T(cudaGetErrorString(cudaGetLastError())));
	_trace(message1);
	OutputDebugString(_T("\n"));*/
	
	//printf("%s\n",cudaGetErrorString(cudaGetLastError()));//in case we want to see that every part of the cuda implementation is right

	for(int frame=0;frame<n_frames;frame++)//copying the output
		cudaMemcpyAsync(filtered[frame],temp_filtered+(n_pixels*frame),(frame_size/channels),cudaMemcpyDeviceToHost);
	//printf("%s\n",cudaGetErrorString(cudaGetLastError()));//in case we want to see that every part of the cuda implementation is right

	free(R_tao);//free the value containing the tao correction
	free(d_taos);//free the value containing the tao values
	free(roots);//free the value containing the root values
	free(d_partials);//free the value containing the partial means
	free(d_Ds);//free the value containing the intensity correntions

	//freeing GPU for next process if needed
	cudaFree(RGB_intensities);
	cudaFree(temp_filtered);
	cudaFree(partials);
	cudaFree(taos);
	cudaFree(D);
	cudaFree(d_roots);
	//printf("%s\n",cudaGetErrorString(cudaGetLastError()));//in case we want to see that every part of the cuda implementation is right
	//system("pause");
	
	//OutputDebugString(_T("========>Retinal enhancer FINISHED ....!!!!!\n"));
	return 0;//exits without any problem
}




__global__ void pixel_correction(unsigned char* intensities, unsigned char *RGB, int* Ds,int maximum, int minimum, int width, int height, const int channels, const int n_frames, const int pix_dist){
	unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;	
	int size = height*width;//size of the frame which has a defined number of channels (RGB, or Gray)

	if(idx>=size) return;

	for(int frame=0;frame<n_frames;frame++)//correcting the intensities of every pixel in every frame
	{
		if(channels>1){//RGB channels
			if(pix_dist == 1){//interleaved
				intensities[idx+frame*size] = (unsigned char)(0.2126*Ds[(int)RGB[idx*3+frame*size*channels] + frame*THREADS] + 
												0.7152*Ds[(int)RGB[idx*3+1+frame*size*channels] + frame*THREADS]+ 
												0.0722*Ds[(int)RGB[idx*3+2+frame*size*channels] + frame*THREADS]);
			}
			else{//planar
				intensities[idx+frame*size] = (unsigned char)(0.2126*Ds[(int)RGB[idx + frame*size*channels] + frame*THREADS] + 
												0.7152*Ds[(int)RGB[idx + size + frame*size*channels] + frame*THREADS]+ 
												0.0722*Ds[(int)RGB[idx + 2*size + frame*size*channels] + frame*THREADS]);
			}
		}
		else//only one channel (gray-scale image)
			intensities[idx+frame*size] = (unsigned char)Ds[(int)RGB[idx*3+frame*size] + frame*THREADS];
	}
}



__global__ void estimator(float* taos, float *average, const int maximum, const int minimum, const int n_frames, const int ave_size){
	unsigned int idx = threadIdx.x;
	float kT, kP;
	__shared__ float approx[THREADS];
	for(int frame=0;frame<n_frames;frame++){		
		kT = (2*__logf(average[0+frame*ave_size])-__logf((float)minimum)-logf((float)maximum))/(__logf((float)maximum)-__logf((float)minimum));//
		kT = 0.4*pow(2,kT);			
		taos[idx+frame*THREADS]=0.5*(idx+1);
		kP = (__logf((average[0+frame*ave_size]+taos[idx+frame*THREADS])/(minimum+taos[idx+frame*THREADS]))/__logf((maximum+taos[idx+frame*THREADS])/(minimum+taos[idx+frame*THREADS])));
		approx[idx] = abs(kP-kT);
		__syncthreads();
	
		if(idx==0)
		{
			int id=0;
			for(int i=1;i<blockDim.x;i++)
				if(approx[i]<approx[i-1])
					id=i;
			taos[0+frame*THREADS]=taos[id+frame*THREADS];
			taos[1+frame*THREADS]=approx[id];
			taos[2+frame*THREADS]=kT;
			taos[3+frame*THREADS]=(__logf((average[0+frame*ave_size]+taos[id+frame*THREADS])/(minimum+taos[id+frame*THREADS])) /
				__logf((maximum+taos[id+frame*THREADS])/(minimum+taos[id+frame*THREADS])));
		}
		__syncthreads();
	}
}




__global__ void pixel_compression(unsigned char *intensities, float* average, float* roots, int size, const int channels, const int n_frames, const int pix_dist)
{
	__shared__ float partial_mul[THREADS];
	for(int frame=0;frame<n_frames;frame++){//process all the frames at once, once single thread process n_frames
		float temp=1;
		int idx = threadIdx.x;
		for (size_t i = blockIdx.x*blockDim.x + idx;i < size;i += blockDim.x*gridDim.x) {
			if(channels>1){//RGB channels
				if(pix_dist == 1)//interleaved
					temp *= (roots[(int)intensities[i*channels+(frame*size*channels)]] + roots[(int)intensities[i*channels+1+(frame*size*channels)]] + 
						roots[(int)intensities[i*channels+2+(frame*size*channels)]])/3;//RGB values
				else//planar
					temp *= (roots[(int)intensities[i + (frame*size*channels)]] + roots[(int)intensities[i + size + (frame*size*channels)]] + 
					roots[(int)intensities[i + 2*size + (frame*size*channels)]])/3;//RGB values
			}
			else
				temp *= roots[(int)intensities[i*channels+(frame*size*channels)]];//only the first channel
		}
		partial_mul[idx] = temp;
		__syncthreads();

		for (int activeThreads = blockDim.x>>1; activeThreads; activeThreads >>= 1) {
			if ( idx < activeThreads ) {
				partial_mul[idx] *= partial_mul[idx+activeThreads];
			}
			__syncthreads();
		}

		if ( idx == 0 ) {
			average[blockIdx.x + frame*gridDim.x] = partial_mul[0];//since we compute block-partial averages per frame
		}
		__syncthreads();
	}
}
#include "execute.cuh"
#include "cuda_includes.h"
#include "Saliency_classes.h"
#include "atlstr.h"
#include <tchar.h>
#include "CIMG_disp.h"

int initialize_gpu(){//device initialization
	int n_gpus;
	cudaGetDeviceCount(&n_gpus);
	if(n_gpus>1){
		for(int i=0;i<n_gpus;i++){
			cudaSetDevice(i);
			cudaFree(0);
		}
	}
	else
		cudaFree(0);

	return 0;
}
int MultiGPUs(unsigned char *frames[], unsigned char *maps[],const int width, const int height, const int n_frames, const int channels, const int orientations, 
	const int pyramid_levels, float *ages, unsigned char *temp_output[], unsigned char *output[], size_t frame_size){

	/*
		In this part since we need two maps (static and dynamic) we have only 2 CPU_threads calling two 
		different functions executing in different devices if there is more than one device in the rig
	*/
	int execution_problem = false;
	omp_set_num_threads(2);//two maps, every thread controls a device
    #pragma omp parallel
	{
		int t_id = omp_get_thread_num();
		if(!t_id){// t_id == 0	STATIC PATH		
			time_t start = clock(), end;
			//printf("Static path started at: %f ms.\n",((float)(start)/CLOCKS_PER_SEC)*1000);
			/*
				Denoising(hanning filtering) the input of this part is the output of the retinal filtering and it is only implemented for the static
				path since we dont need it for the dynamic, with this we want to remove noise by slightly bluring the image
			*/
			memcpy(temp_output[0],output[0],frame_size);//making a copy of the frames, for the static path (in case we want to visualize more than just the saliency maps)
			
			if(hanning_mask::noise_reduction(temp_output,width,height,1,t_id)){
				printf("problem with Hanning mask...!!!\n");
				system("pause");
				execution_problem = true;//flag that tells us the program had a problem in the noise reduction part
			}
			else{		
				/* 
					static path send the device Id if there is more than one device, otherwise the same device if any will do the processing
				*/
				unsigned char **sta;
				sta = new unsigned char*[1];//the number of frames to have be processed by the static pathway
				for(int frame=0;frame<1;frame++)
				{
					sta[frame] = (unsigned char*)malloc(frame_size);//the size of each frame in bytes
				}

				time_t start2=clock();
				if(static_path::static_map(temp_output,sta,width,height,1,t_id,orientations,pyramid_levels)){//main function of the pathway
					printf("problem with Static Map...!!!\n");
					system("pause");
					execution_problem = true;//flag that tells us the program had a problem in the noise reduction part
				}else{
					for(int frame=0;frame<1;frame++)
					{
						memcpy(maps[frame],sta[frame],frame_size);
						free(sta[frame]);
					}
					delete sta;

					end = clock();
					//printf("Static path finished in: %4.3f ms.\n",((float)(end-start)/CLOCKS_PER_SEC)*1000);

					for(int i=0;i<1;i++){//copying the denoised frames in case we want to do visualize it
						free(temp_output[i]);//freeing the memory just used for the temporal copy
					}
				}
				printf("Time spent in Spatial map: %4.4f\n", ((float)(clock()-start2)/CLOCKS_PER_SEC)*1000);

				
				//for(int i=0;i<height;i++){
				//	for(int j=0;j<width;j++){
				//		printf("%i\t",maps[0][i*width + j]);//printing frame
				//	}
				//	printf("\n");
				//}
				//printf("\n");
				//system("pause");
			}
		}else{// t_id == 1 DYNAMIC PATH
			time_t start = clock(), end;
			//printf("Dynamic path started at: %f ms.\n",((float)(start)/CLOCKS_PER_SEC)*1000);

			/* 
				Dynamic path "motion estimation" send the device Id if there is more than one device
			*/
			unsigned char **dyna;
			dyna = new unsigned char*[1];
			for(int frame=0;frame<1;frame++)
			{
				dyna[frame] = (unsigned char*)malloc(frame_size);
			}

			time_t start2=clock();
			if(dynamic_path::dynamic_map(output,dyna,width,height,n_frames,t_id,ages)){
				printf("problem with Dynamic Map...!!!\n");
				system("pause");
				execution_problem = true;//flag that tells us the program had a problem in the noise reduction part
			}
			else{
				for(int frame=0;frame<1;frame++){
					memcpy(maps[1 + frame],dyna[frame],frame_size);//copying the dynamic maps
					free(dyna[frame]);
				}
				delete dyna;
				end = clock();
				//printf("Dynamic path finished in: %f ms.\n",((float)(end-start)/CLOCKS_PER_SEC)*1000);
			
				for(int i=0;i<n_frames;i++){//freeing the temprary data
					free(output[i]);//freeing the memory just used for the temporal copy
				}
			}
			printf("Time spent in Dynamic map: %4.4f\n", ((float)(clock()-start2)/CLOCKS_PER_SEC)*1000);

			//for(int i=0;i<height;i++){
			//		for(int j=0;j<width;j++){
			//			printf("%i\t",maps[1][i*width + j]);//printing frame
			//		}
			//		printf("\n");
			//	}
			//	printf("\n");
			//	system("pause");

		}
		#pragma omp barrier ///synchronization of all threads	
	}

	return execution_problem;//exits with a code of 0 if no problem found ... 1 otherwise

}


int SingleGPU(unsigned char *frames[], unsigned char *maps[],const int width, const int height,	const int n_frames, const int channels, const int orientations, const int pyramid_levels, 
	float *ages, unsigned char *temp_output[], unsigned char *output[], size_t frame_size){

	int execution_problem = false;
	{

		{// STATIC PATH		
			time_t start = clock(), end;
			//printf("Static path started at: %f ms.\n",((float)(start)/CLOCKS_PER_SEC)*1000);
			/*
				Denoising(hanning filtering) the input of this part is the output of the retinal filtering and it is only implemented for the static
				path since we dont need it for the dynamic, with this we want to remove noise by slightly bluring the image
			*/
			memcpy(temp_output[0],output[0],frame_size);//making a copy of the frames, for the static path (in case we want to visualize more than just the saliency maps)
			
			if(hanning_mask::noise_reduction(temp_output,width,height,1,0)){
				printf("problem with Hanning mask...!!!\n");
				system("pause");
				execution_problem = true;//flag that tells us the program had a problem in the noise reduction part
			}
			else{		
				/* 
					static path send the device Id if there is more than one device, otherwise the same device if any will do the processing
				*/
				unsigned char **sta;
				sta = new unsigned char*[1];//the number of frames to have be processed by the static pathway
				for(int frame=0;frame<1;frame++)
				{
					sta[frame] = (unsigned char*)malloc(frame_size);//the size of each frame in bytes
				}

				if(static_path::static_map(temp_output,sta,width,height,1,0,orientations,pyramid_levels)){//main function of the pathway
					printf("problem with Static Map...!!!\n");
					system("pause");
					execution_problem = true;//flag that tells us the program had a problem in the noise reduction part
				}else{
					for(int frame=0;frame<1;frame++)
					{
						memcpy(maps[frame],sta[frame],frame_size);
						free(sta[frame]);
					}
					delete sta;

					end = clock();
					//printf("Static path finished in: %4.3f ms.\n",((float)(end-start)/CLOCKS_PER_SEC)*1000);

					for(int i=0;i<1;i++){//copying the denoised frames in case we want to do visualize it
						free(temp_output[i]);//freeing the memory just used for the temporal copy
					}
				}

				
				//for(int i=0;i<height;i++){
				//	for(int j=0;j<width;j++){
				//		printf("%i\t",maps[0][i*width + j]);//printing frame
				//	}
				//	printf("\n");
				//}
				//printf("\n");
				//system("pause");
			}
			/*disp_uchar_pic(maps[0],height,width,1);
			system("pause");*/
		}


		{// 1 DYNAMIC PATH
			time_t start = clock(), end;
			//printf("Dynamic path started at: %f ms.\n",((float)(start)/CLOCKS_PER_SEC)*1000);

			/* 
				Dynamic path "motion estimation" send the device Id if there is more than one device
			*/
			unsigned char **dyna;
			dyna = new unsigned char*[1];
			for(int frame=0;frame<1;frame++)
			{
				dyna[frame] = (unsigned char*)malloc(frame_size);
			}

			if(dynamic_path::dynamic_map(output,dyna,width,height,n_frames,0,ages)){
				printf("problem with Dynamic Map...!!!\n");
				system("pause");
				execution_problem = true;//flag that tells us the program had a problem in the noise reduction part
			}
			else{
				for(int frame=0;frame<1;frame++){
					memcpy(maps[1 + frame],dyna[frame],frame_size);//copying the dynamic maps
					free(dyna[frame]);
				}
				delete dyna;
				end = clock();
				//printf("Dynamic path finished in: %f ms.\n",((float)(end-start)/CLOCKS_PER_SEC)*1000);
			
				for(int i=0;i<n_frames;i++){//freeing the temprary data
					free(output[i]);//freeing the memory just used for the temporal copy
				}
			}

			//for(int i=0;i<height;i++){
			//		for(int j=0;j<width;j++){
			//			printf("%i\t",maps[1][i*width + j]);//printing frame
			//		}
			//		printf("\n");
			//	}
			//	printf("\n");
			//	system("pause");
			/*disp_uchar_pic(maps[1],height,width,1);
			system("pause");*/
		}
	}

	
	return execution_problem;//exits with a code of 0 if no problem found ... 1 otherwise
}
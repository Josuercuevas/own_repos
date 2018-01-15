/* 
	SaliencyLIB.cpp : Defines the exported functions for the DLL application of the Video Saliency Map library.
*/

#include "stdafx.h"
#include "saliency_functions.h"
#include "cpp_includes.h"
#include "cuda_includes.h"
#include "defines_values.h"
#include "Saliency_classes.h"
#include "CIMG_disp.h"


/*
	Function in charge of initializing the Device to avoid lazy context initialization on the GPU
*/
SALIENCYLIB_API int saliency::Device_Initialization(){
	return Execute::initialization();
}



/*
	Function in charge in compensating the motion of the frames in the sequence
	input:
			1. The frames to be processed by the function in a vector of pointers for unsigned char format *pointer[]
			2. The width of the frame
			3. The height of the frame
			4. The number of frames to be processed
			5. number of channels in the frames (since it could be mono or RGB)
			6. A flag to determine if we need to do the motion compensation for the whole sequence (only the first time) or a newly added frame 
			(subsequent sequences)
		Ouput:
			Motion compensated frames
*/
SALIENCYLIB_API int saliency::Motion_compensation(unsigned char *frames[], const int width, const int height, const int n_frames, const int channels, bool first_frame){
	
	if(n_frames>MAX_FRAMES){
		printf("The number of frames to be processed is to large, please notice that only 50 frames at the time can be processed...!");
		system("pause");
		return 1;//exits with an error code of 1
	}

	return motion_compensation::compensation(frames,width,height,n_frames,channels,first_frame);
}






/*
	Function that calculate the MAPS for static and dynamic paths: In charge of using the functions needed to calculate the saliency map for a frame or a sequence of frames
	the parameters are as follow:
		
		input:
			1. The frames to be processed by the library in a vector of pointers for unsigned char format *pointer[]
			2. The width of the frame
			3. The height of the frame
			4. The number of frames to be processed
			5. number of channels in the frames (since it could be mono or RGB)
			
			Gabor filter for static and dynamic maps
				6. number of orientations for the static maps ( where the degree is = (180/orientations) degrees )
				7. number of levels for the gaussian pyramids in the static maps (
		Ouput:
			Saliency maps which are constructed from the static and dynamic pathways of the frames, but here we dont fusion them. There are a total of 2*n_frames maps
			(static + dynamics)
*/
SALIENCYLIB_API int saliency::Saliency_maps(unsigned char *frames[], unsigned char *maps[],const int width, const int height, const int n_frames, const int channels
	, const int orientations, const int pyramid_levels, float *ages, pixel_arrangement pix_dist){
	
		if(n_frames>MAX_FRAMES){
		printf("The number of frames to be processed is to large, please notice that only 50 frames at the time can be processed...!");
		system("pause");
		return 1;//exits with an error code of 1
	}

	//we will make a temporal copy of the frames to be processed so we dont change the original ones
	unsigned char *output[MAX_FRAMES], *temp_output[1];
	size_t frame_size = sizeof(unsigned char)*width*height;//size of the frame in bytes
	for(int i=0;i<n_frames;i++){
		output[i] = (unsigned char*)malloc(frame_size);//allocating the memory of the temporal data (for dynamic path)
		if(i==0)
			temp_output[i] = (unsigned char*)malloc(frame_size);//allocating the memory of the temporal data (for dynamic path)
	}

	/*
		Enhancing the quality of the frame for building the saliency maps
	*/

	/*disp_uchar_pic(frames[0],height,width,3);
	system("pause");
	disp_uchar_pic(frames[1],height,width,3);
	system("pause");
	disp_uchar_pic(frames[2],height,width,3);
	system("pause");*/

	if(retinal_filter::frame_enhancement(frames,output,width,height,n_frames,channels, pix_dist))
		return 1;//problem while doing the frame enhancement.

	/*disp_uchar_pic(output[0],height,width,1);
	system("pause");
	disp_uchar_pic(output[1],height,width,1);
	system("pause");*/
	//disp_uchar_pic(output[2],height,width,1);
	//system("pause");

	int n_gpus;
	cudaGetDeviceCount(&n_gpus);

	if(n_gpus>1)
		return Execute::Run_MultiGPUs(frames,maps,width,height,n_frames,channels,orientations,pyramid_levels,ages,temp_output,output,frame_size);
	else
		return Execute::Run_SingleGPU(frames,maps,width,height,n_frames,channels,orientations,pyramid_levels,ages,temp_output,output,frame_size);
		
}



/*
	This function is in charge of fusion the maps coming from the saliency pathways, by using the average of both images, a weighted average, normalized summation
	of the maps coming from the previous step
	Input:
		1. Video Salency maps from static and dynamic pathways
		2. Type of fusion to be performed:
			a. AVERAGE --> (Static(x,y)+Dynamic(x,y))/2
			b. WEIGHTED_AVERAGE --> w*Static(x,y)+(1-w)*Dynamic(x,y) --> default of w=0.3
			c. NORM_SUM (default) --> val(x,y) = sum (static(x,y)+dynamic(x,y)) --> Normaliztion(x,y) = ((val(x,y)-min)/(max-min)), where max and min are estimated from the sum.
*/
SALIENCYLIB_API int saliency::Fusion_maps(unsigned char *maps[], unsigned char *before_fusion[],	const int width, const int height, const int n_maps, fusion_type _type){
	//this switch statement is in charge of selecting the right function to perform the fusion of the maps
	switch (_type){
		case AVERAGE:
			/*the fusion is done using the average of the dynamic and static maps*/
			return fusions::average_map(maps, before_fusion, width, height, n_maps);
		case WEIGHTED_AVERAGE:
			/*the fusion is done using the weighted average of the dynamic and static maps, with default of 25% for the static map*/
			return fusions::weighted_average_map(maps, before_fusion, width, height, n_maps);
		case NORM_SUM:
			/*the fusion is done using the normalized sum of the dynamic and static maps*/
			return fusions::normalized_sum(maps, before_fusion, width, height, n_maps);
		default:
			printf("This type of fusion is not supported by the library..!!\n");
			printf("Please try the Following types:\n\t1. AGERAGE\n\t2. WEIGHTED_AVERAGE\n\t3. NORM_SUM\n");
			system("pause");
			return 1;
	}
}



/*
	Function in charge of fading the strength of the pixels in the map according to their age, since in read life an observer may loose interest
	in certain regions as time goes by. Therefore, this function uses the pixel's age previously estimated during the mapping process. The pixel's age is 
	basically increased by one (1) if the intensity value of that particular pixel doesn't change from frame to frame. The age is then substracted from the intensity
	of the map to make it darker (with a limit in the value of 0).
*/
SALIENCYLIB_API int saliency::Fading_maps(unsigned char *maps[], const int width, const int height, const int n_maps, float *ages){
	/*function in charge of estimating the age*/
	return fading_tracker::fade_activation(maps, width, height, n_maps, ages);
}

/*
	Function in the API in charge of doing the alpha blending of the maps just generated, for every frame in the process. Therefore, we superimpose
	every map on top of the frames of the video. This option may be deactivated if the user wants to do so.
*/
SALIENCYLIB_API int saliency::Frames_Maps_Blend(unsigned char *blend_frames[], unsigned char *frames[], unsigned char *maps[], const int width, const int height, const int n_maps, const int channels, 
	pixel_arrangement pix_dist){
	//calling the function to control the blending
	return Blending::Blending_activated(blend_frames, frames, maps, width, height, n_maps, channels, pix_dist);
}


/*
	Function in charge of converting the YUV data to RGB for a better processing in the Saliency library
*/
SALIENCYLIB_API int saliency::YUVtoRGB(unsigned char *YUV_frames[], unsigned char *RGB_frames[], int Y_height, int Y_width, int Cb_height, int Cb_width, int Cr_height, int Cr_width, int n_frames,
	pixel_arrangement pix_dist){
	return YUV_handling::YUVtoRGB_conversion(YUV_frames,RGB_frames,Y_height,Y_width,Cb_height,Cb_width,Cr_height,Cr_width,n_frames,pix_dist);
}

/*
	Function in charge of converting the RGB data to YUV for a better processing in the Saliency library
*/
SALIENCYLIB_API int saliency::RGBtoYUV(unsigned char *YUV_frames[], unsigned char *RGB_frames[], int Y_height, int Y_width, int Cb_height, int Cb_width, int Cr_height, int Cr_width, int n_frames,
	pixel_arrangement pix_dist){
	return YUV_handling::RGBtoYUV_conversion(YUV_frames,RGB_frames,Y_height,Y_width,Cb_height,Cb_width,Cr_height,Cr_width,n_frames,pix_dist);
}
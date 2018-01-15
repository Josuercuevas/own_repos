#include "stdafx.h"
#include "cpp_includes.h"
#include "hanning_functions.cuh"
#include "Saliency_classes.h"
#include "retinal_header.cuh"
#include "dynamic_map_headers.cuh"
#include "static_map_header.cuh"
#include "fusion_types_header.cuh"
#include "age_tracker_header.cuh"
#include "blending_header.cuh"
#include "motion_compensation_header.cuh"
#include "execute.cuh"
#include "YUV_conversion_header.cuh";

/************************ MOTION COMPENSATION **************************************/
//constructor to be used in case something has to be initialize in this class
motion_compensation::motion_compensation(void){
	return;
}

//destructor to be used in case something has to be destroy in this class
motion_compensation::~motion_compensation(void){
	return;
}

/*
	Back-bone of this class, which is in charge of compensate the motion of consecutive frames, by minimizing the MAE of two consecutive frames (t, and t-1)
	input: 
		1. The frames to be processed by the library in a pointer to vector format *pointer[]
		2. The width of the frame
		3. The height of the frame
		4. The number of frames to be processed
		5. Channels in the frames
	Ouput:
		A motion compensated array of frames

*/
int motion_compensation::compensation(unsigned char *frames[], const int width, const int height, const int n_frames, const int channels, bool first_frame){
	//time_t start = clock(), end;	
	//calling the functions in a .cu file in the library, visible only to this .cpp file
	return compensation_main(frames,width,height,n_frames,channels,first_frame);
	//return the frames after processing
	//end=clock();
	//printf("The time elapsed for the motion compensation processes was: %f ms.\n",((float)(end-start)/CLOCKS_PER_SEC)*1000);
}









/******************************RETINAL FILTER CLASS******************************/
//constructor to be used in case something has to be initialize in this class
retinal_filter::retinal_filter(void){
	return;
}

//destructor to be used in case something has to be destroy in this class
retinal_filter::~retinal_filter(void){
	return;
}

/*
	Back-bone of this class, which is in charge of reducing the noise of the incoming frames
	input: 
		1. The frames to be processed by the library in a pointer to vector format *pointer[]
		2. The width of the frame
		3. The height of the frame
		4. The number of frames to be processed
		5. Channels in the frames
	Ouput:
		An enhanced array of frames

*/
int retinal_filter::frame_enhancement(unsigned char *frames[], unsigned char *filtered[],const int width, const int height, const int n_frames, const int channels, const int pix_dist){
	//time_t start = clock(), end;	
	//calling the functions in a .cu file in the library, visible only to this .cpp file
	return retinal_main(frames,filtered,width,height,n_frames,channels, pix_dist);
	//return the frames after processing
	//end=clock();
	//printf("The time elapsed for the retinal filtering processes was: %f ms.\n",((float)(end-start)/CLOCKS_PER_SEC)*1000);
}

/******************************************************************************/


/*************************HANNING FILTERING CLASS*******************************/
//constructor to be used in case something has to be initialize in this class
hanning_mask::hanning_mask(void){
	return;
}

//destructor to be used in case something has to be destroy in this class
hanning_mask::~hanning_mask(void){
	return;
}

/*
Back-bone of this class, which is in charge of reducing the noise of the incoming frames
input: 
	1. The frames to be processed by the library in a pointer to vector format *pointer[]
	2. The width of the frame
	3. The height of the frame
	4. The number of frames to be processed
Ouput:
	A denoised array of frames

*/
int hanning_mask::noise_reduction(unsigned char *frames[],const int width, const int height, const int n_frames, const int GPU_id){
	//variables needed
	//time_t start = clock(), end;	
	//calling the functions in a .cu file in the library, visible only to this .cpp file
	return(hanny_main(frames,width,height,n_frames,GPU_id));
	//return the frames after processing
	//end=clock();
	//printf("The time elapsed for the denoising processes was: %f ms.\n",((float)(end-start)/CLOCKS_PER_SEC)*1000);
}
/******************************************************************************/


/************************ STATIC PATHWAY **************************************/
//constructor to be used in case something has to be initialize in this class
static_path::static_path(void){
	return;
}

//destructor to be used in case something has to be destroy in this class
static_path::~static_path(void){
	return;
}

/*
	This part of the code is in charge or estimate the static saliency map, e.i. determines
	saliency object taking into account texture, intensity, color etc. the parameters for this function are:

	Input:
		1. Frames to be processed by the function
		2. Cointainer to store the map
		3. Width of each frame
		4. Height of each frame
		5. Number of frames to be processed
		6. The device to be used to process these frames (in case that we have multiple devices)
		
		Gabor filtering
			7. Orientations to be considered
			8. Scales to be used for the filtering

	Output
		1. Static Saliency maps which is 1 per frame (RGB or gray), but remember that we construct 
			(orientations*py_levels) temporarely maps in the process, to later fusion them into a single 
			static map (using normalized average).
*/
int static_path::static_map(unsigned char *frames[], unsigned char *static_maps[],const int width, const int height,const int n_maps, const int GPU_id, const int orientations, 
	const int py_levels){
	//time_t start = clock(), end;	
	//calling the functions in a .cu file in the library, visible only to this .cpp file
	return (static_map_main(frames, static_maps,width,height,n_maps, GPU_id,orientations,py_levels));
	//return the frames after processing
	//end=clock();
	//printf("The time elapsed for the dynamic_path processes was: %f ms.\n",((float)(end-start)/CLOCKS_PER_SEC)*1000);
}
/******************************************************************************/


/************************ DYNAMIC PATHWAY **************************************/
//constructor to be used in case something has to be initialize in this class
dynamic_path::dynamic_path(void){
	return;
}

//destructor to be used in case something has to be destroy in this class
dynamic_path::~dynamic_path(void){
	return;
}

/*
	This part of the code is in charge or estimate the dynamic saliency maps, e.i. determines
	saliency object taking into account only their motion. the parameters for this function are:

	Input:
		1. Frames to be processed by the function
		2. Cointainer to store the map
		3. Width of each frame
		4. Height of each frame
		5. Number of frames to be processed
		6. The device to be used to process these frames (in case that we have multiple devices)

	Output
		1. Dynamic Saliency map which is 1 per frame (RGB or gray)
*/
int dynamic_path::dynamic_map(unsigned char *frames[], unsigned char *dynamic_map[],const int width, const int height,const int n_frames, const int GPU_id, float *ages){
	//variables needed
	//time_t start = clock(), end;	
	//calling the functions in a .cu file in the library, visible only to this .cpp file
	return dynamic_map_main(frames, dynamic_map,width,height,n_frames, GPU_id, ages);
	//return the frames after processing
	//end=clock();
	//printf("The time elapsed for the dynamic_path processes was: %f ms.\n",((float)(end-start)/CLOCKS_PER_SEC)*1000);
}
/******************************************************************************/








/******************************************* FUSION IMPLEMENTATION ****************************************************/
//constructor to be used in case something has to be initialize in this class
fusions::fusions(){
	return;
}

//destructor to this class
fusions::~fusions(){
	return;
}



/*
	This fusion type averages both maps (dynamic and static) and estimates the master saliency maps for visual
	output, for the frame just processed taking into account n_frames ahead of the current frame.

	Input:
		1. Maps to be fusioned
		2. Width of the frames
		3. Height of the frames
		4. Number of master maps to be output (usually 1 master map)

	Output
		1. The master Saliency Map of the frame processed
*/
int fusions::average_map(unsigned char *maps[], unsigned char *before_fusion[],	const int width, const int height, const int n_maps){
	return average_fusion_main(maps, before_fusion, width, height, n_maps);
}

/*
	This fusion type averages both maps (dynamic and static) by having a weight for each map, and estimates the master saliency maps for visual
	output, for the frame just processed taking into account n_frames ahead of the current frame.

	Input:
		1. Maps to be fusioned
		2. Width of the frames
		3. Height of the frames
		4. Number of master maps to be output (usually 1 master map)

	Output
		1. The master Saliency Map of the frame processed
*/
int fusions::weighted_average_map(unsigned char *maps[], unsigned char *before_fusion[],	const int width, const int height, const int n_maps){
	return w_average_fusion_main(maps, before_fusion, width, height, n_maps);
}

/*
	This fusion type sums up both maps (dynamic and static) and estimates the master saliency maps for visual
	output by normilizing the values by its maximum and minimum, for the frame just processed taking into account n_frames ahead of the current frame.

	Input:
		1. Maps to be fusioned
		2. Width of the frames
		3. Height of the frames
		4. Number of master maps to be output (usually 1 master map)

	Output
		1. The master Saliency Map of the frame processed
*/
int fusions::normalized_sum(unsigned char *maps[], unsigned char *before_fusion[],	const int width, const int height, const int n_maps){
	return n_sum_fusion_main(maps, before_fusion, width, height, n_maps);
}
/******************************************* FUSION IMPLEMENTATION ****************************************************/


/********************************************** PIXEL FADING FUNCTION **************************************/
fading_tracker::fading_tracker(){
	return;
}

fading_tracker::~fading_tracker(){
	return;
}

/*
	Is in charge of controlling the fading of the pixels in the static maps only.

	Input:
		1. Maps to fade
		2. Width of the frame
		3. Height of the frame
		4. Number of maps to be processed

	Output
		1. The master Saliency Map after implementing the pixel fading part
*/
int fading_tracker::fade_activation(unsigned char *maps[], const int width, const int height, const int n_maps, float *ages){
	//call the functions to do the aging in cuda
	return fade_tracker_main(maps,width,height,n_maps,ages);
}
/********************************************** PIXEL FADING FUNCTION **************************************/


/********************************************** ALPHA BLENDING FUNCTION **************************************/
Blending::Blending(){
	return;
}

Blending::~Blending(){
	return;
}

/*
	Is in charge of blending the master saliency map with the original frame for a funsioned visualization of both
	images at the same time and have a better understanding of what the saliency map is capable of

	Input:
		1. Master Saliency Map
		2. Original Frames from the video file
		3. Width of the frame
		4. Height of the frame
		5. Number of maps to be used for the alpha blending (just one master map)
		6. Number of Channels in the original frames (RGB or GRAY)

	Output
		1. Blended frames of the new video sequence
*/
int Blending::Blending_activated(unsigned char *blend_frames[], unsigned char *frames[], unsigned char *maps[], const int width, const int height, const int n_maps, const int channels, 
	const int pix_dist){
	//call the main functions to do the blending in cuda
	return alpha_blending_main(blend_frames, frames,maps,width,height,n_maps,channels, pix_dist);
}
/********************************************** ALPHA BLENDING FUNCTION **************************************/



/******************************** GPU recognition ****************************************/
Execute::Execute(){
	return;
}

Execute::~Execute(){
	return;
}

//for initialization of the devices, and avoid lazy context
int Execute::initialization(){
	return initialize_gpu();
}

//multiple devices
int Execute::Run_MultiGPUs(unsigned char *frames[], unsigned char *maps[],const int width, const int height, const int n_frames, const int channels, const int orientations, 
	const int pyramid_levels, float *ages, unsigned char *temp_output[], unsigned char *output[], size_t frame_size){
		//call the main function to be executed
		return MultiGPUs(frames,maps,width,height,n_frames,channels,orientations,pyramid_levels,ages,temp_output,output,frame_size);
}

//single device
int Execute::Run_SingleGPU(unsigned char *frames[], unsigned char *maps[],const int width, const int height, const int n_frames, const int channels, const int orientations, 
	const int pyramid_levels, float *ages, unsigned char *temp_output[], unsigned char *output[], size_t frame_size){
		//call the main function to be executed
		return SingleGPU(frames,maps,width,height,n_frames,channels,orientations,pyramid_levels,ages,temp_output,output,frame_size);
}

/******************************** GPU recognition ****************************************/



/****************************** YUV file handling **********************************/
//constructor
YUV_handling::YUV_handling(){
	return;
}

/*destructor*/
YUV_handling::~YUV_handling(){
	return;
}


/*

	RGBtoYUV_conversion:
		Function to handle YUV frames coming from compressed data
		
			Input: 
				1. YUV_frames
				2. RGB_frames
				3. Y_height
				4. Y_width
				5. Cb_height
				6. Cb_width
				7. Cr_height
				8. Cr_width

	*/
int YUV_handling::YUVtoRGB_conversion(unsigned char *YUV_frames[], unsigned char *RGB_frames[], int Y_height, int Y_width, int Cb_height, int Cb_width, int Cr_height, int Cr_width, int n_frames,
	const int pix_dist){
	return convert_YUVtoRGB(YUV_frames,RGB_frames,Y_height,Y_width,Cb_height,Cb_width,Cr_height,Cr_width,n_frames,pix_dist);
}


/*

	RGBtoYUV_conversion:
		Function to handle YUV frames coming from compressed data
		
			Input: 
				1. YUV_frames
				2. RGB_frames
				3. Y_height
				4. Y_width
				5. Cb_height
				6. Cb_width
				7. Cr_height
				8. Cr_width

	*/
int YUV_handling::RGBtoYUV_conversion(unsigned char *YUV_frames[], unsigned char *RGB_frames[], int Y_height, int Y_width, int Cb_height, int Cb_width, int Cr_height, int Cr_width, int n_frames,
	const int pix_dist){
	return convert_RGBtoYUV(YUV_frames,RGB_frames,Y_height,Y_width,Cb_height,Cb_width,Cr_height,Cr_width,n_frames,pix_dist);
}

/****************************** YUV file handling **********************************/
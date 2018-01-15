/*
								VIDEO SALIENCY MAP API
Summary:
This is the library which is in charge of estimating the saliency maps for a given video, 
including a static and dynamic path. It deals with the problem by separating the static
and dynamic parts of the problems into two different GPUs. However in case that only one GPU
is present, it will process both maps sequentially.

Idea:
Input an array of images (frames) and ouput the static and dynamic saliency maps. In other
words the library determines which parts of a given image deserve attention when we analize
one single frame (image), and several frames (array of images) for motion detection.


*/

/*symbol in charge of exporting the library*/
#ifdef SALIENCYLIB_EXPORTS
#define SALIENCYLIB_API __declspec(dllexport)
#else
#define SALIENCYLIB_API __declspec(dllimport)
#endif


/*This namespace is exported from the saliencyLIB.dll*/
namespace saliency{
	enum fusion_type{
		AVERAGE = 0,
		WEIGHTED_AVERAGE,
		NORM_SUM
	};


	enum pixel_arrangement{
		PLANAR = 0,
		INTERLEAVED
	};



	/*
		Function used by the user in order to compensate the motion of consecutive frames
		in the sequence to be used
	*/
	SALIENCYLIB_API int Motion_compensation(unsigned char *frames[], const int width, const int height, 
		const int n_frames,	const int channels, bool first_frame);


	/*
		Function accessed by the user to obtain the INDIVIDUAL saliency maps for the static and the dynamic
		pathways, this will return two maps which could be later unified by the user
		or studied individualy
	*/
	SALIENCYLIB_API int Saliency_maps(unsigned char *frames[], unsigned char *maps[],
		const int width, const int height, const int n_frames, const int channels, const int orientations,
		const int pyramid_levels, float *ages, pixel_arrangement pix_dist);

	/*
		Function accessed by the user to obtain the MASTER saliency map FROM the static and the dynamic
		pathways, this will return one maps determining the saliency of each squence of frames
	*/
	SALIENCYLIB_API int Fusion_maps(unsigned char *maps[], unsigned char *before_fusion[],	const int width, 
		const int height, const int n_maps, fusion_type _type = NORM_SUM);


	/*
		Function in charge of taking care of the pixels fading given their age in the process of 
		Saliency mapping
	*/
	SALIENCYLIB_API int Fading_maps(unsigned char *maps[], const int width, const int height, const int n_maps,
		float *ages);

	/*
		Function in the saliency API in charge of doing the blending, if the user wants to perform it, or 
		it could be deactivated otherwise
	*/
	SALIENCYLIB_API int Frames_Maps_Blend(unsigned char *blend_frames[], unsigned char *frames[], unsigned char *maps[], 
		const int width, const int height, const int n_maps, const int channels, pixel_arrangement pix_dist);

	/*
		Function in charge of initialization of the device
	*/
	SALIENCYLIB_API int Device_Initialization();



	/*

	YUVtoRGB:
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
	SALIENCYLIB_API int YUVtoRGB(unsigned char *YUV_frames[], unsigned char *RGB_frames[], int Y_height, 
		int Y_width, int Cb_height, int Cb_width, int Cr_height, int Cr_width, int n_frames, pixel_arrangement pix_dist);


	/*

	RGBtoYUV:
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
	SALIENCYLIB_API int RGBtoYUV(unsigned char *YUV_frames[], unsigned char *RGB_frames[], int Y_height, 
		int Y_width, int Cb_height, int Cb_width, int Cr_height, int Cr_width, int n_frames, pixel_arrangement pix_dist);

};
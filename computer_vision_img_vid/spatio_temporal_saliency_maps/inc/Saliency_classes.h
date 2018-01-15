/***************motion compensation*****************************************************/
class motion_compensation{
public:
	motion_compensation(void);
	~motion_compensation(void);		
	//in charge of enhance the frames for a better postprocessing
	static int compensation(unsigned char *frames[], const int width, const int height, 
		const int n_frames, const int channels, bool first_frame);
};
/***************************************************************************************/

/***************image preprocessing*****************************************************/
class retinal_filter{
public:
	retinal_filter(void);
	~retinal_filter(void);		
	//in charge of enhance the frames for a better postprocessing
	static int frame_enhancement(unsigned char *frames[], unsigned char *filtered[],const int width,
		const int height, const int n_frames, const int channels, const int pix_dist);
};

class hanning_mask{
public:
	hanning_mask(void);
	~hanning_mask(void);
	//denosing the frames
	static int noise_reduction(unsigned char *frames[],const int width, const int height, 
		const int n_frames, const int GPU_id);
};
/***************************************************************************************/


/*****************************STATIC PATHWAY CLASSES********************************************/
class static_path{
public:
	static_path();
	~static_path();
	//estimating motion with gabor filters
	static int static_map(unsigned char *frames[], unsigned char *static_maps[],const int width, const int height, 
		const int n_maps, const int GPU_id, const int orientations, const int py_levels);
};
/***************************************************************************************/



/*****************************DYNAMIC PATHWAY CLASSES********************************************/
class dynamic_path{
public:
	dynamic_path();
	~dynamic_path();
	//estimating motion with gabor filters
	static int dynamic_map(unsigned char *frames[], unsigned char *dynamic_map[],const int width, 
		const int height, const int n_frames, const int GPU_id, float *ages);
};
/***************************************************************************************/



/******************************************** FUSION *************************************/
class fusions{
public:
	fusions();
	~fusions();
	//fusion of the maps through these functions
	static int average_map(unsigned char *maps[], unsigned char *before_fusion[],	const int width, 
		const int height, const int n_maps);
	static int weighted_average_map(unsigned char *maps[], unsigned char *before_fusion[],	const int width, 
		const int height, const int n_maps);
	static int normalized_sum(unsigned char *maps[], unsigned char *before_fusion[],	const int width, 
		const int height, const int n_maps);
};

/******************************************** / *************************************/

/********************** FADING FUNCTION FOR THE PIXELS *******************************/
class fading_tracker{
public:
	fading_tracker();
	~fading_tracker();
	static int fade_activation(unsigned char *maps[], const int width, const int height, const int n_frames, 
		float *ages);
};
/************************* PIXEL FADING FUNCTION ********************************/

/********************** BLENDING FUNCTION FOR THE PIXELS *******************************/
class Blending{
public:
	Blending();
	~Blending();
	static int Blending_activated(unsigned char *blend_frames[], unsigned char *frames[], unsigned char *maps[], 
		const int width, const int height, const int n_frames, const int channels, const int pix_dist);
};
/************************* PIXEL BLENDING FUNCTION ********************************/


/******************************** GPU recognition ****************************************/
class Execute{
public:
	Execute();
	~Execute();
	static int initialization();
	static int Run_MultiGPUs(unsigned char *frames[], unsigned char *maps[],const int width, const int height, 
		const int n_frames, const int channels, const int orientations, const int pyramid_levels, float *ages,
		unsigned char *temp_output[], unsigned char *output[], size_t frame_size);
	static int Run_SingleGPU(unsigned char *frames[], unsigned char *maps[],const int width, const int height, 
		const int n_frames, const int channels, const int orientations, const int pyramid_levels, float *ages,
		unsigned char *temp_output[], unsigned char *output[], size_t frame_size);
};
/******************************** GPU recognition ****************************************/



/****************************** YUV file handling **********************************/

class YUV_handling{
public:
	YUV_handling();
	~YUV_handling();
	static int YUVtoRGB_conversion(unsigned char *YUV_frames[], unsigned char *RGB_frames[], int Y_height, 
		int Y_width, int Cb_height, int Cb_width, int Cr_height, int Cr_width, int n_frames, const int pix_dist);
	static int RGBtoYUV_conversion(unsigned char *YUV_frames[], unsigned char *RGB_frames[], int Y_height, 
		int Y_width, int Cb_height, int Cb_width, int Cr_height, int Cr_width, int n_frames, const int pix_dist);
};

/****************************** YUV file handling **********************************/
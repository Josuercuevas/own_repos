#include "cuda_includes.h"
#include "defines_values.h"


/*convert_YUVtoRGB*/
int convert_YUVtoRGB(unsigned char *YUV_frames[], unsigned char *RGB_frames[], int Y_height, int Y_width, 
	int Cb_height, int Cb_width, int Cr_height, int Cr_width, int n_frames, const int pix_dist);

/*helper functions*/
extern __global__ void YUVconverter(unsigned char *YUV_frames, unsigned char *RGB_frames, int Y_height, int Y_width, 
	int Cb_height, int Cb_width, int Cr_height, int Cr_width, int n_frames, const int pix_dist);







/*convert_RGBtoYUV*/
int convert_RGBtoYUV(unsigned char *YUV_frames[], unsigned char *RGB_frames[], int Y_height, int Y_width, int Cb_height, 
	int Cb_width, int Cr_height, int Cr_width, int n_frames, const int pix_dist);

/*helper functions*/
extern __global__ void RGBconverter(unsigned char *YUV_frames, unsigned char *RGB_frames, int Y_height, int Y_width, 
	int Cb_height, int Cb_width, int Cr_height, int Cr_width, int n_frames, const int pix_dist);
#include "stdafx.h"
#include "cpp_includes.h"
#include "CImg.h"
#include "CIMG_disp.h"

using namespace cimg_library;
CImgDisplay disp;

void disp_int_pic(int *data, int height, int width, int channels){
	CImg<int> frame(data,width,height,1,channels);
	disp.display(frame);
}

void disp_float_pic(float *data, int height, int width, int channels){
	CImg<float> frame(data,width,height,1,channels);
	disp.display(frame);
}

void disp_uchar_pic(unsigned char *data, int height, int width, int channels){
	CImg<unsigned char> frame(data,width,height,1,channels);
	disp.display(frame);
}
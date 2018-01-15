/*
Main entry functions

Copyright (C) <2015>  <Josue R. Cuevas>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include <cstdlib>
#include <iostream>
#include <cstdio>
#include "colormark_structures.h"
#include "canny_functions.cuh"



#include <opencv\cv.h>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>



using namespace cv;
using namespace std;

void canny_main(int* intensities,int* d_intensities, int* G_direction, int *Gx_strength, int *Gy_strength, const int Threads, const int size, const int width){
	const int blocks = (Threads+size-1)/Threads;
	int *G_strength, *G_direction_round, min_T = 300, max_T = min_T*2;

	cudaMalloc((void**)&G_direction_round,sizeof(int)*size);//will contain the gradients direction in 4 values (0, 45, 90, 135) just for this part of the program
	cudaMalloc((void**)&G_strength,sizeof(int)*size);//will contain the gradients strength


	blur<<<blocks,Threads>>>(d_intensities, width,size);

	/*****************************************blur visualization*********************************************/
	//int *intensities_grad;
	//intensities_grad = (int*)malloc(sizeof(int)*size);
	//cudaMemcpyAsync(intensities_grad,d_intensities,size*sizeof(int),cudaMemcpyDeviceToHost);
	//Mat blurred(size/width,width,CV_8UC1);//you can make it on the GPU too
	//int counter = 0;
	//for(int i=0;i<(size/width);i++)
	//	for(int j=0;j<width;j++)
	//		/*if(intensities_grad[counter]>=0 && intensities_grad[counter]<256)
	//			blurred.at<uchar>(i,j)=(uchar)(intensities_grad[counter++]);
	//		else{*/
	//			blurred.at<uchar>(i,j)=(uchar)(intensities_grad[counter++]);
	//			/*cout<<intensities_grad[counter-1]<<endl;*///}
	//free(intensities_grad);
	//imwrite("C:/Users/Josue/Documents/Visual Studio 2010/Projects/Colormark/testbed/results/blurred image.jpg",blurred);
	/*****************************************blur visualization*********************************************/

	gradients<<<blocks, Threads>>>(d_intensities, G_strength, Gx_strength, Gy_strength, G_direction, G_direction_round, width,size);//calcualtes the gradient

	non_max_supr<<<blocks, Threads>>>(d_intensities,G_strength, G_direction_round, width, max_T, min_T,size);//surpresses the non-maxima values

	cudaMemcpyAsync(intensities,d_intensities,size*sizeof(int),cudaMemcpyDeviceToHost);//gets the canny map

	cudaFree(G_strength);
	cudaFree(G_direction_round);
}

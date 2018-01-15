/*
Main implementations

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
#include <cstdio>
#include "colormark_structures.h"
#include "canny_functions.cuh"
#include "math_functions.h"


__global__ void non_max_supr(int* intensities, int* G_strengths, int* G_directions, int width, int max_T,int min_T, int size)
{
	const int winsize = 3;
	const int height = size/width;
	const int shift = (winsize/2), extra_left = shift, extra_right = shift;
	int position;
	float pi = 3.14159265358979323846;

	unsigned int idx=threadIdx.x + blockIdx.x*blockDim.x;//thread index
	unsigned int tid = threadIdx.x;

	__shared__ int pixel_grad[winsize][extra_left+256+extra_right];//copying into shared memory and is +2 because of the extra values we need to copy when having positions 0 and 255
	__shared__ int pixel_angle[256];//copying into shared memory and is +2 because of the extra values we need to copy when having positions 0 and 255


	int x = (idx)%width, y = (idx)/width;
	if(x>shift && y>shift && x<(width-shift) && y<(height-shift)){//dont take into account boundaries including the window size
		if(tid>0 && tid<(blockDim.x-1))//copying the values before reaching the overlaping part of the shared window
		{
			pixel_angle[tid] = G_directions[idx];
			for(int i=0;i<winsize;i++)
			{
				position = idx + ((i-(shift))*width);
				pixel_grad[i][tid+extra_left] = G_strengths[position];//copies only up and down of the current pixels
			}
		}
		else{//this part is the boundary of the window therefore we need to copy up and down as well as left for tid=0 and right for tid=255
			pixel_angle[tid] = G_directions[idx];
			if(tid==0){//means tid=255, therefore, copy left
				for(int j=0;j<=extra_left;j++)
				{
					for(int i=0;i<winsize;i++)
					{
						position = idx + ((i-(shift))*width) - j;
						pixel_grad[i][extra_left-j] = G_strengths[position];
					}
				}
			}
			else if(tid==(blockDim.x-1)){//means tid=255, therefore, copy right                                                                                                                                                                                                                                                                                                                                                                                                                s only up and down of the pixels to the right of the 255-shift position
				for(int j=0;j<=extra_right;j++)
				{
					for(int i=0;i<winsize;i++)
					{
						position = idx + ((i-(shift))*width) + j;
						pixel_grad[i][(tid+extra_left)+j] = G_strengths[position];
					}
				}
			}
		}
	}
	__syncthreads();


	if(idx < size){
		if(x>shift && y>shift && x<(width-shift) && y<(height-shift)){//dont take into account boundaries including the window size
			if(pixel_grad[shift][(tid+extra_left)]>=max_T)//pixels with larger values than the maximum are considered as strong edges
			{
				switch(pixel_angle[tid])//will determine if the pixel belongs to an edge or not
				{
					case 0:
						if(pixel_grad[shift][(tid+extra_left)]>pixel_grad[shift][(tid+extra_left)-shift] && pixel_grad[shift][(tid+extra_left)]>pixel_grad[shift][(tid+extra_left)+shift])
							intensities[idx] = 255;
						else
							intensities[idx] = 0;//is not a maxima gradient
						break;

					case 45:
						if(pixel_grad[shift][(tid+extra_left)]>=pixel_grad[shift-shift][(tid+extra_left)-shift] && pixel_grad[shift][(tid+extra_left)]>=pixel_grad[shift+shift][(tid+extra_left)+shift])
							intensities[idx] = 255;
						else
							intensities[idx] = 0;//is not a maxima gradient
						break;

					case 90:
						if(pixel_grad[shift][(tid+extra_left)]>pixel_grad[shift+shift][(tid+extra_left)] && pixel_grad[shift][(tid+extra_left)]>pixel_grad[shift-shift][(tid+extra_left)])
							intensities[idx] = 255;
						else
							intensities[idx] = 0;//is not a maxima gradient
						break;

					case 135:
						if(pixel_grad[shift][(tid+extra_left)]>=pixel_grad[shift-shift][(tid+extra_left)+shift] && pixel_grad[shift][(tid+extra_left)]>=pixel_grad[shift+shift][(tid+extra_left)-shift])
							intensities[idx] = 255;
						else
							intensities[idx] = 0;//is not a maxima gradient
						break;

					default: intensities[idx] = 0;//just in case
				}
			}
			else if (pixel_grad[shift][(tid+extra_left)]>min_T){//implementing hysteresis
				switch(pixel_angle[tid])//will determine if the pixel belongs to an esge or not
				{
					case 0:
						if(pixel_grad[shift][(tid+extra_left)-shift]>=max_T && pixel_grad[shift][(tid+extra_left)+shift]>=max_T)
							intensities[idx] = 255;
						else
							intensities[idx] = 0;//is not a maxima gradient
						break;

					case 45:
						if(pixel_grad[shift-shift][(tid+extra_left)-shift]>=max_T && pixel_grad[shift+shift][(tid+extra_left)+shift]>=max_T)
							intensities[idx] = 255;
						else
							intensities[idx] = 0;//is not a maxima gradient
						break;

					case 90:
						if(pixel_grad[shift+shift][(tid+extra_left)]>=max_T && pixel_grad[shift-shift][(tid+extra_left)]>=max_T)
							intensities[idx] = 255;
						else
							intensities[idx] = 0;//is not a maxima gradient
						break;

					case 135:
						if(pixel_grad[shift-shift][(tid+extra_left)+shift]>=max_T && pixel_grad[shift+shift][(tid+extra_left)-shift]>=max_T)
							intensities[idx] = 255;
						else
							intensities[idx] = 0;//is not a maxima gradient
						break;

					default: intensities[idx] = 0;//just in case
				}
			}
			else
				intensities[idx] = 0;//no need to check angle since it isnt an edge at all
		}
	}
}




/************************PERFORMS GRADIENT CALCULATION**********************************/
__global__ void gradients(int* intensities, int* G_strength, int* Gx_str, int *Gy_str, int *G_direction, int *G_dir, int width, int size){
	const int winsize = 5;
	const int height = size/width;
	const int shift = (winsize/2), extra_left = shift, extra_right = shift;
	int position;
	float pi = 3.14159265358979323846;

	unsigned int idx=threadIdx.x + blockIdx.x*blockDim.x;//thread index
	unsigned int tid = threadIdx.x;

	__shared__ int pixel_inte[winsize][extra_left+256+extra_right];//copying into shared memory and is +2 because of the extra values we need to copy when having positions 0 and 255
	__shared__ int Gx[winsize][winsize];//the mask for gradient in x
	__shared__ int Gy[winsize][winsize];//the mask for gradient in y


	if(tid>0 && tid<(blockDim.x-1))//copying the values before reaching the overlaping part of the shared window
	{
		for(int i=0;i<winsize;i++)
		{
			position = idx + ((i-(shift))*width);
			pixel_inte[i][tid+extra_left] = intensities[position];//copies only up and down of the current pixels
		}
	}
	else{//this part is the boundary of the window therefore we need to copy up and down as well as left for tid=0 and right for tid=255
		if(tid==0){//copies only up and down of the pixels to the left of the 0+shift position
			for(int j=0;j<=extra_left;j++)
			{
				for(int i=0;i<winsize;i++)
				{
					position = idx + ((i-(shift))*width) - j;
					pixel_inte[i][extra_left-j] = intensities[position];
				}
			}
		}
		else if(tid==(blockDim.x-1)){//means tid=255, therefore, copie                                                                                                                                                                                                                                                                                                                                                                                                                         s only up and down of the pixels to the right of the 255-shift position
			for(int j=0;j<=extra_right;j++)
			{
				for(int i=0;i<winsize;i++)
				{
					position = idx + ((i-(shift))*width) + j;
					pixel_inte[i][(tid+extra_left)+j] = intensities[position];
				}
			}
		}
	}

	if(tid==0){//the first thread of every block makes this instantiation
		/**************X-gradient mask**************/
		Gx[0][0] = 1;	Gx[0][1] = 2;	Gx[0][2] = 0;	Gx[0][3] = -2;	Gx[0][4] = -1;
		Gx[1][0] = 4;	Gx[1][1] = 8;	Gx[1][2] = 0;	Gx[1][3] = -8;	Gx[1][4] = -4;
		Gx[2][0] = 6;	Gx[2][1] = 12;	Gx[2][2] = 0;	Gx[2][3] = -12;	Gx[2][4] = -6;
		Gx[3][0] = 4;	Gx[3][1] = 8;	Gx[3][2] = 0;	Gx[3][3] = -8;	Gx[3][4] = -4;
		Gx[4][0] = 1;	Gx[4][1] = 2;	Gx[4][2] = 0;	Gx[4][3] = -2;	Gx[4][4] = -1;

		/**************Y-gradient mask**************/
		Gy[0][0] = 1;	Gy[0][1] = 4;	Gy[0][2] = 6;	Gy[0][3] = 4;	Gy[0][4] = 1;
		Gy[1][0] = 2;	Gy[1][1] = 8;	Gy[1][2] = 12;	Gy[1][3] = 8;	Gy[1][4] = 2;
		Gy[2][0] = 0;	Gy[2][1] = 0;	Gy[2][2] = 0;	Gy[2][3] = 0;	Gy[2][4] = 0;
		Gy[3][0] = -2;	Gy[3][1] = -8;	Gy[3][2] = -12;	Gy[3][3] = -8;	Gy[3][4] = -2;
		Gy[4][0] = -1;	Gy[4][1] = -4;	Gy[4][2] = -6;	Gy[4][3] = -4;	Gy[4][4] = -1;
	}
	__syncthreads();


	if(idx < size){

		float Ix=0,Iy=0;//has the summation of the gradients of x and y

		/**********************************instantiating the filter***********************************/
		for(int i=-shift;i<=shift;i++)//filling the filter
		{
			for(int j=-shift;j<=shift;j++){
				/**************calculating the strenth of the gradient on each axis****************/
				Ix += Gx[shift+i][shift+j]*pixel_inte[shift+i][(tid+extra_left)+j];
				Iy += Gy[shift+i][shift+j]*pixel_inte[shift+i][(tid+extra_left)+j];
			}
		}


		int x = (idx)%width, y = (idx)/width;
		if(x>=shift && y>=shift && x<=(width-shift) && y<=(height-shift)){//dont take into account boundaries including the window size

			/*Calculating the strength and angle of the gradient on that particular pixel*/
			float angle = (atan2(Iy, Ix)*180.0)/pi;//argument of the square root
			G_direction[idx] = angle;
			Gx_str[idx] = (Ix);
			Gy_str[idx] = (Iy);


			/********************* Convert actual edge direction to approximate value**********************/
			if ( ( (angle <= 22.5) && (angle > -22.5) ) || (angle > 157.5) || (angle <= -157.5) )
			{
				G_dir[idx] = 0;//angle rounded to the only 4 possible values in an image
				G_strength[idx] = (int)sqrt(Ix*Ix + Iy*Iy);
			}
			else if ( ( (angle > 22.5) && (angle <= 67.5) ) || ( (angle <= -112.5) && (angle > -157.5) ) ){
				G_dir[idx] = 45;//angle rounded to the only 4 possible values in an image
				G_strength[idx] = (int)sqrt(Ix*Ix + Iy*Iy);
			}
			else if ( ( (angle > 67.5) && (angle <= 112.5) ) || ( (angle <= -67.5) && (angle > -112.5) ) ){
				G_dir[idx] = 90;//angle rounded to the only 4 possible values in an image
				G_strength[idx] = (int)sqrt(Ix*Ix + Iy*Iy);
			}
			else if ( ( (angle > 112.5) && (angle <= 157.5) ) || ( (angle <= -22.5) && (angle > -67.5) ) ){
				G_dir[idx] = 135;//angle rounded to the only 4 possible values in an image
				G_strength[idx] = (int)sqrt(Ix*Ix + Iy*Iy);
			}
		}
	}
}





/************************PERFORMS THE GAUSSIAN BLURRING FOR THE IMAGE**********************************/
__global__ void blur(int* intensities, int width, int size){
	const int winsize = 5;
	const int height = size/width;
	const int shift = (winsize/2), extra_left = shift, extra_right = shift;
	int position;

	unsigned int idx=threadIdx.x + blockIdx.x*blockDim.x;//thread index
	unsigned int tid = threadIdx.x;

	__shared__ int pixel_inte[winsize][extra_left+256+extra_right];//copying into shared memory and is +2 because of the extra values we need to copy when having positions 0 and 255


	//copying the block on the shared data consisting of [extra_left + 256 +extra_right] elements
	if(tid>0 && tid<(blockDim.x-1))//copying the values before reaching the overlaping part of the shared window
	{
		for(int i=0;i<winsize;i++)
		{
			position = idx + ((i-(shift))*width);
			pixel_inte[i][tid+extra_left] = intensities[position];//copies only center, up and down of the current pixels
		}
	}
	else{//this part is the boundary of the window therefore we need to copy up and down as well as left for tid=0 and right for tid=255
		if(tid==0){//means tid=0, therefore, copy the extra values to the left
			for(int j=0;j<=extra_left;j++)
			{
				for(int i=0;i<winsize;i++)
				{
					position = idx + ((i-(shift))*width) - j;
					pixel_inte[i][extra_left-j] = intensities[position];
				}
			}
		}
		else if(tid==(blockDim.x-1)){//means tid=255, therefore, copy the extra values to the right                                                                                                                                                                                                                                                                                                                                                                                                                        s only up and down of the pixels to the right of the 255-shift position
			for(int j=0;j<=extra_right;j++)
			{
				for(int i=0;i<winsize;i++)
				{
					position = idx + ((i-(shift))*width) + j;
					pixel_inte[i][(tid+extra_left)+j] = intensities[position];
				}
			}
		}
	}
	__syncthreads();

	//estimating the blurred value of the pixel at location idx
	int filter_sum=159;
	float filter_val=0;

	if(idx < size){
			int filter[winsize][winsize];

			/**********************************instantiating the filter***********************************/
			filter[0][0] = 2;	filter[0][1] = 4;	filter[0][2] = 5;	filter[0][3] = 4;	filter[0][4] = 2;
			filter[1][0] = 4;	filter[1][1] = 9;	filter[1][2] = 12;	filter[1][3] = 9;	filter[1][4] = 4;
			filter[2][0] = 5;	filter[2][1] = 12;	filter[2][2] = 15;	filter[2][3] = 12;	filter[2][4] = 5;
			filter[3][0] = 4;	filter[3][1] = 9;	filter[3][2] = 12;	filter[3][3] = 9;	filter[3][4] = 4;
			filter[4][0] = 2;	filter[4][1] = 4;	filter[4][2] = 5;	filter[4][3] = 4;	filter[4][4] = 2;

			for(int i=-shift;i<=shift;i++)//convoluting with the filter
			{
				for(int j=-shift;j<=shift;j++){
					/******************************performing convolution******************************************/
					filter_val += (pixel_inte[shift+i][(tid+extra_left)+j]*filter[i+shift][j+shift]);//calculate the average blurring of the pixel
				}
			}
	}
	__syncthreads();

	//blurred value of the pixel
	int x = (idx)%width, y = (idx)/width;
	if(x>=shift && y>=shift && x<=(width-shift) && y<=(height-shift))//dont take into account boundaries including the window size
		intensities[idx] = (filter_val/filter_sum);//assigning the new value to the image

}

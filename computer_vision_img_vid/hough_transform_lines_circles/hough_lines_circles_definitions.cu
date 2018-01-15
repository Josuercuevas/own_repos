/*
Entry functions to lines and circles

Copyright (C) <2016>  <Josue R. Cuevas>

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

#include <vector>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include <cstdlib>
#include <iostream>
#include "hough_transform_functions.cuh"
#include <cmath>



#include <opencv\cv.h>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>
using namespace cv;



/************************************************************************will find circles in the image******************************************************************/
void hough_circles_main(int *d_intensities, int *d_gradients_dir, circle_par &circles, const int width, const int height, const int Threads,cudaStream_t stream, float *d_sines, float *d_cosines){


	int possible_circles = Threads*2, pic_size = width*height, *indices, norm_fact=255;
	int min_R = 30, max_R = min_R + 120;
	const int Blocks = ((width*height) + (Threads) - 1) / (Threads);

	int *accumulators, *d_accumulators, *R_space, *d_global_max, *d_local_maxes;
	int *d_x_values,*d_y_values,*d_votes_values, *d_R_values;
	int *x_values,*y_values, *R_values,*votes_values;


	//binding textures needed for the computation
	bind(d_gradients_dir,pic_size,1);


	accumulators = (int*)malloc(sizeof(int)*pic_size);//reserve memory at the CPU
	x_values = (int*)malloc(sizeof(int)*possible_circles);//reserve memory at the CPU for location of center
	y_values = (int*)malloc(sizeof(int)*possible_circles);//reserve memory at the CPU for location of center
	R_values = (int*)malloc(sizeof(int)*possible_circles);//reserve memory at the CPU for location of center
	votes_values = (int*)malloc(sizeof(int)*possible_circles);//reserve memory at the CPU for location of center VOTING



	cudaMalloc((void**)&indices,sizeof(int)*1);//reserve memory at the GPU
	cudaMalloc((void**)&d_accumulators,sizeof(int)*pic_size);//reserve memory at the GPU
	cudaMalloc((void**)&R_space,sizeof(int)*pic_size);//reserve memory at the GPU
	cudaMalloc((void**)&d_global_max,sizeof(int));//reserve memory at the GPU
	cudaMalloc((void**)&d_local_maxes,sizeof(int)*possible_circles);//reserve memory at the GPU
	cudaMalloc((void**)&d_R_values,sizeof(int)*possible_circles);//reserve memory at the GPU
	cudaMalloc((void**)&d_x_values,sizeof(int)*possible_circles);//reserve memory at the GPU
	cudaMalloc((void**)&d_y_values,sizeof(int)*possible_circles);//reserve memory at the GPU
	cudaMalloc((void**)&d_votes_values,sizeof(int)*possible_circles);//reserve memory at the GPU


	/*for Taubin*/
	int *d_x_mean, *d_y_mean, *d_neighbors;
	float *d_Mxx, *d_Mxy, *d_Myy, *d_Mxz, *d_Myz, *d_Mzz;
	cudaMalloc((void**)&d_x_mean,sizeof(int)*possible_circles);//reserve memory at the GPU
	cudaMalloc((void**)&d_y_mean,sizeof(int)*possible_circles);//reserve memory at the GPU
	cudaMalloc((void**)&d_neighbors,sizeof(int)*possible_circles);//reserve memory at the GPU
	cudaMalloc((void**)&d_Mxx,sizeof(float)*possible_circles);//reserve memory at the GPU
	cudaMalloc((void**)&d_Mxy,sizeof(float)*possible_circles);//reserve memory at the GPU
	cudaMalloc((void**)&d_Myy,sizeof(float)*possible_circles);//reserve memory at the GPU
	cudaMalloc((void**)&d_Mxz,sizeof(float)*possible_circles);//reserve memory at the GPU
	cudaMalloc((void**)&d_Myz,sizeof(float)*possible_circles);//reserve memory at the GPU
	cudaMalloc((void**)&d_Mzz,sizeof(float)*possible_circles);//reserve memory at the GPU

	//initialization of the accumulator
	cudaMemset(d_accumulators,0,sizeof(int)*pic_size);
	cudaMemset(R_space,0,sizeof(int)*pic_size);
	cudaMemset(d_local_maxes,0,sizeof(int)*possible_circles);
	cudaMemset(indices,0,sizeof(int)*1);
	cudaMemset(d_global_max,0,sizeof(int)*1);
	cudaMemset(d_votes_values,0,sizeof(int)*possible_circles);
	cudaMemset(d_x_values,0,sizeof(int)*possible_circles);
	cudaMemset(d_y_values,0,sizeof(int)*possible_circles);
	cudaMemset(d_x_mean,0,sizeof(int)*possible_circles);
	cudaMemset(d_y_mean,0,sizeof(int)*possible_circles);
	cudaMemset(d_neighbors,0,sizeof(int)*possible_circles);

	cudaMemset(d_Mxx,0,sizeof(float)*possible_circles);
	cudaMemset(d_Mxy,0,sizeof(float)*possible_circles);
	cudaMemset(d_Myy,0,sizeof(float)*possible_circles);
	cudaMemset(d_Mxz,0,sizeof(float)*possible_circles);
	cudaMemset(d_Myz,0,sizeof(float)*possible_circles);
	cudaMemset(d_Mzz,0,sizeof(float)*possible_circles);

	//for detecting circles in the image
	accumulator_counter<<<Blocks,Threads,0,stream>>>(d_intensities,d_accumulators,R_space,d_sines,d_cosines,min_R,max_R,width,pic_size);
	maximum_R_counters<<<Blocks,Threads,0,stream>>>(d_accumulators,d_global_max,d_local_maxes,Threads,width,pic_size);
	maxes_vals<<<Blocks,Threads,0,stream>>>(d_accumulators,d_global_max,R_space,d_x_values,d_y_values,d_votes_values,d_R_values,possible_circles,width,pic_size,indices);

	cudaMemcpyAsync(x_values,d_x_values,sizeof(int)*possible_circles,cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(y_values,d_y_values,sizeof(int)*possible_circles,cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(R_values,d_R_values,sizeof(int)*possible_circles,cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(votes_values,d_votes_values,sizeof(int)*possible_circles,cudaMemcpyDeviceToHost);


	for(int i=0;i<possible_circles;i++){
		if(votes_values[i]>0){
			if(circles.Xo.size()==0)
			{//saving location
				circles.Xo.push_back(x_values[i]);
				circles.Yo.push_back(y_values[i]);
				circles.radious.push_back(R_values[i]);
				circles.votes.push_back(votes_values[i]);
			}
			else
			{
				bool repeated = false;

				for(int j=0;j<circles.Xo.size();j++){
					if(sqrt((float)((circles.Xo[j]-x_values[i])*(circles.Xo[j]-x_values[i]) ) + (float)(((circles.Yo[j]-y_values[i])*(circles.Yo[j]-y_values[i])))) < circles.radious[j]){
						if(votes_values[i] > circles.votes[j]){
							circles.Xo[j] = x_values[i];
							circles.Yo[j] = y_values[i];
							circles.votes[j] = votes_values[i];
							circles.radious[j] = R_values[i];
						}//better fit
						repeated = true;
						break;
					}
				}

				if(!repeated){
					circles.Xo.push_back(x_values[i]);
					circles.Yo.push_back(y_values[i]);
					circles.radious.push_back(R_values[i]);
					circles.votes.push_back(votes_values[i]);
				}

			}
		}
		//cout<<votes_values[i]<<" "<<x_values[i]<<" "<<y_values[i]<<" "<<R_values[i]<<endl;
	}

	for(int j=0;j<circles.Yo.size();j++){//updating the values for faster access in TAUBIN
		x_values[j] = circles.Xo[j];
		y_values[j] = circles.Yo[j];
		R_values[j] = circles.radious[j];
		//printf("circle at:\tx: %i\ty: %i\tRadius: %i\tvotes: %i\n",x_values[j] , y_values[j],R_values[j],circles.votes[j]);
	}
	cudaMemcpyAsync(d_x_values,x_values,sizeof(int)*possible_circles,cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_y_values,y_values,sizeof(int)*possible_circles,cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_R_values,R_values,sizeof(int)*possible_circles,cudaMemcpyHostToDevice);

	/*using taubin for fitting circles*/
	//calculate the summation of the mean estimation
	sum_for_mean<<<Blocks,Threads,0,stream>>>(d_intensities,d_x_values,d_y_values,d_x_mean, d_y_mean, d_neighbors, width, circles.Xo.size(),d_R_values,pic_size);

	//calculates the mean
	mean_calculation<<<1,possible_circles,0,stream>>>(d_x_mean,d_y_mean,d_neighbors,circles.Xo.size());

	//computes the moment of the for circle fitting
	computing_moments<<<Blocks,Threads,0,stream>>>(d_intensities,d_x_values, d_y_values,d_x_mean,d_y_mean,d_Mxx,d_Myy,d_Mxy,d_Mxz,d_Myz,d_Mzz,circles.Xo.size(),d_R_values,width,pic_size);

	//fits the circles with newton raphson method
	circle_fitting<<<1,possible_circles,0,stream>>>(d_x_values, d_y_values, d_R_values, d_x_mean, d_y_mean, d_Mxx, d_Myy, d_Mxy, d_Mxz, d_Myz, d_Mzz, circles.Xo.size(), d_neighbors,height,width);

	cudaMemcpyAsync(R_values,d_R_values,sizeof(int)*possible_circles,cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(x_values,d_x_values,sizeof(int)*possible_circles,cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(y_values,d_y_values,sizeof(int)*possible_circles,cudaMemcpyDeviceToHost);

	for(int j=0;j<circles.Xo.size();j++)//updating centers after circle fitting
	{
		circles.radious[j] = R_values[j];
		circles.Xo[j] = x_values[j];
		circles.Yo[j] = y_values[j];
	}

	/*****************************************************printing the circle voting space**********************************/

	//Mat original;
	//original = imread("C:/Users/Josue/Documents/Visual Studio 2010/Projects/Colormark/testbed/results/original.jpg",1);
	//Point pt;

	//for(int i=0;i<circles.Xo.size();i++){
	//	pt.x = circles.Xo[i];
	//	pt.y = circles.Yo[i];
	//	circle(original,pt,circles.radious[i],Scalar(0,0,255), 2, CV_AA);
	//	line(original,pt,pt,Scalar(0,0,255), 2, CV_AA);
	//}

	//imwrite("C:/Users/Josue/Documents/Visual Studio 2010/Projects/Colormark/testbed/results/centers detected.jpg",original);

	/*****************************************************printing the circle voting space**********************************/



	cudaFree(d_x_values);
	cudaFree(d_y_values);
	cudaFree(d_R_values);
	cudaFree(d_votes_values);
	cudaFree(d_accumulators);
	cudaFree(R_space);
	cudaFree(d_global_max);
	cudaFree(d_local_maxes);

	/*for taubin*/
	cudaFree(d_x_mean);
	cudaFree(d_y_mean);
	cudaFree(d_neighbors);
	cudaFree(d_Mxx);
	cudaFree(d_Mxy);
	cudaFree(d_Myy);
	cudaFree(d_Mxz);
	cudaFree(d_Mzz);


	free(x_values);
	free(y_values);
	free(votes_values);
	free(accumulators);
	free(R_values);

	//unbind textures needed
	unbind(d_gradients_dir,1);
}





















/***************************************************************will find lines in the image*********************************************************************/
void hough_lines_main(int *d_intensities, int *d_gradients_dir, line_par &lines, const int width, const int height, const int Threads,cudaStream_t stream, float *d_sines, float *d_cosines){
	//float pi = 3.14159265358979323846;
	int pic_size = width*height;
	int line_Thres = 60;
	int block = ((pic_size)+Threads-1)/Threads; //blocks to launch according to the image size

	//binding texture needed for computation
	bind(d_gradients_dir,pic_size,1);

	/*we need to know the diagonal size for when rho is less than zero. As for the accumulator is of size 4*(2d) since the diagonal is twice as big
	to accept negative rho values, additionally, is 4 because we have 4 possible values for the gradient direction: 0, 45, 90 ,135 degrees*/
	int *cpu_lines, *d_lines, *d_accumulator, diagonal_size = sqrt((float)(width*width + height*height)), *indices;
	size_t accum_size = sizeof(int)*(360*diagonal_size*2); //since we have 360 angles and we the size of the diagonal for taking only positive values of rho
	size_t line_container = sizeof(int)*((diagonal_size)+360)*3; //we save three parameters, the angle, rho and voting result

	cpu_lines = (int*)malloc(line_container);


	cudaMalloc((void**)&d_lines,line_container);//is going to contain the parameter of each line, maximum of the number of threads to avoid overhead
	cudaMalloc((void**)&d_accumulator,accum_size);//contains the votes for all the possible values of rho and theta
	cudaMalloc((void**)&indices,sizeof(int)*1);//contains the votes for all the possible values of rho and theta

	/*initializing the container in cuda*/
	cudaMemset(d_lines,0,line_container);
	cudaMemset(indices,0,sizeof(int)*1);
	cudaMemset(d_accumulator,0,accum_size);

	/*initiating voting process*/
	hough_lines<<<block,Threads,0,stream>>>(d_intensities,d_cosines, d_sines, d_accumulator, diagonal_size, width, height);
	extraction<<<((accum_size/sizeof(int))+Threads-1)/Threads,Threads,0,stream>>>(d_accumulator, d_lines, line_Thres, diagonal_size,(accum_size/sizeof(int)),indices);


	cudaMemcpy(cpu_lines,d_lines,line_container,cudaMemcpyDeviceToHost);

	int idx=-1, lid=-1;//"idx" will have the index of the line after extraction and "lid" will have the index of the line in the vector of parameters
	int shitf_rho = (line_container/sizeof(int))/3;
	int shift_counter = (shitf_rho)*2;

	int counter = (line_container/sizeof(int))/3;

	while(idx<counter)//storing the parameters of each line, taking care that we have a unique line within a given range
	{
		idx++;
		if(cpu_lines[idx]>-360 && cpu_lines[idx]<360){

			bool entered = false;
			int temp_id = -1;

			if(cpu_lines[idx]==90 && (cpu_lines[idx+shitf_rho]<(height-5) && cpu_lines[idx+shitf_rho]>5) && cpu_lines[idx+shift_counter]>0 && cpu_lines[idx+shift_counter]<5000){//5 pixels from the borders of the image
				if(lid<0){//first element to be inserted
					lines.theta.push_back(cpu_lines[idx]);
					lines.rho.push_back(cpu_lines[idx+shitf_rho]);
					lines.votes.push_back(cpu_lines[idx+shift_counter]);
					lid++;
				}
				else
				{
					for(int c=0;c<lines.rho.size();c++)//searching in all the current lines in the vector
						if(abs(lines.rho[c]-cpu_lines[idx+shitf_rho])<=5 && abs(lines.theta[c]-cpu_lines[idx])<=2)
						{
							entered = true;
							temp_id = c;
						}
					if(!entered){//make sure we dont have almost identical lines in our vector of line parameters
						lines.theta.push_back(cpu_lines[idx]);
						lines.rho.push_back(cpu_lines[idx+shitf_rho]);
						lines.votes.push_back(cpu_lines[idx+shift_counter]);
						lid++;
					}
					else if(lines.votes[temp_id]<cpu_lines[idx+shift_counter]){//leaving the line with more votes
						lines.theta[temp_id] = cpu_lines[idx];
						lines.rho[temp_id] = cpu_lines[idx+shitf_rho];
						lines.votes[temp_id] = cpu_lines[idx+shift_counter];
					}
				}
			}
			else if(cpu_lines[idx]==-90 && (cpu_lines[idx+shitf_rho]<(height-5) && cpu_lines[idx+shitf_rho]>5) && cpu_lines[idx+shift_counter]>0 && cpu_lines[idx+shift_counter]<5000){////5 pixels from the borders of the image
				if(lid<0){//first element to be inserted
					lines.theta.push_back(cpu_lines[idx]);
					lines.rho.push_back(cpu_lines[idx+shitf_rho]);
					lines.votes.push_back(cpu_lines[idx+shift_counter]);
					lid++;
				}
				else
				{
					for(int c=0;c<lines.rho.size();c++)//searching in all the current lines in the vector
						if(abs(lines.rho[c]-cpu_lines[idx+shitf_rho])<=5 && abs(lines.theta[c]-cpu_lines[idx])<=2)
						{
							entered = true;
							temp_id = c;
						}
					if(!entered){//make sure we dont have almost identical lines in our vector of line parameters
						lines.theta.push_back(cpu_lines[idx]);
						lines.rho.push_back(cpu_lines[idx+shitf_rho]);
						lines.votes.push_back(cpu_lines[idx+shift_counter]);
						lid++;
					}
					else if(lines.votes[temp_id]<cpu_lines[idx+shift_counter]){//leaving the line with more votes
						lines.theta[temp_id] = cpu_lines[idx];
						lines.rho[temp_id] = cpu_lines[idx+shitf_rho];
						lines.votes[temp_id] = cpu_lines[idx+shift_counter];
					}
				}
			}
			else if(cpu_lines[idx]==0 && (cpu_lines[idx+shitf_rho]<(width-5) && cpu_lines[idx+shitf_rho]>5 && cpu_lines[idx+shift_counter]>0 && cpu_lines[idx+shift_counter]<5000)){////5 pixels from the borders of the image
				if(lid<0){//first element to be inserted
					lines.theta.push_back(cpu_lines[idx]);
					lines.rho.push_back(cpu_lines[idx+shitf_rho]);
					lines.votes.push_back(cpu_lines[idx+shift_counter]);
					lid++;
				}
				else
				{
					for(int c=0;c<lines.rho.size();c++)//searching in all the current lines in the vector
						if(abs(lines.rho[c]-cpu_lines[idx+shitf_rho])<=5 && abs(lines.theta[c]-cpu_lines[idx])<=2)
						{
							entered = true;
							temp_id = c;
						}
					if(!entered){//make sure we dont have almost identical lines in our vector of line parameters
						lines.theta.push_back(cpu_lines[idx]);
						lines.rho.push_back(cpu_lines[idx+shitf_rho]);
						lines.votes.push_back(cpu_lines[idx+shift_counter]);
						lid++;
					}
					else if(lines.votes[temp_id]<cpu_lines[idx+shift_counter]){//leaving the line with more votes
						lines.theta[temp_id] = cpu_lines[idx];
						lines.rho[temp_id] = cpu_lines[idx+shitf_rho];
						lines.votes[temp_id] = cpu_lines[idx+shift_counter];
					}
				}
			}
			else if(cpu_lines[idx]!=0 && cpu_lines[idx]!=90 && cpu_lines[idx]!=-90 && cpu_lines[idx+shift_counter]>0 && cpu_lines[idx+shift_counter]<5000){
				if(lid<0){//first element to be inserted
					lines.theta.push_back(cpu_lines[idx]);
					lines.rho.push_back(cpu_lines[idx+shitf_rho]);
					lines.votes.push_back(cpu_lines[idx+shift_counter]);
					lid++;
				}
				else
				{
					for(int c=0;c<lines.rho.size();c++)//searching in all the current lines in the vector
						if(abs(lines.rho[c]-cpu_lines[idx+shitf_rho])<=5 && abs(lines.theta[c]-cpu_lines[idx])<=2)
						{
							entered = true;
							temp_id = c;
						}
					if(!entered){//make sure we dont have almost identical lines in our vector of line parameters
						lines.theta.push_back(cpu_lines[idx]);
						lines.rho.push_back(cpu_lines[idx+shitf_rho]);
						lines.votes.push_back(cpu_lines[idx+shift_counter]);
						lid++;
					}
					else if(lines.votes[temp_id]<cpu_lines[idx+shift_counter]){//leaving the line with more votes
						lines.theta[temp_id] = cpu_lines[idx];
						lines.rho[temp_id] = cpu_lines[idx+shitf_rho];
						lines.votes[temp_id] = cpu_lines[idx+shift_counter];
					}
				}
			}
		}
	}

	free(cpu_lines);
	cudaFree(d_accumulator);
	cudaFree(d_lines);
	cudaFree(indices);

	//unbinding texture needed for computation
	unbind(d_gradients_dir,1);
}

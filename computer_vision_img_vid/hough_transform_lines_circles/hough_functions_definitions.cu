/*
Main function implementations

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

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include <cstdlib>
#include <sm_20_atomic_functions.h>
#include "hough_transform_functions.cuh"
#include <cmath>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

using namespace cv;
using namespace std;

texture<int,1,cudaReadModeElementType> gradir;


/*****************************************************************************HOUGH LINES*************************************************************************/


__global__ void extraction(int *accumulator,int* parameters, int line_thres, int diagonal_size,int size,int *l_id){
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

	if(idx < size){//avoid over operating with inactive threads
		int val = accumulator[idx];
		if(val >=line_thres){//line found
			int angle, diago = diagonal_size, loc_ang = idx/(2*diago), rho = idx%(2*diago), shitft_rho=(diago+360), shift_count=shitft_rho*2;//control positions
			bool enter = true;
			angle = loc_ang;
			if(rho < diago)
					rho = rho;
				else
					rho -= diago;

			if((angle>176 && angle<180) || (angle>=180 && angle<270) || (angle==270) || (angle>270 && rho>(diago-100)))
				enter = false;

			if(enter){
				int id = atomicAdd(l_id,1);
				parameters[id+shift_count] = val;//third part of the array containing votes keeping only the local maxes
				parameters[id] = angle;//first part of the array containing theta
				parameters[id+shitft_rho] = rho;//second part of the array containing rho
			}
		}
	}
}


__global__ void hough_lines(int* intensities, float *cosines, float* sines, int *accumulator, const int diagonal_size, const int width, const int height){
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

	if(idx < width*height){
		int x, y;
		x = (idx%width);/*x location of pixel*/
		y = (idx/width);/*y location of pixel and shifting to lower origin since we read pixels from top to bottom left to right*/

		if(intensities[idx]==255 && x>10 && x<(width-10) && y>10 && y<(height-10))//only edges are considered
		{
			int rho;
			float cosine, sine;//contains the values of the cosine and sine
			int direction = tex1Dfetch(gradir,idx);
			int angle_norm = direction;
			int angle_opp = (direction+180);
			int temp_ang;

			/***** at a given degree ****/
			for(int j=-5;j<5;j++){//sweeps on an ark of 5 degrees
				temp_ang = angle_norm+j;
				if((temp_ang<179 || temp_ang>269) && temp_ang<360 && temp_ang>-360){
					if(temp_ang<0)
						temp_ang = 360+temp_ang;
					cosine = cosines[temp_ang];//
					sine = sines[temp_ang];//
					rho = (x*cosine + y*sine);
					if(((temp_ang*(diagonal_size*2))+(rho+diagonal_size))>=0 && ((temp_ang*(diagonal_size*2))+(rho+diagonal_size))<360*(diagonal_size*2))
						atomicAdd((accumulator+((temp_ang*(diagonal_size*2))+(rho+diagonal_size))),1);
				}

				temp_ang = angle_opp+j;
				if((temp_ang<179 || temp_ang>269) && temp_ang<360 && temp_ang>-360){
					if(temp_ang<0)
						temp_ang = 360+temp_ang;
					cosine = cosines[temp_ang];//
					sine = sines[temp_ang];//
					rho = (x*cosine + y*sine);
					if(((temp_ang*(diagonal_size*2))+(rho+diagonal_size))>=0 && ((temp_ang*(diagonal_size*2))+(rho+diagonal_size))<360*(diagonal_size*2))
						atomicAdd((accumulator+((temp_ang*(diagonal_size*2))+(rho+diagonal_size))),1);
				}
			}
		}
	}
}
/**********************************************************************************************************HOUGH LINES*********************************************************************************/























__global__ void accumulator_counter(int *intensities, int *accumulators, int *R_candidates, float *sines, float *cosines,int min_radius, int max_radius,int width, int size){
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int pic_size = size;

	if(idx<pic_size){//avoids accessing invalid locations
		volatile int x = idx%width, y=idx/width;
		if(intensities[idx] == 255 && (x>10 && x<(width-10)) && (y>10 && y<(((size)/width)-10))){//only considers the edges
			//float pi = 3.14159265358979323846;
			int xo;
			int yo;
			int direction = tex1Dfetch(gradir,idx);
			int angle_norm = direction;
			int angle_opp = (direction+180);
			int center, temp_ang;

			for(int i=min_radius;i<max_radius;i+=5){//doing the voting for all the possible radious with the edge pixel
				for(int j=-2;j<2;j++){//sweep on an ark of 5 degrees arc

					temp_ang = angle_norm+j;
					//voting in one direction according to the gradient direction
					if(temp_ang>-360 && temp_ang<360){
						if(temp_ang<0)
							temp_ang = 360+temp_ang;
						xo = x + (int)i*cosines[temp_ang];
						yo = y + (int)i*sines[temp_ang];
					}

					if((xo>0 || xo<width) && (yo>0 || yo<(size/width))){
						center = xo + yo*width;//center calculation
						if(center<(pic_size) && center>0){
							atomicAdd(accumulators + center,1);
							atomicAdd(R_candidates + center, i);
						}
					}


					temp_ang = angle_opp+j;
					//voting in the opposite direction according to the gradient direction
					if(temp_ang>-360 && temp_ang<360){
						if(temp_ang<0)
							temp_ang = 360+temp_ang;
						xo = x + (int)i*cosines[temp_ang];
						yo = y + (int)i*sines[temp_ang];
					}

					if((xo>0 || xo<width) && (yo>0 || yo<(size/width))){
						center = xo + yo*width;//center calculation
						if(center<(pic_size) && center>0){
							atomicAdd(accumulators + center,1);
							atomicAdd(R_candidates + center, i);
						}
					}


				}

			}
		}
	}
}





__global__ void maximum_R_counters(int *accumulators,int *global_max,int *local_max,int n_locals, int width,int size){
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx<size)//we dont need to use the over-scheduled threads
	{
		int loc_idx = idx%n_locals;
		int x = (idx)%width, y = (idx)/width;//to check location in x and y plane to not exceed the limits of the image in case of incomplete rectangles having extended boundaries
		if((x>10 && x<(width-10)) && (y>10 && y<(((size)/width)-10)))
		{
			int accumulator_val = accumulators[idx], old_max = atomicMax(&local_max[loc_idx],accumulator_val);
			if(old_max<accumulator_val)
				atomicMax(global_max,accumulator_val);
		}
	}
}



__global__ void maxes_vals(int *accumulator, int *maximum, int *R_candidates,int *x_vals,int *y_vals, int *accum_votes, int *radius, int possible_circles, int width,int size,int *id){
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

	if(idx<size)
	{
		int x=idx%width, y=idx/width;
		if(x>10 && x<(width-10) && y>10 && y<((size/width)-10)){
			float accum = accumulator[idx];
			if((accum/(*maximum))>0.65){//avoids processing the zero values
				int c_id = atomicAdd(id,1);
				if(c_id>=possible_circles && c_id<0){
					c_id = 0;
					while(true && c_id<possible_circles){
						if((abs(x_vals[c_id]-x)+abs(y_vals[c_id]-y))<10)
						{
							atomicExch(&accum_votes[c_id],accum);
							atomicExch(&x_vals[c_id],x);
							atomicExch(&y_vals[c_id],y);
							atomicExch(&radius[c_id],R_candidates[idx]/accum);
							break;
						}
						c_id++;
					}
				}
				else{
					accum_votes[c_id] = accum;
					x_vals[c_id] = x;
					y_vals[c_id] = y;
					radius[c_id] = R_candidates[idx]/accum;
				}
			}
		}
	}
}




__global__ void sum_for_mean(int *intensities, int *x_vals,int *y_vals,int *mean_x, int *mean_y, int *N,int width, int centers, int *radious_max, int size){
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

	if(idx<size)
	{
		if(intensities[idx]==255)
		{
			int x = idx%width, y = idx/width;
			for(int i=0;i<centers;i++){
				int r = radious_max[i];//estimation of the radius found
				if(sqrt((float)((x-x_vals[i])*(x-x_vals[i])) + ((y-y_vals[i])*(y-y_vals[i])))<=r+3 && sqrt((float)((x-x_vals[i])*(x-x_vals[i])) + ((y-y_vals[i])*(y-y_vals[i])))>=r-3)//pixels sorrounding the centers found
				{
					atomicAdd(N+i,1);
					atomicAdd(mean_x+i,x);//sum of the x values
					atomicAdd(mean_y+i,y);//sum of the y values
				}
			}
		}
	}
}


__global__ void mean_calculation(int *x_mean,int *y_mean,int *N, int centers){
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx<centers){
		x_mean[idx] /= N[idx];
		y_mean[idx] /= N[idx];
	}
}


__global__ void computing_moments(int *intensities, int *x_vals, int *y_vals,int *x_mean,int *y_mean,float *Mxx,float *Myy,float *Mxy,float *Mxz,float *Myz,float *Mzz,int centers,int *max_R, int width,int size){
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

	if(idx<size)
	{
		if(intensities[idx]==255)
		{
			int x = idx%width, y = idx/width;
			for(int i=0;i<centers;i++){
				volatile int r = max_R[i];
				if(sqrt((float)((x-x_vals[i])*(x-x_vals[i])) + ((y-y_vals[i])*(y-y_vals[i])))<=r+3 && sqrt((float)((x-x_vals[i])*(x-x_vals[i])) + ((y-y_vals[i])*(y-y_vals[i])))>=r-3)//pixels sorrounding the centers found
				{
					float x_diff = (float)(x-x_mean[i]), y_diff = (float)(y-y_mean[i]), sum_diff_sqr = x_diff*x_diff + y_diff*y_diff;
					atomicAdd(Mxy+i,(x_diff*y_diff));
					atomicAdd(Mxx+i,(x_diff*x_diff));
					atomicAdd(Myy+i,(y_diff*y_diff));
					atomicAdd(Mxz+i,(x_diff*sum_diff_sqr));
					atomicAdd(Myz+i,(sum_diff_sqr*y_diff));
					atomicAdd(Mzz+i,(sum_diff_sqr*sum_diff_sqr));
				}
			}
		}
	}
}


__global__ void circle_fitting(int *x_vals, int *y_vals, int *radius,int *x_mean,int *y_mean,float *Mxx,float *Myy,float *Mxy,float *Mxz,float *Myz,float *Mzz,int circles,int *N, int height, int width){
	unsigned int idx = threadIdx.x;

	if(idx<circles)
	{
		//printf("\nNeighbors: %i	",N[idx]);
		float Cov_xy,Var_z,A0,A1,A2,A3,A22,A33,Dy,xnew,ynew,x,y,DET,Xcenter,Ycenter;
		int iter;
		int neighbors = N[idx];
		float loc_Mxx = Mxx[idx]/neighbors,loc_Myy = Myy[idx]/neighbors,loc_Mxy = Mxy[idx]/neighbors,loc_Mxz = Mxz[idx]/neighbors,loc_Myz = Myz[idx]/neighbors,loc_Mzz = Mzz[idx]/neighbors,loc_Mz;

		loc_Mz = loc_Mxx + loc_Myy;
		Cov_xy = loc_Mxx*loc_Myy - loc_Mxy*loc_Mxy;
		Var_z = loc_Mzz - loc_Mz*loc_Mz;
		A3 = 4*loc_Mz;
		A2 = -3*loc_Mz*loc_Mz - loc_Mzz;
		A1 = Var_z*loc_Mz + 4*Cov_xy*loc_Mz - loc_Mxz*loc_Mxz - loc_Myz*loc_Myz;
		A0 = loc_Mxz*(loc_Mxz*loc_Myy - loc_Myz*loc_Mxy) + loc_Myz*(loc_Myz*loc_Mxx - loc_Mxz*loc_Mxy) - Var_z*Cov_xy;
		A22 = A2+A2;
		A33 = A3+A3+A3;

		y = A0;
		x=0;

		for(iter=0;iter<10;iter++)//newton rapson method for estimation, maximum of 10 iterations
		{
			Dy = A1 + x*(A22 + x*A33);//derivative
			xnew = x - y/Dy;
			if((xnew==x) || (!isfinite(xnew))) break;
			ynew = A0 + xnew*(A1 + xnew*(A2 + xnew*A3));
			if(abs(ynew) >= abs(y)) break;
			y = ynew;
			x = xnew;
		}//finish estimation and fitting

		if(iter>0){
			DET = x*x - x*loc_Mz + Cov_xy;
			Xcenter = (loc_Mxz*(loc_Myy-x)-loc_Myz*loc_Mxy)/DET/2;
			Ycenter = (loc_Myz*(loc_Mxx-x)-loc_Mxz*loc_Mxy)/DET/2;
			x_vals[idx] = Xcenter + x_mean[idx];
			y_vals[idx] = Ycenter + y_mean[idx];
			radius[idx] = sqrt((float)(Xcenter*Xcenter + Ycenter*Ycenter + loc_Mz));
		}
	}
}



void bind(int *vals, const int size, const int id){
	switch(id){
		case 1:
			cudaBindTexture(NULL,gradir,vals,sizeof(int)*size);
			break;
		default:
			break;
	}
}


void unbind(int *vals, const int id){
	switch(id){
		case 1:
			cudaUnbindTexture(gradir);
			break;
		default:
			break;
	}
}

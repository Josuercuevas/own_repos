/*
Function prototypes declarations

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
#include "colormark_structures.h"
#include <cstdlib>
#include <cstdio>

void hough_main(int*,int*,int*,int*,int*,int*,int*, line_par&, circle_par&, const int, const int, const int,float*,float*);//the main function of the hough transform that uses the gradient and edges to determine the lines that fit very well in the image

void hough_circles_main(int*,int*,circle_par&,const int,const int,const int,cudaStream_t,float*,float*);//to control the circle fitting
extern __global__ void accumulator_counter(int*,int*,int*, float*,float*,int,int,int,int);//collects the amount of edges pointing to a center
extern __global__ void maximum_R_counters(int *,int *,int *,int , int ,int );//getting the maximum circles counters
extern __global__ void maxes_vals(int*,int*,int*,int*,int*,int*, int*,int,int,int,int*);//determines the centers with the voting

void hough_lines_main(int*,int*,line_par&,const int,const int,const int,cudaStream_t,float*,float*);//to control the line fitting
extern __global__ void hough_lines(int*, float*, float*, int*, const int, const int, const int);//will search for lines in the image and collect their contribution in the counter
extern __global__ void extraction(int*,int*,int,int,int,int*);//will have a flag and a threshold and the array that will contain the values found from the accumulator

void corners_main(int*,int*,int* , int *, int *, int *, int *,const int ,const int ,const int ,cudaStream_t );//to determine corners
extern __global__ void grad_at_edges(int*,int*,int*,int*,int*,int*,int,int);//computes the gradient at edges
extern __global__ void non_max_supp(int *,int ,int );
extern __global__ void maximum_gradients(int *,int *,int *,int , int ,int );

/*TAUBIN*/
extern __global__ void sum_for_mean(int*,int*, int*,int*,int*,int*,int,int,int*,int);//calculate the sum for taubin
extern __global__ void mean_calculation(int*,int*,int*,int);//calculate mean for taubin
extern __global__ void computing_moments(int*,int*,int*,int*,int*,float*,float*,float*,float*,float*,float*,int,int*,int,int);//will compute the moments for taubin
extern __global__ void circle_fitting(int*, int*, int*,int*,int*,float*,float*,float*,float*,float*,float*,int ,int*, int,int);//will estimate the true center and radius of the circle

/*for texture management*/
void bind(int *vals, const int size, const int id);
void unbind(int *vals,const int id);

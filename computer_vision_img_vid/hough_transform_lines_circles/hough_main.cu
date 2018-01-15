/*
Main entry function

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
#include <iostream>
#include "hough_transform_functions.cuh"


using namespace std;



void hough_main(int *d_intensities, int *d_gradients_dir,int *d_inner_grad,int *gmax, int *G_at_edges, int *Gx_strength, int *Gy_strength, line_par &lines, circle_par &circles, const int width, const int height, const int Threads, float *d_sines, float *d_cosines){
	/*resolution of the radius of the circles for hough*/
	cudaStream_t stream1,stream2,stream3;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);


	/*calling functions*/
	hough_circles_main(d_intensities, d_gradients_dir,circles,width,height,Threads,stream1,d_sines,d_cosines);//will report the circles found
	hough_lines_main(d_intensities, d_gradients_dir, lines, width, height, Threads,stream2,d_sines,d_cosines);//send the last stream to leave the other free for circle detection
	corners_main(d_intensities,d_inner_grad,gmax, G_at_edges, Gx_strength, Gy_strength, d_gradients_dir,Threads,width,height,stream3);

	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
	cudaStreamDestroy(stream3);
}

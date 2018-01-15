/*
Prototypes declarations

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


void canny_main(int*,int*,int*, int*, int*,const int,const int, const int);//will handle the canny edge detector operation with paramters defined inside the function
extern __global__ void blur(int*, int,int);//will apply the gassian convolution for blurring or swelling the edges
extern __global__ void gradients(int*, int*, int*, int*, int*, int*, int,int);//will calculate the gradient of the image
extern __global__ void non_max_supr(int*, int*, int*, int, int,int,int);//will perform the non maximum supression in the matrix of gradients

/*
Main cuda function implementations for tone mapping

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
#include <iostream>
#include "colormark_structures.h"
#include <tchar.h>
#include "math_functions.h"



__global__ void Dvalues(int* intensities, unsigned char *RGB, int* Ds,int maximum, int minimum, int width, int height){
	unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;

	if(idx<width*height){
			intensities[idx] = (int)(0.2126*Ds[RGB[idx*3]] + 0.7152*Ds[RGB[idx*3+1]]+0.0722*Ds[RGB[idx*3+2]]);
	}

	/*THIS PART CAN BE IMPLEMENTED IF THE IMAGE IS TOO SMALL

	float temp1 = __logf(((float)intensities[idx]+tao)/((float)minimum+tao)), temp2 = __logf(((float)maximum+tao)/((float)minimum+tao));
	if(temp1>0 && temp2>0)
		intensities[idx] = (int)ceil(255*temp1/temp2);
	else
		intensities[idx] = intensities[idx];

	*/
}



__global__ void estimator(float* taos, float average, int maximum, int minimum){
	volatile extern __shared__ float approx[];
	volatile float kT = (2*__logf(average)-__logf((float)minimum)-logf((float)maximum))/(__logf((float)maximum)-__logf((float)minimum));
	kT = 0.4*pow(2,kT);
	volatile unsigned int idx = threadIdx.x;
	taos[idx]=0.5*(idx+1);
	volatile float kP = (__logf((average+taos[idx])/(minimum+taos[idx]))/__logf((maximum+taos[idx])/(minimum+taos[idx])));
	approx[idx] = abs(kP-kT);
	__syncthreads();

	if(idx==0)
	{
		int id;
		for(int i=1;i<blockDim.x;i++)
			if(approx[i]<approx[i-1])
				id=i;

		taos[0]=taos[id];
		taos[1]=approx[id];
		taos[2]=kT;
		taos[3]=(__logf((average+taos[id])/(minimum+taos[id]))/__logf((maximum+taos[id])/(minimum+taos[id])));
	}
	__syncthreads();
}





__global__ void compression(unsigned char *intensities, float* average, float* roots, int size)
{
	volatile extern __shared__ float partial_mul[];
	volatile float temp=1;
	volatile int idx = threadIdx.x;

	for (size_t i = blockIdx.x*blockDim.x + idx;i < size;i += blockDim.x*gridDim.x) {
		temp *= (roots[intensities[i*3]] + roots[intensities[i*3+1]] + roots[intensities[i*3+2]])/3;//RGB
    }
    partial_mul[idx] = temp;
    __syncthreads();

    for (int activeThreads = blockDim.x>>1; activeThreads; activeThreads >>= 1) {
        if ( idx < activeThreads ) {
            partial_mul[idx] *= partial_mul[idx+activeThreads];
        }
        __syncthreads();
    }

    if ( idx == 0 ) {
        average[blockIdx.x] = partial_mul[0];
    }
	__syncthreads();
}

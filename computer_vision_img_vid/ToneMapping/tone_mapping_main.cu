/*
Tone mapping main entry function

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
#include <time.h>
#include <vector>
#include "tone_map_functions.cuh"
#include <math.h>
#include "colormark_structures.h"

using namespace std;

void tone_map_main(unsigned char *intensities, int* mapped, image_information &image_info, const int Threads, int *d_intensities){

	/**********************************************GPU Implementation*****************************************************/
	float roots[256];//calculate the roots on the CPU for the 256 values
	int D[256];//calculate the compressed values and rounds them to the nearest intenger
	for(int i=0;i<256;i++)
	{
		roots[i]=pow((float)i+1,(float)1/image_info.pixels);
	}

	int blocks = (image_info.pixels+Threads-1)/Threads, *d_Ds;
	size_t size_bytes = sizeof(int)*image_info.pixels, sharedM = sizeof(float)*Threads, size_partials = sizeof(float)*blocks;

	/*space to be allocated in the GPU*/
	float *partials, *d_partials, *d_roots;
	unsigned char *d_RGB_intensities;

	/*calculation of the theoretical K for the tone mapping approach*/
	const int precision=256;
	float *d_taos;
	float taos[precision];

	partials = (float*)malloc(size_partials);
	cudaMalloc((void**)&d_RGB_intensities,sizeof(unsigned char)*image_info.pixels*3);//since is RGB
	cudaMalloc((void**)&d_partials,size_partials);
	cudaMalloc((void**)&d_taos,sizeof(float)*precision);
	cudaMalloc((void**)&d_roots,sizeof(float)*256);
	cudaMalloc((void**)&d_Ds,sizeof(int)*256);

	/*copying the required information*/
	cudaMemcpyAsync(d_roots,roots,sizeof(float)*256,cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_RGB_intensities,intensities,sizeof(unsigned char)*image_info.pixels*3,cudaMemcpyHostToDevice);//copied from the host to device

	/*calling the kernel for computation of average intensity*/
	compression<<<blocks, Threads, sharedM>>>(d_RGB_intensities, d_partials, d_roots, image_info.pixels);



	cudaMemcpy(partials,d_partials,size_partials,cudaMemcpyDeviceToHost);

	/*partial products for the array containing the blocks' products*/
	for(int i=1;i<blocks;i++)//calculating the real mean
	{
		partials[0]*=partials[i];
	}

	/*calling kernel for computation od tao*/
	estimator<<<1,precision,sizeof(float)*precision>>>(d_taos, partials[0],image_info.max_intensity,image_info.min_intensity);//does a estimation of the taos
	cudaMemcpy(taos,d_taos,sizeof(float)*precision,cudaMemcpyDeviceToHost);

	int R_tao =(int)ceil(taos[0]);//tao found rounded up

	//calculation of Ds
	for(int i=0;i<256;i++)
	{
		float tem =log(((float)i+R_tao)/((float)image_info.min_intensity+R_tao))/log(((float)image_info.max_intensity+R_tao)/((float)image_info.min_intensity+R_tao));//pow((float)i+1,(float)1/image_info[2]);
		if(tem>=0 && tem<256)
			D[i]=(int)ceil(255*tem);
		else
			D[i] = i;
	}
	cudaMemcpy(d_Ds,D,sizeof(int)*Threads,cudaMemcpyHostToDevice);

	/*calling kernel for adjusting the intensity levels*/
	Dvalues<<<(image_info.pixels+precision-1)/precision,precision>>>(d_intensities, d_RGB_intensities, d_Ds,image_info.max_intensity,image_info.min_intensity,image_info.width,image_info.height);
	cudaMemcpy(mapped,d_intensities,size_bytes,cudaMemcpyDeviceToHost);//to stored the mapped intensities, FOR IMPLEMENTATION ON INSPECTION THIS PART IS NOT NEEDED SINCE U ARE NOT GOING TO VISUALIZE ANYTHING
	/********************************************** finished GPU Implementation******************************************************/


	cudaFree(d_taos);
	cudaFree(d_partials);
	cudaFree(d_roots);
	cudaFree(d_Ds);
	cudaFree(d_RGB_intensities);
	image_info.average_intensity = partials[0];//records the average intensity found in this method
	free(partials);
}

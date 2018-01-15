/*
Tone Mapping prototype functions definitions

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
#include <tchar.h>


#ifndef _tone_map_functions
#define _tone_map_functions

/*fucntion for tone mapping*/
extern void tone_map_main(unsigned char*, int*, image_information&,const int, int*);//process that does the tone mapping
extern __global__ void compression(unsigned char*,float*, float*, int);//calculate the mean according to the paper
extern __global__ void estimator(float*,  float, int, int);//will determine the best value for the equality solution
extern __global__ void Dvalues(int*,unsigned char*, int*,int, int, int,int);//will change the intensities according to the Tao found earlier


#endif

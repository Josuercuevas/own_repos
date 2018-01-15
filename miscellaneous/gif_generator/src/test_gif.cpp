/*
 * test_gif.cpp
 *
 *  Created on: Oct 6, 2015
 *      Author: josue

 	 Copyright (C) <2018>  <Josue R. Cuevas>

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

#include <CImg.h>
#include "../inc/gif.h"
#include "../inc/resizer.h"

using namespace cimg_library;

int main(int argc, char *argv[]){
	CImg<unsigned char> first_image("PATH_TO_FIRST_FRAME");
	int h=first_image._height,w=first_image._width;
	int new_h=0.2*h, new_w=0.2*w;
	GifWriter *container;
	container = (GifWriter*)malloc(sizeof(GifWriter));
	if(!GifBegin(container, "PATH_TO_DST_GIF_FILE", new_w, new_h, 20)){
		printf("Error while creating the writer for the gif file\n");
		return -1;
	}

	uint8_t *packed_image=NULL;
	uint8_t *packed_image_resized=NULL;
	packed_image=(uint8_t*)malloc(sizeof(uint8_t)*h*w*4);/*RGBA*/
	packed_image_resized=(uint8_t*)malloc(sizeof(uint8_t)*new_h*new_w*4);/*RGBA*/

	for(int i=0;i<4;i++){
		char pic_name[128]={0};
		sprintf(pic_name, "%s-%d.gif", "FOLDER_PATH_WHERE_FRAMES_ARE",
				i);
		CImg<unsigned char> images(pic_name);
		for(int i=0;i<h;i++){
			for(int j=0;j<w;j++){
				packed_image[4*(i*w + j)]=images._data[i*w + j];/*R*/
				packed_image[4*(i*w + j) + 1]=images._data[i*w + j + w*h];/*G*/
				packed_image[4*(i*w + j) + 2]=images._data[i*w + j + 2*w*h];/*B*/
				packed_image[4*(i*w + j) + 3]=0;/*A -> I dont care*/
			}
		}

		resize(packed_image, packed_image_resized, w, h, new_w, new_h, "box");


		if(!GifWriteFrame(container, packed_image_resized, new_w, new_h, 20)){
			printf("Error while writting the frame to the gif file\n");
			return -1;
		}
	}

	if(!GifEnd(container)){
		printf("There is nothing to free ... exiting\n");
	}

	free(packed_image);

	printf("Press any key...\n");
	getchar();

	return 0;
}

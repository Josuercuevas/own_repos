/*
Main structures used in this project

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

#include <cstdlib>
#include <vector>

using namespace std;

#ifndef __colormark_structuras
#define __colormark_structuras

struct image_information{//contains the information of the image
	int height;//height of the image in pixels
	int width;//width of the image in pixels
	int pixels;//size of the image in pixels (height*width)
	int max_intensity;//maximum intensity of the image (defined by user in case we dont want to estimate it from the image)
	int min_intensity;//minimum intensity of the image (defined by user in case we dont want to estimate it from the image)
	int average_intensity;//average intensity of the image, defined after gray tone mapping
};

struct intensities_proccessed{//contains the intensities after processing
	unsigned char *originals;//all the intensity values of the image in the RGB spectrum retrieved from the camera
	int *tone_mapped;//intensities in the gray level after performing the gray tone mapping (global)
	int *cannied;//determine the edges of the image for later processes
	int *on_device;//this values will be on the device for processes throughout the entire execution of the program (a single iteration), therefore we dont need to the data from CPU to GPU multiple times
	int *gradient_dir_gpu;//total gradient strength of every pixel in the image
	int *gradient_str_gpu_Gx;//x gradient strength of every pixel in the image
	int *gradient_str_gpu_Gy;//y gradient strength of every pixel in the image
	int* G_at_edges;//inner negative gradient correlation of the edge points
	int *Gmax;//maximum inner negative inner corralation of the edge points
	int *d_inner_grad;//direction of the inner negative gradient correlation at the edges
};

struct line_par{//in reality we round them to integers because we are dealing with images
	vector<int> rho;//distance from the origin
	vector<int> theta;//angle of the line perpendicular to the line
	vector<int> votes;//contains the votes of each line
};

struct circle_par{//in reality we round them to integers because we are dealing with images
	vector<int> votes;//radious of the circle
	vector<int> radious;//radious of the circle
	vector<int> Xo;//origin on x
	vector<int> Yo;//origin on y
};

struct rect_par{
	vector<int> corners[2][4];//will have 4 corners per rectangle found with x-y coordinate
	vector<int> center[2];//only 1 center per rectangle found with x-y coordinate
	vector<int> Width;//width of the rectangle estimated by the distance between corners (it is usually the largest side)
	vector<int> Heigth;//height of the rectangle estimated by the distance between corners (it is usually the smallest side)
	vector<int> Orientation;//orientation of the rectangle estimated by the it's width and height
};

#endif

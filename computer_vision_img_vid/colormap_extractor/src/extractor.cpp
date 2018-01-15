/*
 * extractor.c
 *
 *  Created on: Aug 20, 2015
 *      Author: josue

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


#include <stdlib.h>
#include <stdio.h>
#include <CImg.h>
#include <string.h>
#include "../inc/extractor.h"

using namespace cimg_library;

typedef struct
{
	int Min[3];
	int Max[3];
	double Average[3];
	long NumPixels;
	long Volume;
} bbox;


static long BoxVolume(bbox Box);
static void MedianSplit(bbox *NewBox, bbox *SplitBox,
	const unsigned char *RgbImage, long NumPixels);


int Rgb2Ind(unsigned char *Dest, unsigned char *Palette, int NumColors,
	const unsigned char *RgbImage, long NumPixels){
	float Merit, MaxMerit;
	const long NumEl = 3*NumPixels;
	long i, Dist, MinDist, Diff;
	int k, BestBox = 0, Channel, NumBoxes = 1;
	bbox Box[256];

	if(!Dest || !Palette || NumColors > 256 || !RgbImage || NumPixels <= 0)
		return 0;

	/* Determine the smallest box containing all pixels */
	Box[0].Min[0] = Box[0].Min[1] = Box[0].Min[2] = 255;
	Box[0].Max[0] = Box[0].Max[1] = Box[0].Max[2] = 0;

	for(i = 0; i < NumEl; i += 3)
		for(Channel = 0; Channel < 3; Channel++)
		{
			if(Box[0].Min[Channel] > RgbImage[i + Channel])
				Box[0].Min[Channel] = RgbImage[i + Channel];
			if(Box[0].Max[Channel] < RgbImage[i + Channel])
				Box[0].Max[Channel] = RgbImage[i + Channel];
		}

	Box[0].NumPixels = NumPixels;
	Box[0].Volume = BoxVolume(Box[0]);

	while(NumBoxes < NumColors)
	{
	   MaxMerit = 0;

	   /* Select a box to split */
	   if(NumBoxes % 4 > 0)        /* Split according to NumPixels */
	   {
		   for(k = 0; k < NumBoxes; k++)
			   if(Box[k].Volume > 2
				   && (Merit = (float)Box[k].NumPixels) > MaxMerit)
			   {
				   MaxMerit = Merit;
				   BestBox = k;
			   }
	   }
	   else                        /* Split according to NumPixels*Volume */
		   for(k = 0; k < NumBoxes; k++)
			   if(Box[k].Volume > 2
				   && (Merit = ((float)Box[k].NumPixels)
				   * ((float)Box[k].Volume)) > MaxMerit)
			   {
				   MaxMerit = Merit;
				   BestBox = k;
			   }

	   /* Split the box */
	   MedianSplit(&Box[NumBoxes], &Box[BestBox], RgbImage, NumPixels);
	   NumBoxes++;
   }

   for(k = 0; k < NumBoxes; k++)
   {
	   Box[k].Average[0] = Box[k].Average[1] = Box[k].Average[2] = 0;
	   Box[k].NumPixels = 0;
   }

   /* Compute box averages */
   for(i = 0; i < NumEl; i += 3)
   {
	   for(k = 0; k < NumBoxes; k++)
		   if(Box[k].Min[0] <= RgbImage[i + 0]
			   && RgbImage[i + 0] <= Box[k].Max[0]
			   && Box[k].Min[1] <= RgbImage[i + 1]
			   && RgbImage[i + 1] <= Box[k].Max[1]
			   && Box[k].Min[2] <= RgbImage[i + 2]
			   && RgbImage[i + 2] <= Box[k].Max[2])
			   break;

	   if(k == NumBoxes)
	   {
		   fprintf(stderr, "Color (%d,%d,%d) unassigned\n",
				   RgbImage[i + 0], RgbImage[i + 1], RgbImage[i + 2]);
		   k = 0;
	   }
	   else
	   {
		   /* Accumate the average color for each box */
		   Box[k].Average[0] += RgbImage[i + 0];
		   Box[k].Average[1] += RgbImage[i + 1];
		   Box[k].Average[2] += RgbImage[i + 2];
	   }

	   Box[k].NumPixels++;
   }

   /* Fill Palette with the box averages */
   for(k = 0; k < NumBoxes; k++)
	   if(Box[k].NumPixels > 0)
		   for(Channel = 0; Channel < 3; Channel++)
		   {
			   Box[k].Average[Channel] /= Box[k].NumPixels;

			   if(Box[k].Average[Channel] < 0.5)
				   Palette[3*k + Channel] = 0;
			   else if(Box[k].Average[Channel] >= 254.5)
				   Palette[3*k + Channel] = 255;
			   else
				   Palette[3*k + Channel] =
					   (unsigned char)(Box[k].Average[Channel] + 0.5);
		   }
	   else
		   for(Channel = 0; Channel < 3; Channel++)
			   Palette[3*k + Channel] = 0;

   /* Assign palette indices to quantized pixels */
   for(i = 0; i < NumEl; i += 3)
   {
	   /* Find the closest palette color */
	   for(k = 0, MinDist = 1000000; k < NumBoxes; k++)
	   {
		   Diff = ((long)RgbImage[i + 0]) - Palette[3*k + 0];
		   Dist = Diff * Diff;
		   Diff = ((long)RgbImage[i + 1]) - Palette[3*k + 1];
		   Dist += Diff * Diff;
		   Diff = ((long)RgbImage[i + 2]) - Palette[3*k + 2];
		   Dist += Diff * Diff;

		   if(MinDist > Dist)
		  {
			   MinDist = Dist;
			   BestBox = k;
		   }
	   }

	   *Dest = BestBox;
	   Dest++;
   }

   return 1;
}


static long BoxVolume(bbox Box){
   return (Box.Max[0] - Box.Min[0] + 1)
	   * (Box.Max[1] - Box.Min[1] + 1)
	   * (Box.Max[2] - Box.Min[2] + 1);
}


static void MedianSplit(bbox *NewBox, bbox *SplitBox,
   const unsigned char *RgbImage, long NumPixels){
   bbox Box = *SplitBox;
   const long NumEl = 3*NumPixels;
   long i, Accum, Hist[256];
   int Length, MaxLength, MaxDim;

   /* Determine the longest box dimension */
   MaxLength = MaxDim = 0;

   for(i = 0; i < 3; i++)
	   if((Length = Box.Max[i] - Box.Min[i] + 1) > MaxLength)
	   {
		   MaxLength = Length;
		   MaxDim = i;
	   }

   /* Build a histogram over MaxDim for pixels within Box */
   memset(Hist, 0, sizeof(long)*256);

   for(i = 0; i < NumEl; i += 3)
	   if(Box.Min[0] <= RgbImage[i + 0] && RgbImage[i + 0] <= Box.Max[0]
		   && Box.Min[1] <= RgbImage[i + 1] && RgbImage[i + 1] <= Box.Max[1]
		   && Box.Min[2] <= RgbImage[i + 2] && RgbImage[i + 2] <= Box.Max[2])
		   Hist[RgbImage[i + MaxDim]]++;

   Accum = Hist[i = Box.Min[MaxDim]];

   /* Set i equal to the median */
   while(2*Accum < Box.NumPixels && i < 254)
	   Accum += Hist[++i];

   /* Adjust i so that the median is included with the larger partition */
   if(i > Box.Min[MaxDim]
	   && ((i - Box.Min[MaxDim]) < (Box.Max[MaxDim] - i - 1)))
	   Accum -= Hist[i--];

   /* Adjust i to ensure that boxes are not empty */
   for(; i >= Box.Max[MaxDim]; i--)
	   Accum -= Hist[i];

   /* Split the boxes */
   *NewBox = Box;
   NewBox->Max[MaxDim] = i;
   NewBox->NumPixels = Accum;
   NewBox->Volume = BoxVolume(*NewBox);

   SplitBox->Min[MaxDim] = i + 1;
   SplitBox->NumPixels = Box.NumPixels - Accum;
   SplitBox->Volume = BoxVolume(*SplitBox);
}

#define FLT_EPSILON (1E-12)
static void RGBtoHSV_fly(uint8_t r, uint8_t g, uint8_t b, uint8_t *Hnew, uint8_t *Snew, uint8_t *Vnew){
	//HSV
	float hscale = ( 180.f/360.f );
	float h, s, v;
	float vmin, diff;
	v=vmin=r;
	v=((v<g)?((g<b)?b:g):(v<b)?b:v);
	diff=v-vmin;
	s=diff/(v+FLT_EPSILON);
	diff=(60.f/(diff+FLT_EPSILON));
	h=(v==r?((float)(g - b)*diff):(v==g?((float)(b -
		r)*diff + 120.f):((float)(r - g)*diff + 240.f)));
	h += (h<0?360.f:0.f);
	*Hnew=(uint8_t)(h*hscale);
	*Snew=(uint8_t)(s*255.f);
	*Vnew=(uint8_t)v;
	return;
}


int Clamp(int i){
  if (i < 0) return 0;
  if (i > 255) return 255;
  return i;
}
static void HSVtoRGB_fly(float h, float S, float V, uint8_t *r, uint8_t *g, uint8_t *b){
	float H = h;
	while (H < 0) { H += 360; };
	while (H >= 360) { H -= 360; };
	float R, G, B;
	if (V <= 0){
		R = G = B = 0;
	}else if (S <= 0){
		R = G = B = V;
	}else{
		float hf = H / 60.0;
		int i = (int)floor(hf);
		float f = hf - i;
		float pv = V * (1 - S);
		float qv = V * (1 - S * f);
		float tv = V * (1 - S * (1 - f));
		switch(i){

		  // Red is the dominant color

		  case 0:
			R = V;
			G = tv;
			B = pv;
			break;

		  // Green is the dominant color

		  case 1:
			R = qv;
			G = V;
			B = pv;
			break;
		  case 2:
			R = pv;
			G = V;
			B = tv;
			break;

		  // Blue is the dominant color

		  case 3:
			R = pv;
			G = qv;
			B = V;
			break;
		  case 4:
			R = tv;
			G = pv;
			B = V;
			break;

		  // Red is the dominant color

		  case 5:
			R = V;
			G = pv;
			B = qv;
			break;

		  // Just in case we overshoot on our math by a little, we put these here. Since its a switch it won't slow us down at all to put these here.

		  case 6:
			R = V;
			G = tv;
			B = pv;
			break;
		  case -1:
			R = V;
			G = pv;
			B = qv;
			break;

		  // The color is not defined, we should throw an error.

		  default:
			R = G = B = V; // Just pretend its black/white
			break;
		}
	}

	*r = (uint8_t)Clamp((int)(R * 255.0));
	*g = (uint8_t)Clamp((int)(G * 255.0));
	*b = (uint8_t)Clamp((int)(B * 255.0));
}
















int main(int argc, char *argv[]){
	CImg<unsigned char> image_to_map;
	image_to_map.load(argv[1]);//loading image
	int pic_size=(image_to_map._height*image_to_map._width);
	colormap color_table;
	color_table.intensities=256;
	color_table.temp_map = (int*)malloc(sizeof(int)*color_table.intensities);
	color_table.Cmap = (uint8_t*)calloc(color_table.intensities*3, sizeof(uint8_t));
	uint8_t h,s,v;
	uint8_t r,g,b;

	/*initial flag*/
	for(int i=0;i<color_table.intensities;i++){
		color_table.temp_map[i]=-1;
	}

	/***********debuging************/
	CImg<unsigned char> gray(image_to_map._width, image_to_map._height, 1, 1);/*GRAY*/
	/***********************************/


	float intensity;
	uint8_t pos=0;
	for(int i=0;i<pic_size;i++){
		intensity = fmin((0.2989*(float)image_to_map._data[i] + 0.5870*(float)image_to_map._data[i + pic_size] +
				0.1140*(float)image_to_map._data[i + 2*pic_size]), 255.0);

		if(color_table.intensities>256){
			if((intensity-(int)(intensity))<0.5){
				pos = ((int)(intensity))*2;
			}else{
				pos = ((int)(intensity))*2 + 1;
			}
		}else{
			pos = ((int)(intensity));
		}


		//convert to hsv values and store them
		RGBtoHSV_fly(image_to_map._data[i], image_to_map._data[i + pic_size],
				image_to_map._data[i + 2*pic_size], &h, &s, &v);

		if(color_table.temp_map[pos]==-1){
			color_table.temp_map[pos]=1;/*removing unused flag*/
			/*storing the corresponding RGB values with the gray intensity*/
			color_table.Cmap[pos*3]=h;/*H*/
			color_table.Cmap[pos*3 + 1]=s;/*S*/
			color_table.Cmap[pos*3 + 2]=v;/*V*/

			/***********debuging************/
			printf("%f (%f): [%d, %d, %d]\n", intensity, (intensity-(int)(intensity)), color_table.Cmap[pos*3], color_table.Cmap[pos*3+1], color_table.Cmap[pos*3+2]);
			/***********************************/
		}

		/***********debuging************/
		gray._data[i]=(unsigned char)intensity;
		/***********************************/
	}


	int roll, unroll;
	printf("\n\n\n");
	for(int i=0;i<color_table.intensities;i++){
		printf("<%d -> %i [%d, %d, %d]>\n", i, color_table.temp_map[i],
				color_table.Cmap[i*3], color_table.Cmap[i*3+1], color_table.Cmap[i*3+2]);
		if(color_table.temp_map[i]==-1){
			roll=i;
		}
		while(color_table.temp_map[roll]==-1){
			roll++;
		}
		unroll=roll;
		while(unroll>=i){
			color_table.temp_map[unroll] = color_table.temp_map[roll];
			color_table.Cmap[unroll*3] = color_table.Cmap[roll*3];
			color_table.Cmap[unroll*3+1] = color_table.Cmap[roll*3+1];
			color_table.Cmap[unroll*3+2] = color_table.Cmap[roll*3+2];
			unroll--;
		}
	}
	printf("\n\n\n");


	/***********debuging************/
	CImg<unsigned char> constructed(image_to_map._width, image_to_map._height, 1, 3);/*constructed RGB*/
	for(int i=0;i<pic_size;i++){
		intensity = fmin((0.2989*(float)image_to_map._data[i] + 0.5870*(float)image_to_map._data[i + pic_size] +
				0.1140*(float)image_to_map._data[i + 2*pic_size]), 255.0);

		if(color_table.intensities>256){
			if((intensity-(int)(intensity))<0.5){
				pos = ((int)(intensity))*2;
			}else{
				pos = ((int)(intensity))*2 + 1;
			}
		}else{
			pos = ((int)(intensity));
		}

		constructed._data[i] = color_table.Cmap[pos*3];//H
		constructed._data[i + pic_size] = color_table.Cmap[pos*3 + 1];//S
		constructed._data[i + pic_size*2] = color_table.Cmap[pos*3 + 2];//V

		HSVtoRGB_fly((float)constructed._data[i], (float)constructed._data[i + pic_size],
				(float)constructed._data[i + 2*pic_size], &r, &g, &b);

		constructed._data[i] = r;//R
		constructed._data[i + pic_size] = g;//G
		constructed._data[i + pic_size*2] = b;//B


	}

//	image_to_map.display("Original Image");
//	gray.display("Gray Image");
//	constructed.display("Reconstructed Image");
	CImgDisplay window;
	window.display((image_to_map, constructed));
	window.wait(5000);
	constructed.save("test.jpg");
	/***********************************/

	FILE *table=fopen("pseudocolor_table", "w+b");
	fwrite((void*)color_table.Cmap, sizeof(uint8_t), color_table.intensities*3, table);
	fclose(table);

	free(color_table.temp_map);
	free(color_table.Cmap);
	return 0;
}

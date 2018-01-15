/*
 * BB_estimation.cpp
 *
 *  Created on: Oct 28, 2014
 *      Author: josue
 *
 *  Routines and functions in charge of estimating the bounding box of the region with large activity
 *  this one is just an estimation, we have to make sure this frames have some sort of activity or most approaches
 *  will fail in detecting a bounding box that encloses the object under study
 */

#include "BB_estimation.h"



//==========================================================================================================//
//Bounding Box extractor

uint32_t estimate_BB(SBufferInfo *pDstInfo, uint64_t *image){
	int i,j;
	uint64_t top_max = 0;
	uint64_t *temp_row;
	int im_h = pDstInfo->UsrData.sSystemBuffer.iHeight;
	int im_w = pDstInfo->UsrData.sSystemBuffer.iWidth;
	pDstInfo->BBox.nboxes = 0;
	float counter = 0;


	temp_row = (uint64_t*)malloc(sizeof(uint64_t)*im_h*im_w);//for speed

	if(_DEBUG_BBOX){
		printf("\t\t---------------------------------------------------------------------\n");
		printf("\t\tPicture size: %ix%i\n", im_w, im_h);
	}


	for(i=0; i<im_h; i++){
		memcpy(temp_row+(i*im_w), image+(i*im_w), sizeof(uint64_t)*im_w);
		for(j=0; j<im_w; j++){
			if(top_max < temp_row[i*im_w+j])
				top_max = temp_row[i*im_w+j];
		}
	}

	if(_DEBUG_BBOX)
		printf("\t\tMax voting value found : %i\n", top_max);

	for(i=0; i<im_h; i++){
		for(j=0; j<im_w; j++){
			if(temp_row[i*im_w+j] < (float)top_max*0.05)
				temp_row[i*im_w+j] = 0;//false alarm
			else{
				temp_row[i*im_w+j] = 255;//true foreground
				counter+=1;
			}
		}
	}



	//call the labeling part to determine how many bounding boxes
	if(Connected_Comp_Label(temp_row, im_w, im_h, pDstInfo->BBox.BBx, pDstInfo->BBox.BBy, pDstInfo->BBox.BBw, pDstInfo->BBox.BBh,
			&(pDstInfo->BBox.nboxes))){
		printf("Error detecting blobs...\n");
		return -1;
	}else{
		if(_DEBUG_BBOX)
			printf("\t\t---------------------------------------------------------------------\n");
	}

	counter /= im_h*im_w;
	pDstInfo->Saliency_value = fmin(counter*5, 1.0);

	free(temp_row);
	return 0;
}

//==========================================================================================================//






uint32_t Connected_Comp_Label(uint64_t *img_data, int width, int height, int *BB_x, int *BB_y, int *BB_w, int *BB_h, int *n_boxes)
{
   int i, j, k, z, max, used_num[256]={0};
   struct min_max boxes[256];
   int mx, pix[9];

   if(_DEBUG_BBOX)
	   printf("\t\tEstimating connected components for blob detection...!!\n");

   *n_boxes = 0;//no boxes at the beginning

   // prepare to binary image format
	for(i=0;i<height;i++)
	{
		for(j=0;j<width;j++)
		{
			if (img_data[i*width + j] == 255)
				img_data[i*width + j] = 0;
			else
				img_data[i*width + j] = 1;
		}
		if(i<256){
			boxes[i].max_x = -1;
			boxes[i].max_y = -1;
			boxes[i].min_x = 999999;
			boxes[i].min_y = 999999;
		}
	}


   z = 0;
   max = 2;
   for(i=1;i<height-1;i++)
   {
      for(j=1;j<width-1;j++)
      {
          pix[0] = img_data[i*width + j];
          pix[1] = img_data[(i-1)*width + j];
          pix[2] = img_data[(i-1)*width + (j+1)];
           pix[3] = img_data[(i*width)+(j+1)];
          pix[4] = img_data[((i+1)*width)+(j+1)];
          pix[5] = img_data[((i+1)*width)+(j)];
          pix[6] = img_data[((i+1)*width)+(j-1)];
          pix[7] = img_data[(i*width)+(j-1)];
          pix[8] = img_data[((i-1)*width)+(j-1)];
          if (pix[0] == 1)
          {
             // find maximum of pixeles.
             mx = 0;
             for(k=1;k<9;k++)
             {
               if (pix[k] > mx)
                  mx = pix[k];
             }
             if (mx == 1) // if neighbours are equal to ONE, set
                          // it to max and increase max by ONE
             {
                  img_data[i*width + j] = max;
                  max++;
             }
             else
             {
                pix[0] = 0;
                QuickSort(pix,0, 8);
                k=8;
                while((k>0) && (pix[k] != 1))
                   k--;
                if (k < 8)
                   img_data[i*width + j] = pix[k+1];
                else
                   img_data[i*width + j] = pix[k];
             }
         }
     }
   }

   k = 0;
   for(i=0;i<height;i++)
   {
      for(j=0;j<width;j++)
      {
        mx = img_data[i*width + j];
        if (mx > 0)
        {
          if (not_in_array(mx, used_num, k))
          {
             used_num[k] = mx;
             k++;
          }
        }
      }
   }

   for(i=0;i<height;i++)
   {
      for(j=0;j<width;j++)
      {
         if (img_data[i*width + j] == 10){
            img_data[i*width + j] = 0;
            k=0;
            if(j > boxes[k].max_x)
			 boxes[k].max_x = j;
			if(i > boxes[k].max_y)
			 boxes[k].max_y = i;
			if(j < boxes[k].min_x)
			 boxes[k].min_x = j;
			if(i < boxes[k].min_y)
			 boxes[k].min_y = i;
         }
         else if (img_data[i*width + j] == 9){
            img_data[i*width + j] = 25;
            k=1;
			if(j > boxes[k].max_x)
			 boxes[k].max_x = j;
			if(i > boxes[k].max_y)
			 boxes[k].max_y = i;
			if(j < boxes[k].min_x)
			 boxes[k].min_x = j;
			if(i < boxes[k].min_y)
			 boxes[k].min_y = i;
         }
         else if (img_data[i*width + j] == 8){
            img_data[i*width + j] = 50;
            k=2;
			if(j > boxes[k].max_x)
			 boxes[k].max_x = j;
			if(i > boxes[k].max_y)
			 boxes[k].max_y = i;
			if(j < boxes[k].min_x)
			 boxes[k].min_x = j;
			if(i < boxes[k].min_y)
			 boxes[k].min_y = i;
         }
         else if (img_data[i*width + j] == 7){
            img_data[i*width + j] = 75;
            k=3;
			if(j > boxes[k].max_x)
			 boxes[k].max_x = j;
			if(i > boxes[k].max_y)
			 boxes[k].max_y = i;
			if(j < boxes[k].min_x)
			 boxes[k].min_x = j;
			if(i < boxes[k].min_y)
			 boxes[k].min_y = i;
         }
         else if (img_data[i*width + j] == 6){
            img_data[i*width + j] = 100;
            k=4;
			if(j > boxes[k].max_x)
			 boxes[k].max_x = j;
			if(i > boxes[k].max_y)
			 boxes[k].max_y = i;
			if(j < boxes[k].min_x)
			 boxes[k].min_x = j;
			if(i < boxes[k].min_y)
			 boxes[k].min_y = i;
         }
         else if (img_data[i*width + j] == 5){
            img_data[i*width + j] = 125;
            k=5;
			if(j > boxes[k].max_x)
			 boxes[k].max_x = j;
			if(i > boxes[k].max_y)
			 boxes[k].max_y = i;
			if(j < boxes[k].min_x)
			 boxes[k].min_x = j;
			if(i < boxes[k].min_y)
			 boxes[k].min_y = i;
         }
         else if (img_data[i*width + j] == 4){
            img_data[i*width + j] = 100;
            k=6;
			if(j > boxes[k].max_x)
			 boxes[k].max_x = j;
			if(i > boxes[k].max_y)
			 boxes[k].max_y = i;
			if(j < boxes[k].min_x)
			 boxes[k].min_x = j;
			if(i < boxes[k].min_y)
			 boxes[k].min_y = i;
         }
         else if (img_data[i*width + j] == 3){
            img_data[i*width + j] = 150;
            k=7;
			if(j > boxes[k].max_x)
			 boxes[k].max_x = j;
			if(i > boxes[k].max_y)
			 boxes[k].max_y = i;
			if(j < boxes[k].min_x)
			 boxes[k].min_x = j;
			if(i < boxes[k].min_y)
			 boxes[k].min_y = i;
         }
         else if (img_data[i*width + j] == 2){
            img_data[i*width + j] = 200;
            k=8;
			if(j > boxes[k].max_x)
			 boxes[k].max_x = j;
			if(i > boxes[k].max_y)
			 boxes[k].max_y = i;
			if(j < boxes[k].min_x)
			 boxes[k].min_x = j;
			if(i < boxes[k].min_y)
			 boxes[k].min_y = i;
         }
         else if (img_data[i*width + j] == 1){
            img_data[i*width + j] = 225;
            k=9;
			if(j > boxes[k].max_x)
			 boxes[k].max_x = j;
			if(i > boxes[k].max_y)
			 boxes[k].max_y = i;
			if(j < boxes[k].min_x)
			 boxes[k].min_x = j;
			if(i < boxes[k].min_y)
			 boxes[k].min_y = i;
         }
         else{
            img_data[i*width + j] = 255;
            k=10;
			if(j > boxes[k].max_x)
			 boxes[k].max_x = j;
			if(i > boxes[k].max_y)
			 boxes[k].max_y = i;
			if(j < boxes[k].min_x)
			 boxes[k].min_x = j;
			if(i < boxes[k].min_y)
			 boxes[k].min_y = i;
         }
      }
   }

   //fixing the borders of the image
   for(k=0; k<11; k++){
	   if(boxes[k].max_x > width-15)
		   boxes[k].max_x = width-15;
	   if(boxes[k].max_y > height-15)
		   boxes[k].max_y = height-15;
	   if(boxes[k].min_x < 15)
		   boxes[k].min_x = 15;
	   if(boxes[k].min_y < 15)
		   boxes[k].min_y = 15;
   }


   //filtering boxes according to size
   for(k=0; k<MAX_BOXES; k++){
	   boxes[k].height = boxes[k].max_y - boxes[k].min_y;
	   boxes[k].width = boxes[k].max_x - boxes[k].min_x;
	   boxes[k].size = boxes[k].width * boxes[k].height;
	   if(boxes[k].size > 0.10*(width*height) && boxes[k].size < 0.65*(width*height) && ((float)boxes[k].height/boxes[k].width)>0.2 && ((float)boxes[k].height/boxes[k].width)<3.0){
		   BB_x[*n_boxes] = boxes[k].min_x;
		   BB_y[*n_boxes] = boxes[k].min_y;
		   BB_h[*n_boxes] = boxes[k].height;
		   BB_w[*n_boxes] = boxes[k].width;
		   (*n_boxes)++;
		   if(_DEBUG_BBOX)
			   printf("\t\t---REPORT FROM BB ESTIMATION---==> X: %i, Y: %i, W: %i, H: %i, Z: %i\n", boxes[k].min_x, boxes[k].min_y, boxes[k].width, boxes[k].height, boxes[k].size);
	   }

   }

   if(_DEBUG_BBOX){
	   printf("\t\t%i bounding boxes found..!!\n", *n_boxes);
	   printf("\t\tdone estimating components..!!\n");
   }

   return 0;
}

uint32_t not_in_array(int x, int diz[], int n)
{
	int i=0;
	while( (i<n) && (diz[i] != x))
		i++;
	if (diz[i] != x)
		return 1;
	return 0;
}

void QuickSort(int a[], int L, int R)
{

	int j, i, x, k;

	i = L;
	j = R;
	x = a[(L+R) / 2];
	do
	{
		while (a[i] < x)
			i++;
		while (a[j] > x)
			j--;
		if (i <= j)
		{
			k = a[i];
			a[i] = a[j];
			a[j] = k;
			i++;
			j--;
		}
	}
	while (i < j);
	if (L < j)
		QuickSort(a,L,j);
	if (i < R)
		QuickSort(a,i,R);
}

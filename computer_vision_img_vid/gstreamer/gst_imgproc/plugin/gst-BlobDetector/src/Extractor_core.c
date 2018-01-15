/*
 * Extractor_core.c
 *
 *  Created on: Jan 6, 2015
 *      Author: josue
 *
 *      Function definitions for the blob extraction process of the API
 */

#include <glib.h>
#include <gst/gstinfo.h>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/video/gstvideofilter.h>
#include <setjmp.h>
#include <stdlib.h>
#include "includes/Extractor.h"

GST_DEBUG_CATEGORY_STATIC (gst_skindetector_debug_category3);
#define GST_CAT_DEFAULT gst_skindetector_debug_category3


void debug_ini_extraction(){
	GST_DEBUG_CATEGORY_INIT (gst_skindetector_debug_category3, "Extraction", 0,
					"debug category for extraction part of blobdetector");
}

/*
 * Main function in charge of calling helper definitions to extract all the allowable
 * blobs in the detector under study
 * */
int count=0;
int recursion=0;

uint8_t extract_blobs(GstBlobDetector *detector){
	uint8_t ierror = blob_success;

	GST_DEBUG_OBJECT(detector, "Extracting the blobs..!!");
	ierror = extract(detector);
	return ierror;
}


uint8_t extract(GstBlobDetector *detector){
	int i,j,k = 2, x, y, found=0;
	uint8_t ierror=blob_success;
	uint32_t xmin=999999999, xmax=0, ymin=999999999, ymax=0;
	uint32_t width, height, cheight, cwidth, ysize, uvsize;

	if(detector->channels_format == planar){//ONLY ONCE
		width = detector->width; height =detector->height;
		if(detector->labels==NULL){
			detector->labels = (uint32_t*)g_malloc0(height*width*sizeof(uint32_t));
		}else{
			memset(detector->labels, 0, height*width*sizeof(uint32_t));
		}
	}
	else{//assumes RGB interleaved, //ONLY ONCE
		width = detector->width>>2;//since we work with 32 bits
		height =detector->height;
		if(detector->labels==NULL){
			detector->labels = (uint32_t*)g_malloc0(height*width*sizeof(uint32_t));
		}else{
			memset(detector->labels, 0, height*width*sizeof(uint32_t));
		}
	}

	gfloat ratio;
	RECTANGLE bounds;

	GST_DEBUG_OBJECT(detector, "Preparing container for labels <W: %i, H: %i>..!!, ", width, height);
	prepare_image(detector, detector->labels);

	/*
	 * Estimating rectangle size, and reduce it every time
	 * where only the unprocessed pixels are to be used
	 * the once already labeling are not touched therefore
	 * just skipped
	 * */
	detector->n_blobs=0;//counting the BBs

	//frame region to scan for blobs, start at whole region
	bounds.xmax = width;
	bounds.ymax = height;
	bounds.xmin = 0;
	bounds.ymin = 0;


	//sort of divide and conquer
	while(TRUE){
		/*
		 * shrinking Bounding box containing possible objects
		 * */
		GetRectangle(&bounds, detector->labels, width);
		GST_DEBUG_OBJECT(detector, "Dimensions: (%d, %d, %d, %d) (%d, %d)\n", bounds.xmin, bounds.xmax, bounds.ymin, bounds.ymax, bounds.width, bounds.height);
		if((int)bounds.width<detector->minwidth || bounds.xmax==0 || bounds.ymax==0 || (int)bounds.height<detector->minheight ||
				bounds.width>bounds.xmax || bounds.height>bounds.ymax || bounds.width>width || (int)bounds.height>height){
			GST_DEBUG_OBJECT(detector, "**************** Early drop..!! ********************");
			break;//early drop
		}

		found=0;
		i=bounds.ymin;
		for (j=bounds.xmin;j<bounds.xmax;j++){
			//check if the pixel is unprocessed, then flood-fill it
			if (detector->labels[i*width + j] == 1){//1 means this pixel is unlabeled and is skin
				k++;

				//this label is pure white, dont consider SAFETY ONLY
				if(k==255)
					k++;


				//printf("NEW..\t");
				xmin=999999999; xmax=0; ymin=999999999; ymax=0;
//				g_print("before size filter Region <BBw: %i, BBh: %i, X: %i, Y: %i>\n", bounds.xmax-bounds.xmin, bounds.ymax-bounds.ymin, j, i);


//				Iter_floodfill(detector, j, i, MAXLABEL, k, width, height, &xmin, &xmax, &ymin, &ymax);

				{
					detector->pixel_density = 0;
					FAST_floodfill(detector->labels, j, i, MAXLABEL, k, width, height, &xmin, &xmax, &ymin, &ymax, GST_VIDEO_FRAME_COMP_DATA(detector->inframe, 0),
							detector);
					detector->pixel_density /= (7.0*((xmax-xmin)*(ymax-ymin)) + FLT_EPSILON);
				}

				//CHECK COUNT AND ESTIMATE BLOB IF ABOVE THRESHOLD
				if(count>detector->minarea && count<detector->maxarea){
					detector->blobs[detector->n_blobs].xmin = xmin;
					detector->blobs[detector->n_blobs].ymin = ymin;
					detector->blobs[detector->n_blobs].xmax = xmax;
					detector->blobs[detector->n_blobs].ymax = ymax;
					detector->blobs[detector->n_blobs].width = detector->blobs[detector->n_blobs].xmax - detector->blobs[detector->n_blobs].xmin;
					detector->blobs[detector->n_blobs].height = detector->blobs[detector->n_blobs].ymax - detector->blobs[detector->n_blobs].ymin;
					detector->blobs[detector->n_blobs].Blob_Pel_Density = detector->pixel_density;

					ratio = (gfloat)detector->blobs[detector->n_blobs].height / (gfloat)detector->blobs[detector->n_blobs].width;

					//validating the BBs
					if(detector->blobs[detector->n_blobs].ymax>0 && detector->blobs[detector->n_blobs].xmax>0 && ratio>MIN_ASPECTRATIO && ratio<MAX_ASPECTRATIO &&
							detector->blobs[detector->n_blobs].height>detector->minheight &&	detector->blobs[detector->n_blobs].width>detector->minwidth &&
							detector->blobs[detector->n_blobs].height<detector->maxheight && detector->blobs[detector->n_blobs].width<detector->maxwidth ){

						found=1;
						if((xmax-xmin)>0 || (ymax-ymin)>0){
							GST_INFO_OBJECT(detector, "PASSED !!:  BLOB_FOUND WITH DIMENSIONS: Recursion: %i Area: %i, Width: %i, Height: %i, Density: %f",
													recursion, count, detector->blobs[detector->n_blobs].width, detector->blobs[detector->n_blobs].height,
													detector->blobs[detector->n_blobs].Blob_Pel_Density);



//							g_print("PASSED !!:  BLOB_FOUND WITH DIMENSIONS: Recursion: %i Area: %i, Width: %i, Height: %i, Density: %f\n",
//													recursion, count, detector->blobs[detector->n_blobs].width, detector->blobs[detector->n_blobs].height,
//													detector->blobs[detector->n_blobs].Blob_Pel_Density);
						}

					}else{

						GST_WARNING_OBJECT(detector, "FAILED !!:"
								"Aspect ratio: <%f -> (%f, %f)>, Width: <%i -> (%i, %i)>, Height: <%i -> (%i, %i)>, Area: <%i -> (%i, %i)>, Density: %f",
								ratio, MIN_ASPECTRATIO, MAX_ASPECTRATIO, detector->blobs[detector->n_blobs].width, detector->minwidth, detector->maxwidth,
								detector->blobs[detector->n_blobs].height, detector->minheight, detector->maxheight, count,
								detector->minarea, detector->maxarea, detector->blobs[detector->n_blobs].Blob_Pel_Density);


//						g_print("FAILED !!:"
//								"Aspect ratio: <%f -> (%f, %f)>, Width: <%i -> (%i, %i)>, Height: <%i -> (%i, %i)>, Area: <%i -> (%i, %i)>, Density: %f\n",
//								ratio, MIN_ASPECTRATIO, MAX_ASPECTRATIO, detector->blobs[detector->n_blobs].width, detector->minwidth, detector->maxwidth,
//								detector->blobs[detector->n_blobs].height, detector->minheight, detector->maxheight, count,
//								detector->minarea, detector->maxarea, detector->blobs[detector->n_blobs].Blob_Pel_Density);

						detector->n_blobs--;//decreasing BBs counter in case is removed

					}
					detector->n_blobs++;//increasing the number of BBs
				}

				count=0; // reset count, this variable will count no of pixels for current color
				recursion=0;

				if(found)
					break;//early drop
			}
		}
	}

	GST_DEBUG_OBJECT(detector, "%i blobs found .. (%i groups)!!", detector->n_blobs, k);
	GST_DEBUG_OBJECT(detector, "Preparing the output buffer..!!");
	GST_DEBUG_OBJECT(detector, "srcformat: %d!!", detector->image_type);

	if(detector->image_type == GST_VIDEO_FORMAT_I420){
		process_I420(detector, k);
	}else if(detector->image_type == GST_VIDEO_FORMAT_GRAY8){
		process_gray8(detector, k);
	}else if(detector->image_type == GST_VIDEO_FORMAT_xRGB || detector->image_type == GST_VIDEO_FORMAT_ARGB
			|| detector->image_type == GST_VIDEO_FORMAT_xBGR || detector->image_type == GST_VIDEO_FORMAT_ABGR){
		process_0RGB(detector, k);
	}else if(detector->image_type == GST_VIDEO_FORMAT_RGBx || detector->image_type == GST_VIDEO_FORMAT_RGBA
			|| detector->image_type == GST_VIDEO_FORMAT_BGRx || detector->image_type == GST_VIDEO_FORMAT_BGRA){
		process_RGB0(detector, k);
	}else{
		GST_ERROR_OBJECT(detector, "This image format is not supported..!!");
		ierror = blob_unknown;
		return ierror;
	}

	if(detector->n_blobs>1){
//		for(i=0;i<detector->n_blobs-1;i++){
//			/* Filtering out blobs with higher density per pel*/
//			if(detector->blobs[0].Blob_Pel_Density > detector->blobs[i+1].Blob_Pel_Density){
//				//(detector->blobs[0].height*detector->blobs[0].width) < (detector->blobs[i+1].height*detector->blobs[i+1].width)
//				detector->blobs[0].height = detector->blobs[i+1].height;
//				detector->blobs[0].width = detector->blobs[i+1].width;
//				detector->blobs[0].ymax = detector->blobs[i+1].ymax;
//				detector->blobs[0].xmax = detector->blobs[i+1].xmax;
//				detector->blobs[0].xmin = detector->blobs[i+1].xmin;
//				detector->blobs[0].ymin = detector->blobs[i+1].ymin;
//				detector->blobs[0].Blob_Pel_Density = detector->blobs[i+1].Blob_Pel_Density;
//			}
//		}

		for(i=0;i<detector->n_blobs;i++){
			/*
			 * Filtering the blobs according to the texture threshold < Texture_threshold%
			 * larger than that we considered as too high and we discard this
			 * blob as a true skin
			 * */
			if(detector->blobs[i].Blob_Pel_Density < detector->text_thres){
				GST_DEBUG_OBJECT(detector, "Blobs texture value: %f (PASSED!!)", detector->blobs[i].Blob_Pel_Density);
				detector->blobs[i].passed = TRUE;
			}else{
				GST_DEBUG_OBJECT(detector, "Blobs texture value: %f (FAILED!!)", detector->blobs[i].Blob_Pel_Density);
				detector->blobs[i].passed = FALSE;
			}

//			detector->blobs[i].passed = TRUE;
//			detector->blobs[i].height = detector->blobs[0].height;
//			detector->blobs[i].width = detector->blobs[0].width;
//			detector->blobs[i].ymax = detector->blobs[0].ymax;
//			detector->blobs[i].xmax = detector->blobs[0].xmax;
//			detector->blobs[i].xmin = detector->blobs[0].xmin;
//			detector->blobs[i].ymin = detector->blobs[0].ymin;
		}
//		detector->n_blobs = 1;
	}else if(detector->n_blobs==1){
		detector->blobs[0].passed = TRUE;//since we have only one
	}

	GST_DEBUG_OBJECT(detector, "=====================\n"
			"FINAL BLOB LEFT: (x: %d, y: %d), W: %d, H: %d, Density: %f\n"
			"======================", detector->blobs[0].xmin, detector->blobs[0].ymin,
			detector->blobs[0].width, detector->blobs[0].height, detector->blobs[0].Blob_Pel_Density);

	return ierror;
}


/********************************** Recursive FloodFill **************************************/
/*
 * Recursive fast flood fill algorithm, however not suitable
 * for ARM
 * */
static guint8 ind, TD=1;
guint32 pel_pos;
void FAST_floodfill(uint32_t* pixels, int x,int y,int oldColor,int newColor, uint32_t width, uint32_t height,
		uint32_t *xmin, uint32_t *xmax, uint32_t *ymin, uint32_t *ymax, guint8 *srcpixels, GstBlobDetector *detector){

	recursion++;

	pel_pos = y*width + x;

	if(oldColor==MAXLABEL)
		oldColor = *(pixels+pel_pos);

	if(oldColor == newColor)
		return;

	if((*(pixels+pel_pos)) != oldColor)
		return;

	int y1;

	//draw current scanline from start position to the top
	y1 = y;
	if(y1 < height){
		pel_pos = y1*width + x;
		while((*(pixels+pel_pos)) == oldColor)
		{
			count++;

			//constant tracking of BB dimensions
			if(*xmin>x)
				*xmin = x;
			if(*xmax<x)
				*xmax = x;
			if(*ymin>y)
				*ymin = y;
			if(*ymax<y)
				*ymax = y;

			(*(pixels+pel_pos)) = newColor;

			for(ind=1; ind<8;ind++){
				/*
				 * shifting one bit and check if the texture
				 * flag was activated during skin detection
				 * */
				if(((*(srcpixels+y1*width + x))>>ind) & TD)
					detector->pixel_density+=1;//increasing accumulator
			}

			y1++;
			if(y1 >= height)
				break;
			pel_pos = y1*width + x;
		}
	}

	//draw current scanline from start position to the bottom
	y1 = y - 1;
	if(y1 >= 0){
		pel_pos = y1*width + x;
		while((*(pixels+pel_pos)) == oldColor)
		{
			count++;

			//constant tracking of BB dimensions
			if(*xmin>x)
				*xmin = x;
			if(*xmax<x)
				*xmax = x;
			if(*ymin>y)
				*ymin = y;
			if(*ymax<y)
				*ymax = y;

			(*(pixels+pel_pos)) = newColor;

			for(ind=1; ind<8;ind++){
				/*
				 * shifting one bit and check if the texture
				 * flag was activated during skin detection
				 * */
				if(((*(srcpixels+(y1*width + x)))>>ind) & TD)
					detector->pixel_density+=1;//increasing accumulator
			}

			y1--;
			if(y1 < 0)
				break;
			pel_pos = y1*width + x;
		}
	}

	//test for new scanlines to the left
	y1 = y;
	if(y1 < height){
		pel_pos = y1*width + x;
		while((*(pixels+pel_pos)) == newColor)
		{
			if(x > 0){
				pel_pos = y1*width + (x-1);
				if((*(pixels+pel_pos)) == oldColor){
					FAST_floodfill(pixels, x - 1, y1, oldColor, newColor,width, height,
							xmin, xmax, ymin, ymax, srcpixels, detector);
				}
			}

			y1++;
			if(y1 >= height)
				break;
			pel_pos = y1*width + x;
		}
	}

	y1 = y - 1;
	if(y1 >= 0){
		pel_pos = y1*width + x;
		while((*(pixels+pel_pos)) == newColor)
		{
			if(x > 0){
				pel_pos = y1*width + (x-1);
				if((*(pixels+pel_pos)) == oldColor)
				{
					FAST_floodfill(pixels, x - 1, y1, oldColor, newColor, width, height,
							xmin, xmax, ymin, ymax, srcpixels, detector);
				}
			}
			y1--;
			if(y1 < 0)
				break;
			pel_pos = y1*width + x;
		}
	}

	//test for new scanlines to the right
	y1 = y;
	if(y1 < height){
		pel_pos = y1*width + x;
		while((*(pixels+pel_pos)) == newColor)
		{
			if(x < width - 1){
				pel_pos = y1*width + (x+1);
				if((*(pixels+pel_pos)) == oldColor)
				{
					FAST_floodfill(pixels, x + 1, y1, oldColor, newColor, width, height,
							xmin, xmax, ymin, ymax, srcpixels, detector);
				}
			}

			y1++;
			if(y1 >= height)
				break;
			pel_pos = y1*width + x;
		}
	}


	y1 = y - 1;
	if(y1 >= 0){
		pel_pos = y1*width + x;
		while((*(pixels+pel_pos)) == newColor)
		{
			if(x < width - 1){
				pel_pos = y1*width + (x+1);
				if((*(pixels+pel_pos)) == oldColor)
				{
					FAST_floodfill(pixels, x + 1, y1, oldColor, newColor, width, height,
							xmin, xmax, ymin, ymax, srcpixels, detector);
				}
			}

			y1--;
			if(y1 < 0)
				break;
			pel_pos = y1*width + x;
		}
	}
}
/********************************** Recursive FloodFill **************************************/













/*
 * Helper functions for the floodfill, and also recovery functions since it is likely
 * to fail when the blob is too large
 * */
/* pixel variables */
static int currRow;         /* current row */
static uint32_t *labelsPtr; /* pointer to labels */
static guint8 *srcptr; /*to get the density per pel*/
static guint32 pel_dens=0;

static void doflood(GstBlobDetector *detector, int x,int y,int oldColor,int newColor, uint32_t width, uint32_t height,
		uint32_t *xmin, uint32_t *xmax, uint32_t *ymin, uint32_t *ymax);
static void visitrows(GstBlobDetector *detector, uint32_t width, uint32_t height, int oldcolor, int newcolor,
		uint32_t *xmin, uint32_t *xmax, uint32_t *ymin, uint32_t *ymax);
static void create_rows(GstBlobDetector *detector, int lft, int rgt);
static void makerow(GstBlobDetector *detector);
static void newrow(GstBlobDetector *detector, int slft, int srgt, int srow, int prow);
static void cliprow(GstBlobDetector *detector, int lft, int rgt, int row, GList *line);
static void freerows(GList *list);
static void init_row(int ypos);
static int check_pixel(int xpos, int ypos, int MaxYpos, int MaxXpos, int oldcolor, int newcolor,
		uint32_t *xmin, uint32_t *xmax, uint32_t *ymin, uint32_t *ymax);

/********************************** Iterative FloodFill **************************************/
void Iter_floodfill(GstBlobDetector *detector, int x,int y,int oldColor,int newColor, uint32_t width, uint32_t height,
		uint32_t *xmin, uint32_t *xmax, uint32_t *ymin, uint32_t *ymax){
	if(oldColor==MAXLABEL)
		oldColor = detector->labels[y*width + x];

	if(newColor == oldColor)
		return; //avoid infinite loop

	init_row(y);

	/*this contains the density, one plane only*/
	pel_dens=0;//reset pixel density since are starting a new blob
	srcptr = GST_VIDEO_FRAME_COMP_DATA(detector->inframe, 0);

	doflood(detector, x, y, oldColor, newColor, width, height, xmin, xmax, ymin, ymax);

	/*
	 * Recording the pixel density of this blob, but as a value ready for thresholding
	 * divided by 7 because we are working with 7 planes, and "FLT_EPSILON" for stability
	 * purposes
	 * */
	detector->pixel_density = ((gfloat)pel_dens)/(7.0*((*xmax-*xmin)*(*ymax-*ymin)) + FLT_EPSILON);
	return;
}


/*
 * Byte far initialize current column and row
 * */
static void init_row(int ypos){
   currRow = ypos - 1;
}


/*
 * examine one pixel
 * */
static int check_pixel(int xpos, int ypos, int MaxYpos, int MaxXpos, int oldcolor, int newcolor,
		uint32_t *xmin, uint32_t *xmax, uint32_t *ymin, uint32_t *ymax){
	unsigned char bitmask;
	guint32 position;
	gint i, j, texture_detected=1;
	gboolean reject=TRUE;
	gint window=7;

	if ((ypos < 3) || (ypos > MaxYpos-3))
	  return FALSE;

	if ((xpos < 3) || (xpos > MaxXpos-3))
	  return FALSE;

	currRow = ypos;

	position = ypos*MaxXpos + xpos;
	if ((*(labelsPtr+position)) != oldcolor){
	  return FALSE;
	}

	/*
	 * FIXME: josue 4-17-2015
	 *
	 * Problem with this algorithm is that it doesn't have the power
	 * to check previous lines, therefore going backwards. To achieve this
	 * behavior I check 8-neighbors (concavity and convexity reasons) to see
	 * if I still in the same blob, otherwise this is a new blob and I just keep skipping leaving it unlabeled for the
	 * next run
	 * */
	if(*ymax>0 && *xmax>0){
		for(i=-window;i<=window;i++){
			for(j=-window;j<=window;j++){
				position = (ypos+i)*MaxXpos + (xpos+j);

				if (((ypos+i) < 3) || ((ypos+i) > MaxYpos-3))
				  break;//early drop

				if (((xpos+j) < 3) || ((xpos+j) > MaxXpos-3))
				  break;//early drop

				if ((*(labelsPtr+position)) == newcolor){
					reject = FALSE;
					break;//early drop
				}
			}
		}
		if(reject)
			return FALSE;//early drop
	}
	/********************************************************************/

	count++;
	position = ypos*MaxXpos + xpos;
	(*(labelsPtr+position)) = newcolor;//Lft or rgt

	/*
	 * we need to check all the higher bit planes that were
	 * used during skin detection to determine texture
	 * shifting one bit and check if the texture
	 * flag was activated during skin detection
	 * */
	if(((*(srcptr+position))>>1) & texture_detected)// 1/2 resolution
		pel_dens++;//increasing accumulator
	if(((*(srcptr+position))>>2) & texture_detected)// 1/4 resolution
			pel_dens++;//increasing accumulator
	if(((*(srcptr+position))>>3) & texture_detected)// 1/8 resolution
			pel_dens++;//increasing accumulator
	if(((*(srcptr+position))>>4) & texture_detected)// 1/16 resolution
			pel_dens++;//increasing accumulator
	if(((*(srcptr+position))>>5) & texture_detected)// 1/32 resolution
			pel_dens++;//increasing accumulator
	if(((*(srcptr+position))>>6) & texture_detected)// 1/64 resolution
			pel_dens++;//increasing accumulator
	if(((*(srcptr+position))>>7) & texture_detected)// 1/128 resolution
			pel_dens++;//increasing accumulator


	if(*ymin > ypos){
		*ymin = ypos;
	}
	if(*ymax < ypos){
		*ymax = ypos;
	}

	if(*xmin > xpos){
		*xmin = xpos;
	}
	if(*xmax < xpos){
		*xmax = xpos;
	}
	return TRUE;
}


/*
 * release row nodes
 * */
static void freerows(GList *list){
   GList *t;
   while(list){
      t = list->next;
      list->next = NULL;
      g_free(list->data);
      g_list_free(list);
      list = t;
   }
   list = NULL;
}

/*
 * make new row
 * */
static void newrow(GstBlobDetector *detector, int slft, int srgt, int srow, int prow){
   GList *new=NULL;

	if ((new = detector->freeHead) != NULL)
	   detector->freeHead = detector->freeHead->next;
	else if ( ( new = g_list_append(new,(gpointer)g_malloc(sizeof(tracker_data))) ) == NULL) {
		freerows(detector->rowHead);
		freerows(detector->pendHead);
		GST_ERROR_OBJECT(detector, "problem creating the list..!!");
	}

	((tracker_data*)(new->data))->lft = slft;  ((tracker_data*)new->data)->rgt = srgt;
	((tracker_data*)new->data)->row = srow;  ((tracker_data*)new->data)->par = prow;
	((tracker_data*)new->data)->ok = TRUE;
	new->next = detector->pendHead;
	detector->pendHead = new;
}


/* make list of all rows on one row */
static void makerow(GstBlobDetector *detector){
	GList *s=NULL, *t=NULL, *u=NULL;
	t = detector->pendHead;
	detector->pendHead = NULL;
	while ((s = t) != NULL) {
		t = t->next;
		if (((tracker_data*)(s->data))->ok) {
			if (detector->rowHead == NULL) {
				currRow = ((tracker_data*)(s->data))->row;
				s->next = NULL;
				detector->rowHead = s;
			}
			else if (((tracker_data*)(s->data))->row == currRow) {
				if (((tracker_data*)(s->data))->lft <= ((tracker_data*)(detector->rowHead->data))->lft) {
					s->next = detector->rowHead;
					detector->rowHead = s;
				}
				else {
					for (u = detector->rowHead; u->next; u = u->next)
						if (((tracker_data*)(s->data))->lft <= ((tracker_data*)(u->next->data))->lft)
							break;
					s->next = u->next;
					u->next = s;
				}
			}
			else {
				s->next = detector->pendHead;
				detector->pendHead = s;
			}
		}
		else {
			s->next = detector->freeHead;
			detector->freeHead = s;
		}
	}
}


/* make new row(s) that don't overlap lines */
static void cliprow(GstBlobDetector *detector, int lft, int rgt, int row, GList *line){
   if (lft < (((tracker_data*)line->data)->lft - 1))
	   newrow(detector, lft, ((tracker_data*)line->data)->lft - 2, row, ((tracker_data*)line->data)->row);
   if (rgt > (((tracker_data*)line->data)->rgt + 1))
	   newrow(detector, ((tracker_data*)line->data)->rgt + 2, rgt, row, ((tracker_data*)line->data)->row);
}


/*
 * discard row segments which overlap lines
 * */
static void removeoverlap(GstBlobDetector *detector, GList *row){
   GList *child;

   for (child = detector->pendHead; child; child = child->next){
	   if ((((tracker_data*)child->data)->row == ((tracker_data*)row->data)->par)) {

		  cliprow(detector, ((tracker_data*)child->data)->lft, ((tracker_data*)child->data)->rgt,
				  ((tracker_data*)child->data)->row, row);

		  if (((tracker_data*)row->data)->rgt > (((tracker_data*)child->data)->rgt + 1))
			  ((tracker_data*)row->data)->lft = ((tracker_data*)child->data)->rgt + 2;
		 else
			 ((tracker_data*)row->data)->ok = FALSE;

		  ((tracker_data*)child->data)->ok = FALSE;

			return;
	  }
   }
}


/*
 * make rows of one child line
 * */
static void create_rows(GstBlobDetector *detector, int lft, int rgt){
   GList *child;

   /*
    * Check where the child should point to
    * */
   if (currRow > ((tracker_data*)detector->seedShadow->data)->par){
	   newrow(detector, lft, rgt, currRow + 1, currRow);
//	   newrow(detector, lft, rgt, currRow - 1, currRow);
	   cliprow(detector, lft, rgt, currRow - 1, detector->seedShadow);
   }
   else if (currRow < ((tracker_data*)detector->seedShadow->data)->par){
	   newrow(detector, lft, rgt, currRow - 1, currRow);
//	   newrow(detector, lft, rgt, currRow + 1, currRow);
	   cliprow(detector, lft, rgt, currRow + 1, detector->seedShadow);
   }
   else{
	   newrow(detector, lft, rgt, currRow + 1, currRow);
	   newrow(detector, lft, rgt, currRow - 1, currRow);
   }

   /*
    * Avoid overlaping labels since this will consume computation
    * */
   for (child = detector->rowHead; child && (((tracker_data*)child->data)->lft <= rgt); child = child->next){
      if (((tracker_data*)child->data)->ok && !detector->Allowoverlap){
         removeoverlap(detector, child);
      }
   }
}



/* visit all child lines found within one row */
static void visitrows(GstBlobDetector *detector, uint32_t width, uint32_t height, int oldcolor, int newcolor,
		uint32_t *xmin, uint32_t *xmax, uint32_t *ymin, uint32_t *ymax){
	int col, lft;
	for (col = ((tracker_data*)(detector->seedShadow->data))->lft; col <= ((tracker_data*)(detector->seedShadow->data))->rgt; col++) {
	  if (check_pixel(col, currRow, height, width, oldcolor, newcolor, xmin, xmax, ymin, ymax)) {
		 if ((lft == col == ((tracker_data*)(detector->seedShadow->data))->lft)){

			while (check_pixel(--lft, currRow, height, width, oldcolor, newcolor, xmin, xmax, ymin, ymax)) ;
			lft++;
		 }

		 while (check_pixel(++col, currRow, height, width, oldcolor, newcolor, xmin, xmax, ymin, ymax)) ;
		 create_rows(detector, lft, col - 1);
	  }
	}
}



/* flood visit */
static void doflood(GstBlobDetector *detector, int x,int y,int oldColor,int newColor, uint32_t width, uint32_t height,
		uint32_t *xmin, uint32_t *xmax, uint32_t *ymin, uint32_t *ymax){

	labelsPtr = detector->labels;
	/* check if we need to scan here */
	detector->pendHead = detector->rowHead = detector->freeHead = NULL;
	newrow(detector, x, x, y, y);

	/*
	 * Iterative part of the code, where the stopping criterion is
	 * given by the presence of more elements in the double
	 * linked list
	 * */
	while (detector->pendHead){
	  makerow(detector);
	  while (detector->rowHead){
		  detector->seedShadow = detector->rowHead;
		  detector->rowHead = detector->rowHead->next;
		 if (((tracker_data*)(detector->seedShadow->data))->ok){
			 visitrows(detector, width, height, oldColor, newColor, xmin, xmax, ymin, ymax);
		 }
		 detector->seedShadow->next = detector->freeHead;
		 detector->freeHead = detector->seedShadow;
	  }
	}

	/* Freeing the lists */
	freerows(detector->freeHead);
	freerows(detector->rowHead);
	freerows(detector->pendHead);
}
/********************************** Iterative FloodFill **************************************/

void process_gray8(GstBlobDetector *detector, int groups){

	gint height = detector->height, fsize;
	guint8 *GRAY_data = NULL;
	gint GRAY_stride = 0;
	guint position;
	gint GRAY_comp = 0;
	guint8 label, pixel_stride=4;

	if (detector->outframe->buffer != NULL && GST_IS_BUFFER(detector->outframe->buffer)) {
		GRAY_data = GST_VIDEO_FRAME_COMP_DATA(detector->outframe, GRAY_comp);
		GRAY_stride = GST_VIDEO_FRAME_COMP_STRIDE(detector->outframe, GRAY_comp);
		fsize = height*GRAY_stride;

		/*since we manipulate the bits constantly, if we dont do this we will have troubles when setting the bits in the loop*/
		memset(GRAY_data, 0,GST_VIDEO_FRAME_COMP_WIDTH(detector->outframe, GRAY_comp)*GST_VIDEO_FRAME_COMP_HEIGHT(detector->outframe, GRAY_comp));

		position=0;
		while(position<fsize-pixel_stride){

			/*unrolling this part*/
			/*
			 * Is not the same value, it to make sure that the labels can be stored in 8-bit GRAY8
			 * frame which is the one going down the pipeline
			 * */
			label = (guint8)(((float)(*(detector->labels+position))/(float)groups)*255);

			/*likely to have same label in the neighborhood*/
			*(GRAY_data+position) |= label;//operating the bits directly
			*(GRAY_data+position+1) |= label;//operating the bits directly
			*(GRAY_data+position+2) |= label;//operating the bits directly
			*(GRAY_data+position+3) |= label;//operating the bits directly


			position+=pixel_stride;
		}
	} else {
		GST_WARNING("buffer null?%d or not a buffer", detector->outframe->buffer == NULL);
	}
}

void process_I420(GstBlobDetector *detector, int groups){
	gint height = detector->height, width = detector->width, uvsize;
	gint i, j, ysize = height*width, cheight = height>>1, cwidth = width>>1;
	GstMapInfo OutInfo;

	ysize = height*width;
	uvsize = cheight*cwidth;

	detector->outframe->buffer =  gst_buffer_make_writable(detector->outframe->buffer);
	gst_buffer_map(detector->outframe->buffer, &OutInfo, GST_MAP_WRITE);

	//respecting the YUV channels
	for (i=0;i<height;i++)
	{
		for (j=0;j<width;j++)
		{
			OutInfo.data[i*width + j] = detector->normalize_frame ? (guint8)(((float)detector->labels[i*width + j]/(float)groups)*255) :
					(guint8)(((float)detector->labels[i*width + j]/(float)groups)*255);//Y
			OutInfo.data[(i/2)*cwidth + (j/2) + ysize] = 128;//U
			OutInfo.data[(i/2)*cwidth + (j/2) + ysize + uvsize] = 128;//V
		}
	}

	gst_buffer_unmap(detector->outframe->buffer, &OutInfo);
}

void process_0RGB(GstBlobDetector *detector, int groups){
	gint height = detector->height, width = detector->width;
	gint i, j;
	GstMapInfo OutInfo;

	detector->outframe->buffer =  gst_buffer_make_writable(detector->outframe->buffer);
	gst_buffer_map(detector->outframe->buffer, &OutInfo, GST_MAP_WRITE);

	//respecting the RGB channels
	for (i=0;i<height;i++)
	{
		for (j=0;j<width;j+=3)
		{
			OutInfo.data[i*width + j + 1] = detector->normalize_frame ? (guint8)(((float)detector->labels[i*width + j]/(float)groups)*255) :
					(guint8)(((float)detector->labels[i*width + j]/(float)groups)*255);//R
			OutInfo.data[i*width + j + 2]  = OutInfo.data[i*width + j];//G
			OutInfo.data[i*width + j + 3] = OutInfo.data[i*width + j];//B
		}
	}

	gst_buffer_unmap(detector->outframe->buffer, &OutInfo);
}

void process_RGB0(GstBlobDetector *detector, int groups){
	gint height = detector->height, width = detector->width;
	gint i, j;
	GstMapInfo OutInfo;

	detector->outframe->buffer =  gst_buffer_make_writable(detector->outframe->buffer);
	gst_buffer_map(detector->outframe->buffer, &OutInfo, GST_MAP_WRITE);

	//respecting the RGB channels
	for (i=0;i<height;i++)
	{
		for (j=0;j<width;j+=3)
		{
			OutInfo.data[i*width + j] = detector->normalize_frame ? (guint8)(((float)detector->labels[i*width + j]/(float)groups)*255) :
					(guint8)(((float)detector->labels[i*width + j]/(float)groups)*255);//R
			OutInfo.data[i*width + j + 1]  = OutInfo.data[i*width + j];//G
			OutInfo.data[i*width + j + 2] = OutInfo.data[i*width + j];//B
		}
	}

	gst_buffer_unmap(detector->outframe->buffer, &OutInfo);
}

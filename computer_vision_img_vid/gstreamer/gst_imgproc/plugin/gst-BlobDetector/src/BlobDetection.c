/*
 * BlobDetection.c
 *
 *  Created on: Jan 6, 2015
 *      Author: josue
 *
 *      Main part of the detector which uses and performs the operations
 *      to do blob detection
 */

#include <gst/gstinfo.h>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/video/gstvideofilter.h>
#include "includes/BlobDetection.h"
#include "includes/Extractor.h"


GST_DEBUG_CATEGORY_STATIC (gst_skindetector_debug_category2);
#define GST_CAT_DEFAULT gst_skindetector_debug_category2


void debug_ini_detection(){
	debug_ini_extraction();
	GST_DEBUG_CATEGORY_INIT (gst_skindetector_debug_category2, "Detection", 0,
					"debug category for detection part of blobdetector");
}


uint8_t Blob_error_handler(uint8_t ierror){
	switch(ierror){
		case blob_success:
			GST_DEBUG("Blob Detector finished the processing successfully..!!");
			break;
		case blob_nobinary:
			GST_DEBUG("The image has not been binarized, please make sure to input a binary image..!!");
			break;
		case blob_mallocerror:
			GST_DEBUG("Error allocating memory with MALLOC...");
			break;
		default:
			GST_DEBUG("Unknown Error code ..!!");
			break;
	}

	return ierror;
}


uint8_t perform_blobdetection(GstBlobDetector *detector){
	uint8_t ierror = blob_success;
	detector->blobs = NULL;
	detector->n_blobs = 0;

	//determine if we have a binary image
	if(detector->channels>1 || detector->channels<1){
		ierror = blob_nobinary;
		return ierror;
	}

	//creating memory to store blobs info
	detector->blobs = (BLOB*)g_malloc(sizeof(BLOB)*MAXBLOBS);
	if(detector->blobs==NULL){
		ierror = blob_mallocerror;
		return ierror;
	}

	//extracting blobs
	ierror = extract_blobs(detector);

	return ierror;//error code return if any
}


void GetRectangle(RECTANGLE *rect, uint32_t *labels, uint32_t width){
	/* to define if we still inside the box*/
	guint32 xmin=999999999, xmax=0, ymin=999999999, ymax=0;

	/* locations in the box */
	guint32 iter, up=rect->ymin, down=rect->ymax-1, left=rect->xmin, right=rect->xmax-1, pos;

	/* flags to determine termination */
	gboolean up_done=FALSE, down_done=FALSE, left_done=FALSE, right_done=FALSE;


	while(TRUE){
		//=============== search up from left to right moving downwards
		for(left=rect->xmin;left<right;left++){
			pos=up*width+left;//moving from left to right in the line
			if((*(labels+pos)) == 1){//when we get the first white pixel
				//we just update the minimum Y-vals
				ymax = ymax<up?up:ymax;
				up_done=TRUE;
				break;//early termination
			}
		}
		if(!up_done && up<down){//not yet done with minY
			up++;//moving downwards
		}else{
			break;//terminate this search
		}
	}


	while(TRUE){
		//=============== search down from left to right moving upwards
		for(left=rect->xmin;left<right;left++){
			pos=down*width+left;//moving from left to right in the line
			if((*(labels+pos)) == 1){//when we get the first white pixel
				//we just update the maximum Y-vals
				ymin = ymin>down?down:ymin;
				down_done=TRUE;
				break;//early termination
			}
		}
		if(!down_done && down>up){//not yet done with minY
			down--;//moving upwards
		}else{
			break;//terminate this search
		}
	}


	left=rect->xmin;
	right=rect->xmax;
	while(TRUE){
		//=============== search left up to down moving rightwards
		for(up=rect->ymin;up<down;up++){
			pos=up*width+left;//moving from left to right in the line
			if((*(labels+pos)) == 1){//when we get the first white pixel
				//we just update the minimum X-vals
				xmax = xmax<left?left:xmax;
				left_done=TRUE;
				break;//early termination
			}
		}
		if(!left_done && left<right){//not yet done with minY
			left++;//moving rightwards
		}else{
			break;//terminate this search
		}
	}


	left=rect->xmin;
	right=rect->xmax;
	while(TRUE){
		//=============== search right up to down moving leftwards
		for(up=rect->ymin;up<down;up++){
			pos=up*width+right;//moving from left to right in the line
			if((*(labels+pos)) == 1){//when we get the first white pixel
				//we just update the maximum X-vals
				xmin = xmin>right?right:xmin;
				right_done=TRUE;
				break;//early termination
			}
		}
		if(!right_done && right>left){//not yet done with minY
			right--;//moving rightwards
		}else{
			break;//terminate this search
		}
	}


	guint32 temp;
	temp = xmax;
	xmax = xmin;
	xmin = temp;
	temp = ymax;
	ymax = ymin;
	ymin = temp;

	//bounds to search for blobs dont consider other parts
	rect->xmax = xmax;
	rect->ymax = ymax;
	rect->xmin = xmin;
	rect->ymin = ymin;
	rect->width = xmax-xmin;
	rect->height = ymax-ymin;
}



void prepare_image(GstBlobDetector *detector, uint32_t *labels){
	uint32_t pos;
	uint32_t height, width, fsize, pix_stride;
	guint32 mask = 0xFF;//most significants
	guint8 skin=1;
	GstMapInfo FrameInfo;


	gint xmax=0, xmin=9999999, ymax=0,ymin=999999999;

	gst_buffer_map(detector->inframe->buffer, &FrameInfo, GST_MAP_READ);
	if(detector->image_type == GST_VIDEO_FORMAT_GRAY8 || detector->image_type == GST_VIDEO_FORMAT_I420){
		/*
		 * For speed we read 4 bytes from the buffer, therefore accessing 4 pixels at the same time
		 * if the format is gray or I420
		 * */
		guint32 pixels;
		height= detector->height, width = detector->width;
		fsize = height*width;
		pix_stride=4;//reading blocks of 4 bytes
		while(pos<fsize){
			/*
			 * Reading 4 pixels in the first plane in case it is I420
			 * or continuous reading 4 pixels in GRAY plane
			 * */
			pixels = *((guint32*)(FrameInfo.data + pos));

			/*
			 * Now accessing the 4 pixels at the same time, but checking
			 * only the first bit of each one of them
			 * */
			*(labels+pos) = ((pixels>>24) & skin)>0 ? 1 : 0;//shift to read first pixel
			*(labels+pos+1) = ((pixels>>16 & mask) & skin)>0  ? 1 : 0;//shift and mask to read second pixel
			*(labels+pos+2) = ((pixels>>8 & mask) & skin)>0  ? 1 : 0;//shift and mask to read third pixel
			*(labels+pos+3) = ((pixels & mask) & skin)>0  ? 1 : 0;//shift and mask to read fourth pixel

			pos += pix_stride;
		}
	}else{//RGBA packed (32 bits)
		/*
		 * This part is tricky since all 4 channels are packed, so we cannot access the skin detection
		 * result in the same way as above, need to think about it ... or not use this at all!! (waste of resources)
		 * */
		guint32 pixels;
		height= detector->height, width = detector->width;
		fsize = height*width;
		pix_stride=4;//reading blocks of 4 bytes
		while(pos<fsize){
			pixels = *((guint32*)(FrameInfo.data + pos));

			/*
			 * Now accessing the first channels no matter which one it is
			 * it always comes with the result from skin detector
			 * */
			*(labels+pos/4) = ((pixels>>24) & skin)>0  ? 1 : 0;

			pos += pix_stride;
		}
	}
	gst_buffer_unmap(detector->inframe->buffer, &FrameInfo);
}


























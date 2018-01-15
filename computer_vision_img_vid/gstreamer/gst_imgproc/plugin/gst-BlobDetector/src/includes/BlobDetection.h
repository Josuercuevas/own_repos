/*
 * BlobDetection.h
 *
 *  Created on: Jan 6, 2015
 *      Author: josue
 *
 *      Only visible header to the user, to be used in order
 *      to interact with the BlobDetection API. HERE is important
 *      to mention that only Binarized images are handled, or
 *      in other words single channel ones. The user may use skin detection first
 *      and then dump the result to later be the input of this API
 */

#ifndef BLOBDETECTION_H_
#define BLOBDETECTION_H_

#if __cplusplus
extern "C"{
#endif

#include "../../../common/metainfo.h"
#include "../gstblobdetector.h"

//For blob detector, basically filtering by size and aspect ratio
#define MAX_ASPECTRATIO (2.0) //MAX_HEIGHT_
#define MIN_ASPECTRATIO (0.5) //_HEIGHT
#define BLOB_MAXHEIGHT (1024)
#define BLOB_MAXWIDTH (1024)
#define BLOB_MINHEIGHT (150)
#define BLOB_MINWIDTH (150)
#define HEIGHT_BLOB_FRAME_RATIO (0.9)
#define WIDTH_BLOB_FRAME_RATIO (0.9)
#define BLOB_AREA_SIZE_THRES (1024)

enum channels_format{
	planar=0,//for planar images/frames
	interleaved//for interleaved channels
};

typedef struct _rect{
	uint32_t xmin, xmax, ymin, ymax;
	uint32_t height, width;
}RECTANGLE;


uint8_t perform_blobdetection(GstBlobDetector *detector);

/*
 * Error handler function, which can be called to determine the error
 * this will be used as a log function in case the user wants to determine crash
 * reasons when using the API
 * */
uint8_t Blob_error_handler(uint8_t ierror);


/*
 * Estimation of the bounds to be scanned for blob
 * detection
 * */
void GetRectangle(RECTANGLE *rect, uint32_t *labels, uint32_t width);
void prepare_image(GstBlobDetector *detector, uint32_t *labels);

/*
 * for debugging purposes
 * */
void debug_ini_detection();


#if __cplusplus
}
#endif

#endif /* BLOBDETECTION_H_ */

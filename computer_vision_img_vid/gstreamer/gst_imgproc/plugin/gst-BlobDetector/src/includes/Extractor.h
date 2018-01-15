/*
 * Extractor.h
 *
 *  Created on: Jan 6, 2015
 *      Author: josue
 *
 *      Main header contaning function prototypes and structures
 *      to be used during the blob detection process of the API
 *
 */

#ifndef EXTRACTOR_H_
#define EXTRACTOR_H_

#include "BlobDetection.h"
#include <math.h>

/*
 * Main function in charge of calling helper definitions to extract all the allowable
 * blobs in the image under study
 * */
uint8_t extract_blobs(GstBlobDetector *detector);

/*
 * Function to extract all the blobs from the frame
 * */
uint8_t extract(GstBlobDetector *detector);

/*
 * Fast version of the Flood Fill algorithm, more stable
 * stack friendly implementation, and it is much faster than
 * original version, though very large blobs may cause
 * stack overflow as well (however still recursive)
 * */
void FAST_floodfill(uint32_t* pixels, int x,int y,int oldColor,int newColor, uint32_t width, uint32_t height,
		uint32_t *xmin, uint32_t *xmax, uint32_t *ymin, uint32_t *ymax, guint8 *srcpixels, GstBlobDetector *detector);

/*
 * Stack friendly Floodfill algorithm where the idea is to use
 * two double linked lists, and manage the tracking process
 * from the next and previous elements
 * */
void Iter_floodfill(GstBlobDetector *detector, int x,int y,int oldColor,int newColor, uint32_t width, uint32_t height,
		uint32_t *xmin, uint32_t *xmax, uint32_t *ymin, uint32_t *ymax);


/*
 * Planar structures that we manage independently
 * */
void process_gray8_2(GstBlobDetector* detector, int groups);
void process_gray8(GstBlobDetector *detector, int groups);
void process_I420(GstBlobDetector *detector, int groups);

/*
 * only this two since we dont care about the order, except if the alpha channels
 * is full or empty
 * */
void process_0RGB(GstBlobDetector *detector, int groups);
void process_RGB0(GstBlobDetector *detector, int groups);


/*
 * for debugging purposes
 * */
void debug_ini_extraction();

#endif /* EXTRACTOR_H_ */

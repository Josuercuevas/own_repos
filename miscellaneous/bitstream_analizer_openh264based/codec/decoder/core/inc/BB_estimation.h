/*
 * BB_estimation.h
 *
 *  Created on: Oct 28, 2014
 *      Author: josue
 *
 *  Rutines in charge of estimating the Bounding Boxes to be used to the TLD
 *  algorithm, in here we do the estimation as well as the refinement and filtering
 *  for an easier and faster processing
 */

#ifndef SRC_BB_ESTIMATION_H_
#define SRC_BB_ESTIMATION_H_

#include "typedefs.h"
#include "wels_common_basis.h"
#include "decoder_context.h"

//REQUIRED STRUCTURES
struct min_max{
	int min_y, min_x, max_x, max_y;
	int size, width, height;
};


//------------------------Bounding box extraction
uint32_t estimate_BB(SBufferInfo *pDstInfo, uint64_t *image);

//----------------------------------------- component labeling
uint32_t Connected_Comp_Label(uint64_t *img_data, int width, int height, int *BB_x, int *BB_y, int *BB_w, int *BB_h,
		int *n_boxes);
void QuickSort(int a[], int L, int R);
uint32_t not_in_array(int x, int diz[], int n);
//-------------------------------------------------FILTERING OF THE BOUNDING BOXES DETECTED





#endif /* SRC_BB_ESTIMATION_H_ */

/*
 * SaliencyEstimator.h
 *
 *  Created on: Nov 24, 2014
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

#ifndef CODEC_DECODER_PLUS_INC_SALIENCYESTIMATOR_H_
#define CODEC_DECODER_PLUS_INC_SALIENCYESTIMATOR_H_

#include "typedefs.h"
#include "wels_common_basis.h"
#include "decoder_context.h"

using namespace WelsDec;
using namespace WelsCommon;

/***********************************************
 * Function prototypes for the estimation of the saliency maps
 * ***************************************/

/**
 * Spatial Saliency estimator function, where the macroblock YUV values is taken if they have significant number of bits
 * and then a Gabor like filters is implemented in order to estimate the maps, the fusion is performed by the
 * average normalization of the individual maps (where only, intensity, color and rotation is considered)
 * */
int SpaMapEstimation(PWelsDecoderContext pCtx, int8_t *SPaExst);

/**
 * The parameters for the Gabor filter are:
 * 		1. sigma = 1.7;
 * 		2. lambda = 2.5;
 * 		3. psi = 1.5;
 * 		4. gamma = 1.5;
 * 		5. theta_step = (180/10);
*/

#define PI (3.14159265358979323846)

/********************************* SPATIAL MAP STUFF ************************/

typedef struct filter_container{
	float sigma;// sigma parameter
	float lambda;//lambda parameter (wave length)
	float psi[2];// phase parameters for real and imaginary parts
	float gamma;//gamma parameter (aspect ratio)
	float theta_steps_size;//theta steps
	int theta_step;// steps to be used for saliency map
	int filter_size;//filter size
	int bw;// bandwidth
	int n_steps;// steps to be used for saliency map
	float *filter;// container of the filter values
}FILTER_CONTAINER;

/**
 * Functions used to help for the estimation of the Spatial Saliency maps
 * */
void filter_extraction(FILTER_CONTAINER f_pars);
int32_t map_estimation(PDqLayer pCurLayer, FILTER_CONTAINER filter, uint8_t *MB_Y, uint8_t *MB_U, uint8_t *MB_V, float *MBSal,
		float *max_val, float *min_val,	float *additional_cost);
void PartDec(PWelsDecoderContext pCtx);


/********************************* TEMPORAL MAP STUFF ************************/

/**
 * Function in charge of estimating the temporal saliency of the macroblock
 * */
int32_t TSmapEstimation(PWelsDecoderContext pCtx, int8_t *TempExst);
int32_t VelocityApprox(PDqLayer pCurLayer, uint8_t *currLuma, uint8_t *PrevLuma, uint8_t *TempMap);

//!Optical flow estimation (Not implemented here but the prototype is here if any future work on it)
//int Optical_flow(Slice *currSlice, Macroblock *currMB, const int mb_loc_x, const int mb_loc_y,
	//float alpha, const int iterations, const int n_frames, const int buff_size);


/********************************* FUSION MAP STUFF ************************/
int32_t MapsFusion(PWelsDecoderContext pCtx);

/*resetting*/
enum mapid{
	SPAMB=0,
	TEMPMB,
	FUSIONMB,
	YUVMB
};
uint32_t resetMB(PWelsDecoderContext pCtx, uint32_t maptype);

#endif /* CODEC_DECODER_PLUS_INC_SALIENCYESTIMATOR_H_ */

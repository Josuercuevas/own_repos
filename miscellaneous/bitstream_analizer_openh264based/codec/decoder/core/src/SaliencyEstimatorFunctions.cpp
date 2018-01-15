/*
 * SaliencyEstimatorFunctions.cpp
 *
 *  Created on: Nov 24, 2014
 *      Author: josue
 *
 *  Part of the Program which is in charge of estimating the saliency maps
 *  after the macroblocks have been filtered by the API. it is important to mention
 *  that it is a MB based approach therefore, features are scale can't be handled by
 *  the bitstream saliency map

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

#include "SaliencyEstimator.h"



/************************************************************************	SPATIAL SALIENCY MAP ESTIMATION *****************************************************************/

static inline float ffmax(float a, float b)
{
  return ((a) > (b)) ? (a) : (b);
}

static inline float ffmin(float a, float b)
{
  return ((a) < (b)) ? (a) : (b);
}

static inline int8_t int8min(int8_t a, int8_t b)
{
  return ((a) < (b)) ? (a) : (b);
}

static inline int64_t int64max(int64_t a, int64_t b)
{
  return ((a) > (b)) ? (a) : (b);
}


uint32_t resetMB(PWelsDecoderContext pCtx, uint32_t maptype){
	uint8_t *PartDec[3];
	uint8_t *spa, *tempMap, *FusionMap, *VotingFrames;
	uint64_t *votes;
	uint64_t maxIntensity=-1;
	PDqLayer pCurLayer = pCtx->pCurDqLayer;//current layer info
	uint32_t MBx = pCurLayer->iMbX, MBy = pCurLayer->iMbY, i, j;//starting points of the MBs
	uint32_t MAPDecStrideL = pCurLayer->pDec->iWidthInPixel, MAPDecStrideC = pCurLayer->pDec->iWidthInPixel>>1;//line size of Chroma and Luma for Maps

	//maps
	int32_t MAPOffsetL = (MBx + MBy * MAPDecStrideL) << 4;//since is a 16x16 MB
	int32_t MAPOffsetC = (MBx + MBy * MAPDecStrideC) << 3;//since is a 8x8 MB

	//chuncks to copy
	int32_t iCopySizeY  = (sizeof (uint8_t) << 4);//16 ELEMENTS OF uint8_t SIZE EACH
	int32_t iCopySizeUV = (sizeof (uint8_t) << 3);//8 ELEMENTS OF uint8_t SIZE EACH

	switch(maptype){
		case SPAMB:
			spa = pCtx->Saliency_Maps.SpaMap + MAPOffsetL;//spatial map
			//spatial map
			for(i=0; i<16; i++){
				memset(spa, 0, iCopySizeY);//Y
				spa+=MAPDecStrideL;
			}
			break;
		case TEMPMB:
			tempMap = pCtx->Saliency_Maps.TempMap + MAPOffsetL;//spatial map
			//votes = pCtx->Saliency_Maps.voting + MAPOffsetL;
			//VotingFrames = pCtx->Saliency_Maps.VotingFrames + MAPOffsetL;

			//temporal map
			for(i=0; i<16; i++){
				/*for(j=0; j<16; j++){
					if((*(votes+(i*MAPDecStrideL + j)))>0){
						(*(votes+(i*MAPDecStrideL + j)))-=1;//reducing votes
						maxIntensity = int64max(maxIntensity, (*(votes+(i*MAPDecStrideL + j))));
					}
				}*/
				memset(tempMap, 0, iCopySizeY);//Y
				tempMap+=MAPDecStrideL;
			}


			//for visualization of the voting
			/*for (i = 0; i < 16; i++) {
				for (j = 0; j < 16; j++) {
					(*(VotingFrames+j)) = (uint8_t)((((float)(*(votes+j)))/((float)maxIntensity + 1))*255.0f);
				}
				votes += MAPDecStrideL;
				VotingFrames += MAPDecStrideL;
			}*/

			break;
		case FUSIONMB:
			FusionMap = pCtx->Saliency_Maps.MasterMap + MAPOffsetL;//spatial map
			votes = pCtx->Saliency_Maps.voting + MAPOffsetL;
			VotingFrames = pCtx->Saliency_Maps.VotingFrames + MAPOffsetL;
			//fusion map
			for(i=0; i<16; i++){
				for(j=0; j<16; j++){
					if((*(votes+(i*MAPDecStrideL + j)))>0){
						(*(votes+(i*MAPDecStrideL + j)))-=1;//reducing votes
						maxIntensity = int64max(maxIntensity, (*(votes+(i*MAPDecStrideL + j))));
					}
				}
				memset(FusionMap, 0, iCopySizeY);//Y
				FusionMap+=MAPDecStrideL;
			}

			//for visualization of the voting
			for (i = 0; i < 16; i++) {
				for (j = 0; j < 16; j++) {
					(*(VotingFrames+j)) = (uint8_t)((((float)(*(votes+j)))/((float)maxIntensity + 1))*255.0f);
				}
				votes += MAPDecStrideL;
				VotingFrames += MAPDecStrideL;
			}

			break;
		case YUVMB:
			//partial decoded frame
			PartDec[0] = pCtx->Saliency_Maps.PartDec[0]+MAPOffsetL;//Y
			PartDec[1] = pCtx->Saliency_Maps.PartDec[1]+MAPOffsetC;//U
			PartDec[2] = pCtx->Saliency_Maps.PartDec[2]+MAPOffsetC;//V

			//Y
			for(i=0; i<16; i++){
				memset(PartDec[0], 16, iCopySizeY);//Y
				PartDec[0]+=MAPDecStrideL;
			}

			//U
			for(i=0; i<8; i++){
				memset(PartDec[1], 128, iCopySizeUV);//Y
				PartDec[1]+=MAPDecStrideC;
			}

			//V
			for(i=0; i<8; i++){
				memset(PartDec[2], 128, iCopySizeUV);//Y
				PartDec[2]+=MAPDecStrideC;
			}
			break;
		default:
			printf("This MB is not recognized...\n");
			return -1;
			break;
	}
}



//it copies that partially decoded frame
void PartDec(PWelsDecoderContext pCtx){
	uint8_t *PartDec[3];
	uint8_t *pDec[3];
	PDqLayer pCurLayer = pCtx->pCurDqLayer;//current layer info
	uint32_t MBx = pCurLayer->iMbX, MBy = pCurLayer->iMbY, i;//starting points of the MBs
	uint32_t iDecStrideL = pCurLayer->pDec->iLinesize[0], iDecStrideC = pCurLayer->pDec->iLinesize[1];//line size of Chroma and Luma for original
	uint32_t MAPDecStrideL = pCurLayer->pDec->iWidthInPixel, MAPDecStrideC = pCurLayer->pDec->iWidthInPixel>>1;//line size of Chroma and Luma for Maps

	//original
	int32_t iOffsetL = (MBx + MBy * iDecStrideL) << 4;//since is a 16x16 MB
	int32_t iOffsetC = (MBx + MBy * iDecStrideC) << 3;//since is a 8x8 MB

	//maps
	int32_t MAPOffsetL = (MBx + MBy * MAPDecStrideL) << 4;//since is a 16x16 MB
	int32_t MAPOffsetC = (MBx + MBy * MAPDecStrideC) << 3;//since is a 8x8 MB

	//chuncks to copy
	int32_t iCopySizeY  = (sizeof (uint8_t) << 4);//16 ELEMENTS OF uint8_t SIZE EACH
	int32_t iCopySizeUV = (sizeof (uint8_t) << 3);//8 ELEMENTS OF uint8_t SIZE EACH

	//partial decoded frame
	PartDec[0] = pCtx->Saliency_Maps.PartDec[0]+MAPOffsetL;//Y
	PartDec[1] = pCtx->Saliency_Maps.PartDec[1]+MAPOffsetC;//U
	PartDec[2] = pCtx->Saliency_Maps.PartDec[2]+MAPOffsetC;//V

	//original data
	pDec[0] = pCurLayer->pDec->pData[0]+iOffsetL;//Y
	pDec[1] = pCurLayer->pDec->pData[1]+iOffsetC;//U
	pDec[2] = pCurLayer->pDec->pData[2]+iOffsetC;//V

	//Y
	for(i=0; i<16; i++){
		memcpy(PartDec[0], pDec[0], iCopySizeY);//Y
		pDec[0]+=iDecStrideL;
		PartDec[0]+=MAPDecStrideL;
	}

	//U
	for(i=0; i<8; i++){
		memcpy(PartDec[1], pDec[1], iCopySizeUV);//Y
		pDec[1]+=iDecStrideC;
		PartDec[1]+=MAPDecStrideC;
	}

	//V
	for(i=0; i<8; i++){
		memcpy(PartDec[2], pDec[2], iCopySizeUV);//Y
		pDec[2]+=iDecStrideC;
		PartDec[2]+=MAPDecStrideC;
	}
}



/*
 * Spatial Saliency Estimation
 * */
int SpaMapEstimation(PWelsDecoderContext pCtx, int8_t *SPaExst){
	/*
	 * The saliency map of a single MB is an array of floats but when copied to the decoder context it has to be an
	 * 8-bit image, therefore normalization is required
	 * */

	//local vars needed
	uint8_t *MB_Y=NULL, *MB_U=NULL, *MB_V=NULL;
	uint32_t MBHeight=1<<4, MBWidth=1<<4, i, j;
	uint32_t PicHeight=pCtx->iPicHeightReq, PicWidth=pCtx->iPicWidthReq;
	FILTER_CONTAINER filter;

	//get current layer
	PDqLayer pCurLayer = pCtx->pCurDqLayer;//current layer info
	uint32_t MBx = pCurLayer->iMbX, MBy = pCurLayer->iMbY;//starting points of the MBs
	uint32_t PRED_TYPE = pCurLayer->pIntraPredMode[pCurLayer->iMbXyIndex][7];//prediction mode to determine the angle
	uint32_t MBtype = pCurLayer->pMbType[pCurLayer->iMbXyIndex];//MB type
	uint32_t SUBMBtype[4];

	//get data to use for saliency
	uint8_t *pDec[3];
	uint32_t iDecStrideL = pCurLayer->pDec->iLinesize[0], iDecStrideC = pCurLayer->pDec->iLinesize[1];//line size of Chroma and Luma for original
	uint32_t MAPDecStrideL = pCurLayer->pDec->iWidthInPixel;
	//original
	int32_t MAPOffsetL = (MBx + MBy * MAPDecStrideL) << 4;//since is a 16x16 MB
	int32_t iOffsetL = (MBx + MBy * iDecStrideL) << 4;//since is a 16x16 MB
	int32_t iOffsetC = (MBx + MBy * iDecStrideC) << 3;//since is a 8x8 MB
	//chunck size to copy
	int32_t iCopySizeY  = (sizeof (uint8_t) << 4);//16 ELEMENTS OF uint8_t SIZE EACH
	int32_t iCopySizeUV = (sizeof (uint8_t) << 3);//8 ELEMENTS OF uint8_t SIZE EACH

	//a buffer for the saliency map
	uint8_t *Spatial_map;

	//additional cost depends on the type of the macroblock (if it has subblocks or not)
	float maxval=-100000000.0f, minval=100000000.0f, additionalcost=1, range;
	float *MBSal=NULL;

	/* Making a temporal copy of the basic structures of the context*/

	SUBMBtype[0] = pCurLayer->pSubMbType[pCurLayer->iMbXyIndex][0];//First 8x8 SUBMB
	SUBMBtype[1] = pCurLayer->pSubMbType[pCurLayer->iMbXyIndex][1];//Second 8x8 SUBMB
	SUBMBtype[2] = pCurLayer->pSubMbType[pCurLayer->iMbXyIndex][2];//Third 8x8 SUBMB
	SUBMBtype[3] = pCurLayer->pSubMbType[pCurLayer->iMbXyIndex][3];//Fourth 8x8 SUBMB

	MB_Y = (uint8_t*)malloc(sizeof(uint8_t)*MBHeight*MBWidth);//size of the MB
	MB_U = (uint8_t*)malloc(sizeof(uint8_t)*(MBHeight>>1)*(MBWidth>>1));//size of the MB half the size of Y
	MB_V = (uint8_t*)malloc(sizeof(uint8_t)*(MBHeight>>1)*(MBWidth>>1));//size of the MB half the size of Y
	MBSal = (float*)malloc(sizeof(float)*MBHeight*MBWidth);//size of the MB same as Y
	memset(MBSal, 0, sizeof(float)*(MBHeight)*(MBWidth));

	if(MB_Y==NULL || MB_U==NULL || MB_V==NULL || MBSal==NULL)
		goto MALLOC_ERROR;


	//copy the MBs
	pDec[0] = pCurLayer->pDec->pData[0]+iOffsetL;//location in the Y place to copy the chunk
	pDec[1] = pCurLayer->pDec->pData[1]+iOffsetC;//location in the U place to copy the chunk
	pDec[2] = pCurLayer->pDec->pData[2]+iOffsetC;//location in the V place to copy the chunk

	//traking map
	Spatial_map = pCtx->Saliency_Maps.SpaMap+MAPOffsetL;

	//Y
	for(i=0; i<16; i++){
		memcpy(MB_Y+(i*iCopySizeY), pDec[0], iCopySizeY);//Y
		pDec[0]+=iDecStrideL;
	}
	//U
	for(i=0; i<8; i++){
		memcpy(MB_U+(i*iCopySizeUV), pDec[1], iCopySizeUV);//Y
		pDec[1]+=iDecStrideC;
	}
	//V
	for(i=0; i<8; i++){
		memcpy(MB_V+(i*iCopySizeUV), pDec[2], iCopySizeUV);//Y
		pDec[2]+=iDecStrideC;
	}

	filter.bw = 1;// bandwidth
	filter.n_steps = 9;// steps to be used for saliency map
	filter.lambda = 2.5f;//lambda parameter
	filter.gamma = 0.5f;//gamma parameter
	filter.psi[0] = 0;// phase parameter 1 for real part
	filter.psi[1] = PI/4;// phase parameter 2 for imaginary part
	filter.sigma = filter.lambda/PI*sqrt(log(2.0f)/2.0f)*pow(2.0f,filter.bw+1)/pow(2.0f,filter.bw-1);// sigma parameter

	filter.theta_steps_size = PI/filter.n_steps;//theta steps
	filter.filter_size = 3;//filter size

	filter.filter = (float*)malloc(sizeof(float)*filter.filter_size*filter.filter_size);


	/****************
	 * determine the additional cost of splitting the macroblock into smaller chunks
	 * ******************/
	if(MBtype == MB_TYPE_16x8 || MBtype == MB_TYPE_8x16 || SUBMBtype[0] == SUB_MB_TYPE_8x8 || SUBMBtype[2] == SUB_MB_TYPE_8x8 || SUBMBtype[2] == SUB_MB_TYPE_8x8
			|| SUBMBtype[3] == SUB_MB_TYPE_8x8)
		additionalcost += 1;//equivalent to have double the weight for this MB
	else if(MBtype == MB_TYPE_INTRA8x8 || MBtype == MB_TYPE_8x8 || MBtype == MB_TYPE_8x8_REF0 || SUBMBtype[0] == SUB_MB_TYPE_8x8)
		additionalcost += 2;//equivalent to have triple the weight for this MB
	else if(MBtype == MB_TYPE_INTRA4x4 || SUBMBtype[0] == SUB_MB_TYPE_8x4 || SUBMBtype[0] == SUB_MB_TYPE_4x8)
		additionalcost += 3;//equivalent to have four the weight for this MB
	else if(MBtype == MB_TYPE_INTRA4x4 || SUBMBtype[0] == SUB_MB_TYPE_4x4)
		additionalcost += 4;//equivalent to have five the weight for this MB


	/***************
	 * Estimation of the Gabor like filter for the saliency map computation
	 * *****************/
	if(PRED_TYPE==I16_PRED_H){//0 degrees prediction
		//printf("I16_PRED_H..\t");
		filter.theta_step = 0;// degrees
		filter_extraction(filter);//filter extraction
		if(map_estimation(pCurLayer, filter, MB_Y, MB_U, MB_V, MBSal, &maxval, &minval, &additionalcost))
			goto MAP_ERROR;

		free(filter.filter);
	}
	else if(PRED_TYPE==I16_PRED_V){//90 degrees prediction
		//printf("I16_PRED_V..\t");
		filter.theta_step = 8;// degrees
		filter_extraction(filter);//filter extraction
		if(map_estimation(pCurLayer, filter, MB_Y, MB_U, MB_V, MBSal, &maxval, &minval, &additionalcost))
			goto MAP_ERROR;

		free(filter.filter);
	}
	else if(PRED_TYPE==I16_PRED_P){//45 degrees prediction and 135 degrees
		//printf("I16_PRED_P..\t");
		filter.theta_step = 4;// degrees
		filter_extraction(filter);//filter extraction
		if(map_estimation(pCurLayer, filter, MB_Y, MB_U, MB_V, MBSal, &maxval, &minval, &additionalcost))
			goto MAP_ERROR;

		filter.theta_step = 12;// degrees
		filter_extraction(filter);//filter extraction
		if(map_estimation(pCurLayer, filter, MB_Y, MB_U, MB_V, MBSal, &maxval, &minval, &additionalcost))
			goto MAP_ERROR;

		free(filter.filter);
	}
	else{
		//sweeping
		//printf("SWEEP..\t");
		filter.theta_step = 0;//0
		filter_extraction(filter);//filter extraction
		if(map_estimation(pCurLayer, filter, MB_Y, MB_U, MB_V, MBSal, &maxval, &minval, &additionalcost))
			goto MAP_ERROR;

		filter.theta_step = 4;//45
		filter_extraction(filter);//filter extraction
		if(map_estimation(pCurLayer, filter, MB_Y, MB_U, MB_V, MBSal, &maxval, &minval, &additionalcost))
			goto MAP_ERROR;

		filter.theta_step = 8;//90
		filter_extraction(filter);//filter extraction
		if(map_estimation(pCurLayer, filter, MB_Y, MB_U, MB_V, MBSal, &maxval, &minval, &additionalcost))
			goto MAP_ERROR;

		filter.theta_step = 12;//135
		filter_extraction(filter);//filter extraction
		if(map_estimation(pCurLayer, filter, MB_Y, MB_U, MB_V, MBSal, &maxval, &minval, &additionalcost))
			goto MAP_ERROR;

		free(filter.filter);
	}


	//printf("Min and Max: < %4.2f, %4.2f>\t",minval[0],maxval[0]);

	//printf("<\t");
	range = ffmax(1.0,(maxval-minval));

	//printf("-----------------+++++++++++++++++++++++--------------------------\n");
	//printf("%4.2f, %4.2f, %4.2f, %i, %i, %i\t", maxval, minval, range, filter.filter_size, MBHeight, MBWidth);

	//printf("<<<<<\n");
	for(i=0;i<MBHeight;i++){//y
		for(j=0;j<MBWidth;j++){//x

			if(MBSal[i*MBWidth + j] < 0.001*maxval)
				MBSal[i*MBWidth + j] = 0;
			else{
				MBSal[i*MBWidth + j] = fabs((maxval - MBSal[i*MBWidth + j])/range);
				MBSal[i*MBWidth + j] *= 255.0f;
			}
			//if(MBSal[i*MB_BLOCK_SIZE + j]>2.0)
			//printf("%4.2f\t", MBSal[i*MBWidth + j]);

			*(Spatial_map+j) = (uint8_t)MBSal[i*MBWidth + j];
		}
		Spatial_map+=MAPDecStrideL;//advance according to stride value
		//printf("\n");
	}
	//printf(">>>>>\n\n\n");
	//getchar();


	//printf("cost: %4.5f\t",additionalcost);

	//freeing the pointers used
	free(MB_Y); free(MB_U); free(MB_V); free(MBSal);

	*SPaExst=1;
	return ERR_NONE;//no error encountered during the estimation of the map

MALLOC_ERROR://this will happen if we failed in requesting memory for the saliency maps
	printf("Problem Allocating the memory required for the Saliency Maps, check "
			"SaliencyEstimatorFunctions.cpp..!!\n");
	return ERR_MALLOC_FAILED;

MAP_ERROR://when the program fails in estimating the saliency map in the spatial domain
	printf("Problem estimating the Saliency Map in the spatial domain ..!!\n");
	return ERR_API_FAILED;
}


void filter_extraction(FILTER_CONTAINER filter){
	float *filter_vals = filter.filter;
	float sigma_x = filter.sigma;
	float sigma_y = filter.sigma/filter.gamma;
	float lambda= filter.lambda;
	float *PSI = filter.psi;
	float theta;
	int theta_step = filter.theta_step;
	int f_size = filter.filter_size;
	float x_theta, y_theta;
	int i, j, w, h, off;
	int wsize = f_size/2;// from - to +

	h=0;
	off = 1;
	theta = theta_step*filter.theta_steps_size;
	//printf("Angle: %4.2f\n",theta);
	for(j=-off*wsize; j<=off*wsize; j+=off){
		w=0;
		for(i=-off*wsize; i<=off*wsize; i+=off){
			x_theta = i*cosf(theta) + j*sinf(theta);
			y_theta = -i*sinf(theta) + j*cosf(theta);
			filter_vals[h*f_size + w] = expf(-0.5*((x_theta*x_theta)/(sigma_x*sigma_x) + (y_theta*y_theta)/(sigma_y*sigma_y))) * cosf((2*PI/lambda*x_theta) + PSI[0]);// +
										//expf(-0.5*((x_theta*x_theta)/(sigma_x*sigma_x) + (y_theta*y_theta)/(sigma_y*sigma_y))) * cosf((2*PI/lambda*x_theta) + PSI[1]);

			//printf("%4.4f\t",filter_vals[h*f_size + w]);
			w++;
		}
		//printf("\n");
		h++;
	}
	//printf("\n\n");
	//system("pause");
}



/*Main function in charge of estimating the saliency maps*/
/*map_estimation(pCurLayer, filter, MB_Y, MB_U, MB_V, &MBSal, &maxval, &minval, &additionalcost)*/
int32_t map_estimation(PDqLayer pCurLayer, FILTER_CONTAINER filter, uint8_t *MB_Y, uint8_t *MB_U, uint8_t *MB_V, float *MBSal, float *max_val, float *min_val,
		float *additional_cost){
	float *filter_vals;
	uint8_t *temp_MB_Y, *temp_MB_U, *temp_MB_V;
	float temp;
	int i,j,k,l,h,w;
	int MBHeight = 1<<4;
	int MBWidth = 1<<4;
	int wsize;//size of the filter from - to +

	filter_vals = filter.filter;
	wsize = filter.filter_size/2;

	//padding issue problem that is why we use 2*wsize
	temp_MB_Y = (uint8_t*)malloc(sizeof(uint8_t)*(MBHeight+2*wsize)*(MBWidth+2*wsize));
	temp_MB_U = (uint8_t*)malloc(sizeof(uint8_t)*((MBHeight>>1)+2*wsize)*((MBWidth>>1)+2*wsize));
	temp_MB_V = (uint8_t*)malloc(sizeof(uint8_t)*((MBHeight>>1)+2*wsize)*((MBWidth>>1)+2*wsize));

	int PADWIDTH = MBWidth + 2*wsize, PADHEIGTH = MBHeight + 2*wsize;
	int PADWIDTHC = (MBWidth/2 + 2*wsize);
	int PADHEIGTHC = (MBHeight/2 + 2*wsize);

	k=0;
	int in=0;
	for(i=0;i<PADHEIGTH;i++){//y
		l=0;
		for(j=0;j<PADWIDTH;j++){//x
			if(j<wsize || j>=(PADWIDTH-wsize) || i<wsize || i>=(PADHEIGTH-wsize))
				temp_MB_Y[i*PADWIDTH + j] = 0;
			else{
				temp_MB_Y[i*PADWIDTH + j] = MB_Y[k*MBWidth + l];
				in=1;
				l++;
			}
			//printf("%i\t", temp_MB_Y[i*PADWIDTH + j]);
		}
		if(in){
			k++;
			in=0;
		}
		//printf("\n");
	}
	//printf("\n");

	k=0;
	in=0;
	//printf("%i, %i, %i, %i\t",PADHEIGTHC, PADWIDTHC, PADHEIGTH, PADWIDTH);
	for(i=0;i<PADHEIGTHC;i++){//y
		l=0;
		for(j=0;j<PADWIDTHC;j++){//x
			if(j<wsize || j>=(PADWIDTHC-wsize) || i<wsize || i>=(PADWIDTHC-wsize)){
				temp_MB_U[i*PADWIDTHC + j] = 0;
				temp_MB_V[i*PADWIDTHC + j] = 0;
			}
			else{
				temp_MB_U[i*PADWIDTHC + j] = MB_U[k*(MBWidth>>1) + l];
				temp_MB_V[i*PADWIDTHC + j] = MB_U[k*(MBWidth>>1) + l];
				in=1;
				l++;
			}
			//printf("%i\t", temp_MB_U[i*PADWIDTHC + j]);
		}
		if(in){
			k++;
			in=0;
		}
		//printf("\n");
	}
	//printf("\n");


	if(wsize>MBHeight || wsize>MBWidth || wsize>(MBHeight>>1) || wsize>(MBWidth>>1)){
		printf("The size of the GABOR filter is larger than the MB, please choose a "
				"smaller filter, in order to convolve the MB\n");
		return ERR_INVALID_PARAMETERS;
	}



	//printf("<%i, %i, %i>\t", MBHeight, MBWidth, wsize);

	//printf("<<<<<\n");
	//for Y only Full resolution
	int hh=0, ww;
	for(i=wsize;i<PADHEIGTH-wsize;i++){//y
		ww=0;
		for(j=wsize;j<PADWIDTH-wsize;j++){//x
			temp = 0;
			//convolution per pixel
			h=0;
			for(k=-wsize; k<=wsize; k++){//y
				w=0;
				for(l=-wsize; l<=wsize; l++){//x
					//convolving the pixel for all the Y values
					temp = 0.6*((float)temp_MB_Y[((i+k)*PADWIDTH) + (j+l)] * filter_vals[h*filter.filter_size + w]);//Y-plane full resolution
					//convolving the pixel for all the UV values
					temp += 0.2*(temp_MB_U[(((i/2) + k)*PADWIDTHC) + ((j/2)+l)] * filter_vals[h*filter.filter_size + w]);//U-plane half the resolution of Y
					temp += 0.2*(temp_MB_V[(((i/2) + k)*PADWIDTHC) + ((j/2)+l)] * filter_vals[h*filter.filter_size + w]);//V-plane half the resolution of Y
					w++;
				}
				h++;
			}

			if(temp>0){
				MBSal[hh*MBWidth + ww] += fabs(temp+(*additional_cost));
				//saliency[i*MB_BLOCK_SIZE + j] /= 6;
				//addition of the cost in case the MB has been splitted to several subblocks
				if(MBSal[hh*MBWidth + ww] < min_val[0]){
					min_val[0] = MBSal[hh*MBWidth + ww];
				}
				if(MBSal[hh*MBWidth + ww] > max_val[0])
					max_val[0] = MBSal[hh*MBWidth + ww];
				//printf("%4.2f\t", MBSal[hh*MBWidth + ww]);
				//printf("%i, %i, %i\t", MB_Y[i*MBWidth + j], MB_U[(i/2)*(MBWidth/2) + (j/2)], MB_V[(i/2)*(MBWidth/2) + (j/2)]);
			}else{//Borders
				if(hh==(MBHeight-wsize) && ww<(MBWidth-wsize)){//DOWN
					MBSal[hh*MBWidth + ww] += MBSal[(hh-wsize)*MBWidth + ww];
					//saliency[i*MB_BLOCK_SIZE + j] /= 6;
					//addition of the cost in case the MB has been splitted to several subblocks
					if(MBSal[hh*MBWidth + ww] < min_val[0]){
						min_val[0] = MBSal[hh*MBWidth + ww];
					}
					if(MBSal[hh*MBWidth + ww] > max_val[0])
						max_val[0] = MBSal[hh*MBWidth + ww];
					//printf("%4.2f\t", MBSal[hh*MBWidth + ww]);
					//printf("%i, %i, %i\t", MB_Y[i*MBWidth + j], MB_U[(i/2)*(MBWidth/2) + (j/2)], MB_V[(i/2)*(MBWidth/2) + (j/2)]);
				}else if(ww==(MBWidth-wsize) && hh<(MBHeight-wsize)){//RIGHT
					MBSal[hh*MBWidth + ww] += MBSal[hh*MBWidth + (ww-wsize)];
					//saliency[i*MB_BLOCK_SIZE + j] /= 6;
					//addition of the cost in case the MB has been splitted to several subblocks
					if(MBSal[hh*MBWidth + ww] < min_val[0]){
						min_val[0] = MBSal[hh*MBWidth + ww];
					}
					if(MBSal[hh*MBWidth + ww] > max_val[0])
						max_val[0] = MBSal[hh*MBWidth + ww];
					//printf("%4.2f\t", MBSal[hh*MBWidth + ww]);
					//printf("%i, %i, %i\t", MB_Y[i*MBWidth + j], MB_U[(i/2)*(MBWidth/2) + (j/2)], MB_V[(i/2)*(MBWidth/2) + (j/2)]);
				}else if(ww==(MBWidth-wsize) && hh==(MBHeight-wsize)){//DOWN CORNER RIGHT
					MBSal[hh*MBWidth + ww] += MBSal[(hh-wsize)*MBWidth + (ww-wsize)];
					//saliency[i*MB_BLOCK_SIZE + j] /= 6;
					//addition of the cost in case the MB has been splitted to several subblocks
					if(MBSal[hh*MBWidth + ww] < min_val[0]){
						min_val[0] = MBSal[hh*MBWidth + ww];
					}
					if(MBSal[hh*MBWidth + ww] > max_val[0])
						max_val[0] = MBSal[hh*MBWidth + ww];
					//printf("%4.2f\t", MBSal[hh*MBWidth + ww]);
					//printf("%i, %i, %i\t", MB_Y[i*MBWidth + j], MB_U[(i/2)*(MBWidth/2) + (j/2)], MB_V[(i/2)*(MBWidth/2) + (j/2)]);
				}
			}
			ww++;
		}
		hh++;
		//printf("\n");
	}
	//printf("\n");
	//printf(">>>>>>\n");

	free(temp_MB_Y);
	free(temp_MB_U);
	free(temp_MB_V);
	//free(filter_vals);


	return ERR_NONE;//no problem encountered
}








/**************************************************** TEMPORAL SALIENCY MAP AND FUNCTIONS ******************************************************/
int32_t TSmapEstimation(PWelsDecoderContext pCtx, int8_t *TempExst){
	uint8_t *TempMap, *currLuma, *PrevLuma;
	PDqLayer pCurLayer = pCtx->pCurDqLayer;
	int32_t iMbXy = pCurLayer->iMbXyIndex;
	int32_t i,j;
	int16_t temp;
	float magnitude=0;
	uint32_t MBx = pCurLayer->iMbX, MBy = pCurLayer->iMbY;//starting points of the MBs
	uint32_t MBHeight=1<<4, MBWidth=1<<4;
	uint32_t iDecStrideL = pCurLayer->pDec->iLinesize[0];//stride in the luma
	uint32_t MAPDecStride = pCurLayer->pDec->iWidthInPixel;//line size for Maps

	//maps
	int32_t MAPOffset = (MBx + MBy * MAPDecStride) << 4;//since is a 16x16 MB

	//Luma
	int32_t iOffsetL = (MBx + MBy * iDecStrideL) << 4;//since is a 16x16 MB

	//maps
	TempMap = pCtx->Saliency_Maps.TempMap+MAPOffset;//to be located in the right MB
	PrevLuma = pCtx->Saliency_Maps.prev_frame+MAPOffset;//to be located in the right MB

	//original
	currLuma = pCurLayer->pDec->pData[0]+iOffsetL;//to be located in the right MB


	//printf("<<<<<<<<<\n");
	for (i = 0; i < 16; i++) {
		temp = *(pCurLayer->pMv[0][iMbXy][i]);
		magnitude += ((float)temp*(float)temp);
	  //printf("%i\t", *(pCurLayer->pMv[0][iMbXy][i]));
	}
	magnitude = sqrt(magnitude);
	//printf("\nMag: %4.2f\t",magnitude);
	//printf("\n");
	//printf(">>>>>>>\n");



	if(magnitude>0){
		//printf("to velocity..\t");

		*TempExst= VelocityApprox(pCurLayer, currLuma, PrevLuma, TempMap);
	}

	return ERR_NONE;//no problem encountered
}


int32_t VelocityApprox(PDqLayer pCurLayer, uint8_t *currLuma, uint8_t *PrevLuma, uint8_t *TempMap){
	int32_t iMbXy = pCurLayer->iMbXyIndex;
	int32_t i,j;
	float magnitude=0;
	float alpha = 1.0f;
	uint8_t n_frame=1, iterations=5;
	uint32_t MBx = pCurLayer->iMbX, MBy = pCurLayer->iMbY;//starting points of the MBs
	uint32_t MBHeight=1<<4, MBWidth=1<<4;
	uint32_t iDecStrideL = pCurLayer->pDec->iLinesize[0];//stride in the luma
	uint32_t MAPDecStride = pCurLayer->pDec->iWidthInPixel;//line size for Maps

	uint8_t **cLMB, **pLMB;
	cLMB = (uint8_t**)malloc(sizeof(uint8_t*)*MBHeight);//current luma
	pLMB = (uint8_t**)malloc(sizeof(uint8_t*)*MBHeight);//past luma

	int f,w,h,it;//local variables for the estimation of the optical flow
	float **Ex, **Ey, **Et, *Vx, *Vy, Vxbar, Vybar, temp;//gradient in x, y and temporal between frames


	for(i=0;i<MBHeight;i++){
		cLMB[i] = (uint8_t*)malloc(sizeof(uint8_t)*MBWidth);//current luma
		pLMB[i] = (uint8_t*)malloc(sizeof(uint8_t)*MBWidth);//past luma
		memcpy(cLMB[i], currLuma, sizeof(uint8_t)*MBWidth);
		memcpy(pLMB[i], PrevLuma, sizeof(uint8_t)*MBWidth);
		currLuma+=iDecStrideL;//is according to decoder's stride
		PrevLuma+=MAPDecStride;//is according to our stride
	}

	Vx = (float*)calloc(MBWidth*MBWidth,sizeof(float));// Estimate optical flow in X
	Vy = (float*)calloc(MBWidth*MBWidth,sizeof(float));// Optical flow in Y
	Ex = (float**)malloc(sizeof(float*)*n_frame);// the number of values to be estimated is equal to the number of frames
	Ey = (float**)malloc(sizeof(float*)*n_frame);// the number of values to be estimated is equal to the number of frames
	Et = (float**)malloc(sizeof(float*)*n_frame);// the number of values to be estimated is equal to the number of frames

	for(f=0;f<n_frame;f++){
		Ex[f] = (float*)calloc(MBWidth*MBWidth,sizeof(float));
		Ey[f] = (float*)calloc(MBWidth*MBWidth,sizeof(float));
		Et[f] = (float*)calloc(MBWidth*MBWidth,sizeof(float));


		for(h=1;h<MBHeight-1;h+=4){//omitting the values at the border of the MB
			for(w=1;w<MBWidth-1;w+=4){//omitting values at the border of the MB
				if(f<(n_frame-1)){//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
					/*
						The calculation is as follows: "using the lastly decoded frames - BUT THIS SHOULD BE ANOTHER BUFFER ... TOO MUCH MEMORY"

						1. Ex[y,x,k] = ( ((I[y+1,x+1,k] - I[y+1,x,k]) + (I[y,x+1,k] - I[y,x,k])) + ((I[y+1,x+1,k+1] - I[y+1,x,k+1]) + (I[y,x+1,k+1] - I[y,x,k+1])) ) / 4;

						2. Ey[y,x,k] = ( ((I[y,x,k] - I[y+1,x,k]) + (I[y,x+1,k] - I[y+1,x+1,k])) + ((I[y,x,k+1] - I[y+1,x,k+1]) + (I[y,x+1,k+1] - I[y+1,x+1,k+1])) ) / 4;

						3. Et[y,x,k] = ( ((I[y+1,x,k+1] - I[y+1,x,k]) + (I[y,x,k+1] - I[y,x,k])) + ((I[y+1,x+1,k+1] - I[y+1,x+1,k]) + (I[y,x+1,k+1] - I[y,x+1,k])) ) / 4;
					*/

					//Estimation of the gradient in X
					//*NOT YET THERE IS A PROBLEM WHEN IMPLEMENTING THIS PART ... NEED MORE THINKING*/
				}else{
					/*
						The calculation is as follows: "we use the lastly decoded frame and the one we are decoding now"

						1. Ex[y,x,k] = ( ((I[y+1,x+1,k] - I[y+1,x,k]) + (I[y,x+1,k] - I[y,x,k])) + ((I[y+1,x+1,k+1] - I[y+1,x,k+1]) + (I[y,x+1,k+1] - I[y,x,k+1])) ) / 4;

						2. Ey[y,x,k] = ( ((I[y,x,k] - I[y+1,x,k]) + (I[y,x+1,k] - I[y+1,x+1,k])) + ((I[y,x,k+1] - I[y+1,x,k+1]) + (I[y,x+1,k+1] - I[y+1,x+1,k+1])) ) / 4;

						3. Et[y,x,k] = ( ((I[y+1,x,k+1] - I[y+1,x,k]) + (I[y,x,k+1] - I[y,x,k])) + ((I[y+1,x+1,k+1] - I[y+1,x+1,k]) + (I[y,x+1,k+1] - I[y,x+1,k])) ) / 4;
					*/

					//Estimation of the gradient in X
					Ex[f][h*MBWidth + w] = (
						//current reference frame
						((pLMB[h + 1][w + 1] - pLMB[h + 1][w]) + (pLMB[h][w + 1] - pLMB[h][w]))
						+
						//next reference frame
						((cLMB[h + 1][w + 1] - cLMB[h + 1][w]) + (cLMB[h][w + 1] - cLMB[h][w])) )/4.0f;

					//Estimation of the gradient in Y
					Ey[f][h*MBWidth + w] = (
						//current reference frame
						((pLMB[h][w] - pLMB[h + 1][w]) + (pLMB[h][w + 1] - pLMB[h + 1][w + 1]))
						+
						//next reference frame
						((cLMB[h][w] - cLMB[h + 1][w]) + (cLMB[h][w + 1] - cLMB[h + 1][w + 1])) )/4.0f;

					//Estimation of the gradient interframe
					Et[f][h*MBWidth + w] = (
						//current reference frame
						((cLMB[h + 1][w] - pLMB[h + 1][w]) + (cLMB[h][w] - pLMB[h][w]))
						+
						//next reference frame
						((cLMB[h + 1][w + 1] - pLMB[h + 1][w + 1]) + (cLMB[h][w + 1] - pLMB[h][w + 1])) )/4.0f;
				}
			}
		}


		/*
			*******************************************************************************************************************************************************************
			Estimation of the Ex, Ey, Et for the MB
		*/

		for(it=0;it<iterations;it++){//iterative search
			for(h=1;h<MBHeight-1;h+=4){//omitting the values at the border of the MB
				for(w=1;w<MBWidth-1;w+=4){//omitting values at the border of the MB

					//Estimation of the mean for the X direction
					Vxbar = ((Vx[(h-1)*MBWidth + w] + Vx[h*MBWidth + (w+1)] + Vx[(h+1)*MBWidth + w] + Vx[h*MBWidth + (w-1)])/6.0f) +
						((Vx[(h-1)*MBWidth + (w-1)] + Vx[(h-1)*MBWidth + (w+1)] + Vx[(h+1)*MBWidth + (w+1)] + Vx[(h+1)*MBWidth + (w-1)])/12.0f);

					//Estimation of the mean for the X direction
					Vybar = ((Vy[(h-1)*MBWidth + w] + Vy[h*MBWidth + (w+1)] + Vy[(h+1)*MBWidth + w] + Vy[h*MBWidth + (w-1)])/6.0f) +
						((Vy[(h-1)*MBWidth + (w-1)] + Vy[(h-1)*MBWidth + (w+1)] + Vy[(h+1)*MBWidth + (w+1)] + Vy[(h+1)*MBWidth + (w-1)])/12.0f);

					//Horn's estimation of the velocities
					temp = (Ex[f][h*MBWidth + w]*Vxbar + Ey[f][h*MBWidth + w]*Vybar + Et[f][h*MBWidth + w]) /
						(alpha*alpha + Ex[f][h*MBWidth + w]*Ex[f][h*MBWidth + w] + Ey[f][h*MBWidth + w]*Ey[f][h*MBWidth + w]);

					//updating the velocities iteratively
					Vx[h*MBWidth + w] = Vxbar - Ex[f][h*MBWidth + w]*temp;
					Vy[h*MBWidth + w] = Vybar - Ey[f][h*MBWidth + w]*temp;
				}
			}
		}

		free(Ex[f]);
		free(Ey[f]);
		free(Et[f]);
	}

	//Estimation of the magnitude for the velocities at this particular MB
	temp = 0;
	for(h=1;h<MBHeight-1;h+=4){//omitting the values at the border of the MB
		for(w=1;w<MBWidth-1;w+=4){//omitting values at the border of the MB
			temp += (float)(Vx[h*MBWidth + w]*Vx[h*MBWidth + w] + Vy[h*MBWidth + w]*Vy[h*MBWidth + w]);
		}
	}

	for(h=0;h<MBHeight;h++){//assigning the value to the MB
		for(w=0;w<MBWidth;w++){
			if(temp>30)
				*(TempMap+w) = (int8_t)temp;
			else
				*(TempMap+w) = 0;
		}
		TempMap+=MAPDecStride;//moving according to stride
	}

	//free the containers
	free(Ex); free(Ey); free(Et); free(Vx); free(Vy);

	for(i=0;i<MBHeight;i++){
		free(cLMB[i]); free(pLMB[i]);
	}
	free(cLMB); free(pLMB);

	if(temp>10)
		return 1;//there is a map
	else
		return 0;//there is no a map
}









/**************************************************** FUSION SALIENCY MAP AND FUNCTIONS ******************************************************/
int32_t MapsFusion(PWelsDecoderContext pCtx){
	uint8_t *spatial, *temporal, *fusion, *buff, *votingframe;
	uint64_t *votes;
	PDqLayer pCurLayer = pCtx->pCurDqLayer;
	int32_t iMbXy = pCurLayer->iMbXyIndex;
	int32_t buff_index = pCtx->Saliency_Maps.prev_map_index;
	int32_t i,j;
	int16_t temp;
	uint32_t MBx = pCurLayer->iMbX, MBy = pCurLayer->iMbY;//starting points of the MBs
	uint32_t MBHeight=1<<4, MBWidth=1<<4;
	uint32_t MAPDecStride = pCurLayer->pDec->iWidthInPixel;//line size for Maps
	uint32_t n_IMBs=pCtx->n_IMBs, n_PMBs=pCtx->n_PMBs, nmb_MB= n_IMBs+n_PMBs;
	uint8_t MB_type = pCurLayer->pMbType[iMbXy];//MB type
	uint64_t currBitRate = pCurLayer->pBitStringAux->iBits;//total num of bits in this NAL
	uint64_t maxbrightness = -1, maxfusb=-1;
	float a, b, c;
	float temp2;
	float weight=1.0;
	float *TOTAL_SAL;

	//maps
	int32_t MAPOffset = (MBx + MBy * MAPDecStride) << 4;//since is a 16x16 MB

	//implementing offset
	spatial = pCtx->Saliency_Maps.SpaMap + MAPOffset;
	temporal = pCtx->Saliency_Maps.TempMap + MAPOffset;
	fusion = pCtx->Saliency_Maps.MasterMap + MAPOffset;
	buff = pCtx->Saliency_Maps.Buff_MasterMap[pCtx->Saliency_Maps.prev_map_index] + MAPOffset;
	votes = pCtx->Saliency_Maps.voting + MAPOffset;
	votingframe = pCtx->Saliency_Maps.VotingFrames + MAPOffset;
	TOTAL_SAL = &(pCtx->Frame_saliency);


	//printf("%i, %i\t", pCtx->prevbirate , currBitRate);

	//printf("<Is: %i, Ps: %i>\t", n_IMBs, n_PMBs);
	if(nmb_MB>0){//check total number of Mbs
		//Weight for spatial map ... I frames
		a = ((float)n_IMBs/(4.0f*nmb_MB) + 0.083f)*weight;//
		//Weight for temporal map ... P frames
		b = ((float)(n_PMBs)/(4.0f*nmb_MB) + 0.083f)*weight;//
	}

	c = (float)(pCurLayer->pDec->iHeightInPixel * pCurLayer->pDec->iWidthInPixel) * 8;//bits in this frame (CBR)
	if(MB_type>0 && MB_type<5){
		//I-MB
		c = (((fabs((float)pCtx->prevbirate - (float)currBitRate)/c) + (0.125f)) * (0.25f) + (0.083f))*weight;//I frames
	}else if(MB_type>5 && MB_type<16){
		//P-MB
		c = (((fabs((float)pCtx->prevbirate - (float)currBitRate)/c) + (0.25f)) * (0.25f) + (0.083f))*weight;//P frames
	}
	else{
		printf("MB type is UNKNOWN..\t");
	}

	//printf("< %4.5f, %4.5f, %4.5f>\t", a, b, c);

	//Fusion
	//printf("<<<<<<<<<\n");
	for (i = 0; i < MBHeight; i++) {
		for (j = 0; j < MBWidth; j++) {
			temp2 = (*(buff+j))*c;//taking into account the previous saliency maps
			temp2 += (*(temporal+j))*b;
			if((*(temporal+j))>30){
				temp2 += (*(spatial+j))*a;
				(*TOTAL_SAL) += temp2*1.5;
				*(fusion+(i*MAPDecStride+j)) = (uint8_t)ffmin(255.0,(temp2*1.5)+0.5);
				(*(votes+(i*MAPDecStride+j)))+=10;//increasing votes
			}
			else{
				*(fusion+(i*MAPDecStride+j)) = 0;
				if((*(votes+(i*MAPDecStride+j)))>0){
					(*(votes+(i*MAPDecStride+j)))-=1;//reducing votes
				}
			}
			maxfusb = ffmax(255.0, *(fusion+(i*MAPDecStride+j)));
			maxbrightness = int64max(maxbrightness, (*(votes+(i*MAPDecStride+j))));
			//printf("<%i, %i, %i, %i, %4.5f>\t", *(fusion+j), (*(buff+j)), (*(spatial+j)), (*(temporal+j)), (temp2*1.5));
		}

		//fusion += MAPDecStride;//moving according to stride
		buff += MAPDecStride;//moving according to stride
		temporal += MAPDecStride;//moving according to stride
		spatial += MAPDecStride;//moving according to stride
		//votes += MAPDecStride;
		//printf("\n");
	}
	//printf("Max: %i\n", maxbrightness);
	//printf("\n");
	//printf(">>>>>>>\n");

	//for visualization of the voting
	for (i = 0; i < MBHeight; i++) {
		for (j = 0; j < MBWidth; j++) {
			(*(votingframe+j)) = (uint8_t)((((float)(*(votes+j)))/((float)maxbrightness + 1))*255.0f);
			(*(fusion+j)) = (uint8_t)((((float)(*(fusion+j)))/((float)maxfusb + 1))*255.0f);
		}
		votes += MAPDecStride;
		votingframe += MAPDecStride;
		fusion += MAPDecStride;
	}

	return ERR_NONE;//no problem encountered
}

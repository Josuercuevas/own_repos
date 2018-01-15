/*
 * MomentNormalizer.c
 *
 *  Created on: Jan 7, 2015
 *      Author: josue
 *
 *      Main entry source code to use the normalizer API, we need to make sure we
 *      can handle all errors for user friendliness
 */

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/video/gstvideofilter.h>
#include "includes/MomentNormalizer.h"
#include "includes/Normalizer.h"

GST_DEBUG_CATEGORY_STATIC (gst_momentnormalization_debug_category2);
#define GST_CAT_DEFAULT gst_momentnormalization_debug_category2

void normalizer_debug_init(){
	GST_DEBUG_CATEGORY_INIT (gst_momentnormalization_debug_category2, "Normalizer", 0,
					"debug category for Normalizer part");
}


#define max_val(a,b) \
  ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
    _a > _b ? _a : _b; })

#define min_val(a,b) \
  ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
    _a < _b ? _a : _b; })

uint8_t Normalization_error_handler(uint8_t ierror){
	switch(ierror){
		case normalization_success:
			GST_DEBUG("Normalization finished the processing successfully..!!\n\n");
			break;
		case normalization_nobinary:
			GST_ERROR("The image has not been binarized, please make sure to input a binary image..!!\n\n");
			break;
		case normalization_mallocerror:
			GST_ERROR("Error allocating memory with MALLOC...\n\n");
			break;
		default:
			GST_ERROR("Unknown Error code ..!!\n\n");
			break;
	}

	return ierror;
}


uint8_t perform_normalization(GstMomentNormalization *normalizer){
	uint8_t ierror = normalization_success;
	uint32_t i,j,k;
	gfloat accum, accum2;
	gfloat diffmean, diffmean2;
	double **moments=NULL;//always 9 moments to estimate per patch

	//determine if we have a binary image
	if(normalizer->channels>1 || normalizer->channels<1){
		return normalization_nobinary;
	}

	//malloc function for patch
	moments = (double**)g_malloc(sizeof(double*)*normalizer->n_blobs);//how nany patch we need to process
	for(i=0;i<normalizer->n_blobs;i++){
		PATCH_PADDING_X[i] =  PATCH_PADDING_X[i]-normalizer->blobs[i].width;
		if(PATCH_PADDING_X[i]<=0){
			GST_WARNING("Pad_X = %i is not enough... leaving with the current width = %i..\n", PATCH_PADDING_X[i], normalizer->blobs[i].width);
			PATCH_PADDING_X[i] = 0;
		}

		PATCH_PADDING_Y[i] =  PATCH_PADDING_Y[i]-normalizer->blobs[i].height;
		if(PATCH_PADDING_Y[i]<=0){
			GST_WARNING("Pad_Y = %i is not enough... leaving with the current height = %i..\n", PATCH_PADDING_Y[i], normalizer->blobs[i].height);
			PATCH_PADDING_Y[i]=0;
		}

		moments[i] = (double*)g_malloc0(sizeof(double)*10);//always 7 moments + two central moments + and AREA
		normalizer->blobs[i].patch = NULL;
		normalizer->blobs[i].patch = (uint8_t*)g_malloc0((normalizer->blobs[i].height+PATCH_PADDING_Y[i])*(normalizer->blobs[i].width+PATCH_PADDING_X[i])*sizeof(uint8_t));
		normalizer->blobs[i].transformed = (uint8_t*)g_malloc0((normalizer->blobs[i].height+PATCH_PADDING_Y[i])*(normalizer->blobs[i].width+PATCH_PADDING_X[i])*sizeof(uint8_t));
		if(normalizer->blobs[i].patch == NULL){
			return normalization_mallocerror;
		}
	}


	//read patches
	GST_DEBUG_OBJECT(normalizer, "-------------- Copy patches --------------..\n");
	ierror = copy_patches(normalizer);

	//process each patch
	/*
	 * Each individual patch has to be normalized where the size of
	 * it may change therefore when compared with benchmark patches we need to be careful
	 * */
	accum = 0;
	accum2=0;
	for(i=0;i<normalizer->n_blobs;i++){
		//moments estimations
		GST_DEBUG_OBJECT(normalizer, "-------------- Estimate moments-------------- ..\n");
		ierror = Patch_Moments(&(normalizer->blobs[i]), moments[i], i);
		if(ierror!=normalization_success)
			return ierror;
		normalizer->blobs[i].moments = moments[i];

		//Eigen estimation
		GST_DEBUG_OBJECT(normalizer, "-------------- Calculating Eigen values and vectors --------------..\n");
		ierror = Patch_Eigen(&(normalizer->blobs[i]));
		if(ierror!=normalization_success)
			return ierror;

		//Compactification matrix
		GST_DEBUG_OBJECT(normalizer, "-------------- Compactification process --------------..\n");
		ierror = Patch_Compactification(&(normalizer->blobs[i]), i);
		if(ierror!=normalization_success)
			return ierror;

		//Tensor Values
		GST_DEBUG_OBJECT(normalizer, "-------------- Tensor estimation --------------..\n");
		ierror = Patch_Tensor_angle(&(normalizer->blobs[i]));
		if(ierror!=normalization_success)
			return ierror;

		//Normalization
		GST_DEBUG_OBJECT(normalizer, "-------------- Normalization of compact image --------------..\n");
		ierror = Patch_Normalization(&(normalizer->blobs[i]), i);
		if(ierror!=normalization_success)
			return ierror;

		//we dont need anymore
		g_free(moments[i]);
		g_free(normalizer->blobs[i].patch);
		normalizer->blobs[i].patch = NULL;
		normalizer->blobs[i].moments = NULL;
	}

	GST_DEBUG_OBJECT(normalizer, "-------------- Estimating difference per blob --------------..\n");
	ierror = frames_diff(normalizer);
	if(ierror!=normalization_success)
		return ierror;



	GST_DEBUG_OBJECT(normalizer, "-------------- Copying transformed frames --------------..\n");
	put_blobs_in_outbuffer(normalizer);

	return ierror;//returns error code if any
}


uint8_t frames_diff(GstMomentNormalization *normalizer){
	guint8 i,j,k;
	guint32 blob_size;

	if(!normalizer->any_blob){
		/*
		 * perhaps it is the first frame to be processed by the element
		 * so we create the memory which is to be dynamic since the number of blobs to be
		 * processed varies according to the blob detector
		 * */
		normalizer->Prev_Norm_Blobs=NULL;
		normalizer->Prev_Norm_Blobs = (PreviousBlobs*)g_malloc(normalizer->n_blobs*sizeof(PreviousBlobs));

		if(normalizer->Prev_Norm_Blobs==NULL){
			GST_ERROR_OBJECT(normalizer, "Memory for the Previous buffers was not created.. stopping pipeline...!!");
			return normalization_mallocerror;
		}

		normalizer->n_prev_blobs = normalizer->n_blobs;/* n_previous blobs */
		for(i=0;i<normalizer->n_blobs;i++){
			/*
			 * Creating the container for the data contained on each normalized blob
			 * */
			//WxH
			normalizer->Prev_Norm_Blobs[i].height = normalizer->blobs[i].height+PATCH_PADDING_Y[i];
			normalizer->Prev_Norm_Blobs[i].width = normalizer->blobs[i].width+PATCH_PADDING_X[i];

			//PADX x PADy
			normalizer->Prev_Norm_Blobs[i].paddinx = PATCH_PADDING_X[i];
			normalizer->Prev_Norm_Blobs[i].paddiny = PATCH_PADDING_Y[i];


			blob_size = (normalizer->blobs[i].height+PATCH_PADDING_Y[i])*(normalizer->blobs[i].width+PATCH_PADDING_X[i]);
			normalizer->Prev_Norm_Blobs[i].NormalizedBlob = (guint8*)g_malloc(blob_size*sizeof(guint8));
			/*
			 * Making a copy of the data
			 * */
			memcpy(normalizer->Prev_Norm_Blobs[i].NormalizedBlob, normalizer->blobs[i].transformed, blob_size*sizeof(guint8));
		}

		/*
		 * Activating the flag
		 * */
		normalizer->n_diff_blobs = 1;
		normalizer->any_blob = TRUE;
	}else{
		/*
		 * We already have some blobs in store, therefore we can estimate the difference and make sure
		 * if there is any motion, first check the union and intersection of every blob with
		 * every previously stored one, remember they are all normalized therefore size is not an
		 * issue here. They all have the same size
		 * */
		guint32 BBxmin[15], BBymin[15], BBymax[15], BBxmax[15];
		normalizer->diff_acumm_skin=0.f;
		normalizer->diff_acumm_nonskin=0.f;
		normalizer->diff_count=0;
		normalizer->diff_count_nonskin=0;
		guint8 *unions=NULL, *intersections=NULL;
		gboolean memory_created=FALSE;
		guint8** temp_diffs=NULL;
		gfloat *correspondances_vals=NULL;
		guint32 curr_Height, curr_Width, curr_Bsize, pos;
		guint32 old_H, old_W;

		gfloat T_unions, T_inter, T_ratio;
		gint *correspondence_loc;
		guint x,y;
		guint minX=9999999, maxX=0, minY=9999999, maxY=0;

		correspondence_loc = (gint*)g_malloc(sizeof(gint)*min_val((normalizer->n_blobs), (normalizer->n_prev_blobs))*2);
		temp_diffs = (guint8**)g_malloc(sizeof(guint8*)*min_val((normalizer->n_blobs), (normalizer->n_prev_blobs)));
		correspondances_vals = (gfloat*)g_malloc(((normalizer->n_blobs)*(normalizer->n_prev_blobs))*sizeof(gfloat));

		if(temp_diffs==NULL || correspondances_vals==NULL){
			GST_ERROR_OBJECT(normalizer, "Memory for temporal difference estimator was not created.. stopping pipeline...!!");
			return normalization_mallocerror;
		}



		for(i=0;i<normalizer->n_blobs;i++){//current blobs
			for(j=0;j<normalizer->n_prev_blobs;j++){//previous blobs
				BBxmin[j]=99999, BBymin[j]=99999, BBymax[j]=0, BBxmax[j]=0;
				T_unions=0.f, T_inter=0.f, T_ratio;
				curr_Height = min_val(normalizer->Prev_Norm_Blobs[j].height, normalizer->blobs[i].height+PATCH_PADDING_Y[i]);
				curr_Width = min_val(normalizer->Prev_Norm_Blobs[j].width, normalizer->blobs[i].width+PATCH_PADDING_X[i]);
				curr_Bsize = curr_Height*curr_Width;
				pos=0;
				if(!memory_created){
					unions = (guint8*)g_malloc(sizeof(guint8)*curr_Bsize);
					intersections = (guint8*)g_malloc(sizeof(guint8)*curr_Bsize);
					if(unions==NULL || intersections==NULL){
						GST_ERROR_OBJECT(normalizer, "Cannot allocate memory for the binary operators.. stopping pipeline...!!");
						return normalization_mallocerror;
					}
					old_H = curr_Height; old_W = curr_Width;
					memory_created = TRUE;
				}else if(old_H!=curr_Height || old_W!=curr_Width){
					/* reallocating memory*/
					unions = (guint8*)g_realloc(unions, sizeof(guint8)*curr_Bsize);
					intersections = (guint8*)g_realloc(intersections, sizeof(guint8)*curr_Bsize);
					if(unions==NULL || intersections==NULL){
						GST_ERROR_OBJECT(normalizer, "Cannot allocate memory for the binary operators.. stopping pipeline...!!");
						return normalization_mallocerror;
					}
					old_H = curr_Height; old_W = curr_Width;
				}

				/*
				 * now we proceed to estimate intersection and union of blobs
				 * */
				while(pos<curr_Bsize){
					/************************* UNROLLING ************************/
					//pixel 1
					if((*(normalizer->Prev_Norm_Blobs[j].NormalizedBlob+pos)) || (*(normalizer->blobs[i].transformed+pos))){
						*(unions+pos) = (*(normalizer->Prev_Norm_Blobs[j].NormalizedBlob+pos)) | (*(normalizer->blobs[i].transformed+pos));
						*(intersections+pos) = (*(normalizer->Prev_Norm_Blobs[j].NormalizedBlob+pos)) & (*(normalizer->blobs[i].transformed+pos));
						T_unions += *(unions+pos); T_inter += *(intersections+pos);
						if(*(normalizer->blobs[i].transformed+pos)){
							x = pos%curr_Width; y = pos/curr_Width;
							BBxmin[j] = x<BBxmin[j] ? x : BBxmin[j];//x-min
							BBymin[j] = y<BBymin[j] ? y : BBymin[j];//y-min
							BBxmax[j] = x>BBxmax[j] ? x : BBxmax[j];//x-max
							BBymax[j] = y>BBymax[j] ? y : BBymax[j];//y-max
						}
					}
					pos++;

					//pixel 2
					if((*(normalizer->Prev_Norm_Blobs[j].NormalizedBlob+pos)) || (*(normalizer->blobs[i].transformed+pos))){
						*(unions+pos) = (*(normalizer->Prev_Norm_Blobs[j].NormalizedBlob+pos)) | (*(normalizer->blobs[i].transformed+pos));
						*(intersections+pos) = (*(normalizer->Prev_Norm_Blobs[j].NormalizedBlob+pos)) & (*(normalizer->blobs[i].transformed+pos));
						T_unions += *(unions+pos); T_inter += *(intersections+pos);
						if(*(normalizer->blobs[i].transformed+pos)){
							x = pos%curr_Width; y = pos/curr_Width;
							BBxmin[j] = x<BBxmin[j] ? x : BBxmin[j];//x-min
							BBymin[j] = y<BBymin[j] ? y : BBymin[j];//y-min
							BBxmax[j] = x>BBxmax[j] ? x : BBxmax[j];//x-max
							BBymax[j] = y>BBymax[j] ? y : BBymax[j];//y-max
						}
					}
					pos++;

					//pixel 3
					if((*(normalizer->Prev_Norm_Blobs[j].NormalizedBlob+pos)) || (*(normalizer->blobs[i].transformed+pos))){
						*(unions+pos) = (*(normalizer->Prev_Norm_Blobs[j].NormalizedBlob+pos)) | (*(normalizer->blobs[i].transformed+pos));
						*(intersections+pos) = (*(normalizer->Prev_Norm_Blobs[j].NormalizedBlob+pos)) & (*(normalizer->blobs[i].transformed+pos));
						T_unions += *(unions+pos); T_inter += *(intersections+pos);
						if(*(normalizer->blobs[i].transformed+pos)){
							x = pos%curr_Width; y = pos/curr_Width;
							BBxmin[j] = x<BBxmin[j] ? x : BBxmin[j];//x-min
							BBymin[j] = y<BBymin[j] ? y : BBymin[j];//y-min
							BBxmax[j] = x>BBxmax[j] ? x : BBxmax[j];//x-max
							BBymax[j] = y>BBymax[j] ? y : BBymax[j];//y-max
						}
					}
					pos++;

					//pixel 4
					if((*(normalizer->Prev_Norm_Blobs[j].NormalizedBlob+pos)) || (*(normalizer->blobs[i].transformed+pos))){
						*(unions+pos) = (*(normalizer->Prev_Norm_Blobs[j].NormalizedBlob+pos)) | (*(normalizer->blobs[i].transformed+pos));
						*(intersections+pos) = (*(normalizer->Prev_Norm_Blobs[j].NormalizedBlob+pos)) & (*(normalizer->blobs[i].transformed+pos));
						T_unions += *(unions+pos); T_inter += *(intersections+pos);
						if(*(normalizer->blobs[i].transformed+pos)){
							x = pos%curr_Width; y = pos/curr_Width;
							BBxmin[j] = x<BBxmin[j] ? x : BBxmin[j];//x-min
							BBymin[j] = y<BBymin[j] ? y : BBymin[j];//y-min
							BBxmax[j] = x>BBxmax[j] ? x : BBxmax[j];//x-max
							BBymax[j] = y>BBymax[j] ? y : BBymax[j];//y-max
						}
					}
					pos++;

					//pixel 5
					if((*(normalizer->Prev_Norm_Blobs[j].NormalizedBlob+pos)) || (*(normalizer->blobs[i].transformed+pos))){
						*(unions+pos) = (*(normalizer->Prev_Norm_Blobs[j].NormalizedBlob+pos)) | (*(normalizer->blobs[i].transformed+pos));
						*(intersections+pos) = (*(normalizer->Prev_Norm_Blobs[j].NormalizedBlob+pos)) & (*(normalizer->blobs[i].transformed+pos));
						T_unions += *(unions+pos); T_inter += *(intersections+pos);
						if(*(normalizer->blobs[i].transformed+pos)){
							x = pos%curr_Width; y = pos/curr_Width;
							BBxmin[j] = x<BBxmin[j] ? x : BBxmin[j];//x-min
							BBymin[j] = y<BBymin[j] ? y : BBymin[j];//y-min
							BBxmax[j] = x>BBxmax[j] ? x : BBxmax[j];//x-max
							BBymax[j] = y>BBymax[j] ? y : BBymax[j];//y-max
						}
					}
					pos++;

					//pixel 6
					if((*(normalizer->Prev_Norm_Blobs[j].NormalizedBlob+pos)) || (*(normalizer->blobs[i].transformed+pos))){
						*(unions+pos) = (*(normalizer->Prev_Norm_Blobs[j].NormalizedBlob+pos)) | (*(normalizer->blobs[i].transformed+pos));
						*(intersections+pos) = (*(normalizer->Prev_Norm_Blobs[j].NormalizedBlob+pos)) & (*(normalizer->blobs[i].transformed+pos));
						T_unions += *(unions+pos); T_inter += *(intersections+pos);
						if(*(normalizer->blobs[i].transformed+pos)){
							x = pos%curr_Width; y = pos/curr_Width;
							BBxmin[j] = x<BBxmin[j] ? x : BBxmin[j];//x-min
							BBymin[j] = y<BBymin[j] ? y : BBymin[j];//y-min
							BBxmax[j] = x>BBxmax[j] ? x : BBxmax[j];//x-max
							BBymax[j] = y>BBymax[j] ? y : BBymax[j];//y-max
						}
					}
					pos++;

					//pixel 7
					if((*(normalizer->Prev_Norm_Blobs[j].NormalizedBlob+pos)) || (*(normalizer->blobs[i].transformed+pos))){
						*(unions+pos) = (*(normalizer->Prev_Norm_Blobs[j].NormalizedBlob+pos)) | (*(normalizer->blobs[i].transformed+pos));
						*(intersections+pos) = (*(normalizer->Prev_Norm_Blobs[j].NormalizedBlob+pos)) & (*(normalizer->blobs[i].transformed+pos));
						T_unions += *(unions+pos); T_inter += *(intersections+pos);
						if(*(normalizer->blobs[i].transformed+pos)){
							x = pos%curr_Width; y = pos/curr_Width;
							BBxmin[j] = x<BBxmin[j] ? x : BBxmin[j];//x-min
							BBymin[j] = y<BBymin[j] ? y : BBymin[j];//y-min
							BBxmax[j] = x>BBxmax[j] ? x : BBxmax[j];//x-max
							BBymax[j] = y>BBymax[j] ? y : BBymax[j];//y-max
						}
					}
					pos++;

					//pixel 8
					if((*(normalizer->Prev_Norm_Blobs[j].NormalizedBlob+pos)) || (*(normalizer->blobs[i].transformed+pos))){
						*(unions+pos) = (*(normalizer->Prev_Norm_Blobs[j].NormalizedBlob+pos)) | (*(normalizer->blobs[i].transformed+pos));
						*(intersections+pos) = (*(normalizer->Prev_Norm_Blobs[j].NormalizedBlob+pos)) & (*(normalizer->blobs[i].transformed+pos));
						T_unions += *(unions+pos); T_inter += *(intersections+pos);
						if(*(normalizer->blobs[i].transformed+pos)){
							x = pos%curr_Width; y = pos/curr_Width;
							BBxmin[j] = x<BBxmin[j] ? x : BBxmin[j];//x-min
							BBymin[j] = y<BBymin[j] ? y : BBymin[j];//y-min
							BBxmax[j] = x>BBxmax[j] ? x : BBxmax[j];//x-max
							BBymax[j] = y>BBymax[j] ? y : BBymax[j];//y-max
						}
					}
					pos++;
					/************************* UNROLLING ************************/
				}

				/*
				 * Determine rations and correspondence
				 * */
				T_ratio = min_val(T_unions,T_inter)/max_val(T_unions,T_inter);//min/max make sure is <=1
				correspondances_vals[i*(normalizer->n_prev_blobs) + j] = T_ratio;
			}//loop for previous blobs

			/*
			 * we have one memory buffer left
			 * */
			g_free(unions);
			g_free(intersections);
			memory_created = FALSE;
			unions=NULL; intersections=NULL;

		}//loop for current blobs

		/*
		 * The number of blobs is equal to the minimum amount of blobs on the two comparing frames used for
		 * difference estimation
		 * */
		for(i=0;i<min_val((normalizer->n_blobs), (normalizer->n_prev_blobs));i++){
			correspondence_loc[i*2] = -1;
			correspondence_loc[i*2+1] = -1;
			//finding minimum
			gfloat max_ratio=-1.f;
			for(j=0;j<(normalizer->n_blobs);j++){//current frame
				for(k=0;k<(normalizer->n_prev_blobs);k++){//previous frame
					if(max_ratio<correspondances_vals[j*(normalizer->n_prev_blobs) + k]){
						gboolean considered=FALSE;
						gint t;
						for(t=0;t<i;t++){
							if(j==correspondence_loc[t*2] || k==correspondence_loc[t*2+1]){
								considered=TRUE;
							}
						}

						if(!considered){
							max_ratio = correspondances_vals[j*(normalizer->n_prev_blobs) + k];
							/*two blobs correspondence*/
							correspondence_loc[i*2] = j;
							correspondence_loc[i*2 + 1] = k;
						}
					}
				}
			}


			/*
			 * Rejecting similarities smaller than a threshold T, the reason is in case a blob from previous frame and
			 * current are very different, and they do not match at all
			 * */
			if(correspondence_loc[i*2]<0 && correspondence_loc[i*2+1]<0){
				GST_WARNING_OBJECT(normalizer, "No more blobs to select from.");
				temp_diffs[i] = NULL;//since we did not estimate it we signal that is empty
				break;;//early break
			}


			GST_DEBUG_OBJECT(normalizer, "Correspondence is %d (curr) ---> %d (prev) with similarity ratio of: %f, "
					"(Curr_B_size:<x: %d, y: %d>, Prev_B_size<x: %d, y: %d>)", correspondence_loc[i*2], correspondence_loc[i*2+1],
					correspondances_vals[correspondence_loc[i*2]*(normalizer->n_prev_blobs) + correspondence_loc[i*2+1]],
					normalizer->blobs[correspondence_loc[i*2]].width, normalizer->blobs[correspondence_loc[i*2]].height,
					normalizer->Prev_Norm_Blobs[correspondence_loc[i*2+1]].width - normalizer->Prev_Norm_Blobs[correspondence_loc[i*2+1]].paddinx,
					normalizer->Prev_Norm_Blobs[correspondence_loc[i*2+1]].height - normalizer->Prev_Norm_Blobs[correspondence_loc[i*2+1]].paddiny);


			/*
			 * Rejecting similarities smaller than a threshold T, the reason is in case a blob from previous frame and
			 * current are very different, and they do not match at all
			 * */
			if(correspondances_vals[correspondence_loc[i*2]*(normalizer->n_prev_blobs) + correspondence_loc[i*2+1]] < 0.35){
				GST_WARNING_OBJECT(normalizer, "Similarity thresholds (T: %f) will REJECT these paring blobs. (S: %f <Curr_B: %d, Prev_B: %d>)", 0.35,
						correspondances_vals[correspondence_loc[i*2]*(normalizer->n_prev_blobs) + correspondence_loc[i*2+1]],
						correspondence_loc[i*2], correspondence_loc[i*2+1]);
				temp_diffs[i] = NULL;//since we did not estimate it we signal that is empty
				continue;//jump to next blob
			}


			 /*
			 * Estimate differences, which is to be stored in the transformed blob to later be
			 * output when sinking the blobs
			 * */
			curr_Height = min_val(normalizer->blobs[correspondence_loc[i*2]].height+PATCH_PADDING_Y[correspondence_loc[i*2]],
					normalizer->Prev_Norm_Blobs[correspondence_loc[i*2+1]].height);
			curr_Width = min_val(normalizer->blobs[correspondence_loc[i*2]].width+PATCH_PADDING_X[correspondence_loc[i*2]],
					normalizer->Prev_Norm_Blobs[correspondence_loc[i*2+1]].width);
			curr_Bsize = curr_Height*curr_Width;

			pos=0;
			temp_diffs[i] = (guint8*)g_malloc0(sizeof(guint8)*curr_Bsize);
			while(pos<curr_Bsize){
				/* UNROLLING */
				//pixel 1
				if((*(normalizer->Prev_Norm_Blobs[correspondence_loc[i*2+1]].NormalizedBlob+pos)) ||
						(*(normalizer->blobs[correspondence_loc[i*2]].transformed+pos))){
					if( (*(temp_diffs[i]+pos) = ((*(normalizer->Prev_Norm_Blobs[correspondence_loc[i*2+1]].NormalizedBlob+pos))
							!= (*(normalizer->blobs[correspondence_loc[i*2]].transformed+pos)))) ){
						x = pos%curr_Width; y = pos/curr_Width;
						normalizer->diff_count++;
						if( x > BBxmin[correspondence_loc[i*2+1]] && x < BBxmax[correspondence_loc[i*2+1]] &&
								y > BBymin[correspondence_loc[i*2+1]] && y < BBymax[correspondence_loc[i*2+1]] &&
								!(*(normalizer->blobs[correspondence_loc[i*2]].transformed+pos))){
							normalizer->diff_count_nonskin++;
						}

						minX = x < minX ? x : minX;
						maxX = x > maxX ? x : maxX;
						minY = y < minY ? y : minY;
						maxY = y > maxY ? y : maxY;
					}
				}
				pos++;

				//pixel 2
				if((*(normalizer->Prev_Norm_Blobs[correspondence_loc[i*2+1]].NormalizedBlob+pos)) ||
						(*(normalizer->blobs[correspondence_loc[i*2]].transformed+pos))){
					if( (*(temp_diffs[i]+pos) = ((*(normalizer->Prev_Norm_Blobs[correspondence_loc[i*2+1]].NormalizedBlob+pos))
							!= (*(normalizer->blobs[correspondence_loc[i*2]].transformed+pos)))) ){
						x = pos%curr_Width; y = pos/curr_Width;
						normalizer->diff_count++;
						if( x > BBxmin[correspondence_loc[i*2+1]] && x < BBxmax[correspondence_loc[i*2+1]] &&
								y > BBymin[correspondence_loc[i*2+1]] && y < BBymax[correspondence_loc[i*2+1]] &&
								!(*(normalizer->blobs[correspondence_loc[i*2]].transformed+pos))){
							normalizer->diff_count_nonskin++;
						}

						minX = x < minX ? x : minX;
						maxX = x > maxX ? x : maxX;
						minY = y < minY ? y : minY;
						maxY = y > maxY ? y : maxY;
					}
				}
				pos++;

				//pixel 3
				if((*(normalizer->Prev_Norm_Blobs[correspondence_loc[i*2+1]].NormalizedBlob+pos)) ||
						(*(normalizer->blobs[correspondence_loc[i*2]].transformed+pos))){
					if( (*(temp_diffs[i]+pos) = ((*(normalizer->Prev_Norm_Blobs[correspondence_loc[i*2+1]].NormalizedBlob+pos))
							!= (*(normalizer->blobs[correspondence_loc[i*2]].transformed+pos)))) ){
						x = pos%curr_Width; y = pos/curr_Width;
						normalizer->diff_count++;
						if( x > BBxmin[correspondence_loc[i*2+1]] && x < BBxmax[correspondence_loc[i*2+1]] &&
								y > BBymin[correspondence_loc[i*2+1]] && y < BBymax[correspondence_loc[i*2+1]] &&
								!(*(normalizer->blobs[correspondence_loc[i*2]].transformed+pos))){
							normalizer->diff_count_nonskin++;
						}

						minX = x < minX ? x : minX;
						maxX = x > maxX ? x : maxX;
						minY = y < minY ? y : minY;
						maxY = y > maxY ? y : maxY;
					}
				}
				pos++;

				//pixel 4
				if((*(normalizer->Prev_Norm_Blobs[correspondence_loc[i*2+1]].NormalizedBlob+pos)) ||
						(*(normalizer->blobs[correspondence_loc[i*2]].transformed+pos))){
					if( (*(temp_diffs[i]+pos) = ((*(normalizer->Prev_Norm_Blobs[correspondence_loc[i*2+1]].NormalizedBlob+pos))
							!= (*(normalizer->blobs[correspondence_loc[i*2]].transformed+pos)))) ){
						x = pos%curr_Width; y = pos/curr_Width;
						normalizer->diff_count++;
						if( x > BBxmin[correspondence_loc[i*2+1]] && x < BBxmax[correspondence_loc[i*2+1]] &&
								y > BBymin[correspondence_loc[i*2+1]] && y < BBymax[correspondence_loc[i*2+1]] &&
								!(*(normalizer->blobs[correspondence_loc[i*2]].transformed+pos))){
							normalizer->diff_count_nonskin++;
						}

						minX = x < minX ? x : minX;
						maxX = x > maxX ? x : maxX;
						minY = y < minY ? y : minY;
						maxY = y > maxY ? y : maxY;
					}
				}
				pos++;

				//pixel 5
				if((*(normalizer->Prev_Norm_Blobs[correspondence_loc[i*2+1]].NormalizedBlob+pos)) ||
						(*(normalizer->blobs[correspondence_loc[i*2]].transformed+pos))){
					if( (*(temp_diffs[i]+pos) = ((*(normalizer->Prev_Norm_Blobs[correspondence_loc[i*2+1]].NormalizedBlob+pos))
							!= (*(normalizer->blobs[correspondence_loc[i*2]].transformed+pos)))) ){
						x = pos%curr_Width; y = pos/curr_Width;
						normalizer->diff_count++;
						if( x > BBxmin[correspondence_loc[i*2+1]] && x < BBxmax[correspondence_loc[i*2+1]] &&
								y > BBymin[correspondence_loc[i*2+1]] && y < BBymax[correspondence_loc[i*2+1]] &&
								!(*(normalizer->blobs[correspondence_loc[i*2]].transformed+pos))){
							normalizer->diff_count_nonskin++;
						}

						minX = x < minX ? x : minX;
						maxX = x > maxX ? x : maxX;
						minY = y < minY ? y : minY;
						maxY = y > maxY ? y : maxY;
					}
				}
				pos++;

				//pixel 6
				if((*(normalizer->Prev_Norm_Blobs[correspondence_loc[i*2+1]].NormalizedBlob+pos)) ||
						(*(normalizer->blobs[correspondence_loc[i*2]].transformed+pos))){
					if( (*(temp_diffs[i]+pos) = ((*(normalizer->Prev_Norm_Blobs[correspondence_loc[i*2+1]].NormalizedBlob+pos))
							!= (*(normalizer->blobs[correspondence_loc[i*2]].transformed+pos)))) ){
						x = pos%curr_Width; y = pos/curr_Width;
						normalizer->diff_count++;
						if( x > BBxmin[correspondence_loc[i*2+1]] && x < BBxmax[correspondence_loc[i*2+1]] &&
								y > BBymin[correspondence_loc[i*2+1]] && y < BBymax[correspondence_loc[i*2+1]] &&
								!(*(normalizer->blobs[correspondence_loc[i*2]].transformed+pos))){
							normalizer->diff_count_nonskin++;
						}

						minX = x < minX ? x : minX;
						maxX = x > maxX ? x : maxX;
						minY = y < minY ? y : minY;
						maxY = y > maxY ? y : maxY;
					}
				}
				pos++;

				//pixel 7
				if((*(normalizer->Prev_Norm_Blobs[correspondence_loc[i*2+1]].NormalizedBlob+pos)) ||
						(*(normalizer->blobs[correspondence_loc[i*2]].transformed+pos))){
					if( (*(temp_diffs[i]+pos) = ((*(normalizer->Prev_Norm_Blobs[correspondence_loc[i*2+1]].NormalizedBlob+pos))
							!= (*(normalizer->blobs[correspondence_loc[i*2]].transformed+pos)))) ){
						x = pos%curr_Width; y = pos/curr_Width;
						normalizer->diff_count++;
						if( x > BBxmin[correspondence_loc[i*2+1]] && x < BBxmax[correspondence_loc[i*2+1]] &&
								y > BBymin[correspondence_loc[i*2+1]] && y < BBymax[correspondence_loc[i*2+1]] &&
								!(*(normalizer->blobs[correspondence_loc[i*2]].transformed+pos))){
							normalizer->diff_count_nonskin++;
						}

						minX = x < minX ? x : minX;
						maxX = x > maxX ? x : maxX;
						minY = y < minY ? y : minY;
						maxY = y > maxY ? y : maxY;
					}
				}
				pos++;

				//pixel 8
				if((*(normalizer->Prev_Norm_Blobs[correspondence_loc[i*2+1]].NormalizedBlob+pos)) ||
						(*(normalizer->blobs[correspondence_loc[i*2]].transformed+pos))){
					if( (*(temp_diffs[i]+pos) = ((*(normalizer->Prev_Norm_Blobs[correspondence_loc[i*2+1]].NormalizedBlob+pos))
							!= (*(normalizer->blobs[correspondence_loc[i*2]].transformed+pos)))) ){
						x = pos%curr_Width; y = pos/curr_Width;
						normalizer->diff_count++;
						if( x > BBxmin[correspondence_loc[i*2+1]] && x < BBxmax[correspondence_loc[i*2+1]] &&
								y > BBymin[correspondence_loc[i*2+1]] && y < BBymax[correspondence_loc[i*2+1]] &&
								!(*(normalizer->blobs[correspondence_loc[i*2]].transformed+pos))){
							normalizer->diff_count_nonskin++;
						}

						minX = x < minX ? x : minX;
						maxX = x > maxX ? x : maxX;
						minY = y < minY ? y : minY;
						maxY = y > maxY ? y : maxY;
					}
				}
				pos++;
			}

			/*
			 * to avoid numerical issues, we normalize the whole summation using the size of the blob
			 * W*H, so we have a percentage of change with respect to the whole blob, rather than particular
			 * areas inside the blob.
			 * */
			normalizer->diff_acumm_skin += (gfloat)normalizer->diff_count / (gfloat)((maxX-minX)*(maxY-minY));
			normalizer->diff_acumm_nonskin += (gfloat)normalizer->diff_count_nonskin / (gfloat)((maxX-minX)*(maxY-minY));
		}

		/*free the correspondences*/
		g_free(correspondances_vals);
		g_free(correspondence_loc);


		/*
		 * normalizing by the number of blobs since we want it to be
		 * robust no matter how many blobs we have
		 * */
		normalizer->diff_acumm_skin /= normalizer->n_blobs;
		normalizer->diff_acumm_nonskin /= normalizer->n_blobs;


		/*make a copy of the buffers*/
		//free old blobs
		for(j=0;j<normalizer->n_prev_blobs;j++){
			g_free(normalizer->Prev_Norm_Blobs[j].NormalizedBlob);
		}
		g_free(normalizer->Prev_Norm_Blobs);


		//create a copy of new blobs for buffer
		gint n_diffs = min_val((normalizer->n_blobs), (normalizer->n_prev_blobs));
		normalizer->Prev_Norm_Blobs = (PreviousBlobs*)g_malloc(normalizer->n_blobs*sizeof(PreviousBlobs));
		normalizer->n_prev_blobs = normalizer->n_blobs;/* n_previous blobs */
		for(i=0;i<normalizer->n_blobs;i++){
			/*
			 * Creating the container for the data contained on each normalized blob
			 * */
			//WxH
			normalizer->Prev_Norm_Blobs[i].height = normalizer->blobs[i].height+PATCH_PADDING_Y[i];
			normalizer->Prev_Norm_Blobs[i].width = normalizer->blobs[i].width+PATCH_PADDING_X[i];

			//PADX x PADy
			normalizer->Prev_Norm_Blobs[i].paddinx = PATCH_PADDING_X[i];
			normalizer->Prev_Norm_Blobs[i].paddiny = PATCH_PADDING_Y[i];


			blob_size = (normalizer->blobs[i].height+PATCH_PADDING_Y[i])*(normalizer->blobs[i].width+PATCH_PADDING_X[i]);
			normalizer->Prev_Norm_Blobs[i].NormalizedBlob = (guint8*)g_malloc(blob_size*sizeof(guint8));
			/*
			 * Making a copy of the data
			 * */
			memcpy(normalizer->Prev_Norm_Blobs[i].NormalizedBlob, normalizer->blobs[i].transformed, blob_size*sizeof(guint8));

			if(i<n_diffs){
				if(temp_diffs[i]!=NULL){//in case we rejected some pairs
					memcpy(normalizer->blobs[i].transformed, temp_diffs[i], blob_size*sizeof(guint8));
					g_free(temp_diffs[i]);
				}
			}
		}
		g_free(temp_diffs);
		normalizer->n_diff_blobs = n_diffs;
	}

	return normalization_success;
}




















void put_blobs_in_outbuffer(GstMomentNormalization *normalizer){
	if(normalizer->image_type == GST_VIDEO_FORMAT_I420){
		//YUV
		format_YUV(normalizer);
	}else if(GST_VIDEO_FORMAT_GRAY8){
		//GRAY8
		format_GRAY8(normalizer);
		//format_GRAY8_2(normalizer);
	}else if(normalizer->image_type==GST_VIDEO_FORMAT_ARGB || normalizer->image_type==GST_VIDEO_FORMAT_ABGR || normalizer->image_type==GST_VIDEO_FORMAT_xRGB
			|| normalizer->image_type==GST_VIDEO_FORMAT_xBGR){
		//xRGB
		format_xRGB(normalizer);
	}else if(normalizer->image_type==GST_VIDEO_FORMAT_RGBA || normalizer->image_type==GST_VIDEO_FORMAT_BGRA || normalizer->image_type==GST_VIDEO_FORMAT_BGRx
			|| normalizer->image_type==GST_VIDEO_FORMAT_RGBx){
		//RGBx
		format_RGBx(normalizer);
	}
}



void format_YUV(GstMomentNormalization *normalizer){
	gint i,j, y, x, ysize, uvsize;
	gint extrax,extray, bloby, blobx;
	gint width = GST_VIDEO_FRAME_PLANE_STRIDE (normalizer->outframe, 0), height = GST_VIDEO_FRAME_HEIGHT(normalizer->outframe);
	gint dstride[3];
	gint Ydest, Udest, Vdest;
	GstMapInfo OutInfo;
	guint32 xmax=0, ymax=0, xmin=10000, ymin=10000;

	GST_DEBUG_OBJECT(normalizer, "-------------- Preparing output buffer --------------..\n");
	normalizer->outframe->buffer =  gst_buffer_make_writable(normalizer->outframe->buffer);
	gst_buffer_map(normalizer->outframe->buffer, &OutInfo, GST_MAP_WRITE);

	//YUV
	dstride[0] = GST_VIDEO_FRAME_PLANE_STRIDE (normalizer->outframe, 0);
	dstride[1] = GST_VIDEO_FRAME_PLANE_STRIDE (normalizer->outframe, 1);
	dstride[2] = GST_VIDEO_FRAME_PLANE_STRIDE (normalizer->outframe, 2);

	ysize = width*height;
	uvsize = (width>>1)*(height>>1);

	for(i=0;i<normalizer->n_blobs;i++){
		//centering
		extrax = width - (normalizer->blobs[i].width+(PATCH_PADDING_X[i]));//half padding
		extray = height - (normalizer->blobs[i].height+(PATCH_PADDING_Y[i]));//half padding
		extrax /= 2; extray /= 2;

		bloby=0;
		for (y=0;y<height;y++) {
			blobx=0;
			for (x=0;x<width;x++) {
				//locations estimations
				Ydest = y*dstride[0] + x;
				Udest = (y/2)*dstride[1] + (x/2);
				Vdest = (y/2)*dstride[2] + (x/2);

				if(normalizer->patch_normalization){
					if(x>extrax && y>extray && x<(width-extrax) && y<(height-extray)){
						OutInfo.data[Ydest] = normalizer->blobs[i].transformed[bloby*(normalizer->blobs[i].width+PATCH_PADDING_X[i]) + blobx]*255;//Y
						OutInfo.data[Udest + ysize] = 120;
						OutInfo.data[Vdest + ysize + uvsize] = 120;
						if(OutInfo.data[Ydest]>0){
							if(x<xmin){//minx
								xmin=x;
							}
							if(x>xmax){//maxx
								xmax=x;
							}
							if(y<ymin){//miny
								ymin=y;
							}
							if(y>ymax){//maxy
								ymax=y;
							}
						}
						blobx++;
					}else{
						OutInfo.data[Ydest] = 0;//Y
						OutInfo.data[Udest + ysize] = 120;
						OutInfo.data[Vdest + ysize + uvsize] = 120;
					}
				}else{
					if(x>extrax && y>extray && x<(width-extrax) && y<(height-extray)){
						OutInfo.data[Ydest] = normalizer->blobs[i].transformed[bloby*(normalizer->blobs[i].width+PATCH_PADDING_X[i]) + blobx];//Y
						OutInfo.data[Udest + ysize] = 120;
						OutInfo.data[Vdest + ysize + uvsize] = 120;
						if(OutInfo.data[Ydest]>0){
							if(x<xmin){//minx
								xmin=x;
							}
							if(x>xmax){//maxx
								xmax=x;
							}
							if(y<ymin){//miny
								ymin=y;
							}
							if(y>ymax){//maxy
								ymax=y;
							}
						}
						blobx++;
					}else{
						OutInfo.data[Ydest] = 0;//Y
						OutInfo.data[Udest + ysize] = 120;
						OutInfo.data[Vdest + ysize + uvsize] = 120;
					}
				}
			}
			if(y>extray && y<(height-extray)){
				bloby++;
			}
		}
		g_free(normalizer->blobs[i].transformed);
		normalizer->blobs[i].transformed = NULL;
	}
	//freeing used memory
	g_free(normalizer->blobs);

	//store in the first blob only
	normalizer->BBx = (guint32)max_val(xmin, 0);
	normalizer->BBy = (guint32)max_val(ymin, 0);
	normalizer->BBw = (guint32)max_val((xmax - xmin), 0);
	normalizer->BBh = (guint32)max_val((ymax - ymin), 0);

	gst_buffer_unmap(normalizer->outframe->buffer, &OutInfo);
}

void format_GRAY8(GstMomentNormalization *normalizer){
	gint i = 0, j = 0, x = 0, y = 0;
	gint extrax,extray, bloby, blobx;
	gint width = GST_VIDEO_FRAME_PLANE_STRIDE(normalizer->outframe, 0), height = GST_VIDEO_FRAME_HEIGHT(normalizer->outframe);
	gint dstpos, iter, fsize=width*height;
	guint32 xmax=0, ymax=0, xmin=10000, ymin=10000;

	guint8 *GRAY_data = NULL, zero_mask=0;
	gint GRAY_ref = 0, GRAY_comp = 0;

	GST_DEBUG_OBJECT(normalizer, "-------------- Preparing output buffer --------------..\n");
	GRAY_data = GST_VIDEO_FRAME_COMP_DATA(normalizer->outframe, GRAY_comp);

	memset(GRAY_data, 0, fsize);

	for (i = 0; i < normalizer->n_blobs; i++) {
		if(i<normalizer->n_diff_blobs){//since not all blobs in this frame were used for difference estimation
			//centering
			extrax = width - (normalizer->blobs[i].width+(PATCH_PADDING_X[i]));//half padding
			extray = height - (normalizer->blobs[i].height+(PATCH_PADDING_Y[i]));//half padding
			extrax /= 2; extray /= 2;

			blobx = dstpos = bloby = iter = 0;
			while(dstpos<fsize){
				x = dstpos%width;
				y = dstpos/width;

				/* UNROLLING */
				//pixel 1
				if (normalizer->patch_normalization) {
					if(x>extrax && y>extray && x<(width-extrax) && y<(height-extray)){
						*(GRAY_data+dstpos) |= ((*(normalizer->blobs[i].transformed+(bloby*(normalizer->blobs[i].width+PATCH_PADDING_X[i]) + blobx))) * 255);//Y
						if (*(GRAY_data+dstpos)>0) {
							if(x<xmin){//minx
								xmin=x;
							}
							if(x>xmax){//maxx
								xmax=x;
							}
							if(y<ymin){//miny
								ymin=y;
							}
							if(y>ymax){//maxy
								ymax=y;
							}
						}	// (GRAY_data + Idest)
						blobx++;
					}else {
						*(GRAY_data+dstpos) &= zero_mask;	// I
					}
				}else{	// normalizer->patch_normalization
					if(x>extrax && y>extray && x<(width-extrax) && y<(height-extray)){
						*(GRAY_data+dstpos) |= (*(normalizer->blobs[i].transformed+(bloby*(normalizer->blobs[i].width+PATCH_PADDING_X[i]) + blobx)));//Y
						if (*(GRAY_data+dstpos)>0) {
							if(x<xmin){//minx
								xmin=x;
							}
							if(x>xmax){//maxx
								xmax=x;
							}
							if(y<ymin){//miny
								ymin=y;
							}
							if(y>ymax){//maxy
								ymax=y;
							}
						}	// (GRAY_data + Idest)
						blobx++;
					} else {
						*(GRAY_data+dstpos) &= zero_mask;	// I
					}
				}
				dstpos++;
				GRAY_ref++;

				//pixel 2
				if (normalizer->patch_normalization) {
					if(x>extrax && y>extray && x<(width-extrax) && y<(height-extray)){
						*(GRAY_data+dstpos) |= ((*(normalizer->blobs[i].transformed+(bloby*(normalizer->blobs[i].width+PATCH_PADDING_X[i]) + blobx))) * 255);//Y
						if (*(GRAY_data+dstpos)>0) {
							if(x<xmin){//minx
								xmin=x;
							}
							if(x>xmax){//maxx
								xmax=x;
							}
							if(y<ymin){//miny
								ymin=y;
							}
							if(y>ymax){//maxy
								ymax=y;
							}
						}	// (GRAY_data + Idest)
						blobx++;
					}else {
						*(GRAY_data+dstpos) &= zero_mask;	// I
					}
				}else{	// normalizer->patch_normalization
					if(x>extrax && y>extray && x<(width-extrax) && y<(height-extray)){
						*(GRAY_data+dstpos) |= (*(normalizer->blobs[i].transformed+(bloby*(normalizer->blobs[i].width+PATCH_PADDING_X[i]) + blobx)));//Y
						if (*(GRAY_data+dstpos)>0) {
							if(x<xmin){//minx
								xmin=x;
							}
							if(x>xmax){//maxx
								xmax=x;
							}
							if(y<ymin){//miny
								ymin=y;
							}
							if(y>ymax){//maxy
								ymax=y;
							}
						}	// (GRAY_data + Idest)
						blobx++;
					} else {
						*(GRAY_data+dstpos) &= zero_mask;	// I
					}
				}
				dstpos++;
				GRAY_ref++;

				//pixel 3
				if (normalizer->patch_normalization) {
					if(x>extrax && y>extray && x<(width-extrax) && y<(height-extray)){
						*(GRAY_data+dstpos) |= ((*(normalizer->blobs[i].transformed+(bloby*(normalizer->blobs[i].width+PATCH_PADDING_X[i]) + blobx))) * 255);//Y
						if (*(GRAY_data+dstpos)>0) {
							if(x<xmin){//minx
								xmin=x;
							}
							if(x>xmax){//maxx
								xmax=x;
							}
							if(y<ymin){//miny
								ymin=y;
							}
							if(y>ymax){//maxy
								ymax=y;
							}
						}	// (GRAY_data + Idest)
						blobx++;
					}else {
						*(GRAY_data+dstpos) &= zero_mask;	// I
					}
				}else{	// normalizer->patch_normalization
					if(x>extrax && y>extray && x<(width-extrax) && y<(height-extray)){
						*(GRAY_data+dstpos) |= (*(normalizer->blobs[i].transformed+(bloby*(normalizer->blobs[i].width+PATCH_PADDING_X[i]) + blobx)));//Y
						if (*(GRAY_data+dstpos)>0) {
							if(x<xmin){//minx
								xmin=x;
							}
							if(x>xmax){//maxx
								xmax=x;
							}
							if(y<ymin){//miny
								ymin=y;
							}
							if(y>ymax){//maxy
								ymax=y;
							}
						}	// (GRAY_data + Idest)
						blobx++;
					} else {
						*(GRAY_data+dstpos) &= zero_mask;	// I
					}
				}
				dstpos++;
				GRAY_ref++;

				//pixel 4
				if (normalizer->patch_normalization) {
					if(x>extrax && y>extray && x<(width-extrax) && y<(height-extray)){
						*(GRAY_data+dstpos) |= ((*(normalizer->blobs[i].transformed+(bloby*(normalizer->blobs[i].width+PATCH_PADDING_X[i]) + blobx))) * 255);//Y
						if (*(GRAY_data+dstpos)>0) {
							if(x<xmin){//minx
								xmin=x;
							}
							if(x>xmax){//maxx
								xmax=x;
							}
							if(y<ymin){//miny
								ymin=y;
							}
							if(y>ymax){//maxy
								ymax=y;
							}
						}	// (GRAY_data + Idest)
						blobx++;
					}else {
						*(GRAY_data+dstpos) &= zero_mask;	// I
					}
				}else{	// normalizer->patch_normalization
					if(x>extrax && y>extray && x<(width-extrax) && y<(height-extray)){
						*(GRAY_data+dstpos) |= (*(normalizer->blobs[i].transformed+(bloby*(normalizer->blobs[i].width+PATCH_PADDING_X[i]) + blobx)));//Y
						if (*(GRAY_data+dstpos)>0) {
							if(x<xmin){//minx
								xmin=x;
							}
							if(x>xmax){//maxx
								xmax=x;
							}
							if(y<ymin){//miny
								ymin=y;
							}
							if(y>ymax){//maxy
								ymax=y;
							}
						}	// (GRAY_data + Idest)
						blobx++;
					} else {
						*(GRAY_data+dstpos) &= zero_mask;	// I
					}
				}
				dstpos++;
				GRAY_ref++;

				//pixel 5
				if (normalizer->patch_normalization) {
					if(x>extrax && y>extray && x<(width-extrax) && y<(height-extray)){
						*(GRAY_data+dstpos) |= ((*(normalizer->blobs[i].transformed+(bloby*(normalizer->blobs[i].width+PATCH_PADDING_X[i]) + blobx))) * 255);//Y
						if (*(GRAY_data+dstpos)>0) {
							if(x<xmin){//minx
								xmin=x;
							}
							if(x>xmax){//maxx
								xmax=x;
							}
							if(y<ymin){//miny
								ymin=y;
							}
							if(y>ymax){//maxy
								ymax=y;
							}
						}	// (GRAY_data + Idest)
						blobx++;
					}else {
						*(GRAY_data+dstpos) &= zero_mask;	// I
					}
				}else{	// normalizer->patch_normalization
					if(x>extrax && y>extray && x<(width-extrax) && y<(height-extray)){
						*(GRAY_data+dstpos) |= (*(normalizer->blobs[i].transformed+(bloby*(normalizer->blobs[i].width+PATCH_PADDING_X[i]) + blobx)));//Y
						if (*(GRAY_data+dstpos)>0) {
							if(x<xmin){//minx
								xmin=x;
							}
							if(x>xmax){//maxx
								xmax=x;
							}
							if(y<ymin){//miny
								ymin=y;
							}
							if(y>ymax){//maxy
								ymax=y;
							}
						}	// (GRAY_data + Idest)
						blobx++;
					} else {
						*(GRAY_data+dstpos) &= zero_mask;	// I
					}
				}
				dstpos++;
				GRAY_ref++;

				//pixel 6
				if (normalizer->patch_normalization) {
					if(x>extrax && y>extray && x<(width-extrax) && y<(height-extray)){
						*(GRAY_data+dstpos) |= ((*(normalizer->blobs[i].transformed+(bloby*(normalizer->blobs[i].width+PATCH_PADDING_X[i]) + blobx))) * 255);//Y
						if (*(GRAY_data+dstpos)>0) {
							if(x<xmin){//minx
								xmin=x;
							}
							if(x>xmax){//maxx
								xmax=x;
							}
							if(y<ymin){//miny
								ymin=y;
							}
							if(y>ymax){//maxy
								ymax=y;
							}
						}	// (GRAY_data + Idest)
						blobx++;
					}else {
						*(GRAY_data+dstpos) &= zero_mask;	// I
					}
				}else{	// normalizer->patch_normalization
					if(x>extrax && y>extray && x<(width-extrax) && y<(height-extray)){
						*(GRAY_data+dstpos) |= (*(normalizer->blobs[i].transformed+(bloby*(normalizer->blobs[i].width+PATCH_PADDING_X[i]) + blobx)));//Y
						if (*(GRAY_data+dstpos)>0) {
							if(x<xmin){//minx
								xmin=x;
							}
							if(x>xmax){//maxx
								xmax=x;
							}
							if(y<ymin){//miny
								ymin=y;
							}
							if(y>ymax){//maxy
								ymax=y;
							}
						}	// (GRAY_data + Idest)
						blobx++;
					} else {
						*(GRAY_data+dstpos) &= zero_mask;	// I
					}
				}
				dstpos++;
				GRAY_ref++;

				//pixel 7
				if (normalizer->patch_normalization) {
					if(x>extrax && y>extray && x<(width-extrax) && y<(height-extray)){
						*(GRAY_data+dstpos) |= ((*(normalizer->blobs[i].transformed+(bloby*(normalizer->blobs[i].width+PATCH_PADDING_X[i]) + blobx))) * 255);//Y
						if (*(GRAY_data+dstpos)>0) {
							if(x<xmin){//minx
								xmin=x;
							}
							if(x>xmax){//maxx
								xmax=x;
							}
							if(y<ymin){//miny
								ymin=y;
							}
							if(y>ymax){//maxy
								ymax=y;
							}
						}	// (GRAY_data + Idest)
						blobx++;
					}else {
						*(GRAY_data+dstpos) &= zero_mask;	// I
					}
				}else{	// normalizer->patch_normalization
					if(x>extrax && y>extray && x<(width-extrax) && y<(height-extray)){
						*(GRAY_data+dstpos) |= (*(normalizer->blobs[i].transformed+(bloby*(normalizer->blobs[i].width+PATCH_PADDING_X[i]) + blobx)));//Y
						if (*(GRAY_data+dstpos)>0) {
							if(x<xmin){//minx
								xmin=x;
							}
							if(x>xmax){//maxx
								xmax=x;
							}
							if(y<ymin){//miny
								ymin=y;
							}
							if(y>ymax){//maxy
								ymax=y;
							}
						}	// (GRAY_data + Idest)
						blobx++;
					} else {
						*(GRAY_data+dstpos) &= zero_mask;	// I
					}
				}
				dstpos++;
				GRAY_ref++;

				//pixel 8
				if (normalizer->patch_normalization) {
					if(x>extrax && y>extray && x<(width-extrax) && y<(height-extray)){
						*(GRAY_data+dstpos) |= ((*(normalizer->blobs[i].transformed+(bloby*(normalizer->blobs[i].width+PATCH_PADDING_X[i]) + blobx))) * 255);//Y
						if (*(GRAY_data+dstpos)>0) {
							if(x<xmin){//minx
								xmin=x;
							}
							if(x>xmax){//maxx
								xmax=x;
							}
							if(y<ymin){//miny
								ymin=y;
							}
							if(y>ymax){//maxy
								ymax=y;
							}
						}	// (GRAY_data + Idest)
						blobx++;
					}else {
						*(GRAY_data+dstpos) &= zero_mask;	// I
					}
				}else{	// normalizer->patch_normalization
					if(x>extrax && y>extray && x<(width-extrax) && y<(height-extray)){
						*(GRAY_data+dstpos) |= (*(normalizer->blobs[i].transformed+(bloby*(normalizer->blobs[i].width+PATCH_PADDING_X[i]) + blobx)));//Y
						if (*(GRAY_data+dstpos)>0) {
							if(x<xmin){//minx
								xmin=x;
							}
							if(x>xmax){//maxx
								xmax=x;
							}
							if(y<ymin){//miny
								ymin=y;
							}
							if(y>ymax){//maxy
								ymax=y;
							}
						}	// (GRAY_data + Idest)
						blobx++;
					} else {
						*(GRAY_data+dstpos) &= zero_mask;	// I
					}
				}
				dstpos++;
				GRAY_ref++;

				if(GRAY_ref>=width){
					blobx=0;
					GRAY_ref=0;
					if(y>extray && y<(height-extray)){
						bloby++;
					}
				}
			}
		}
		g_free(normalizer->blobs[i].transformed);
		normalizer->blobs[i].transformed = NULL;
	}
	//freeing used memory
	g_free(normalizer->blobs);


	if(!normalizer->any_buffer){
		GstMapInfo Srcmap;
		normalizer->prevBuffer=NULL;
		normalizer->outframe->buffer =  gst_buffer_make_writable(normalizer->outframe->buffer);
		if(!gst_buffer_map (normalizer->outframe->buffer, &Srcmap, GST_MAP_READ)){
			GST_ERROR_OBJECT(normalizer, "Problem mapping memory, of current frame..!!");
			return;
		}
		guint32 sample_count = gst_buffer_get_size(normalizer->outframe->buffer);
		normalizer->prevBuffer = gst_buffer_new_and_alloc(sample_count);

		if(normalizer->prevBuffer == NULL){
			GST_ERROR_OBJECT(normalizer, "Problem allocating memory for the buffer to be used as reference!!");
			return;
		}

		gst_buffer_fill(normalizer->prevBuffer, 0, Srcmap.data, sample_count);
		gst_buffer_unmap(normalizer->outframe->buffer, &Srcmap);
		normalizer->any_buffer = TRUE;
	}else{
		GstMapInfo Srcmap, OldMap;
		normalizer->outframe->buffer =  gst_buffer_make_writable(normalizer->outframe->buffer);
		if(!gst_buffer_map (normalizer->outframe->buffer, &Srcmap, GST_MAP_READ) ||
				!gst_buffer_map (normalizer->prevBuffer, &OldMap, GST_MAP_WRITE)){
			GST_ERROR_OBJECT(normalizer, "Problem mapping memory, for storing in previous buffer..!!");
			return;
		}
		memcpy(OldMap.data, Srcmap.data, Srcmap.size);
		gst_buffer_unmap(normalizer->outframe->buffer, &Srcmap);
		gst_buffer_unmap(normalizer->prevBuffer, &OldMap);
	}



	//store in the first blob only
	normalizer->BBx = (guint32)max_val(xmin, 0);
	normalizer->BBy = (guint32)max_val(ymin, 0);
	normalizer->BBw = (guint32)max_val((xmax - xmin), 0);
	normalizer->BBh = (guint32)max_val((ymax - ymin), 0);
}



void format_xRGB(GstMomentNormalization *normalizer){
	gint i,j, y, x;
	gint extrax,extray, bloby, blobx;
	gint width = GST_VIDEO_FRAME_PLANE_STRIDE (normalizer->outframe, 0), height = GST_VIDEO_FRAME_HEIGHT(normalizer->outframe);
	gint Ydest, Udest, Vdest, Rdest;
	GstMapInfo OutInfo;
	guint32 xmax=0, ymax=0, xmin=10000, ymin=10000;

	GST_DEBUG_OBJECT(normalizer, "-------------- Preparing output buffer --------------..\n");
	normalizer->outframe->buffer =  gst_buffer_make_writable(normalizer->outframe->buffer);
	gst_buffer_map(normalizer->outframe->buffer, &OutInfo, GST_MAP_WRITE);

	//xRGB
	for(i=0;i<normalizer->n_blobs;i++){
		extrax = (width/4) - (normalizer->blobs[i].width+PATCH_PADDING_X[i]);//half padding
		extray = height - (normalizer->blobs[i].height+PATCH_PADDING_Y[i]);//half padding
		extrax /= 2; extray /= 2;

		bloby=0;
		for (y=0;y<height;y++){
			blobx=0;
			for (x=0;x<width;x+=4){
				//locations estimations
				Rdest = y*width + x;

				if(normalizer->patch_normalization){
					if((x/4)>extrax && y>extray && (x/4)<((width/4)-extrax) && y<(height-extray)){
						OutInfo.data[Rdest+1] = normalizer->blobs[i].transformed[bloby*(normalizer->blobs[i].width+PATCH_PADDING_X[i]) + blobx]*255;//R
						OutInfo.data[Rdest+2] = OutInfo.data[Rdest];//G
						OutInfo.data[Rdest+3] = OutInfo.data[Rdest];//B
						if(OutInfo.data[Rdest]>0){
							if(x/4<xmin){//minx
								xmin=x/4;
							}
							if(x/4>xmax){//maxx
								xmax=x/4;
							}
							if(y<ymin){//miny
								ymin=y;
							}
							if(y>ymax){//maxy
								ymax=y;
							}
						}
						blobx++;
					}else{
						OutInfo.data[Rdest+1] = 0;//R
						OutInfo.data[Rdest+2] = 0;
						OutInfo.data[Rdest+3] = 0;
					}
				}else{
					if((x/4)>extrax && y>extray && (x/4)<((width/4)-extrax) && y<(height-extray)){
						OutInfo.data[Rdest+1] = normalizer->blobs[i].transformed[bloby*(normalizer->blobs[i].width+PATCH_PADDING_X[i]) + blobx];//R
						OutInfo.data[Rdest+2] = OutInfo.data[Rdest];//G
						OutInfo.data[Rdest+3] = OutInfo.data[Rdest];//B
						if(OutInfo.data[Rdest]>0){
							if(x/4<xmin){//minx
								xmin=x/4;
							}
							if(x/4>xmax){//maxx
								xmax=x/4;
							}
							if(y<ymin){//miny
								ymin=y;
							}
							if(y>ymax){//maxy
								ymax=y;
							}
						}
						blobx++;
					}else{
						OutInfo.data[Rdest+1] = 0;//R
						OutInfo.data[Rdest+2] = 0;
						OutInfo.data[Rdest+3] = 0;
					}
				}
			}
			if(y>extray && y<(height-extray)){
				bloby++;
			}
		}
		g_free(normalizer->blobs[i].transformed);
		normalizer->blobs[i].transformed = NULL;
	}
	//freeing used memory
	g_free(normalizer->blobs);

	//store in the first blob only
	normalizer->BBx = (guint32)max_val(xmin, 0);
	normalizer->BBy = (guint32)max_val(ymin, 0);
	normalizer->BBw = (guint32)max_val((xmax - xmin), 0);
	normalizer->BBh = (guint32)max_val((ymax - ymin), 0);

	gst_buffer_unmap(normalizer->outframe->buffer, &OutInfo);
}



void format_RGBx(GstMomentNormalization *normalizer){
	gint i,j, y, x;
	gint extrax,extray, bloby, blobx;
	gint width = GST_VIDEO_FRAME_PLANE_STRIDE (normalizer->outframe, 0), height = GST_VIDEO_FRAME_HEIGHT(normalizer->outframe);
	gint Ydest, Udest, Vdest, Rdest;
	GstMapInfo OutInfo;
	guint32 xmax=0, ymax=0, xmin=10000, ymin=10000;

	GST_DEBUG_OBJECT(normalizer, "-------------- Preparing output buffer --------------..\n");
	normalizer->outframe->buffer =  gst_buffer_make_writable(normalizer->outframe->buffer);
	gst_buffer_map(normalizer->outframe->buffer, &OutInfo, GST_MAP_WRITE);

	//xRGB
	for(i=0;i<normalizer->n_blobs;i++){
		extrax = (width/4) - (normalizer->blobs[i].width+PATCH_PADDING_X[i]);//half padding
		extray = height - (normalizer->blobs[i].height+PATCH_PADDING_Y[i]);//half padding
		extrax /= 2; extray /= 2;

		bloby=0;
		for (y=0;y<height;y++){
			blobx=0;
			for (x=0;x<width;x+=4){
				//locations estimations
				Rdest = y*width + x;

				if(normalizer->patch_normalization){
					if((x/4)>extrax && y>extray && (x/4)<((width/4)-extrax) && y<(height-extray)){
						OutInfo.data[Rdest] = normalizer->blobs[i].transformed[bloby*(normalizer->blobs[i].width+PATCH_PADDING_X[i]) + blobx]*255;//R
						OutInfo.data[Rdest+1] = OutInfo.data[Rdest];//G
						OutInfo.data[Rdest+2] = OutInfo.data[Rdest];//B
						if(OutInfo.data[Rdest]>0){
							if(x/4<xmin){//minx
								xmin=x/4;
							}
							if(x/4>xmax){//maxx
								xmax=x/4;
							}
							if(y<ymin){//miny
								ymin=y;
							}
							if(y>ymax){//maxy
								ymax=y;
							}
						}
						blobx++;
					}else{
						OutInfo.data[Rdest] = 0;//R
						OutInfo.data[Rdest+1] = 0;
						OutInfo.data[Rdest+2] = 0;
					}
				}else{
					if((x/4)>extrax && y>extray && (x/4)<((width/4)-extrax) && y<(height-extray)){
						OutInfo.data[Rdest] = normalizer->blobs[i].transformed[bloby*(normalizer->blobs[i].width+PATCH_PADDING_X[i]) + blobx];//R
						OutInfo.data[Rdest+1] = OutInfo.data[Rdest];//G
						OutInfo.data[Rdest+2] = OutInfo.data[Rdest];//B
						if(OutInfo.data[Rdest]>0){
							if(x/4<xmin){//minx
								xmin=x/4;
							}
							if(x/4>xmax){//maxx
								xmax=x/4;
							}
							if(y<ymin){//miny
								ymin=y;
							}
							if(y>ymax){//maxy
								ymax=y;
							}
						}
						blobx++;
					}else{
						OutInfo.data[Rdest] = 0;//R
						OutInfo.data[Rdest+1] = 0;
						OutInfo.data[Rdest+2] = 0;
					}
				}
			}
			if(y>extray && y<(height-extray)){
				bloby++;
			}
		}
		g_free(normalizer->blobs[i].transformed);
		normalizer->blobs[i].transformed = NULL;
	}
	//freeing used memory
	g_free(normalizer->blobs);

	//store in the first blob only
	normalizer->BBx = (guint32)max_val(xmin, 0);
	normalizer->BBy = (guint32)max_val(ymin, 0);
	normalizer->BBw = (guint32)max_val((xmax - xmin), 0);
	normalizer->BBh = (guint32)max_val((ymax - ymin), 0);

	gst_buffer_unmap(normalizer->outframe->buffer, &OutInfo);
}






uint8_t copy_patches(GstMomentNormalization *normalizer){
	uint8_t ierror = normalization_success;
	uint32_t i, j, k;
	uint32_t half_pad_x, half_pad_y;
	GstMapInfo FrameInfo;

	gst_buffer_map(normalizer->inframe->buffer, &FrameInfo, GST_MAP_READ);

	for(i=0;i<normalizer->n_blobs;i++){
		half_pad_x = PATCH_PADDING_X[i]/2;
		half_pad_y = PATCH_PADDING_Y[i]/2;
		for(j=0;j<normalizer->blobs[i].height;j++){
			if(normalizer->channels_format != planar){
				for(k=0;k<normalizer->blobs[i].width;k++){
					normalizer->blobs[i].patch[(j+half_pad_y)*(normalizer->blobs[i].width+PATCH_PADDING_X[i])+half_pad_x+k] =
							FrameInfo.data[(normalizer->blobs[i].y +j)*normalizer->width + (normalizer->blobs[i].x + k)*4];//only taking the first channel
				}
			}else{
				memcpy(normalizer->blobs[i].patch+((j+half_pad_y)*(normalizer->blobs[i].width+PATCH_PADDING_X[i])+half_pad_x),
						FrameInfo.data+((normalizer->blobs[i].y +j)*normalizer->width + normalizer->blobs[i].x),
						sizeof(uint8_t)*normalizer->blobs[i].width);
			}
		}
	}
	gst_buffer_unmap(normalizer->inframe->buffer, &FrameInfo);

	return ierror;//returns error code if any
}


uint8_t Patch_Moments(BLOBMOMENT *patch, double *moments, uint32_t blob_id){
	uint8_t ierror = normalization_success;
	uint32_t i, k, r, x, y;
	double sum_pxy = 0, inv, *fxy, Cx=0, Cy=0, M20=0, M11=0, M02=0, M30=0, M21=0, M12=0, M03=0;
	uint32_t height = (patch->height+PATCH_PADDING_Y[blob_id]), width = (patch->width+PATCH_PADDING_X[blob_id]);
	uint32_t patch_size = (height*width);

	guint32 pixels=0, mask=0xFF, pos, pix_stride=4;
	guint8 skin=1;

	/*
	 * Working with blocks of 4 pixels (bytes) for estimation
	 * of sum
	 * */
	pos=0;
	while(pos<patch_size-pix_stride){
		pixels = *((guint32*)(patch->patch+pos));

		if(pixels>>24){
			sum_pxy+=(double)(pixels>>24);//first pixel
		}
		if((pixels>>16)&mask){
			sum_pxy+=(double)((pixels>>16)&mask);//second pixel
		}
		if((pixels>>8)&mask){
			sum_pxy+=(double)((pixels>>8)&mask);//third pixel
		}
		if(pixels&mask){
			sum_pxy+=(double)(pixels&mask);//fourth pixel
		}

		pos += pix_stride;
	}
	fxy = (double*)g_malloc0(patch_size*sizeof(double));

	/*
	 * for estimation
	 * of density function
	 * */
	pos=0;
	while(pos<patch_size-pix_stride){
		/****************** UNROLLING THIS *******************/
		//pixel 1
		if((*(patch->patch + pos))){//counting only foreground
			*(fxy + pos) = (double)(*(patch->patch + pos));
			(*(patch->patch + pos)) = skin;
		}
		pos++;

		//pixel 2
		if((*(patch->patch + pos))){//counting only foreground
			*(fxy + pos) = (double)(*(patch->patch + pos));
			(*(patch->patch + pos)) = skin;
		}
		pos++;

		//pixel 3
		if((*(patch->patch + pos))){//counting only foreground
			*(fxy + pos) = (double)(*(patch->patch + pos));
			(*(patch->patch + pos)) = skin;
		}
		pos++;

		//pixel 4
		if((*(patch->patch + pos))){//counting only foreground
			*(fxy + pos) = (double)(*(patch->patch + pos));
			(*(patch->patch + pos)) = skin;
		}
		pos++;
		/****************** UNROLLING THIS *******************/
	}


	/*
	 * for estimation
	 * of central moments
	 * */
	pos=0;
	gdouble vall;
	while(pos<patch_size-pix_stride){
		/****************** UNROLLING THIS *******************/
		//pixel 1
		if((*(patch->patch + pos))&skin){//counting only foreground
			x = pos%width;
			y = pos/width;
			vall=(*(fxy+pos));//read once
			Cx += x*vall;
			Cy += y*vall;
		}
		pos++;

		//pixel 2
		if((*(patch->patch + pos))&skin){//counting only foreground
			x = pos%width;
			y = pos/width;
			vall=(*(fxy+pos));//read once
			Cx += x*vall;
			Cy += y*vall;
		}
		pos++;

		//pixel 3
		if((*(patch->patch + pos))&skin){//counting only foreground
			x = pos%width;
			y = pos/width;
			vall=(*(fxy+pos));//read once
			Cx += x*vall;
			Cy += y*vall;
		}
		pos++;

		//pixel 4
		if((*(patch->patch + pos))&skin){//counting only foreground
			x = pos%width;
			y = pos/width;
			vall=(*(fxy+pos));//read once
			Cx += x*vall;
			Cy += y*vall;
		}
		pos++;
		/****************** UNROLLING THIS *******************/
	}


	/*
	 * for estimation
	 * of 2nd and 3rd moments, M20 M11 M02 M30 M21 M12 M03
	 * */
	pos=0;
	inv = 1/sum_pxy;
	gdouble xCx,yCy, val;
	while(pos<patch_size-pix_stride){
		/****************** UNROLLING THIS *******************/
		//pixel 1
		if((*(patch->patch + pos))&skin){//counting only foreground
			x = pos%width;
			y = pos/width;
			xCx = (x-Cx*inv);//estimate once
			yCy = (y-Cy*inv);//estimate once
			val = (*(fxy+pos));//read once
			M20 += xCx*xCx*val;//variance x
			M11 += xCx*yCy*val;//covariance x and y
			M02 += yCy*yCy*val;//variance y
			M30 += xCx*xCx*xCx*val;//skew x
			M21 += xCx*xCx*yCy*val;//skew xy
			M12 += xCx*yCy*yCy*val;//skew yx
			M03 += yCy*yCy*yCy*val;//skew y
		}
		pos++;

		//pixel 2
		if((*(patch->patch + pos))&skin){//counting only foreground
			x = pos%width;
			y = pos/width;
			xCx = (x-Cx*inv);//estimate once
			yCy = (y-Cy*inv);//estimate once
			val = (*(fxy+pos));//read once
			M20 += xCx*xCx*val;//variance x
			M11 += xCx*yCy*val;//covariance x and y
			M02 += yCy*yCy*val;//variance y
			M30 += xCx*xCx*xCx*val;//skew x
			M21 += xCx*xCx*yCy*val;//skew xy
			M12 += xCx*yCy*yCy*val;//skew yx
			M03 += yCy*yCy*yCy*val;//skew y
		}
		pos++;

		//pixel 3
		if((*(patch->patch + pos))&skin){//counting only foreground
			x = pos%width;
			y = pos/width;
			xCx = (x-Cx*inv);//estimate once
			yCy = (y-Cy*inv);//estimate once
			val = (*(fxy+pos));//read once
			M20 += xCx*xCx*val;//variance x
			M11 += xCx*yCy*val;//covariance x and y
			M02 += yCy*yCy*val;//variance y
			M30 += xCx*xCx*xCx*val;//skew x
			M21 += xCx*xCx*yCy*val;//skew xy
			M12 += xCx*yCy*yCy*val;//skew yx
			M03 += yCy*yCy*yCy*val;//skew y
		}
		pos++;

		//pixel 4
		if((*(patch->patch + pos))&skin){//counting only foreground
			x = pos%width;
			y = pos/width;
			xCx = (x-Cx*inv);//estimate once
			yCy = (y-Cy*inv);//estimate once
			val = (*(fxy+pos));//read once
			M20 += xCx*xCx*val;//variance x
			M11 += xCx*yCy*val;//covariance x and y
			M02 += yCy*yCy*val;//variance y
			M30 += xCx*xCx*xCx*val;//skew x
			M21 += xCx*xCx*yCy*val;//skew xy
			M12 += xCx*yCy*yCy*val;//skew yx
			M03 += yCy*yCy*yCy*val;//skew y
		}
		pos++;
		/****************** UNROLLING THIS *******************/
	}

	//copying the normalized moments values
	moments[0] = Cx; moments[1] = Cy; /*M20*/moments[2] = M20; /*M11*/moments[3] = M11;
	/*M02*/moments[4] = M02; /*M30*/moments[5] = M30; /*M21*/moments[6] = M21;
	/*M12*/moments[7] = M12; /*M03*/moments[8] = M03; /*M00*/ moments[9] = sum_pxy;

	GST_DEBUG("Moments: %f, %f,  %f,  %f,  %f,  %f,  %f,  %f,  %f\n", moments[0], moments[1], moments[2],
							moments[3], moments[4], moments[5], moments[6], moments[7], moments[8]);

	g_free(fxy);
	return ierror;//returns error code if any
}



uint8_t Patch_Eigen(BLOBMOMENT *patch){
	uint8_t ierror = normalization_success;
	const gdouble normalizer = 1/patch->moments[9];

	//lambda1
	patch->eigen_vals[0] = ( (patch->moments[2]*normalizer)+(patch->moments[4]*normalizer)+
			sqrt((((patch->moments[2]*normalizer)-(patch->moments[4]*normalizer))*
					((patch->moments[2]*normalizer)-(patch->moments[4]*normalizer))) +
			4*(patch->moments[3]*normalizer)*(patch->moments[3]*normalizer)) ) * 0.5;
	//lambda2
	patch->eigen_vals[1] = ( (patch->moments[2]*normalizer)+(patch->moments[4]*normalizer)-
			sqrt((((patch->moments[2]*normalizer)-(patch->moments[4]*normalizer))*((patch->moments[2]*normalizer)-(patch->moments[4]*normalizer))) +
			4*(patch->moments[3]*normalizer)*(patch->moments[3]*normalizer)) ) * 0.5;

	GST_DEBUG("L1: %f, L2: %f\n", patch->eigen_vals[0], patch->eigen_vals[1]);

	//E1x
	patch->eigen_vec[0][0] = (((patch->moments[3]*normalizer))/sqrt(((patch->eigen_vals[0]-(patch->moments[2]*normalizer))*
			(patch->eigen_vals[0]-(patch->moments[2]*normalizer))) +
			 ((patch->moments[3]*normalizer)*(patch->moments[3]*normalizer))) );
	//E1y
	patch->eigen_vec[0][1] = (((patch->eigen_vals[0]-(patch->moments[2]*normalizer)))/sqrt(((patch->eigen_vals[0]-(patch->moments[2]*normalizer))*
			(patch->eigen_vals[0]-(patch->moments[2]*normalizer))) +
			 ((patch->moments[3]*normalizer)*(patch->moments[3]*normalizer))) );

	//E2x
	patch->eigen_vec[1][0] = -patch->eigen_vec[0][1];/*((patch->moments[3])/sqrt(((patch->eigen_vals[1]-patch->moments[2])*(patch->eigen_vals[1]-patch->moments[2])) +
			 (patch->moments[3]*patch->moments[3])) );*/
	//E2y
	patch->eigen_vec[1][1] = patch->eigen_vec[0][0];/*(((patch->eigen_vals[1]-patch->moments[2]))/sqrt(((patch->eigen_vals[1]-patch->moments[2])*(patch->eigen_vals[1]-patch->moments[2])) +
			 (patch->moments[3]*patch->moments[3])) );*/

	GST_DEBUG("Area:\n"
			"[%f]\n\n", patch->moments[9]);

	GST_DEBUG("Eigen Vectors:\n"
			"[%f, %f;\n%f, %f]\n\n", patch->eigen_vec[0][0], patch->eigen_vec[0][1], patch->eigen_vec[1][0], patch->eigen_vec[1][1]);

	GST_DEBUG("Eigen Values:\n"
			"[%f;\n%f]\n\n", patch->eigen_vals[0], patch->eigen_vals[1]);

	GST_DEBUG("Moments (variance):\n"
			"[%f, %f;\n%f, %f]\n\n", patch->moments[2], patch->moments[3], patch->moments[3], patch->moments[4]);

	GST_DEBUG("Third Moments (before):\n"
			"[%f, %f;\n%f, %f]\n\n", patch->moments[5], patch->moments[6], patch->moments[7], patch->moments[8]);

	return ierror;//returns error code if any
}

uint8_t Patch_Compactification(BLOBMOMENT *patch, uint32_t blob_id){
	uint8_t ierror = normalization_success;
	uint32_t height = patch->height+PATCH_PADDING_Y[blob_id], width = patch->width+PATCH_PADDING_X[blob_id];
	//printf("size: %i, %i\n", height, width);
	double c = min_val(max_val(patch->eigen_vals[0]/patch->eigen_vals[1], patch->eigen_vals[1]/patch->eigen_vals[0])*sqrt(sqrt((width*height))), 70);
	double inv1 = 1/sqrt(patch->eigen_vals[0]);
	double inv2 = 1/sqrt(patch->eigen_vals[1]);

	//A[1][1]
	patch->A[0][0] = (c*patch->eigen_vec[0][0])*inv1;

	//A[1][2]
	patch->A[0][1] = (c*patch->eigen_vec[0][1])*inv1;

	//A[2][1]
	patch->A[1][0] = (c*patch->eigen_vec[1][0])*inv2;

	//A[2][2]
	patch->A[1][1] = (c*patch->eigen_vec[1][1])*inv2;


	GST_DEBUG("\tA:\n\t{<%f, %f>\n\t<%f, %f>}\n", patch->A[0][0], patch->A[0][1], patch->A[1][0], patch->A[1][1]);

	//moments
	//U30 = (A11^3 * M30) + (3 * A11^2 * A12 * M21) + (3 * A11 * A12^2 * M12) + (A12^3 * M03)
	patch->compact_moments[0] = (patch->A[0][0]*patch->A[0][0]*patch->A[0][0]*patch->moments[5]);
	patch->compact_moments[0] += (3*patch->A[0][0]*patch->A[0][0]*patch->A[0][1]*patch->moments[6]);
	patch->compact_moments[0] += (3*patch->A[0][0]*patch->A[0][1]*patch->A[0][1]*patch->moments[7]);
	patch->compact_moments[0] += (patch->A[0][1]*patch->A[0][1]*patch->A[0][1]*patch->moments[8]);

	//U21 = (A11^2 * A21 * M30) + ( (A11^2*A22 + 2*A11*A12*A21) * M21) + ( (2*A11*A12*A22 + A12^2*A21)* M12) + (A21^2 * A22 * M03)
	patch->compact_moments[1] = (patch->A[0][0]*patch->A[0][0]*patch->A[1][0])*patch->moments[5];
	patch->compact_moments[1] += ((patch->A[0][0]*patch->A[0][0]*patch->A[1][1]) + (2*patch->A[0][0]*patch->A[0][1]*patch->A[1][0]))*patch->moments[6];
	patch->compact_moments[1] += ((2*patch->A[0][0]*patch->A[0][1]*patch->A[1][1]) + (patch->A[0][1]*patch->A[0][1]*patch->A[1][0]))*patch->moments[7];
	patch->compact_moments[1] += (patch->A[0][1]*patch->A[0][1]*patch->A[1][1])*patch->moments[8];


	//U12 = (A11 * A21^2 * M30) + ( (A21^2*A12 + 2*A11*A21*A22) * M21) + ( (2*A12*A21*A22 + A22^2*A11)* M12) + (A12 * A22^2 * M03)
	patch->compact_moments[2] = (patch->A[0][0]*patch->A[1][0]*patch->A[1][0])*patch->moments[5];
	patch->compact_moments[2] += ((patch->A[1][0]*patch->A[1][0]*patch->A[0][1]) + (2*patch->A[0][0]*patch->A[1][0]*patch->A[1][1]))*patch->moments[6];
	patch->compact_moments[2] += ((2*patch->A[0][1]*patch->A[1][0]*patch->A[1][1])+(patch->A[1][1]*patch->A[1][1]*patch->A[0][0]))*patch->moments[7];
	patch->compact_moments[2] += (patch->A[0][1]*patch->A[1][1]*patch->A[1][1])*patch->moments[8];

	//U03 = (A21^3 * M30) + (3 * A21^2 * A22 * M21) + (3 * A21 * A22^2 * M12) + (A22^3 * M03)
	patch->compact_moments[3] = (patch->A[1][0]*patch->A[1][0]*patch->A[1][0]*patch->moments[5]);
	patch->compact_moments[3] += (3*patch->A[1][0]*patch->A[1][0]*patch->A[1][1]*patch->moments[6]);
	patch->compact_moments[3] += (3*patch->A[1][0]*patch->A[1][1]*patch->A[1][1]*patch->moments[7]);
	patch->compact_moments[3] += (patch->A[1][1]*patch->A[1][1]*patch->A[1][1]*patch->moments[8]);

	GST_DEBUG("C-value:\n"
			"[%f]\n\n", c);

	GST_DEBUG("W Matrix:\n"
				"[%f, %f;\n%f, %f]\n\n", c/sqrt(patch->eigen_vals[0]), 0.0, 0.0, c/sqrt(patch->eigen_vals[1]) );

	GST_DEBUG("Scale Matrix:\n"
			"[%f, %f;\n%f, %f]\n\n", patch->A[0][0], patch->A[0][1], patch->A[1][0], patch->A[1][1]);

	GST_DEBUG("Third Order moments:\n"
			"[%f, %f;\n%f, %f]\n\n", patch->compact_moments[0], patch->compact_moments[1], patch->compact_moments[2], patch->compact_moments[3]);

	return ierror;//returns error code if any
}



uint8_t Patch_Tensor_angle(BLOBMOMENT *patch){
	uint8_t ierror = normalization_success;
	double pi_val = 3.14159265358979323846;

	//T1 = U12 + U30;
	patch->Tensors[0] = patch->compact_moments[2] + patch->compact_moments[0];

	//T2 = U03 + U21
	patch->Tensors[1] = patch->compact_moments[3] + patch->compact_moments[1];

	patch->angle = atan2(-patch->Tensors[0], patch->Tensors[1]);

	//Tmean
	patch->Tensors[2] = -patch->Tensors[0]*sin(patch->angle) + patch->Tensors[1]*cos(patch->angle);

	GST_DEBUG("TENSOR VALUE: %f\n", patch->Tensors[2]);
	GST_DEBUG("\tBEFORE Rotation angle found (Radians): %f\n", patch->angle);

	//verification
	if(patch->Tensors[2]<0)
		patch->angle += pi_val;

	GST_DEBUG("\tAFTER Rotation angle found (Radians): %f\n", patch->angle);

	return ierror;//returns error code if any
}


uint8_t Patch_Normalization(BLOBMOMENT *patch, uint32_t blob_id){
	uint8_t ierror = normalization_success;
	uint32_t height = patch->height+PATCH_PADDING_Y[blob_id], width = patch->width+PATCH_PADDING_X[blob_id];
	int x, y, new_x, new_y;
	double c = min_val(max_val(patch->eigen_vals[0]/patch->eigen_vals[1], patch->eigen_vals[1]/patch->eigen_vals[0])*sqrt(sqrt((width*height))), 50);
	uint32_t patch_size = (height*width);
	guint32 pos, new_pos, pix_stride=4;
	guint8 skin=1;
	const gdouble norma = 1.f/patch->moments[9];

	GST_DEBUG("\tC-Val: %f\n", c);
	GST_DEBUG("Rotation Matrix:\n"
			"[%f, %f;\n%f, %f]\n\n", cos(patch->angle), sin(patch->angle), -sin(patch->angle), cos(patch->angle));

	GST_DEBUG("Angle:\n"
			"[%f]\n\n", patch->angle);

	/*
	 * Working with blocks of 4 pixels (bytes) for estimation
	 * of sum
	 * */
	pos=0;
	gint locx, locy;
	gdouble c_scale_x, c_scale_y, sin_angle, cos_angle, x_mom, y_mom, eigx, eigy;

	/*
	 * this is done once since it wont change throught the whole
	 * blob
	 * */
	locx = width/2;
	locy = height/2;
	c_scale_x = c/sqrt(patch->eigen_vals[0]);
	c_scale_y = c/sqrt(patch->eigen_vals[1]);
	sin_angle = sin(patch->angle);
	cos_angle = cos(patch->angle);
	x_mom = (patch->moments[0]*norma);
	y_mom = (patch->moments[1]*norma);
	eigx = patch->eigen_vec[0][0];
	eigy = patch->eigen_vec[0][1];

	while(pos<patch_size-pix_stride){
		/****************** UNROLLING THIS *******************/
		if(((*(patch->patch + pos))&skin)){
			x = pos%width;
			y = pos/width;
			//only foreground pixels taken into account
			new_x = locx + c_scale_x*cos_angle*((eigx*(x-x_mom)) + (eigy*(y-y_mom))) +
					c_scale_y*sin_angle*((-eigy*(x-x_mom)) + (eigx*(y-y_mom)));

			new_y = locy + c_scale_x*(-sin_angle)*((eigx*(x-x_mom)) + (eigy*(y-y_mom))) +
					c_scale_y*cos_angle*((-eigy*(x-x_mom)) + (eigx*(y-y_mom)));

			new_pos = new_y*width + new_x;

			if(new_x<width-1 && new_x>0  && new_y<height-1 && new_y>0){
				//validation that we still inside the patch
				(*(patch->transformed+new_pos)) |= skin;
				(*(patch->transformed+new_pos+width)) |= skin;
				(*(patch->transformed+new_pos-width)) |= skin;
				(*(patch->transformed+new_pos+1)) |= skin;
				(*(patch->transformed+new_pos-1)) |= skin;
				(*(patch->transformed+new_pos+width-1)) |= skin;
				(*(patch->transformed+new_pos+width+1)) |= skin;
				(*(patch->transformed+new_pos-width-1)) |= skin;
				(*(patch->transformed+new_pos-width+1)) |= skin;
			}
		}
		pos++;

		if(((*(patch->patch + pos))&skin)){
			x = pos%width;
			y = pos/width;
			//only foreground pixels taken into account
			new_x = locx + c_scale_x*cos_angle*((eigx*(x-x_mom)) + (eigy*(y-y_mom))) +
					c_scale_y*sin_angle*((-eigy*(x-x_mom)) + (eigx*(y-y_mom)));

			new_y = locy + c_scale_x*(-sin_angle)*((eigx*(x-x_mom)) + (eigy*(y-y_mom))) +
					c_scale_y*cos_angle*((-eigy*(x-x_mom)) + (eigx*(y-y_mom)));

			new_pos = new_y*width + new_x;

			if(new_x<width-1 && new_x>0  && new_y<height-1 && new_y>0){
				//validation that we still inside the patch
				(*(patch->transformed+new_pos)) |= skin;
				(*(patch->transformed+new_pos+width)) |= skin;
				(*(patch->transformed+new_pos-width)) |= skin;
				(*(patch->transformed+new_pos+1)) |= skin;
				(*(patch->transformed+new_pos-1)) |= skin;
				(*(patch->transformed+new_pos+width-1)) |= skin;
				(*(patch->transformed+new_pos+width+1)) |= skin;
				(*(patch->transformed+new_pos-width-1)) |= skin;
				(*(patch->transformed+new_pos-width+1)) |= skin;
			}
		}
		pos++;

		if(((*(patch->patch + pos))&skin)){
			x = pos%width;
			y = pos/width;
			//only foreground pixels taken into account
			new_x = locx + c_scale_x*cos_angle*((eigx*(x-x_mom)) + (eigy*(y-y_mom))) +
					c_scale_y*sin_angle*((-eigy*(x-x_mom)) + (eigx*(y-y_mom)));

			new_y = locy + c_scale_x*(-sin_angle)*((eigx*(x-x_mom)) + (eigy*(y-y_mom))) +
					c_scale_y*cos_angle*((-eigy*(x-x_mom)) + (eigx*(y-y_mom)));

			new_pos = new_y*width + new_x;

			if(new_x<width-1 && new_x>0  && new_y<height-1 && new_y>0){
				//validation that we still inside the patch
				(*(patch->transformed+new_pos)) |= skin;
				(*(patch->transformed+new_pos+width)) |= skin;
				(*(patch->transformed+new_pos-width)) |= skin;
				(*(patch->transformed+new_pos+1)) |= skin;
				(*(patch->transformed+new_pos-1)) |= skin;
				(*(patch->transformed+new_pos+width-1)) |= skin;
				(*(patch->transformed+new_pos+width+1)) |= skin;
				(*(patch->transformed+new_pos-width-1)) |= skin;
				(*(patch->transformed+new_pos-width+1)) |= skin;
			}
		}
		pos++;

		if(((*(patch->patch + pos))&skin)){
			x = pos%width;
			y = pos/width;
			//only foreground pixels taken into account
			new_x = locx + c_scale_x*cos_angle*((eigx*(x-x_mom)) + (eigy*(y-y_mom))) +
					c_scale_y*sin_angle*((-eigy*(x-x_mom)) + (eigx*(y-y_mom)));

			new_y = locy + c_scale_x*(-sin_angle)*((eigx*(x-x_mom)) + (eigy*(y-y_mom))) +
					c_scale_y*cos_angle*((-eigy*(x-x_mom)) + (eigx*(y-y_mom)));

			new_pos = new_y*width + new_x;

			if(new_x<width-1 && new_x>0  && new_y<height-1 && new_y>0){
				//validation that we still inside the patch
				(*(patch->transformed+new_pos)) |= skin;
				(*(patch->transformed+new_pos+width)) |= skin;
				(*(patch->transformed+new_pos-width)) |= skin;
				(*(patch->transformed+new_pos+1)) |= skin;
				(*(patch->transformed+new_pos-1)) |= skin;
				(*(patch->transformed+new_pos+width-1)) |= skin;
				(*(patch->transformed+new_pos+width+1)) |= skin;
				(*(patch->transformed+new_pos-width-1)) |= skin;
				(*(patch->transformed+new_pos-width+1)) |= skin;
			}
		}
		pos++;
		/****************** UNROLLING THIS *******************/
	}

	return ierror;//returns error code if any
}



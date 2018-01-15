/*
 * norm.h
 *
 *  Created on: Jan 22, 2015
 *      Author: josue
 *
 *      Normalization of the image so we make sure we dont
 *      overflow the datatype
 */

#ifndef GST_RETINEX_SRC_NORM_H_
#define GST_RETINEX_SRC_NORM_H_

#ifdef __cplusplus
	extern "C"{
#endif

/*
 * main function to normalize the image
 * */
void normalize_mean_dt(float *data, const float *ref, size_t size);

#ifdef __cplusplus
	}
#endif

#endif /* GST_RETINEX_SRC_RETINEXLIB_H_ */

/*
 * MomentNormalizer.h
 *
 *  Created on: Jan 7, 2015
 *      Author: josue
 *
 *      Main header to be used by the user in order to interact
 *      and call the function to perform the moment normalization part
 */

#ifndef MOMENTNORMALIZER_H_
#define MOMENTNORMALIZER_H_

#ifdef __cplusplus
extern "C"{
#endif

#include "../../../common/metainfo.h"
#include "../gstmomentnormalization.h"

/*
 * Local padding designed to avoid the problem of pixels
 * outside the boundary of the image when rotation
 * during the normalization process
 * */
int PATCH_PADDING_X[MAXBLOBS];
int PATCH_PADDING_Y[MAXBLOBS];
#define MAX_PAD (512)

enum channels_format{
	planar=0,//for planar images/frames
	interleaved//for interleaved channels
};

/*
 * Normalizer function, which is the main entry for the user
 * he has to make sure to send the right data and information for
 * the normalizer to work properly
 * */
uint8_t perform_normalization(GstMomentNormalization *normalizer);

/*
 * Error handler function, which can be called to determine the error
 * this will be used as a log function in case the user wants to determine crash
 * reasons when using the API
 * */
uint8_t Normalization_error_handler(uint8_t ierror);


/*
 * For debugging the normalizer routine
 * */
void normalizer_debug_init();



#ifdef __cplusplus
}
#endif

#endif /* MOMENTNORMALIZER_H_ */

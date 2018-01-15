/*
 * RetinexLib.h
 *
 *  Created on: Jan 22, 2015
 *      Author: josue
 *
 *      Using the source code from IPOL website
 *      this retinex routine has the function of enhance the
 *      image contrast and color so when blob detection and skin
 *      detection are performed the result is more consistent
 */

#ifndef GST_RETINEX_SRC_RETINEXLIB_H_
#define GST_RETINEX_SRC_RETINEXLIB_H_

#ifdef __cplusplus
	extern "C"{
#endif

/*
 * Main function of this API for retinex implementation and enhancement
 * of the frame to be studied, where only the value of T has to be given
 *  */
float *retinex_pde(float *data, size_t nx, size_t ny, float t);



/*
 * Helper functions to perform the retinex routine
 * where they are not to be called individually outside,
 * they are used only to help retinex_pde
 * */
float *retinex_poisson_dct(float *data, size_t nx, size_t ny, double m);
double *cos_table(size_t size);
float *discrete_laplacian_threshold(float *data_out, const float *data_in, size_t nx, size_t ny, float t);




#ifdef __cplusplus
	}
#endif

#endif /* GST_RETINEX_SRC_RETINEXLIB_H_ */


/*
 * RetinexLib.c
 *
 *  Created on: Jan 22, 2015
 *      Author: josue
 *
 *      Core development of the retinex routine where it does depend
 *      on LibFFTW3F for performance, multithreadig should be enabled
 *      so we can have better and consistent results
 *
 * @file norm.c
 * @brief array normalization
 *
 * @author Jose-Luis Lisani <joseluis.lisani@uib.es>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "includes/norm.h"

/**
 * @brief compute mean and variance of a float array
 *
 * @param data float array
 * @param size array size
 * @param mean_p, dt_p addresses to store the mean and variance
 */
static void mean_dt(const float *data, size_t size,
                    double *mean_p, double *dt_p)
{
    double mean, dt;
    const float *ptr_data;
    size_t i;

    mean = 0.;
    dt = 0.;
    ptr_data = data;
    for (i = 0; i < size; i++) {
        mean += *ptr_data;
        dt += (*ptr_data) * (*ptr_data);
        ptr_data++;
    }
    mean /= (double) size;
    dt /= (double) size;
    dt -= (mean * mean);
    dt = sqrt(dt);

    *mean_p = mean;
    *dt_p = dt;

    return;
}

/**
 * @brief normalize mean and variance of a float array given a reference
 *        array
 *
 * The normalized array is normalized by an affine transformation
 * to adjust its mean and variance to a reference array.
 *
 * @param data normalized array
 * @param ref reference array
 * @param size size of the arrays
 */
void normalize_mean_dt(float *data, const float *ref, size_t size)
{
    double mean_ref, mean_data, dt_ref, dt_data;  
    double a, b;
    size_t i;
    float *ptr_data;
    float temp;

    /* sanity check */
    if (NULL == data || NULL == ref) {
        fprintf(stderr, "a pointer is NULL and should not be so\n");
        abort();
    }

    /* compute mean and variance of the two arrays */
    mean_dt(ref, size, &mean_ref, &dt_ref);
    mean_dt(data, size, &mean_data, &dt_data);

    /* compute the normalization coefficients */
    a = dt_ref / dt_data;
    b = mean_ref - a * mean_data;

    /* normalize the array */
    ptr_data = data;
    for (i = 0; i < size; i++) {
    	temp = (a * *ptr_data + b);
    	if(temp>=0 && temp<=255)
    		*ptr_data = temp;
    	else{
    		if(temp<0)
    			*ptr_data = 0;
    		else
    			*ptr_data = 255;
    	}
        ptr_data++;
    }

    return;
}

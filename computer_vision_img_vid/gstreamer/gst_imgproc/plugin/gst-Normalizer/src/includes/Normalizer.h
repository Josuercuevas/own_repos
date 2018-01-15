/*
 * Normalizer.h
 *
 *  Created on: Jan 7, 2015
 *      Author: josue
 *
 *      In charge of performing all the computation for the normalization
 *      part of the API
 */

#ifndef NORMALIZER_H_
#define NORMALIZER_H_

#include "MomentNormalizer.h"



/*
 * Extract the specific patches found when blob detector was performed
 * */
uint8_t copy_patches(GstMomentNormalization *normalizer);

/*
 * Moments  estimation for the patch, where 1st ~ 3rd moments
 * of the patch will be estimated
 * MOMENTS: "Cx, Cy , M20 M11 M02 M30 M21 M12 M03"
 * COMPACT MOMENTS: U30 U21 U12 U03
 * */
uint8_t Patch_Moments(BLOBMOMENT *patch, double *moments, uint32_t blob_id);

/*
 * Eigen values and vectors extraction
 * Compactification
 * Tensors and angle
 * */
uint8_t Patch_Eigen(BLOBMOMENT *patch);
uint8_t Patch_Compactification(BLOBMOMENT *patch, uint32_t blob_id);
uint8_t Patch_Tensor_angle(BLOBMOMENT *patch);

/*
 * Normalization for each pixel in the patch
 * basically translating and rotating the pixels
 * */
uint8_t Patch_Normalization(BLOBMOMENT *patch, uint32_t blob_id);


/*
 * In charge of avoiding memory copies
 * */
void put_blobs_in_outbuffer(GstMomentNormalization *normalizer);
void format_YUV(GstMomentNormalization *normalizer);
void format_xRGB(GstMomentNormalization *normalizer);
void format_RGBx(GstMomentNormalization *normalizer);
void format_GRAY8(GstMomentNormalization *normalizer);





/*
 * Image difference part for previous and current blobs,
 * first e need to match the blobs and afterwards determine the
 * differences to see if they are moving. The matching is simply
 * the ratio = Sum(B(i)|A(j)) / Sum(B(i)&A(j)), the closer to 1 the ratio
 * the better meaning that both blobs are alike.
 *
 * (B(i)|A(j)) = Union of the blobs bits
 * (B(i)&A(j)) = Intersetion of the blobs bits
 *
 * Sum(B(i)|A(j)) = Area uniting the two blobs
 * Sum(B(i)&A(j)) = Area where the two blobs intercept
 *
 * ratio = Sum(B(i)|A(j)) / Sum(B(i)&A(j))  -> how the areas are alike (1 is the perfect match)
 * */
uint8_t frames_diff(GstMomentNormalization *normalizer);



#endif /* NORMALIZER_H_ */



/*
 * const Moments M = moments ( I );
+ const double l1 = ( M.mu20 / M.m00 + M.mu02 / M.m00 + sqrt ( ( M.mu20 / M.m00 - M.mu02 / M.m00 ) * ( M.mu20 / M.m00 - M.mu02 / M.m00 ) + 4 * M.mu11 / M.m00 * M.mu11 / M.m00 ) ) / 2;
+ const double l2 = ( M.mu20 / M.m00 + M.mu02 / M.m00 - sqrt ( ( M.mu20 / M.m00 - M.mu02 / M.m00 ) * ( M.mu20 / M.m00 - M.mu02 / M.m00 ) + 4 * M.mu11 / M.m00 * M.mu11 / M.m00 ) ) / 2;
+ const double ex = ( M.mu11 / M.m00 ) / sqrt ( ( l1 - M.mu20 / M.m00 ) * ( l1 - M.mu20 / M.m00 ) + M.mu11 / M.m00 * M.mu11 / M.m00 );
+ const double ey = ( l1 - M.mu20 / M.m00 ) / sqrt ( ( l1 - M.mu20 / M.m00 ) * ( l1 - M.mu20 / M.m00 ) + M.mu11 / M.m00 * M.mu11 / M.m00 );
+ const Matx22d E = Matx22d ( ex, ey, -ey, ex );
+ const double p = sqrt ( I.size().height * I.size().width ) / 8;
+ const Matx22d W = Matx22d ( p / sqrt ( l1 ), 0, 0, p / sqrt ( l2 ) );
+ const Matx21d c = Matx21d ( M.m10 / M.m00, M.m01 / M.m00 );
+ const Matx21d i = Matx21d ( I.size().height / 2, I.size().width / 2 );
+ const Moments N = M & W * E;
+ const double t1 = N.mu12 + N.mu30;
+ const double t2 = N.mu03 + N.mu21;
+ const double phi = atan2 ( -t1, t2 );
+ const double psi = ( -t1 * sin ( phi ) + t2 * cos ( phi ) >= 0 ) ? phi : ( phi + CV_PI );
+ const Matx22d A = Matx22d ( cos ( psi ), sin ( psi ), -sin ( psi ), cos ( psi ) );
+ return ( A * W * E ) | ( i - A * W * E * c );*/

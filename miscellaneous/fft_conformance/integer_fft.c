/*      fix_fft.c - Fixed-point Fast Fourier Transform  */
/*
        fix_fft()       perform FFT or inverse FFT
        window()        applies a Hanning window to the (time) input
        fix_loud()      calculates the loudness of the signal, for
                        each freq point. Result is an integer array,
                        units are dB (values will be negative).
        iscale()        scale an integer value by (numer/denom).
        fix_mpy()       perform fixed-point multiplication.
        Sinewave[1024]  sinewave normalized to 32767 (= 1.0).
        Loudampl[100]   Amplitudes for lopudnesses from 0 to -99 dB.
        Low_pass        Low-pass filter, cutoff at sample_freq / 4.


        All data are fixed-point short integers, in which
        -32768 to +32768 represent -1.0 to +1.0. Integer arithmetic
        is used for speed, instead of the more natural floating-point.

        For the forward FFT (time -> freq), fixed scaling is
        performed to prevent arithmetic overflow, and to map a 0dB
        sine/cosine wave (i.e. amplitude = 32767) to two -6dB freq
        coefficients; the one in the lower half is reported as 0dB
        by fix_loud(). The return value is always 0.

        For the inverse FFT (freq -> time), fixed scaling cannot be
        done, as two 0dB coefficients would sum to a peak amplitude of
        64K, overflowing the 32k range of the fixed-point integers.
        Thus, the fix_fft() routine performs variable scaling, and
        returns a value which is the number of bits LEFT by which
        the output must be shifted to get the actual amplitude
        (i.e. if fix_fft() returns 3, each value of fr[] and fi[]
        must be multiplied by 8 (2**3) for proper scaling.
        Clearly, this cannot be done within the fixed-point short
        integers. In practice, if the result is to be used as a
        filter, the scale_shift can usually be ignored, as the
        result will be approximately correctly normalized as is.


        TURBO C, any memory model; uses inline assembly for speed
        and for carefully-scaled arithmetic.

        Written by:  Tom Roberts  11/8/89
        Made portable:  Malcolm Slaney 12/15/94 malcolm@interval.com

                Timing on a Macintosh PowerBook 180.... (using Symantec C6.0)
                        fix_fft (1024 points)             8 ticks
                        fft (1024 points - Using SANE)  112 Ticks
                        fft (1024 points - Using FPU)    11

*/

/*

I have added omparison between:

  FFTW3_float
  FFT_OWN_implementation
  FFTS
  FFT_FIXED
  KISS_FFT

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


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>
#include "ffts.h"


#define dosFIX_MPY(DEST,A,B)       {       \
        _DX = (B);                      \
        _AX = (A);                      \
        asm imul dx;                    \
        asm add ax,ax;                  \
        asm adc dx,dx;                  \
        DEST = _DX;             }

#define FIX_MPY(DEST,A,B)       DEST = ((long)(A) * (long)(B))>>15

#define N_WAVE          1024    /* dimension of Sinewave[] */
#define LOG2_N_WAVE     10      /* log2(N_WAVE) */
#define N_LOUD          100     /* dimension of Loudampl[] */
#ifndef fixed
#define fixed short
#endif

extern fixed Sinewave[N_WAVE]; /* placed at end of this file for clarity */
extern fixed Loudampl[N_LOUD];
int db_from_ampl(fixed re, fixed im);
fixed fix_mpy(fixed a, fixed b);

/*
        fix_fft() - perform fast Fourier transform.

        if n>0 FFT is done, if n<0 inverse FFT is done
        fr[n],fi[n] are real,imaginary arrays, INPUT AND RESULT.
        size of data = 2**m
        set inverse to 0=dft, 1=idft
*/
int fix_fft(fixed fr[], fixed fi[], int m, int inverse)
{
        int mr,nn,i,j,l,k,istep, n, scale, shift;
        fixed qr,qi,tr,ti,wr,wi,t;

                n = 1<<m;

        if(n > N_WAVE)
                return -1;

        mr = 0;
        nn = n - 1;
        scale = 0;

        /* decimation in time - re-order data */
        for(m=1; m<=nn; ++m) {
                l = n;
                do {
                        l >>= 1;
                } while(mr+l > nn);
                mr = (mr & (l-1)) + l;

                if(mr <= m) continue;
                tr = fr[m];
                fr[m] = fr[mr];
                fr[mr] = tr;
                ti = fi[m];
                fi[m] = fi[mr];
                fi[mr] = ti;
        }

        l = 1;
        k = LOG2_N_WAVE-1;
        while(l < n) {
                if(inverse) {
                        /* variable scaling, depending upon data */
                        shift = 0;
                        for(i=0; i<n; ++i) {
                                j = fr[i];
                                if(j < 0)
                                        j = -j;
                                m = fi[i];
                                if(m < 0)
                                        m = -m;
                                if(j > 16383 || m > 16383) {
                                        shift = 1;
                                        break;
                                }
                        }
                        if(shift)
                                ++scale;
                } else {
                        /* fixed scaling, for proper normalization -
                           there will be log2(n) passes, so this
                           results in an overall factor of 1/n,
                           distributed to maximize arithmetic accuracy. */
                        shift = 1;
                }
                /* it may not be obvious, but the shift will be performed
                   on each data point exactly once, during this pass. */
                istep = l << 1;
                for(m=0; m<l; ++m) {
                        j = m << k;
                        /* 0 <= j < N_WAVE/2 */
                        wr =  Sinewave[j+N_WAVE/4];
                        wi = -Sinewave[j];
                        if(inverse)
                                wi = -wi;
                        if(shift) {
                                wr >>= 1;
                                wi >>= 1;
                        }
                        for(i=m; i<n; i+=istep) {
                                j = i + l;
                                        tr = fix_mpy(wr,fr[j]) -
fix_mpy(wi,fi[j]);
                                        ti = fix_mpy(wr,fi[j]) +
fix_mpy(wi,fr[j]);
                                qr = fr[i];
                                qi = fi[i];
                                if(shift) {
                                        qr >>= 1;
                                        qi >>= 1;
                                }
                                fr[j] = qr - tr;
                                fi[j] = qi - ti;
                                fr[i] = qr + tr;
                                fi[i] = qi + ti;
                        }
                }
                --k;
                l = istep;
        }

        return scale;
}


/*      window() - apply a Hanning window       */
void window(fixed fr[], int n)
{
        int i,j,k;

        j = N_WAVE/n;
        n >>= 1;
        for(i=0,k=N_WAVE/4; i<n; ++i,k+=j)
                FIX_MPY(fr[i],fr[i],16384-(Sinewave[k]>>1));
        n <<= 1;
        for(k-=j; i<n; ++i,k-=j)
                FIX_MPY(fr[i],fr[i],16384-(Sinewave[k]>>1));
}

/*      fix_loud() - compute loudness of freq-spectrum components.
        n should be ntot/2, where ntot was passed to fix_fft();
        6 dB is added to account for the omitted alias components.
        scale_shift should be the result of fix_fft(), if the time-series
        was obtained from an inverse FFT, 0 otherwise.
        loud[] is the loudness, in dB wrt 32767; will be +10 to -N_LOUD.
*/
void fix_loud(fixed loud[], fixed fr[], fixed fi[], int n, int scale_shift)
{
        int i, max;

        max = 0;
        if(scale_shift > 0)
                max = 10;
        scale_shift = (scale_shift+1) * 6;

        for(i=0; i<n; ++i) {
                loud[i] = db_from_ampl(fr[i],fi[i]) + scale_shift;
                if(loud[i] > max)
                        loud[i] = max;
        }
}

/*      db_from_ampl() - find loudness (in dB) from
        the complex amplitude.
*/
int db_from_ampl(fixed re, fixed im)
{
        static long loud2[N_LOUD] = {0};
        long v;
        int i;

        if(loud2[0] == 0) {
                loud2[0] = (long)Loudampl[0] * (long)Loudampl[0];
                for(i=1; i<N_LOUD; ++i) {
                        v = (long)Loudampl[i] * (long)Loudampl[i];
                        loud2[i] = v;
                        loud2[i-1] = (loud2[i-1]+v) / 2;
                }
        }

        v = (long)re * (long)re + (long)im * (long)im;

        for(i=0; i<N_LOUD; ++i)
                if(loud2[i] <= v)
                        break;

        return (-i);
}

/*
        fix_mpy() - fixed-point multiplication
*/
fixed fix_mpy(fixed a, fixed b)
{
        FIX_MPY(a,a,b);
        return a;
}

/*
        iscale() - scale an integer value by (numer/denom)
*/
int iscale(int value, int numer, int denom)
{
#ifdef  DOS
        asm     mov ax,value
        asm     imul WORD PTR numer
        asm     idiv WORD PTR denom

        return _AX;
#else
                return (long) value * (long)numer/(long)denom;
#endif
}

/*
        fix_dot() - dot product of two fixed arrays
*/
fixed fix_dot(fixed *hpa, fixed *pb, int n)
{
        fixed *pa;
        long sum;
        register fixed a,b;
        unsigned int seg,off;

/*      seg = FP_SEG(hpa);
        off = FP_OFF(hpa);
        seg += off>>4;
        off &= 0x000F;
        pa = MK_FP(seg,off);
 */
        sum = 0L;
        while(n--) {
                a = *pa++;
                b = *pb++;
                FIX_MPY(a,a,b);
                sum += a;
        }

        if(sum > 0x7FFF)
                sum = 0x7FFF;
        else if(sum < -0x7FFF)
                sum = -0x7FFF;

        return (fixed)sum;
#ifdef  DOS
        /* ASSUMES hpa is already normalized so FP_OFF(hpa) < 16 */
        asm     push    ds
        asm     lds     si,hpa
        asm     les     di,pb
        asm     xor     bx,bx

        asm     xor     cx,cx

loop:   /* intermediate values can overflow by a factor of 2 without
           causing an error; the final value must not overflow! */
        asm     lodsw
.
        asm     imul    word ptr es:[di]
        asm     add     bx,ax
        asm     adc     cx,dx
        asm     jo      overflow
        asm     add     di,2
        asm     dec     word ptr n
        asm     jg      loop

        asm     add     bx,bx
        asm     adc     cx,cx
        asm     jo      overflow

        asm     pop     ds
        return _CX;

overflow:
        asm     mov     cx,7FFFH
        asm     adc     cx,0

        asm     pop     ds
        return _CX;
#endif

}


#if N_WAVE != 1024
        ERROR: N_WAVE != 1024
#endif
fixed Sinewave[1024] = {
      0,    201,    402,    603,    804,   1005,   1206,   1406,
   1607,   1808,   2009,   2209,   2410,   2610,   2811,   3011,
   3211,   3411,   3611,   3811,   4011,   4210,   4409,   4608,
   4807,   5006,   5205,   5403,   5601,   5799,   5997,   6195,
   6392,   6589,   6786,   6982,   7179,   7375,   7571,   7766,
   7961,   8156,   8351,   8545,   8739,   8932,   9126,   9319,
   9511,   9703,   9895,  10087,  10278,  10469,  10659,  10849,
  11038,  11227,  11416,  11604,  11792,  11980,  12166,  12353,
  12539,  12724,  12909,  13094,  13278,  13462,  13645,  13827,
  14009,  14191,  14372,  14552,  14732,  14911,  15090,  15268,
  15446,  15623,  15799,  15975,  16150,  16325,  16499,  16672,
  16845,  17017,  17189,  17360,  17530,  17699,  17868,  18036,
  18204,  18371,  18537,  18702,  18867,  19031,  19194,  19357,
  19519,  19680,  19840,  20000,  20159,  20317,  20474,  20631,
  20787,  20942,  21096,  21249,  21402,  21554,  21705,  21855,
  22004,  22153,  22301,  22448,  22594,  22739,  22883,  23027,
  23169,  23311,  23452,  23592,  23731,  23869,  24006,  24143,
  24278,  24413,  24546,  24679,  24811,  24942,  25072,  25201,
  25329,  25456,  25582,  25707,  25831,  25954,  26077,  26198,
  26318,  26437,  26556,  26673,  26789,  26905,  27019,  27132,
  27244,  27355,  27466,  27575,  27683,  27790,  27896,  28001,
  28105,  28208,  28309,  28410,  28510,  28608,  28706,  28802,
  28897,  28992,  29085,  29177,  29268,  29358,  29446,  29534,
  29621,  29706,  29790,  29873,  29955,  30036,  30116,  30195,
  30272,  30349,  30424,  30498,  30571,  30643,  30713,  30783,
  30851,  30918,  30984,  31049,
  31113,  31175,  31236,  31297,
  31356,  31413,  31470,  31525,  31580,  31633,  31684,  31735,
  31785,  31833,  31880,  31926,  31970,  32014,  32056,  32097,
  32137,  32176,  32213,  32249,  32284,  32318,  32350,  32382,
  32412,  32441,  32468,  32495,  32520,  32544,  32567,  32588,
  32609,  32628,  32646,  32662,  32678,  32692,  32705,  32717,
  32727,  32736,  32744,  32751,  32757,  32761,  32764,  32766,
  32767,  32766,  32764,  32761,  32757,  32751,  32744,  32736,
  32727,  32717,  32705,  32692,  32678,  32662,  32646,  32628,
  32609,  32588,  32567,  32544,  32520,  32495,  32468,  32441,
  32412,  32382,  32350,  32318,  32284,  32249,  32213,  32176,
  32137,  32097,  32056,  32014,  31970,  31926,  31880,  31833,
  31785,  31735,  31684,  31633,  31580,  31525,  31470,  31413,
  31356,  31297,  31236,  31175,  31113,  31049,  30984,  30918,
  30851,  30783,  30713,  30643,  30571,  30498,  30424,  30349,
  30272,  30195,  30116,  30036,  29955,  29873,  29790,  29706,
  29621,  29534,  29446,  29358,  29268,  29177,  29085,  28992,
  28897,  28802,  28706,  28608,  28510,  28410,  28309,  28208,
  28105,  28001,  27896,  27790,  27683,  27575,  27466,  27355,
  27244,  27132,  27019,  26905,  26789,  26673,  26556,  26437,
  26318,  26198,  26077,  25954,  25831,  25707,  25582,  25456,
  25329,  25201,  25072,  24942,  24811,  24679,  24546,  24413,
  24278,  24143,  24006,  23869,  23731,  23592,  23452,  23311,
  23169,  23027,  22883,  22739,  22594,  22448,  22301,  22153,
  22004,  21855,  21705,  21554,  21402,  21249,  21096,  20942,
  20787,  20631,  20474,  20317,  20159,  20000,  19840,  19680,
  19519,  19357,  19194,  19031,  18867,  18702,  18537,  18371,
  18204,  18036,  17868,  17699,  17530,  17360,  17189,  17017,
  16845,  16672,  16499,  16325,  16150,  15975,  15799,  15623,
  15446,  15268,  15090,  14911,  14732,  14552,  14372,  14191,
  14009,  13827,  13645,  13462,  13278,  13094,  12909,  12724,
  12539,  12353,  12166,  11980,  11792,  11604,  11416,  11227,
  11038,  10849,  10659,  10469,  10278,  10087,   9895,   9703,
   9511,   9319,   9126,   8932,   8739,   8545,   8351,   8156,
   7961,   7766,   7571,   7375,   7179,   6982,   6786,   6589,
   6392,   6195,   5997,   5799,   5601,   5403,   5205,   5006,
   4807,   4608,   4409,   4210,   4011,   3811,   3611,   3411,
   3211,   3011,   2811,   2610,   2410,   2209,   2009,   1808,
   1607,   1406,   1206,   1005,    804,    603,    402,    201,
      0,   -201,   -402,   -603,   -804,  -1005,  -1206,  -1406,
  -1607,  -1808,  -2009,  -2209,  -2410,  -2610,  -2811,  -3011,
  -3211,  -3411,  -3611,  -3811,  -4011,  -4210,  -4409,  -4608,
  -4807,  -5006,  -5205,  -5403,  -5601,  -5799,  -5997,  -6195,
  -6392,  -6589,  -6786,  -6982,  -7179,  -7375,  -7571,  -7766,
  -7961,  -8156,  -8351,  -8545,  -8739,  -8932,  -9126,  -9319,
  -9511,  -9703,  -9895, -10087, -10278, -10469, -10659, -10849,
 -11038, -11227, -11416, -11604, -11792, -11980, -12166, -12353,
 -12539, -12724, -12909, -13094, -13278, -13462, -13645, -13827,
 -14009, -14191, -14372, -14552, -14732, -14911, -15090, -15268,
 -15446, -15623, -15799, -15975, -16150, -16325, -16499, -16672,
 -16845, -17017, -17189, -17360, -17530, -17699, -17868, -18036,
 -18204, -18371, -18537, -18702, -18867, -19031, -19194, -19357,
 -19519, -19680, -19840, -20000, -20159, -20317, -20474, -20631,
 -20787, -20942, -21096, -21249, -21402, -21554, -21705, -21855,
 -22004, -22153, -22301, -22448, -22594, -22739, -22883, -23027,
 -23169, -23311, -23452, -23592, -23731, -23869, -24006, -24143,
 -24278, -24413, -24546, -24679, -24811, -24942, -25072, -25201,
 -25329, -25456, -25582, -25707, -25831, -25954, -26077, -26198,
 -26318, -26437, -26556, -26673, -26789, -26905, -27019, -27132,
 -27244, -27355, -27466, -27575, -27683, -27790, -27896, -28001,
 -28105, -28208, -28309, -28410, -28510, -28608, -28706, -28802,
 -28897, -28992, -29085, -29177, -29268, -29358, -29446, -29534,
 -29621, -29706, -29790, -29873, -29955, -30036, -30116, -30195,
 -30272, -30349, -30424, -30498, -30571, -30643, -30713, -30783,
 -30851, -30918, -30984, -31049, -31113, -31175, -31236, -31297,
 -31356, -31413, -31470, -31525, -31580, -31633, -31684, -31735,
 -31785, -31833, -31880, -31926, -31970, -32014, -32056, -32097,
 -32137, -32176, -32213, -32249, -32284, -32318, -32350, -32382,
 -32412, -32441, -32468, -32495, -32520, -32544, -32567, -32588,
 -32609, -32628, -32646, -32662, -32678, -32692, -32705, -32717,
 -32727, -32736, -32744, -32751, -32757, -32761, -32764, -32766,
 -32767, -32766, -32764, -32761, -32757, -32751, -32744, -32736,
 -32727, -32717, -32705, -32692, -32678, -32662, -32646, -32628,
 -32609, -32588, -32567, -32544, -32520, -32495, -32468, -32441,
 -32412, -32382, -32350, -32318, -32284, -32249, -32213, -32176,
 -32137, -32097, -32056, -32014, -31970, -31926, -31880, -31833,
 -31785, -31735, -31684, -31633, -31580, -31525, -31470, -31413,
 -31356, -31297, -31236, -31175, -31113, -31049, -30984, -30918,
 -30851, -30783, -30713, -30643, -30571, -30498, -30424, -30349,
 -30272, -30195, -30116, -30036, -29955, -29873, -29790, -29706,
 -29621, -29534, -29446, -29358, -29268, -29177, -29085, -28992,
 -28897, -28802, -28706, -28608, -28510, -28410, -28309, -28208,
 -28105, -28001, -27896, -27790, -27683, -27575, -27466, -27355,
 -27244, -27132, -27019, -26905, -26789, -26673, -26556, -26437,
 -26318, -26198, -26077, -25954, -25831, -25707, -25582, -25456,
 -25329, -25201, -25072, -24942, -24811, -24679, -24546, -24413,
 -24278, -24143, -24006, -23869, -23731, -23592, -23452, -23311,
 -23169, -23027, -22883, -22739, -22594, -22448, -22301, -22153,
 -22004, -21855, -21705, -21554, -21402, -21249, -21096, -20942,
 -20787, -20631, -20474, -20317, -20159, -20000, -19840, -19680,
 -19519, -19357, -19194, -19031, -18867, -18702, -18537, -18371,
 -18204, -18036, -17868, -17699, -17530, -17360, -17189, -17017,
 -16845, -16672, -16499, -16325, -16150, -15975, -15799, -15623,
 -15446, -15268, -15090, -14911, -14732, -14552, -14372, -14191,
 -14009, -13827, -13645, -13462, -13278, -13094, -12909, -12724,
 -12539, -12353, -12166, -11980, -11792, -11604, -11416, -11227,
 -11038, -10849, -10659, -10469, -10278, -10087,  -9895,  -9703,
  -9511,  -9319,  -9126,  -8932,  -8739,  -8545,  -8351,  -8156,
  -7961,  -7766,  -7571,  -7375,  -7179,  -6982,  -6786,  -6589,
  -6392,  -6195,  -5997,  -5799,  -5601,  -5403,  -5205,  -5006,
  -4807,  -4608,  -4409,  -4210,  -4011,  -3811,  -3611,  -3411,
  -3211,  -3011,  -2811,  -2610,  -2410,  -2209,  -2009,  -1808,
  -1607,  -1406,  -1206,  -1005,   -804,   -603,   -402,   -201,
};

#if N_LOUD != 100
        ERROR: N_LOUD != 100
#endif
fixed Loudampl[100] = {
  32767,  29203,  26027,  23197,  20674,  18426,  16422,  14636,
  13044,  11626,  10361,   9234,   8230,   7335,   6537,   5826,
   5193,   4628,   4125,   3676,   3276,   2920,   2602,   2319,
   2067,   1842,   1642,   1463,   1304,   1162,   1036,    923,
    823,    733,    653,    582,    519,    462,    412,    367,
    327,    292,    260,    231,    206,    184,    164,    146,
    130,    116,    103,     92,     82,     73,     65,     58,
     51,     46,     41,     36,     32,     29,     26,     23,
     20,     18,     16,     14,     13,     11,     10,      9,
      8,      7,      6,      5,      5,      4,      4,      3,
      3,      2,      2,      2,      2,      1,      1,      1,
      1,      1,      1,      0,      0,      0,      0,      0,
      0,      0,      0,      0,
};













































#define M (3)
#define N (1<<M)




static unsigned int numberOfBitsNeeded(unsigned int p_nSamples){
	int i;
	if(p_nSamples<2){
		return 0;
	}

	i=0;
	while(1){
		//============
		if(p_nSamples & (1<<i)){
			return i;
		}
		i++;
	}
}

static unsigned int reverseBits(unsigned int p_nIndex, unsigned int p_nBits){
	unsigned int i, rev;
	i=rev=0;
	while(1){
		//=================
		if(i >= p_nBits) break;
		rev = (rev << 1) | (p_nIndex & 1);
		p_nIndex >>= 1;
		i++;
	}

	return rev;
}


#define temp_angle_numerator (-2.0 * 3.1415926535897)
void Own_FFT(int p_bInverseTransform, const float *p_lpRealIn, const float *p_lpImagIn, float *p_lpRealOut, float *p_lpImagOut){
	unsigned int NumBits;
	unsigned int i, j, k, n;
	unsigned int BlockSize, BlockEnd;

	float tr, ti;
	float angle_numerator=temp_angle_numerator;

	if(!p_bInverseTransform)
		angle_numerator = -angle_numerator;

	NumBits = numberOfBitsNeeded(N);

	i=0;
	while(1){
		//*************************************************************
		if(i>=N) break;
		//========
		j = reverseBits(i, NumBits);
		*(p_lpRealOut+j) = *(p_lpRealIn++);
		*(p_lpImagOut+j) = (p_lpImagIn == 0) ? 0.0 : *(p_lpImagIn++);
		i++;

		//*************************************************************
		if(i>=N) break;
		//========
		j = reverseBits(i, NumBits);
		*(p_lpRealOut+j) = *(p_lpRealIn++);
		*(p_lpImagOut+j) = (p_lpImagIn == 0) ? 0.0 : *(p_lpImagIn++);
		i++;

		//*************************************************************
		if(i>=N) break;
		//========
		j = reverseBits(i, NumBits);
		*(p_lpRealOut+j) = *(p_lpRealIn++);
		*(p_lpImagOut+j) = (p_lpImagIn == 0) ? 0.0 : *(p_lpImagIn++);
		i++;

		//*************************************************************
		if(i>=N) break;
		//========
		j = reverseBits(i, NumBits);
		*(p_lpRealOut+j) = *(p_lpRealIn++);
		*(p_lpImagOut+j) = (p_lpImagIn == 0) ? 0.0 : *(p_lpImagIn++);
		i++;
	}

	BlockEnd = 1;

	float delta_angle = angle_numerator / (float)BlockSize;
	float sm2, sm1, cm2, cm1, w, ar[3], ai[3];
	sm2 = -sin (-2 * delta_angle);
	sm1 = -sin (-delta_angle);
	cm2 = cos (-2 * delta_angle);
	cm1 = cos (-delta_angle);
	BlockSize=2;

	while(1){
		//=======================================================
		delta_angle = angle_numerator / (float)BlockSize;
		sm2 = -sin (-2 * delta_angle);
		sm1 = -sin (-delta_angle);
		cm2 = cos (-2 * delta_angle);
		cm1 = cos (-delta_angle);
		w = 2*cm1;
		i=0;
		while(1){
			//***************************************************
			*(ar+2) = cm2;
			*(ar+1) = cm1;
			*(ai+2) = sm2;
			*(ai+1) = sm1;
			n=0;j=i;
			while(1){
				//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
				if(n >= BlockEnd) break;
				//--------------------------------------------------
				*(ar+0) = w*(*(ar+1)) - *(ar+2);
				*(ar+2) = *(ar+1);
				*(ar+1) = *(ar+0);
				*(ai+0) = w*(*(ai+1)) - *(ai+2);
				*(ai+2) = *(ai+1);
				*(ai+1) = *(ai+0);
				k = j + BlockEnd;
				tr = (*(ar+0))*(*(p_lpRealOut+k)) - ((*(p_lpImagOut+k))!=0?(*(ai+0))*(*(p_lpImagOut+k)):0.0);
				ti = ((*(p_lpImagOut+k))!=0?(*(ar+0))*(*(p_lpImagOut+k)):0.0) + (*(ai+0))*(*(p_lpRealOut+k));
				*(p_lpRealOut+k) = *(p_lpRealOut+j) - tr;
				*(p_lpImagOut+k) = *(p_lpImagOut+j) - ti;
				*(p_lpRealOut+j) += tr;
				*(p_lpImagOut+j) += ti;
				n++;
				j++;
			}
			i += BlockSize;
			if(i>=N) break;
		}
		BlockEnd = BlockSize;
		BlockSize <<= 1;
		if(BlockSize > N) break;
	}


	if(p_bInverseTransform){
		float denom = 1.0/(float)N;
		i=0;
		while(1){
			//------------------
			if(i>=N) break;
			//==
			(*(p_lpRealOut+i)) *= denom;
//					(*(p_lpImagOut+i)) *= denom;
			//==
			(*(p_lpRealOut+i+1)) *= denom;
//					(*(p_lpImagOut+i+1)) *= denom;
			i+=2;
			//------------------
			if(i>=N) break;
			//==
			(*(p_lpRealOut+i)) *= denom;
//					(*(p_lpImagOut+i)) *= denom;
			//==
			(*(p_lpRealOut+i+1)) *= denom;
//					(*(p_lpImagOut+i+1)) *= denom;
			i+=2;
		}
	}
}









static int PRINT_COEFF=0;
//#define FFTS_SUPPORTED

#if (__arm__)
 /*DONT KNOW*/
#else
uint64_t rdtsc(){
#if (!__arm__)
    unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
#endif
}
#endif







static float reference_mfcc[N];
float extract_mfccs(float *real, float *imag, char *name){
    int i,j;
    float mse=0.0f;

	/* Calculate at startup */
    float *freqs, *lower, *center, *upper, *triangleHeight, *fftFreqs;
    float *ceps;
	float lowestFrequency   = 66.6666666;
	const int linearFilters     = 13;
	float linearSpacing     = 66.66666666;
	const int logFilters        = 27;
	float logSpacing        = 1.0711703;

	float totalFilters      = linearFilters + logFilters;
	float logPower          = 1.0f;


	/* The number of cepstral componenents */
	const int nceps = 20;
	int fftSize = N;
	int samplingRate = 8000;

	/* Set if user want C0 */
	int WANT_C0 = 1;

	/* Allocate space for feature vector */
	if (WANT_C0 == 1) {
		ceps = (float*)calloc(nceps+1, sizeof(float));
	} else {
		ceps = (float*)calloc(nceps, sizeof(float));
	}

	/* Allocate space for local vectors */
	float **mfccDCTMatrix = (float**)calloc(nceps+1, sizeof(float*));
	for (i = 0; i < nceps+1; i++) {
		mfccDCTMatrix[i]= (float*)calloc(totalFilters, sizeof(float));
	}

	float **mfccFilterWeights = (float**)calloc(totalFilters, sizeof(float*));
	for (i = 0; i < totalFilters; i++) {
		mfccFilterWeights[i] = (float*)calloc(fftSize, sizeof(float));
	}


	freqs  = (double*)calloc(totalFilters+2,sizeof(float));

	lower  = (double*)calloc(totalFilters,sizeof(float));
	center = (double*)calloc(totalFilters,sizeof(float));
	upper  = (double*)calloc(totalFilters,sizeof(float));

	triangleHeight = (float*)calloc(totalFilters,sizeof(float));
	fftFreqs       = (float*)calloc(fftSize,sizeof(float));

	for (i = 0; i < linearFilters; i++) {
		freqs[i] = lowestFrequency + ((float)i) * linearSpacing;
	}

	for (i = linearFilters; i < totalFilters+2; i++) {
		freqs[i] = freqs[linearFilters-1] *
			pow(logSpacing, (float)(i-linearFilters+1));
	}

	/* Define lower, center and upper */
	memcpy(lower,  freqs,totalFilters*sizeof(float));
	memcpy(center, &freqs[1],totalFilters*sizeof(float));
	memcpy(upper,  &freqs[2],totalFilters*sizeof(float));

	for (i=0;i<totalFilters;i++){
		triangleHeight[i] = 2./(upper[i]-lower[i]);
	}

	for (i=0;i<fftSize;i++){
		fftFreqs[i] = ((float) i / ((float) fftSize ) *
					   (float) samplingRate);
//			fftFreqs[i] = ( ((double)samplingRate/(double)((fftSize-1)*2)) *
//						   (double)(i>0?i:0));
	}

	/* Build now the mccFilterWeight matrix */
	for(i=0;i<totalFilters;i++){
		for(j=0;j<fftSize;j++){
			if((fftFreqs[j] > lower[i]) && (fftFreqs[j] <= center[i])){

				mfccFilterWeights[i][j] = triangleHeight[i] *
					(fftFreqs[j]-lower[i]) / (center[i]-lower[i]);

			}else{
				mfccFilterWeights[i][j] = 0.0;
			}

			if((fftFreqs[j]>center[i]) && (fftFreqs[j]<upper[i])){

				mfccFilterWeights[i][j] = mfccFilterWeights[i][j]
					+ triangleHeight[i] * (upper[i]-fftFreqs[j])
					/ (upper[i]-center[i]);
			}else{
				mfccFilterWeights[i][j] = mfccFilterWeights[i][j] + 0.0;
			}
		}
	}

	/*
	 * We calculate now mfccDCT matrix
	 * NB: +1 because of the DC component
	 */

	const double pi = 3.14159265358979323846264338327950288;

	for (i = 0; i < nceps+1; i++) {
		for (j = 0; j < totalFilters; j++) {
			mfccDCTMatrix[i][j] = (1./sqrt((float) totalFilters / 2.))
				* cos((float) i * ((float) j + 0.5) / (float) totalFilters * pi);
		}
	}

	for (j = 0; j < totalFilters; j++){
		mfccDCTMatrix[0][j] = (sqrt(2.)/2.) * mfccDCTMatrix[0][j];
	}









	//MFFC estimation
	float earMag[N], outceps[nceps+1];
	float fftmags[N/2 + 1];

	for(i = 0; i < fftSize/2; ++i){
		fftmags[i] = sqrt(real[i]*real[i] + imag[i]*imag[i]);
	}

	for(i = 0; i < totalFilters; ++i){
		earMag[i] = 0.0;
	}

	/* Multiply by mfccFilterWeights */
	for(i = 0; i < totalFilters; i++){
		double tmp = 0.0;
		for(j = 0; j < fftSize/2; j++){
			tmp = tmp + (mfccFilterWeights[i][j] * fftmags[j]);
		}
		if(tmp > 0){
			earMag[i] = log10(tmp);
		}else{
			earMag[i] = 0.0;
		}

		if(logPower != 1.0){
			earMag[i] = pow(earMag[i], logPower);
		}
	}

	/*
	 *
	 * Calculate now the cepstral coefficients
	 * with or without the DC component
	 *
	 */
	if(WANT_C0 == 1){
		if(PRINT_COEFF){
			printf("MFCCs with %s\n", name);
		}
		for(i = 0; i < nceps+1; i++) {
			float tmp = 0.;
			for(j = 0; j < totalFilters; j++){
				tmp = tmp + mfccDCTMatrix[i][j] * earMag[j];
			}
			outceps[i] = tmp;

			if(strcmp(name, "FFTW")==0){
				reference_mfcc[i]=outceps[i];
			}else{
				mse += (outceps[i]-reference_mfcc[i])*(outceps[i]-reference_mfcc[i]);
			}

			if(PRINT_COEFF){
				printf("%4.4f  ", outceps[i]);
			}
		}
		if(PRINT_COEFF){
			printf("\n\n");
		}
	}else{
		if(PRINT_COEFF){
			printf("MFCCs with %s\n", name);
		}
		for(i = 1; i < nceps+1; i++) {
			float tmp = 0.;
			for(j = 0; j < totalFilters; j++){
				tmp = tmp + mfccDCTMatrix[i][j] * earMag[j];
			}
			outceps[i-1] = tmp;

			if(strcmp(name, "FFTW")==0){
				reference_mfcc[i]=outceps[i];
			}else{
				mse += (outceps[i]-reference_mfcc[i])*(outceps[i]-reference_mfcc[i]);
			}

			if(PRINT_COEFF){
				printf("%4.4f  ", outceps[i]);
			}
		}
		if(PRINT_COEFF){
			printf("\n\n");
		}
	}


	return mse/(float)nceps;
}

















































#define MAIN
#ifdef  MAIN


#define fftw_malloc            fftwf_malloc
#define fftw_free              fftwf_free
#define fftw_execute           fftwf_execute
#define fftw_plan_dft_r2c_1d   fftwf_plan_dft_r2c_1d
#define fftw_plan_dft_c2r_1d   fftwf_plan_dft_c2r_1d
#define fftw_plan_dft_1d		fftwf_plan_dft_1d
#define fftw_plan_r2r_1d       fftwf_plan_r2r_1d
#define fftw_plan              fftwf_plan
#define fftw_complex           fftwf_complex
#define fftw_destroy_plan      fftwf_destroy_plan

#define FIXED_POINT (16)

#include "kiss_fft.h"
#include <time.h>

int main(int argc, char *argv[]){
	int repeat = 1;

	int jj=0;
#ifdef FFTS_SUPPORTED
	float statistic[5][2]={0.0f};
	uint64_t cycles[5]={0};
	float mse_mfccs[5]={0.0f};
#else
	float statistic[5][2]={0.0f};
	uint64_t cycles[5]={0};
	float mse_mfccs[5]={0.0f};
#endif
	while(jj<repeat){
		struct timeval init_fftw3, end_fftw3;
		struct timeval init_fixed, end_fixed;
		struct timeval init_own, end_own;
		struct timeval init_kiss, end_kiss;

#ifdef FFTS_SUPPORTED
		struct timeval init_ffts, end_ffts;
#endif

		if(argc>1){
			if(strcmp(argv[1],"print")==0){
				PRINT_COEFF=1;
			}
		}

#ifdef FFTS_SUPPORTED
		/*ffts*/
//		float __attribute__ ((aligned(32))) *ffts_in = valloc(2*N*sizeof(float));
//		float __attribute__ ((aligned(32))) *ffts_out = valloc(2*N*sizeof(float));
//		float __attribute__ ((aligned(32))) *ffts_out2 = valloc(2*N*sizeof(float));
		float ffts_in[2*N], ffts_out[2*N], ffts_out2[2*N];
#endif

		/*kiss fft*/
	    kiss_fft_cpx  *kiss_in = (kiss_fft_cpx*)malloc(N*sizeof(kiss_fft_cpx));
	    kiss_fft_cpx  *kiss_out = (kiss_fft_cpx*)malloc(N*sizeof(kiss_fft_cpx));
	    kiss_fft_cpx  *kiss_out2 = (kiss_fft_cpx*)malloc(N*sizeof(kiss_fft_cpx));

		/*own implementation*/
		float realin[N], imagin[N];
		float realout[N], imagout[N];
		float realout2[N], imagout2[N];

		/*fftw3*/
		fftw_complex fftw3in[N], fftw3in2[N], fftw3out[N];
		float fftlogmags[N]={0.0f};
		fftw_plan plan_backward;
		fftw_plan plan_forward;


		/*16-bit integer fft*/
		fixed real[N]={0.0f}, imag[N]={0.0f}, logmags[N]={0.0f};
		int i;


		for(i=0; i<N; i++){
				real[i] = 10000*cos(i*2*3.1415926535/N);
				imag[i] = 0;
		}

		if(PRINT_COEFF){
			printf("Input:\n");
			for(i=0; i<N; i++){
				printf("<%d: %d, %d>  ", i, real[i], imag[i]);
			}
			printf("\n\n");
		}


		/*copy data to other libs*/
		for(i=0; i<N; i++){
			//fftw3
			fftw3in[i][0] = (float)real[i];
			fftw3in[i][1] = (float)imag[i];

			//own
			realin[i] = (float)real[i];
			imagin[i] = (float)imag[i];

			kiss_in[i].r = (kiss_fft_scalar)real[i];
			kiss_in[i].i = (kiss_fft_scalar)imag[i];


#ifdef FFTS_SUPPORTED
			//ffts
			ffts_in[2*i] = (float)real[i];
			ffts_in[2*i + 1] = (float)imag[i];
#endif
		}




		/*float FFTW3*/
		plan_forward = fftw_plan_dft_1d(N, fftw3in, fftw3out, FFTW_FORWARD, FFTW_ESTIMATE);

		gettimeofday(&init_fftw3, NULL);
#if (!__arm__)
		uint64_t temp1 = rdtsc();
#endif
		fftw_execute(plan_forward);
#if (!__arm__)
		cycles[0] += rdtsc()-temp1;
#endif
		gettimeofday(&end_fftw3, NULL);

/*&&&&&&&&&&&&&&&&&&&&&&&&&*/
		float rr[N], ii[N];
		for(i=0; i<N; i++){
			rr[i] = fftw3out[i][0]/(float)N;
			ii[i] = fftw3out[i][1]/(float)N;
		}
		mse_mfccs[0] += extract_mfccs(rr, ii, "FFTW");
/*&&&&&&&&&&&&&&&&&&&&&&&&&*/

		if(PRINT_COEFF){
			printf("FFTW3-Float:\n");
			for(i=0; i<N; i++){
				fftlogmags[i] = sqrt((fftw3out[i][0]/(float)N)*(fftw3out[i][0]/(float)N) + (fftw3out[i][1]/(float)N)*(fftw3out[i][1]/(float)N));
				fftlogmags[i] = fftlogmags[i]>0?log(fftlogmags[i]):0;
				printf("<%d: %4.4f, %4.4f, %4.4f>  ", i, fftw3out[i][0]/(float)N, fftw3out[i][1]/(float)N,
						fftlogmags[i]);
			}
			printf("\n\n");
		}


		/*FFT*/
		gettimeofday(&init_fixed, NULL);
#if (!__arm__)
		temp1 = rdtsc();
#endif
		fix_fft(real, imag, M, 0);
#if (!__arm__)
		cycles[1] += rdtsc()-temp1;
#endif
		gettimeofday(&end_fixed, NULL);


/*&&&&&&&&&&&&&&&&&&&&&&&&&*/
		for(i=0; i<N; i++){
			rr[i] = (float)real[i];
			ii[i] = (float)imag[i];
		}
		mse_mfccs[1] += extract_mfccs(rr, ii, "FFT_INT");
/*&&&&&&&&&&&&&&&&&&&&&&&&&*/

		if(PRINT_COEFF){
			printf("FFT:\n");
			for(i=0; i<N; i++){
				logmags[i] = (fixed)sqrt(real[i]*real[i]+imag[i]*imag[i]);
				logmags[i] = (fixed)(logmags[i]>0?log(logmags[i]):0);
				printf("<%d: %d, %d, %d>  ", i, real[i], imag[i], logmags[i]);
			}
			printf("\n\n");
		}


		/*OWN*/
		gettimeofday(&init_own, NULL);
#if (!__arm__)
		temp1 = rdtsc();
#endif
		Own_FFT(0, realin, imagin, realout, imagout);
#if (!__arm__)
		cycles[2] += rdtsc()-temp1;
#endif
		gettimeofday(&end_own, NULL);


/*&&&&&&&&&&&&&&&&&&&&&&&&&*/
		for(i=0; i<N; i++){
			rr[i] = realout[i]/(float)N;
			ii[i] = imagout[i]/(float)N;
		}
		mse_mfccs[2] += extract_mfccs(rr, ii, "FFT_OWN");
/*&&&&&&&&&&&&&&&&&&&&&&&&&*/

		if(PRINT_COEFF){
			printf("Own-FFT:\n");
			for(i=0; i<N; i++){
				printf("<%d: %4.4f, %4.4f>  ", i, realout[i]/(float)N, imagout[i]/(float)N);
			}
			printf("\n\n");
		}



		/*kiss fft*/
		kiss_fft_cfg  kiss_cfg = kiss_fft_alloc(N, 0, 0, 0);
		gettimeofday(&init_kiss, NULL);
#if (!__arm__)
		temp1 = rdtsc();
#endif
		kiss_fft(kiss_cfg, kiss_in, kiss_out);
#if (!__arm__)
		cycles[4] += rdtsc()-temp1;
#endif
		gettimeofday(&end_kiss, NULL);



/*&&&&&&&&&&&&&&&&&&&&&&&&&*/
		for(i=0; i<N; i++){
#ifdef FIXED_POINT
			rr[i] = (float)kiss_out[i].r;
			ii[i] = (float)kiss_out[i].i;
#else
			rr[i] = kiss_out[i].r/(float)N;
			ii[i] = kiss_out[i].i/(float)N;
#endif
		}
		mse_mfccs[4] += extract_mfccs(rr, ii, "FFT_KISS");
/*&&&&&&&&&&&&&&&&&&&&&&&&&*/

		if(PRINT_COEFF){
			printf("KISS-FFT:\n");
			for(i=0; i<N; i++){
#ifdef FIXED_POINT
				printf("<%d: %4.4f, %4.4f>  ", i, (float)kiss_out[i].r, (float)kiss_out[i].i);
#else
				printf("<%d: %4.4f, %4.4f>  ", i, kiss_out[i].r/(float)N, kiss_out[i].i/(float)N);
#endif
			}
			printf("\n\n");
		}





#ifdef FFTS_SUPPORTED
		/*FFTS*/
		int sign = POSITIVE_SIGN;
		ffts_plan_t *p = ffts_init_1d(N, sign);
		if(p) {
			gettimeofday(&init_ffts, NULL);
#if (!__arm__)
			temp1 = rdtsc();
#endif
			ffts_execute(p, ffts_in, ffts_out);
#if (!__arm__)
			cycles[3] += rdtsc()-temp1;
#endif
			gettimeofday(&end_ffts, NULL);


/*&&&&&&&&&&&&&&&&&&&&&&&&&*/
		for(i=0; i<N; i++){
			rr[i] = ffts_out[2*i]/(float)N;
			ii[i] = ffts_out[2*i+1]/(float)N;
		}
		mse_mfccs[3] += extract_mfccs(rr, ii, "FFTS");
/*&&&&&&&&&&&&&&&&&&&&&&&&&*/


			if(PRINT_COEFF){
				printf("FFTS:\n");
				for(i=0;i<N;i++){
					printf("<%d: %4.4f %4.4f>  ", i, ffts_out[2*i]/(float)N, ffts_out[2*i+1]/(float)N);
				}
				printf("\n\n");
			}
		}else{
			if(PRINT_COEFF){
				printf("Plan unsupported\n\n");
			}
		}
#endif





		if(PRINT_COEFF){
			printf("DIFF in FFT:\n");
			float mse=0, mseR=0, mseI;
			for(i=0; i<N; i++){
		//		printf("<%d: %4.4f, %4.4f>\n", i, fabs((fftw3out[i][0]/(float)N)-(float)real[i]), fabs((fftw3out[i][1]/(float)N)-(float)imag[i]));
				printf("<%d: %4.4f, %d, %4.4f>\n", i, fftlogmags[i], logmags[i], fabs(fftlogmags[i]-logmags[i]));
				mseR += (((fftw3out[i][0]/(float)N)-(float)real[i]))*((fftw3out[i][0]/(float)N)-(float)real[i]);
				mseI += (((fftw3out[i][1]/(float)N)-(float)imag[i]))*((fftw3out[i][1]/(float)N)-(float)imag[i]);
				mse += (fftlogmags[i]-logmags[i])*(fftlogmags[i]-logmags[i]);
			}
			mse /= (float)N;
			mseR /= (float)N;
			printf("Mean squared error detected was: logMag: %f, Real: %f, Imag: %f\n", mse, mseR, mseI);
			printf("\n\n");


			printf("====================================================================================================================\n\n");
			printf("====================================================================================================================\n\n");
			printf("====================================================================================================================\n\n");
			printf("====================================================================================================================\n\n");
			printf("====================================================================================================================\n\n");
			printf("====================================================================================================================\n\n");
			printf("====================================================================================================================\n\n");
		}

		/*======================================================================================================================================================*/
		/*======================================================================================================================================================*/
		/*======================================================================================================================================================*/
		/*======================================================================================================================================================*/
		/*======================================================================================================================================================*/
		/*======================================================================================================================================================*/
		/*======================================================================================================================================================*/
		/*======================================================================================================================================================*/

		float diff_fftw=0, diff_ffts=0, diff_own=0, diff_int=0, diff_kiss=0;

		/*float Inv FFTW3*/
		plan_backward = fftw_plan_dft_1d(N, fftw3out, fftw3in2, FFTW_BACKWARD, FFTW_ESTIMATE);
		fftw_execute(plan_backward);

		for(i=0; i<N; i++){
			diff_fftw += (fftw3in2[i][0]/(float)N - fftw3in[i][0])*(fftw3in2[i][0]/(float)N - fftw3in[i][0]) +
					(fftw3in2[i][1]/(float)N - fftw3in[i][1])*(fftw3in2[i][1]/(float)N - fftw3in[i][1]);
		}

		if(PRINT_COEFF){
			printf("InvFFTW3-Float:\n");
			for(i=0; i<N; i++){
				printf("<%d: %4.4f, %4.4f>  ", i, fftw3in2[i][0]/(float)N - fftw3in[i][0], fftw3in2[i][1]/(float)N - fftw3in[i][1]);
			}
			printf("\n\n");
		}


		/*INVERSE OWN*/
		fix_fft(real, imag, M, 1);

		for(i=0; i<N; i++){
			diff_int += ((float)real[i] - fftw3in[i][0])*((float)real[i] - fftw3in[i][0]) +
					((float)imag[i] - fftw3in[i][1])*((float)imag[i] - fftw3in[i][1]);
		}

		if(PRINT_COEFF){
			printf("InvFFT:\n");
			for(i=0; i<N; i++){
				printf("<%d: %4.4f, %4.4f>  ", i, (float)real[i] - fftw3in[i][0], (float)imag[i] - fftw3in[i][1]);
			}
			printf("\n\n");
		}


		/*own*/
		Own_FFT(1, realout, imagout, realout2, imagout2);

		for(i=0; i<N; i++){
			diff_own += (realin[i]-realout2[i])*(realin[i]-realout2[i]) +
					(imagin[i]-imagout2[i])*(imagin[i]-imagout2[i]);
		}

		if(PRINT_COEFF){
			printf("Inv-Own-FFT:\n");
			for(i=0; i<N; i++){
				printf("<%d: %4.4f, %4.4f>  ", i, realin[i]-realout2[i], imagin[i]-imagout2[i]);
			}
			printf("\n\n");
		}


		/*Kiss FFt inverse*/
		kiss_fft_cfg  kiss_cfg_inv = kiss_fft_alloc(N, 1, 0, 0);
		kiss_fft(kiss_cfg_inv, kiss_out, kiss_out2);

		for(i=0; i<N; i++){
#ifdef FIXED_POINT
			diff_kiss += (float)(kiss_in[i].r-kiss_out2[i].r/(float)N)*(kiss_in[i].r-kiss_out2[i].r/(float)N) +
					(float)(kiss_in[i].i-kiss_out2[i].i/(float)N)*(kiss_in[i].i-kiss_out2[i].i/(float)N);
#else
			diff_kiss += (kiss_in[i].r-kiss_out2[i].r/(float)N)*(kiss_in[i].r-kiss_out2[i].r/(float)N) +
					(kiss_in[i].i-kiss_out2[i].i/(float)N)*(kiss_in[i].i-kiss_out2[i].i/(float)N);
#endif
		}

		if(PRINT_COEFF){
			printf("Inv-Kiss-FFT:\n");
			for(i=0; i<N; i++){
#ifdef FIXED_POINT
				printf("<%d: %4.4f, %4.4f>  ", i, (float)kiss_in[i].r-(float)kiss_out2[i].r/(float)N,
						(float)kiss_in[i].i-(float)kiss_out2[i].i/(float)N);
#else
				printf("<%d: %4.4f, %4.4f>  ", i, kiss_in[i].r-kiss_out2[i].r/(float)N, kiss_in[i].i-kiss_out2[i].i/(float)N);
#endif
			}
			printf("\n\n");
		}

		if(kiss_cfg){
			free(kiss_cfg);
		}
		if(kiss_cfg_inv){
			free(kiss_cfg_inv);
		}


		if(plan_forward){
			fftw_destroy_plan(plan_forward);
			plan_forward = 0;
		}

#ifdef FFTS_SUPPORTED
		/*FFTS inverse*/
		sign = NEGATIVE_SIGN;
		ffts_plan_t *p2 = ffts_init_1d(N, sign);
		if(p) {
			ffts_execute(p2, ffts_out, ffts_out2);
	//		ffts_execute(p, ffts_out, ffts_out2);

			for(i=0; i<N; i++){
				diff_ffts += (ffts_out2[2*i]/(float)N-ffts_in[2*i])*(ffts_out2[2*i]/(float)N-ffts_in[2*i]) +
						(ffts_out2[2*i+1]/(float)N-ffts_in[2*i+1])*(ffts_out2[2*i+1]/(float)N-ffts_in[2*i+1]);
			}

			if(PRINT_COEFF){
				printf("Inv-FFTS:\n");
				for(i=0;i<N;i++){
					printf("<%d: %4.4f %4.4f>  ", i, ffts_out2[2*i]/(float)N-ffts_in[2*i],
							ffts_out2[2*i+1]/(float)N-ffts_in[2*i+1]);
				}
				printf("\n\n");
			}
		}else{
			if(PRINT_COEFF){
				printf("Plan unsupported\n\n");
			}
		}
#endif





	    free(kiss_in);
	    free(kiss_out);
	    free(kiss_out2);




#ifdef FFTS_SUPPORTED
		/*ffts*/
		if(p && p2){
	//	if(p){
			ffts_free(p2);
			ffts_free(p);

//			free(ffts_in);
//			free(ffts_out);
//			free(ffts_out2);
		}
#endif


#ifdef FFTS_SUPPORTED
		//fftw
		statistic[0][0] += diff_fftw/(float)N;//error mse
		statistic[0][1] += (((float)((end_fftw3.tv_sec-init_fftw3.tv_sec)*1000000 + (end_fftw3.tv_usec-init_fftw3.tv_usec)))/(float)1000000.0)*1000.0;//time

		//int
		statistic[1][0] += diff_int/(float)N;//error mse
		statistic[1][1] += (((float)((end_fixed.tv_sec-init_fixed.tv_sec)*1000000 + (end_fixed.tv_usec-init_fixed.tv_usec)))/(float)1000000.0)*1000.0;//time

		//own
		statistic[2][0] += diff_own/(float)N;//error mse
		statistic[2][1] += (((float)(((end_own.tv_sec-init_own.tv_sec)*1000000) + (end_own.tv_usec-init_own.tv_usec)))/(float)1000000.0)*1000.0;//time

		//ffts
		statistic[3][0] += diff_ffts/(float)N;//error mse
		statistic[3][1] += (((float)((end_ffts.tv_sec-init_ffts.tv_sec)*1000000 + (end_ffts.tv_usec-init_ffts.tv_usec)))/(float)1000000.0)*1000.0;//time

		//kiss
		statistic[4][0] += diff_kiss/(float)N;//error mse
		statistic[4][1] += (((float)(((end_kiss.tv_sec-init_kiss.tv_sec)*1000000) + (end_kiss.tv_usec-init_kiss.tv_usec)))/(float)1000000.0)*1000.0;//time
#else

		//fftw
		statistic[0][0] += diff_fftw/(float)N;//error mse
		statistic[0][1] += (((float)((end_fftw3.tv_sec-init_fftw3.tv_sec)*1000000 + (end_fftw3.tv_usec-init_fftw3.tv_usec)))/(float)1000000.0)*1000.0;//time

		//int
		statistic[1][0] += diff_int/(float)N;//error mse
		statistic[1][1] += (((float)((end_fixed.tv_sec-init_fixed.tv_sec)*1000000 + (end_fixed.tv_usec-init_fixed.tv_usec)))/(float)1000000.0)*1000.0;//time

		//own
		statistic[2][0] += diff_own/(float)N;//error mse
		statistic[2][1] += (((float)(((end_own.tv_sec-init_own.tv_sec)*1000000) + (end_own.tv_usec-init_own.tv_usec)))/(float)1000000.0)*1000.0;//time

		//kiss
		statistic[4][0] += diff_kiss/(float)N;//error mse
		statistic[4][1] += (((float)(((end_kiss.tv_sec-init_kiss.tv_sec)*1000000) + (end_kiss.tv_usec-init_kiss.tv_usec)))/(float)1000000.0)*1000.0;//time
#endif
		jj++;
		printf(".");
	}
	printf("\n\n\n");

#ifdef FFTS_SUPPORTED
	printf("FFTW: %f ms (err: %f) (cycles: %f) (mfcc_err: %f)\n"
			"FIXED: %f ms (err: %f) (cycles: %f) (mfcc_err: %f)\n"
			"OWN: %f ms (err: %f) (cycles: %f) (mfcc_err: %f)\n"
			"FFTS: %f ms (err: %f) (cycles: %f) (mfcc_err: %f)\n"
			"KISS-FFT: %f ms (err: %f) (cycles: %f) (mfcc_err: %f)\n\n",
			statistic[0][1]/(float)jj,
			statistic[0][0]/(float)jj,
#if (!__arm__)
			((float)cycles[0]/(float)jj),
#endif
			mse_mfccs[0]/(float)jj,

			statistic[1][1]/(float)jj,
			statistic[1][0]/(float)jj,
#if (!__arm__)
			((float)cycles[1]/(float)jj),
#endif
			mse_mfccs[1]/(float)jj,

			statistic[2][1]/(float)jj,
			statistic[2][0]/(float)jj,
#if (!__arm__)
			((float)cycles[2]/(float)jj),
#endif
			mse_mfccs[2]/(float)jj,

			statistic[3][1]/(float)jj,
			statistic[3][0]/(float)jj
#if (!__arm__)
			,((float)cycles[3]/(float)jj)
#endif
			,mse_mfccs[3]/(float)jj,

			statistic[4][1]/(float)jj,
			statistic[4][0]/(float)jj
#if (!__arm__)
			,((float)cycles[4]/(float)jj)
#endif
			,mse_mfccs[4]/(float)jj
			);
#else
	printf("FFTW: %f ms (err: %f) (cycles: %f) (mfcc_err: %f)\n"
			"FIXED: %f ms (err: %f) (cycles: %f) (mfcc_err: %f)\n"
			"OWN: %f ms (err: %f) (cycles: %f) (mfcc_err: %f)\n"
			"KISS-FFT: %f ms (err: %f) (cycles: %f) (mfcc_err: %f)\n",
			statistic[0][1]/(float)jj,
			statistic[0][0]/(float)jj,
#if (!__arm__)
			((float)cycles[0]/(float)jj),
#endif
			mse_mfccs[0]/(float)jj,


			statistic[1][1]/(float)jj,
			statistic[1][0]/(float)jj,
#if (!__arm__)
			((float)cycles[1]/(float)jj),
#endif
			mse_mfccs[1]/(float)jj,


			statistic[2][1]/(float)jj,
			statistic[2][0]/(float)jj
#if (!__arm__)
			,((float)cycles[2]/(float)jj)
#endif
			,mse_mfccs[2]/(float)jj,

			statistic[4][1]/(float)jj,
			statistic[4][0]/(float)jj
#if (!__arm__)
			,((float)cycles[4]/(float)jj)
#endif
			,mse_mfccs[4]/(float)jj
			);
#endif

    return 0;
}
#endif  /* MAIN */

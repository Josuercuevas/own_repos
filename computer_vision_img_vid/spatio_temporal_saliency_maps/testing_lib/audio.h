#include "includes_libs.h"

/* Call this instead of exit(), so we can clean up SDL: atexit() is evil. */
static void quit(int rc);

/* NOTE: These RGB conversion functions are not intended for speed,
         only as examples.
*/

void RGBtoYUV(Uint8 * rgb, int *yuv, int monochrome, int luminance);

void ConvertRGBtoYV12(Uint8 *rgb, Uint8 *out, int w, int h,int monochrome, int luminance);

int audio_main(const char* path);
/* vi: set ts=4 sw=4 expandtab: */
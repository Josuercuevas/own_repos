#include "cpp_includes.h"

int initialize_gpu();
int MultiGPUs(unsigned char *frames[], unsigned char *maps[],const int width, const int height, 
		const int n_frames, const int channels, const int orientations, const int pyramid_levels, float *ages,
		unsigned char *temp_output[], unsigned char *output[], size_t frame_size);
int SingleGPU(unsigned char *frames[], unsigned char *maps[],const int width, const int height, 
		const int n_frames, const int channels, const int orientations, const int pyramid_levels, float *ages,
		unsigned char *temp_output[], unsigned char *output[], size_t frame_size);
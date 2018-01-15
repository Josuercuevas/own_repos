# pragma comment (lib, "avformat.lib")
# pragma comment (lib, "avutil.lib")
# pragma comment (lib, "avcodec.lib")
# pragma comment (lib, "swscale.lib")
# pragma comment (lib, "avdevice.lib")
# pragma comment (lib, "SDL2.lib")

extern "C" 
{
    #ifndef __STDC_CONSTANT_MACROS
    #define __STDC_CONSTANT_MACROS
    #endif
    #include <libavcodec\avcodec.h>
    #include <libavformat\avformat.h>
    #include <libswscale\swscale.h>
    #include <libavutil\avutil.h>
	#include <libavdevice\avdevice.h>
	#include <SDL.h>
	#include <SDL_thread.h>
}

//#ifdef __MINGW32__
//#undef main /* Prevents SDL from overriding main() */
//#endif

#define SDL_AUDIO_BUFFER_SIZE 1024
#define MAX_AUDIO_FRAME_SIZE 192000

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <tchar.h>

using namespace std;
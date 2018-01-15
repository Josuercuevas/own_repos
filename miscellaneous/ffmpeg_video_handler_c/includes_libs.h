/*
Video/Camera decoder or frame extractor using FFMPEG library

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
// #define SDL_AUDIO_BUFFER_SIZE 1024
// #define MAX_AUDIO_FRAME_SIZE 192000

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <tchar.h>

using namespace std;

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

#include "includes_libs.h"

int read_video_file(const char* path, AVFormatContext *&pFormatCtx, int &videoStream, AVCodecContext  *&pCodecCtx,
	AVCodec *&pCodec, AVFrame *&pFrame, AVFrame *&pFrameRGB, AVPacket &packet, int &frameFinished,
	int &numBytes, uint8_t *&buffer,AVDictionary *&optionsDict, struct SwsContext *&sws_ctx);

int extract_frame(AVFormatContext *&pFormatCtx,AVPacket &packet,int &videoStream,AVCodecContext  *&pCodecCtx,
	int &frameFinished, struct SwsContext *&sws_ctx, AVFrame *&pFrame, AVFrame *&pFrameRGB);

int camera_main(const char *device_name, AVCodecContext  *&pCodecCtx, AVFormatContext *&pFormatCtx,
	AVCodec *&pCodec, AVInputFormat *&iformat, AVFrame *&pFrame, AVFrame *&pFrameRGB, int &videoStream,
	uint8_t *&buffer, int &numBytes, AVPixelFormat  &pFormat, int &frameFinished);

int extract_camera_frame(AVFormatContext *&pFormatCtx, int &res, AVPacket &packet, int &videoStream,
	AVCodecContext  *&pCodecCtx, AVFrame *&pFrame, AVFrame *&pFrameRGB, int &frameFinished,
	struct SwsContext *&img_convert_ctx);

#include "includes_libs.h"
#include <CImg.h>

using namespace cimg_library;

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
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

#include "video.h"


int read_video_file(const char* path, AVFormatContext *&pFormatCtx, int &videoStream, AVCodecContext  *&pCodecCtx, AVCodec *&pCodec, AVFrame *&pFrame, AVFrame *&pFrameRGB, AVPacket &packet,
	int &frameFinished, int &numBytes, uint8_t *&buffer,AVDictionary *&optionsDict, struct SwsContext *&sws_ctx) {


  int i;
  // Register all formats and codecs
  av_register_all();

  // Open video file
  if(avformat_open_input(&pFormatCtx, path, NULL, NULL)!=0)
    return -10; // Couldn't open file

  // Retrieve stream information
  if(avformat_find_stream_info(pFormatCtx, NULL)<0)
    return -11; // Couldn't find stream information

  // Dump information about file onto standard error
  ///av_dump_format(pFormatCtx, 0, path, 0);

  // Find the first video stream
  videoStream=-12;
  for(i=0; i<pFormatCtx->nb_streams; i++)
    if(pFormatCtx->streams[i]->codec->codec_type==AVMEDIA_TYPE_VIDEO) {
      videoStream=i;
      break;
    }
  if(videoStream==-1)
    return -13; // Didn't find a video stream

  // Get a pointer to the codec context for the video stream
  pCodecCtx=pFormatCtx->streams[videoStream]->codec;

  // Find the decoder for the video stream
  pCodec=avcodec_find_decoder(pCodecCtx->codec_id);
  if(pCodec==NULL) {
    fprintf(stderr, "Unsupported codec!\n");
    return -14; // Codec not found
  }
  // Open codec
  if(avcodec_open2(pCodecCtx, pCodec, &optionsDict)<0)
    return -15; // Could not open codec

  // Allocate video frame
  pFrame=av_frame_alloc();

  // Allocate an AVFrame structure
  pFrameRGB=av_frame_alloc();
  if(pFrameRGB==NULL)
    return -16;

  // Determine required buffer size and allocate buffer
  numBytes=avpicture_get_size(PIX_FMT_RGB24, pCodecCtx->width, pCodecCtx->height);
  buffer=(uint8_t *)av_malloc(numBytes*sizeof(uint8_t));

  sws_ctx =
    sws_getContext
    (
        pCodecCtx->width,
        pCodecCtx->height,
        pCodecCtx->pix_fmt,
        pCodecCtx->width,
        pCodecCtx->height,
        PIX_FMT_RGB24,
        SWS_BILINEAR,
        NULL,
        NULL,
        NULL
    );

  // Assign appropriate parts of buffer to image planes in pFrameRGB
  // Note that pFrameRGB is an AVFrame, but AVFrame is a superset
  // of AVPicture
  avpicture_fill((AVPicture *)pFrameRGB, buffer, PIX_FMT_RGB24, pCodecCtx->width, pCodecCtx->height);

  return 0;//return with no problem
}


int extract_frame(AVFormatContext *&pFormatCtx,AVPacket &packet,int &videoStream,AVCodecContext  *&pCodecCtx, int &frameFinished, struct SwsContext *&sws_ctx, AVFrame *&pFrame,
	AVFrame *&pFrameRGB){

	bool got_frame = false;
	while(av_read_frame(pFormatCtx, &packet)>=0) {
	// Is this a packet from the video stream?
		if(packet.stream_index==videoStream) {
			// Decode video frame
			avcodec_decode_video2(pCodecCtx, pFrame, &frameFinished, &packet);

			// Did we get a video frame?
			if(frameFinished) {
			// Convert the image from its native format to RGB
				sws_scale
				(
					sws_ctx,
					(uint8_t const * const *)pFrame->data,
					pFrame->linesize,
					0,
					pCodecCtx->height,
					pFrameRGB->data,
					pFrameRGB->linesize
				);
				av_free_packet(&packet);
				got_frame = true;
				break;
			}
			av_free_packet(&packet);
		}

		// Free the packet that was allocated by av_read_frame
		//gets here iff the frame was not a video frame
		av_free_packet(&packet);
	}
	if(got_frame)
		return 0;
	else
		return -1;
}



int camera_main(const char *device_name, AVCodecContext  *&pCodecCtx, AVFormatContext *&pFormatCtx, AVCodec *&pCodec, AVInputFormat *&iformat, AVFrame *&pFrame, AVFrame *&pFrameRGB,
	int &videoStream, uint8_t *&buffer, int &numBytes, AVPixelFormat  &pFormat, int &frameFinished){

	// Register all formats and codecs
	avdevice_register_all();
	avcodec_register_all();

	if(avformat_open_input(&pFormatCtx,device_name,iformat,NULL) != 0)
		return -12;//if we cannot open the streamer device

	if(avformat_find_stream_info(pFormatCtx, NULL) < 0)
		return -13;//check if we are able to decode and recognize the frames coming from this device

	av_dump_format(pFormatCtx, 0, device_name, 0);//in case we want to see the information of the device

	for(int i=0; i < pFormatCtx->nb_streams; i++)
    {
        if(pFormatCtx->streams[i]->codec->coder_type==AVMEDIA_TYPE_VIDEO)
        {
            videoStream = i;//video type to be used for decoding
            break;
        }
    }

    if(videoStream == -1)
		return -14;//we were not able to find a suitable decoder from this device

    pCodecCtx = pFormatCtx->streams[videoStream]->codec;//getting the codec for the context

    pCodec =avcodec_find_decoder(pCodecCtx->codec_id);//finding the codec in the database of ffmpeg

    if(pCodec==NULL)
		return -15; //codec not found

    if(avcodec_open2(pCodecCtx,pCodec,NULL) < 0)
		return -16;//we are not able to open the codec

	/*Getting the API ready for video streaming*/
    pFrame    = avcodec_alloc_frame();//allocating the frame
    pFrameRGB = avcodec_alloc_frame();//allocating the RGB frame
    pFormat = PIX_FMT_RGB24;//format of the pixels to be used
    numBytes = avpicture_get_size(pFormat,pCodecCtx->width,pCodecCtx->height);//size of frame in bytes
    buffer = (uint8_t *) av_malloc(numBytes*sizeof(uint8_t));//allocating the memory for the buffer
    avpicture_fill((AVPicture *) pFrameRGB,buffer,pFormat,pCodecCtx->width,pCodecCtx->height);//making a copy for the RGB frame
	/*Getting the API ready for video streaming*/

	return 0;//returns with no problem
}



int extract_camera_frame(AVFormatContext *&pFormatCtx, int &res, AVPacket &packet, int &videoStream, AVCodecContext  *&pCodecCtx, AVFrame *&pFrame, AVFrame *&pFrameRGB, int &frameFinished,
	struct SwsContext *&img_convert_ctx){

	bool got_frame = false;

	while(res = av_read_frame(pFormatCtx,&packet)>=0)
    {
        if(packet.stream_index == videoStream){//is it a video peacket
            avcodec_decode_video2(pCodecCtx,pFrame,&frameFinished,&packet);
            if(frameFinished){//finished decoding frame
                img_convert_ctx = sws_getCachedContext(NULL,pCodecCtx->width, pCodecCtx->height, pCodecCtx->pix_fmt,   pCodecCtx->width, pCodecCtx->height,
					PIX_FMT_RGB24, SWS_BICUBIC, NULL, NULL,NULL);
                sws_scale(img_convert_ctx, ((AVPicture*)pFrame)->data, ((AVPicture*)pFrame)->linesize, 0, pCodecCtx->height, ((AVPicture *)pFrameRGB)->data,
					((AVPicture *)pFrameRGB)->linesize);
				av_free_packet(&packet);
				sws_freeContext(img_convert_ctx);
				got_frame = true;
				break;
            }
        }
		av_free_packet(&packet);
    }

	if(got_frame)
		return 0;
	else
		return -1;

}

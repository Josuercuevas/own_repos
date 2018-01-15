#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "video.h"

#include "saliency_functions.h"
#include <time.h>

using namespace std;
using namespace saliency;

int _tmain()
{
	//video path
	const char *path = "../test_videos/11.mp4";//

	if(path==NULL){
		printf("problem opening the video file :(  !!");
		system("pause");
		return -1;
	}

	bool first_time=true;

	/*reading the video file with ffmpeg*/

	AVFormatContext *pFormatCtx = NULL;
	int             i=0, videoStream;
	AVCodecContext  *pCodecCtx = NULL;
	AVCodec         *pCodec = NULL;
	AVFrame         *pFrame = NULL; 
	AVFrame         *pFrameRGB = NULL;
	AVPacket        packet;
	int             frameFinished;
	int             numBytes;
	uint8_t         *buffer = NULL;

	AVDictionary    *optionsDict = NULL;
	struct SwsContext      *sws_ctx = NULL;

	if(read_video_file(path,pFormatCtx,videoStream,pCodecCtx,pCodec,pFrame,pFrameRGB,packet,frameFinished,numBytes,buffer,optionsDict,sws_ctx))
	{
		printf("problem opening the video or camera using ffmpeg :(  !!");
		system("pause");
		return -1;
	}
	/*reading the video file with ffmpeg*/

	const int n_frames = 4;
	int frame_width, frame_height, frame_channels;
	float *ages;
	unsigned char *blended_frames[1];//blended images coming from combining the original and mapped frames
	unsigned char *inputdata[n_frames];//contains the frames
	unsigned char *outputdata[2];//contains the ouput maps (one static map per frame, and 1 dynamic maps every frame sequence)
	unsigned char *Saliency_maps[1];//contain all the master saliency maps after fusion
	unsigned char *YUV_frames[n_frames];

	int index = 0;
	bool finished_video = false;

	//initialization of the library
	saliency::Device_Initialization();

	CImgDisplay main_window, staticmap, dynamicmap, blended, mastermap, agesmaps;
	for(;;){

		for(int j=0;j<n_frames;j++){
			if(i==0){//allocates the memory just once
				if(extract_frame(pFormatCtx,packet,videoStream,pCodecCtx,frameFinished,sws_ctx,pFrame,pFrameRGB))//reading frame with ffmpeg
					finished_video = true;//ended the processing of the video
				if(finished_video)
					break;
				frame_width = pCodecCtx->width; frame_height = pCodecCtx->height; frame_channels = 3;//for the RGB data


				YUV_frames[j] = (unsigned char*)malloc(sizeof(unsigned char)*((frame_width*frame_height) + (frame_width/2)*(frame_height/2) + (frame_width/2)*(frame_height/2)));
				inputdata[j] = (unsigned char*)malloc(sizeof(unsigned char)*frame_width*frame_height*frame_channels);//an RGB
				memcpy(inputdata[j], pFrameRGB->data[0], sizeof(unsigned char)*frame_width*frame_height*frame_channels);//copying the RGB frame				
			}else{
				if(j<(n_frames-1)){
					memcpy(inputdata[j], inputdata[j+1], sizeof(unsigned char)*frame_width*frame_height*frame_channels);//
				}
				else{
					if(extract_frame(pFormatCtx,packet,videoStream,pCodecCtx,frameFinished,sws_ctx,pFrame,pFrameRGB))//reading frame with ffmpeg
						finished_video = true;//ended the processing of the video
					if(finished_video)
						break;
					memcpy(inputdata[j], pFrameRGB->data[0], sizeof(unsigned char)*frame_width*frame_height*frame_channels);//copying the RGB frame
				}
			}
		}


		/*for(int ht=0;ht<n_frames;ht++){
			CImg<unsigned char> im(inputdata[ht], 3, frame_width, frame_height, 1);
			im.permute_axes("yzcx");
			staticmap.display(im);
			system("pause");
		}*/


		//test plannar frames since FFMPEG gives us interleaved
		unsigned char *temp1[10];
		int pix_dist = saliency::PLANAR;
		//int pix_dist = saliency::INTERLEAVED;
		for(int ht=0;ht<n_frames;ht++){
			temp1[ht] = (unsigned char*)malloc(sizeof(unsigned char)*frame_width*frame_height*frame_channels);
			memcpy(temp1[ht],inputdata[ht],sizeof(unsigned char)*frame_width*frame_height*frame_channels);
		}


		/*for(int ht=0;ht<n_frames;ht++){
			CImg<unsigned char> im(inputdata[ht], 3, frame_width, frame_height, 1);
			im.permute_axes("yzcx");
			staticmap.display(im);
			system("pause");
		}*/



		for(int tt=0; tt<frame_height; tt++){
			for(int vv=0; vv<frame_width; vv++){
				if(i==0){
					for(int ht=0;ht<n_frames;ht++){
						inputdata[ht][tt*frame_width + vv] = temp1[ht][(tt*frame_width + vv)*3 + 0];
						inputdata[ht][(tt*frame_width + vv) + frame_width*frame_height] = temp1[ht][(tt*frame_width + vv)*3 + 1];
						inputdata[ht][(tt*frame_width + vv) + 2*frame_width*frame_height] = temp1[ht][(tt*frame_width + vv)*3 + 2];
					}
				}
				else{
					inputdata[n_frames-1][tt*frame_width + vv] = temp1[n_frames-1][(tt*frame_width + vv)*3 + 0];
					inputdata[n_frames-1][(tt*frame_width + vv) + frame_width*frame_height] = temp1[n_frames-1][(tt*frame_width + vv)*3 + 1];
					inputdata[n_frames-1][(tt*frame_width + vv) + 2*frame_width*frame_height] = temp1[n_frames-1][(tt*frame_width + vv)*3 + 2];
				}
			}
		}

		for(int ht=0;ht<n_frames;ht++){
			free(temp1[ht]);
		}

		/*CImg<unsigned char> im(inputdata[0], frame_width, frame_height, 1,3);
		CImg<unsigned char> im2(inputdata[n_frames-1], frame_width, frame_height,1,3);
		staticmap.display(im);
		dynamicmap.display(im2);
		system("pause");*/


		saliency::RGBtoYUV(YUV_frames, inputdata, frame_height,	frame_width, frame_height/2, frame_width/2, frame_height/2, frame_width/2, n_frames, saliency::PLANAR);
		/*CImg<unsigned char> im3(YUV_frames[n_frames-1], frame_width, frame_height, 1, 1);
		CImg<unsigned char> im4(YUV_frames[n_frames-1]+(frame_width*frame_height), frame_width/2, frame_height/2, 1, 1);
		CImg<unsigned char> im5(YUV_frames[n_frames-1]+((frame_width*frame_height) + ((frame_height/2)*(frame_width/2))), frame_width/2, frame_height/2, 1, 1);

		mastermap.display(im3);
		agesmaps.display(im4);
		main_window.display(im5);
		system("pause");*/
		
		
		saliency::YUVtoRGB(YUV_frames, inputdata, frame_height,	frame_width, frame_height/2, frame_width/2, frame_height/2, frame_width/2, n_frames, saliency::PLANAR);
		//CImg<unsigned char> im3(inputdata[0], frame_width, frame_height, 1, 3);
		//CImg<unsigned char> im4(inputdata[1], frame_width, frame_height, 1, 3);
		////CImg<unsigned char> im5(inputdata[2], frame_width, frame_height, 1, 3);

		//mastermap.display(im3);
		//agesmaps.display(im4);
		////main_window.display(im5);
		//system("pause");


		if(finished_video)
			break;

		//array that will contain the maps coming from the library, which are in gray scale values
		if(i==0){
			for(int frame=0;frame<2;frame++){
				outputdata[frame] = (unsigned char*)malloc(sizeof(unsigned char)*frame_width*frame_height);//for static and dynamic maps
			}
			blended_frames[0] = (unsigned char*)malloc(sizeof(unsigned char)*frame_width*frame_height*frame_channels);//blended  frames
			Saliency_maps[0] = (unsigned char*)malloc(sizeof(unsigned char)*frame_width*frame_height);//for combined maps
			ages = (float*)malloc(sizeof(float)*frame_width*frame_height);
		}


		//calling the saliency map library
		time_t start = clock(), end;
		//if(saliency::Motion_compensation(inputdata,frame_width,frame_height,n_frames,frame_channels,i))
			//return -1;
		if(saliency::Saliency_maps(inputdata, outputdata,frame_width,frame_height,n_frames,frame_channels,6,4,ages,saliency::PLANAR))//estimate the maps per pathway
			return -1;
		if(saliency::Fusion_maps(Saliency_maps, outputdata, frame_width, frame_height, 1,AVERAGE))//fusions the maps
			return -1;
		if(saliency::Fading_maps(Saliency_maps,frame_width,frame_height,1,ages))//fading the pixels for the master map
			return -1;
		if(saliency::Frames_Maps_Blend(blended_frames,inputdata,Saliency_maps,frame_width,frame_height,1,frame_channels,saliency::PLANAR))//blending the original frames and master map
			return -1;

		end=clock();
		printf("The time elapsed for the mapping processes was: %f ms.\n",((float)(end-start)/CLOCKS_PER_SEC)*1000);

		
		if(pix_dist == saliency::INTERLEAVED){
			CImg<unsigned char> original(inputdata[0],frame_channels,frame_width,frame_height,1);
			CImg<unsigned char> blended_Map(blended_frames[0],frame_channels,frame_width,frame_height,1);

			CImg<float> Age_maps(frame_width,frame_height);
			CImg<unsigned char> Static_Map(frame_width,frame_height);		
			CImg<unsigned char> Dynamic_Map(frame_width,frame_height);		
			CImg<unsigned char> master_map(frame_width,frame_height);		
		
			memcpy(Age_maps._data,ages,sizeof(float)*frame_width*frame_height);
			memcpy(Static_Map._data,outputdata[0],sizeof(unsigned char)*frame_width*frame_height);
			memcpy(Dynamic_Map._data,outputdata[1],sizeof(unsigned char)*frame_width*frame_height);
			memcpy(master_map._data,Saliency_maps[0],sizeof(unsigned char)*frame_width*frame_height);
			original.permute_axes("yzcx");
			blended_Map.permute_axes("yzcx");
		
			main_window.display(original);
			//agesmaps.display(Age_maps);
			staticmap.display((Static_Map,Dynamic_Map,master_map));
			//dynamicmap.display(Dynamic_Map);
			//mastermap.display(master_map);
			//blended.display(blended_Map);



			/*char f_name[256];
			char num[10];
			char format[10] = ".bmp";
			strcpy(f_name, "../test_videos/frames/frame");
			sprintf(num, "%06i", i+1);
			strcat(f_name, num);   
			strcat(f_name, format);
			blended_Map.save(f_name);*/

			//dynamicmap.wait(1);
			if(first_time){
				system("pause");
				first_time=false;
			}
		}
		else{//PLANAR
			CImg<unsigned char> original(inputdata[0],frame_width,frame_height,1,frame_channels);
			CImg<unsigned char> blended_Map(blended_frames[0],frame_width,frame_height,1,frame_channels);

			CImg<float> Age_maps(frame_width,frame_height);
			CImg<unsigned char> Static_Map(frame_width,frame_height);		
			CImg<unsigned char> Dynamic_Map(frame_width,frame_height);		
			CImg<unsigned char> master_map(frame_width,frame_height);		
		
			memcpy(Age_maps._data,ages,sizeof(float)*frame_width*frame_height);
			memcpy(Static_Map._data,outputdata[0],sizeof(unsigned char)*frame_width*frame_height);
			memcpy(Dynamic_Map._data,outputdata[1],sizeof(unsigned char)*frame_width*frame_height);
			memcpy(master_map._data,Saliency_maps[0],sizeof(unsigned char)*frame_width*frame_height);
		
			main_window.set_title("Video File");
			main_window.display(original);

			agesmaps.set_title("Age map");
			agesmaps.display(Age_maps);
			
			staticmap.set_title("Static map");
			staticmap.display(Static_Map);
			
			dynamicmap.set_title("Dynamic map");
			dynamicmap.display(Dynamic_Map);
			
			mastermap.set_title("Master map");
			mastermap.display(master_map);
			
			blended.set_title("Blended map");
			blended.display(blended_Map);



			/*char f_name[256];
			char num[10];
			char format[10] = ".bmp";
			strcpy(f_name, "../test_videos/frames/frame");
			sprintf(num, "%06i", i+1);
			strcat(f_name, num);   
			strcat(f_name, format);
			blended_Map.save(f_name);*/

			//dynamicmap.wait(1);
			if(first_time){
				system("pause");
				first_time=false;
			}
		}
		i++;
	}

	free(YUV_frames[0]);



	for(int frame=0;frame<n_frames;frame++){
		free(inputdata[frame]);//freeing the frames		
		if(frame<2){//the maps
			free(outputdata[frame]);//freeing the static map			
		}			
	}
	free(Saliency_maps[0]);//freeing the dynamic map
	free(blended_frames[0]);//blended frames
	free(ages);

	/*this part is when we read video files*/
	// Free the RGB image
	av_free(buffer);
	av_free(pFrameRGB);
  
	// Free the YUV frame
	av_free(pFrame);
  
	// Close the codec
	avcodec_close(pCodecCtx);
  
	// Close the video file
	avformat_close_input(&pFormatCtx);
	/*this part is when we read video files*/

	system("pause");
	return 0;
}



















//
//
//
//int _tmain()
//{
//	std::cerr << "Application started....\n" << std::endl;
//
//	//video path
//	const char *device_name = "video=Logitech HD Webcam C270";
//
//	if(device_name==NULL){
//		printf("problem opening the streaming device :(  !!");
//		system("pause");
//		return -1;
//	}
//
//	/*getting the streaming device with ffmpeg*/
//	avdevice_register_all();
//	avcodec_register_all();
//
//	AVCodecContext  *pCodecCtx;
//	AVFormatContext *pFormatCtx = avformat_alloc_context();
//	AVCodec * pCodec;
//	AVInputFormat *iformat = av_find_input_format("dshow");
//	AVFrame *pFrame, *pFrameRGB;
//	int videoStream, numBytes, frameFinished, res;
//	uint8_t *buffer;
//	AVPixelFormat  pFormat;
//	AVPacket packet;
//	struct SwsContext *img_convert_ctx;
//
//	if(int put = camera_main(device_name,pCodecCtx,pFormatCtx,pCodec,iformat,pFrame,pFrameRGB,videoStream,buffer,numBytes,pFormat,frameFinished))
//	{
//		printf("error %i\n",put);
//		printf("problem opening the video or camera using ffmpeg :(  !!");
//		system("pause");
//		return -1;
//	}
//	/*getting the streaming device with ffmpeg*/
//
//
//	///*output device test*/
//	//int j=0;
//	//CImgDisplay main_window;
//	//while((j++)<1000){
//	//	extract_camera_frame(pFormatCtx,res,packet,videoStream,pCodecCtx,pFrame,pFrameRGB,frameFinished,img_convert_ctx);
//	//	CImg<unsigned char> original(pFrameRGB->data[0],3,pCodecCtx->width,pCodecCtx->height,1);
//	//	original.permute_axes("yzcx");
//	//	main_window.display(original);
//	//	Sleep(42);
//	//}
//
//
//	int i=0;
//	const int n_frames = 3;
//	int frame_width, frame_height, frame_channels;
//	float *ages;
//	unsigned char *blended_frames[1];//blended images coming from combining the original and mapped frames
//	unsigned char *inputdata[n_frames];//contains the frames
//	unsigned char *outputdata[2];//contains the ouput maps (one static map per frame, and 1 dynamic maps every frame sequence)
//	unsigned char *Saliency_maps[1];//contain all the master saliency maps after fusion
//
//	int index = 0;
//	bool finished_video = false;
//
//	CImgDisplay main_window, staticmap, dynamicmap, blended, mastermap, agesmaps;
//	while(true){
//		for(int j=0;j<n_frames;j++){
//
//			if(i==0){//allocates the memory just once
//				if(extract_camera_frame(pFormatCtx,res,packet,videoStream,pCodecCtx,pFrame,pFrameRGB,frameFinished,img_convert_ctx))//reading frame with ffmpeg
//					finished_video = true;//ended the processing of the video
//				if(finished_video)
//					break;
//				frame_width = pCodecCtx->width; frame_height = pCodecCtx->height; frame_channels = 3;//for the RGB data
//				inputdata[j] = (unsigned char*)malloc(sizeof(unsigned char)*frame_width*frame_height*frame_channels);//an RGB
//				blended_frames[0] = (unsigned char*)malloc(sizeof(unsigned char)*frame_width*frame_height*frame_channels);//blended  frames
//				memcpy(inputdata[j], pFrameRGB->data[0], sizeof(unsigned char)*frame_width*frame_height*frame_channels);//copying the RGB frame
//			}else{
//				if(j<(n_frames-1)){
//					memcpy(inputdata[j], inputdata[j+1], sizeof(unsigned char)*frame_width*frame_height*frame_channels);//
//				}
//				else{
//					if(extract_camera_frame(pFormatCtx,res,packet,videoStream,pCodecCtx,pFrame,pFrameRGB,frameFinished,img_convert_ctx))//reading frame with ffmpeg
//						finished_video = true;//ended the processing of the video
//					if(finished_video)
//						break;
//					memcpy(inputdata[j], pFrameRGB->data[0], sizeof(unsigned char)*frame_width*frame_height*frame_channels);//copying the RGB frame
//				}
//			}
//		}
//
//		if(finished_video)
//			break;
//
//		//array that will contain the maps coming from the library, which are in gray scale values
//		if(i==0){
//			for(int frame=0;frame<2;frame++){
//				outputdata[frame] = (unsigned char*)malloc(sizeof(unsigned char)*frame_width*frame_height);//for static and dynamic maps
//			}
//			Saliency_maps[0] = (unsigned char*)malloc(sizeof(unsigned char)*frame_width*frame_height);//for combined maps
//			ages = (float*)malloc(sizeof(float)*frame_width*frame_height);
//		}
//
//
//		//calling the saliency map library
//		time_t start = clock(), end;
//		//if(saliency::Motion_compensation(inputdata,frame_width,frame_height,n_frames,frame_channels,i))
//			//return -1;
//		if(saliency::Saliency_maps(inputdata, outputdata,frame_width,frame_height,n_frames,frame_channels,6,4,ages))//estimate the maps per pathway
//			return -1;
//		if(saliency::Fusion_maps(Saliency_maps, outputdata, frame_width, frame_height, 1,AVERAGE))//fusions the maps
//			return -1;
//		if(saliency::Fading_maps(Saliency_maps,frame_width,frame_height,1,ages))//fading the pixels for the master map
//			return -1;
//		if(saliency::Frames_Maps_Blend(blended_frames,inputdata,Saliency_maps,frame_width,frame_height,1,frame_channels))//blending the original frames and master map
//			return -1;
//
//		end=clock();
//		printf("The time elapsed for the mapping processes was: %f ms.\n",((float)(end-start)/CLOCKS_PER_SEC)*1000);
//
//
//
//		//CImg<unsigned char> original(inputdata[0],frame_channels,frame_width,frame_height,1);
//		CImg<unsigned char> blended_Map(blended_frames[0],frame_channels,frame_width,frame_height,1);
//
//		//CImg<float> Age_maps(frame_width,frame_height);
//		//CImg<unsigned char> Static_Map(frame_width,frame_height);		
//		//CImg<unsigned char> Dynamic_Map(frame_width,frame_height);		
//		//CImg<unsigned char> master_map(frame_width,frame_height);		
//		
//		//memcpy(Age_maps._data,ages,sizeof(float)*frame_width*frame_height);
//		//memcpy(Static_Map._data,outputdata[0],sizeof(unsigned char)*frame_width*frame_height);
//		//memcpy(Dynamic_Map._data,outputdata[1],sizeof(unsigned char)*frame_width*frame_height);
//		//memcpy(master_map._data,Saliency_maps[0],sizeof(unsigned char)*frame_width*frame_height);
//		//original.permute_axes("yzcx");
//		blended_Map.permute_axes("yzcx");
//		
//		//main_window.display(original);
//		//agesmaps.display(Age_maps);
//		//staticmap.display(Static_Map);
//		//dynamicmap.display(Dynamic_Map);
//		//mastermap.display(master_map);
//		blended.display(blended_Map);
//		i++;
//	}
//
//	for(int frame=0;frame<n_frames;frame++){
//		free(inputdata[frame]);//freeing the frames		
//		if(frame<2){//the maps
//			free(outputdata[frame]);//freeing the static map			
//		}			
//	}
//	free(Saliency_maps[0]);//freeing the dynamic map
//	free(blended_frames[0]);//blended frames
//	free(ages);
//
//	avcodec_close(pCodecCtx);
//	av_free(pFrame);
//	av_free(pFrameRGB);
//	avformat_close_input(&pFormatCtx); 
//
//	system("pause");
//	return 0;
//}
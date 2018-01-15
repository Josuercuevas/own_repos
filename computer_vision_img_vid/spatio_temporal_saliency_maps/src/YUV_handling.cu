#include <Windows.h>
#include <tchar.h>
#include "cpp_includes.h"
#include "YUV_conversion_header.cuh"
#include "atlstr.h"


/****************************************************************************************************************************************************************************************/
//========================================================================================================================================================================================
/*Main function for converting YUV to RGB*/
//#ifdef _DEBUG
//bool _trace(TCHAR *format, ...)
//{
//   TCHAR buffer[1000];
//
//   va_list argptr;
//   va_start(argptr, format);
//   wvsprintf(buffer, format, argptr);
//   va_end(argptr);
//
//   OutputDebugString(buffer);
//
//   return true;
//}
//#endif


int convert_YUVtoRGB(unsigned char *YUV_frames[], unsigned char *RGB_frames[], int Y_height, int Y_width, int Cb_height, int Cb_width, int Cr_height, int Cr_width, int n_frames, const int pix_dist){
	//OutputDebugString(_T("========>Converting YUV to RGB ....!!!!!\n"));

	/*USES_CONVERSION;
	TCHAR message1[1000];
	TCHAR message2[1000];
	TCHAR message3[1000];*/
	//printf("%s\n",cudaGetErrorString(cudaGetLastError()));//in case we want to see that every part of the cuda implementation is right

	const int BLOCKS = ((Y_height*Y_width) + THREADS -1) / THREADS;
	int YUVframe_size = Y_height*Y_width + Cb_height*Cb_width + Cr_height*Cr_width;
	int RGBframe_size = Y_height*Y_width*3;

	unsigned char *d_RGB_frames, *d_YUV_frames;
	cudaMalloc((void**)&d_RGB_frames,sizeof(unsigned char)*RGBframe_size*n_frames);
	cudaMalloc((void**)&d_YUV_frames,sizeof(unsigned char)*YUVframe_size*n_frames);

	for(int frame=0;frame<n_frames; frame++){		
		cudaMemcpy(d_YUV_frames + YUVframe_size*frame, YUV_frames[frame], sizeof(unsigned char)*YUVframe_size, cudaMemcpyHostToDevice);
	}
	/*_tcscpy(message1, A2T(cudaGetErrorString(cudaGetLastError())));
	_trace(message1);
	OutputDebugString(_T("\n"));*/


	//int Y, Cr, Cb, Yidx, Cridx, Cbidx, R, G, B;
	//int factor = ceil((float)Y_height/Cb_height);
	//int Ysize = (Y_height*Y_width);//frame size on a single plane
	//int Cbsize = (Cb_height*Cb_width);//frame size on a single plane
	//int Crsize = (Cr_height*Cr_width);//frame size on a single plane




	//for(int frame=0;frame<n_frames; frame++){
	//	for(int i=0;i<Y_height;i++){
	//		for(int j=0;j<Y_width;j++){
	//			Yidx = i*Y_width + j;
	//			Cbidx = Ysize + (i/factor)*(Y_width/factor) + (j/factor);
	//			Cridx = Ysize + Cbsize + (i/factor)*(Y_width/factor) + (j/factor);

	//			Y = YUV_frames[frame][Yidx] - 16;
	//			Cb = YUV_frames[frame][Cbidx] - 128;
	//			Cr = YUV_frames[frame][Cridx] - 128;

	//			R = ( 298 * Y           + 409 * Cr + 128) >> 8; //Y + Cr + (Cr>>2) + (Cr>>3) + (Cr>>5);
	//			R = max(0, min(255, R));

	//			G = ( 298 * Y - 100 * Cb - 208 * Cr + 128) >> 8;//Y - ((Cb>>2) + (Cb>>4) + (Cb>>5)) - ((Cr>>1) + (Cr>>3) + (Cr>>4) + (Cr>>5));
	//			G = max(0, min(255, G));

	//			B = ( 298 * Y + 516 * Cb           + 128) >> 8;//Y + Cb + (Cb>>1) + (Cb>>2) + (Cb>>6) ;
	//			B = max(0, min(255, B));


	//			RGB_frames[frame][Yidx] = R;
	//			RGB_frames[frame][Yidx + Y_height*Y_width] = G;
	//			RGB_frames[frame][Yidx + Y_height*Y_width*2] = B;
	//		}
	//	}
	//}
	/*char f_name[1000];
	char num[150];
	strcpy(f_name, "coming Sizes are: ");
	sprintf(num, "<%03ix%03i, %03ix%03i, %03ix%03i>", Y_height, Y_width, Cbidx/Cb_width, Cbidx%Cb_width, Cr_height, Cr_width);
	strcat(f_name, num);
	
	_tcscpy(message2, A2T(f_name));
	_trace(message2);
	OutputDebugString(_T("\n"));*/

	YUVconverter<<<BLOCKS, THREADS>>>(d_YUV_frames,d_RGB_frames,Y_height,Y_width,Cb_height,Cb_width,Cr_height,Cr_width,n_frames, pix_dist);
	//printf("Error: %s\n",cudaGetErrorString(cudaGetLastError()));
	/*_tcscpy(message2, A2T(cudaGetErrorString(cudaGetLastError())));
	_trace(message2);
	OutputDebugString(_T("\n"));*/



	for(int frame=0;frame<n_frames; frame++){
		cudaMemcpy(RGB_frames[frame], d_RGB_frames + RGBframe_size*frame, sizeof(unsigned char)*RGBframe_size, cudaMemcpyDeviceToHost);
	}
	/*_tcscpy(message3, A2T(cudaGetErrorString(cudaGetLastError())));
	_trace(message3);
	OutputDebugString(_T("\n"));*/



	//for(int frame=0;frame<n_frames; frame++){
		cudaFree(d_RGB_frames);
		cudaFree(d_YUV_frames);
	//}
		//printf("%s\n",cudaGetErrorString(cudaGetLastError()));//in case we want to see that every part of the cuda implementation is right
	//OutputDebugString(_T("=======>Finished conversion ....!!!!!\n"));
	return 0;//if no problem found
}

__global__ void YUVconverter(unsigned char *YUV_frames, unsigned char *RGB_frames, int Y_height, int Y_width, int Cb_height, int Cb_width, int Cr_height, int Cr_width, int n_frames, 
	const int pix_dist){

	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int Ysize = (Y_height*Y_width);//frame size on a single plane
	int Cbsize = (Cb_height*Cb_width);//frame size on a single plane
	int Crsize = (Cr_height*Cr_width);//frame size on a single plane
	int YUV_frame_size = Ysize+Cbsize+Crsize;//size of the YUV container


	if(idx>=Ysize) return;//inactive threads

	int Y, Cr, Cb, R, G, B, Yidx, Cridx, Cbidx;
	int factor = ceil((float)Y_height/Cb_height);

	int i = idx/Y_width, j = idx%Y_width;
	Yidx = idx;
	Cbidx = Ysize + (i/factor)*(Y_width/factor) + (j/factor);
	Cridx = Ysize + Cbsize + (i/factor)*(Y_width/factor) + (j/factor);

	for(int frame=0;frame<n_frames; frame++){
		if(pix_dist == 1){//interlace
			Y = YUV_frames[Yidx + YUV_frame_size*frame]-16;
			Cb = YUV_frames[Cbidx + YUV_frame_size*frame]-128;
			Cr = YUV_frames[Cridx + YUV_frame_size*frame]-128;

			R = (298*Y + 409*Cr) >> 8;//Y + 1.140*Cr //Y + Cr + (Cr>>2) + (Cr>>3) + (Cr>>5);
			R = max(0, min(255, R));

			G = (298*Y - 100*Cb - 208*Cr) >> 8;//Y - 0.395*Cb - 0.581*Cr;//Y - ((Cb>>2) + (Cb>>4) + (Cb>>5)) - ((Cr>>1) + (Cr>>3) + (Cr>>4) + (Cr>>5));
			G = max(0, min(255, G));

			B = (298*Y + 516*Cb) >> 8;//Y + 2.032*Cb;//Y + Cb + (Cb>>1) + (Cb>>2) + (Cb>>6) ;
			B = max(0, min(255, B));

			RGB_frames[Yidx*3 + 0 + frame*Ysize*3] = (unsigned char)R;//R values
			RGB_frames[Yidx*3 + 1 + frame*Ysize*3] = (unsigned char)G;//G values
			RGB_frames[Yidx*3 + 2 + frame*Ysize*3] = (unsigned char)B;//B values
		}
		else{//plannar
			Y = YUV_frames[Yidx + YUV_frame_size*frame]-16;
			Cb = YUV_frames[Cbidx + YUV_frame_size*frame]-128;
			Cr = YUV_frames[Cridx + YUV_frame_size*frame]-128;


			R = (298*Y + 409*Cr + 128) >> 8;;//Y + 1.140*Cr //Y + Cr + (Cr>>2) + (Cr>>3) + (Cr>>5);
			R = max(0, min(255, R));

			G = (298*Y - 100*Cb - 208*Cr + 128) >> 8;//Y - 0.395*Cb - 0.581*Cr;//Y - ((Cb>>2) + (Cb>>4) + (Cb>>5)) - ((Cr>>1) + (Cr>>3) + (Cr>>4) + (Cr>>5));
			G = max(0, min(255, G));

			B = (298*Y + 516*Cb  + 128) >> 8;//Y + 2.032*Cb;//Y + Cb + (Cb>>1) + (Cb>>2) + (Cb>>6) ;
			B = max(0, min(255, B));


			RGB_frames[Yidx + frame*Ysize*3] = (unsigned char)R;//R values
			RGB_frames[Yidx + Ysize + frame*Ysize*3] = (unsigned char)G;//G values
			RGB_frames[Yidx + Ysize*2 + frame*Ysize*3] = (unsigned char)B;//B values
		}
		
	}
}








/****************************************************************************************************************************************************************************************/
//========================================================================================================================================================================================

/*Main function for converting RGB to YUV*/
int convert_RGBtoYUV(unsigned char *YUV_frames[], unsigned char *RGB_frames[], int Y_height, int Y_width, int Cb_height, int Cb_width, int Cr_height, int Cr_width, int n_frames, const int pix_dist){
	//OutputDebugString(_T("========>Converting RGB to YUV ....!!!!!\n"));

	/*USES_CONVERSION;
	TCHAR message1[1000];
	TCHAR message2[1000];
	TCHAR message3[1000];*/
	//printf("%s\n",cudaGetErrorString(cudaGetLastError()));//in case we want to see that every part of the cuda implementation is right

	const int BLOCKS = ((Y_height*Y_width) + THREADS -1) / THREADS;
	int YUVframe_size = Y_height*Y_width + Cb_height*Cb_width + Cr_height*Cr_width;
	int RGBframe_size = Y_height*Y_width*3;

	unsigned char *d_RGB_frames, *d_YUV_frames;

	cudaMalloc((void**)&d_RGB_frames,sizeof(unsigned char)*RGBframe_size*n_frames);
	cudaMalloc((void**)&d_YUV_frames,sizeof(unsigned char)*YUVframe_size*n_frames);
	//printf("Error: %s\n",cudaGetErrorString(cudaGetLastError()));

	for(int frame=0;frame<n_frames; frame++){
		cudaMemcpy(d_RGB_frames + RGBframe_size*frame, RGB_frames[frame], sizeof(unsigned char)*RGBframe_size, cudaMemcpyHostToDevice);
	}
	//printf("Error: %s\n",cudaGetErrorString(cudaGetLastError()));
	/*_tcscpy(message1, A2T(cudaGetErrorString(cudaGetLastError())));
	_trace(message1);
	OutputDebugString(_T("\n"));*/


	//int R, G, B;
	//int Y, Cr, Cb;
	//int Yidx, Cbidx, Cridx;
	//int Ysize = (Y_height*Y_width);//frame size on a single plane
	//int Cbsize = (Cb_height*Cb_width);//frame size on a single plane
	//int Crsize = (Cr_height*Cr_width);//frame size on a single plane



	//for(int frame=0;frame<n_frames; frame++){
	//	for(int i=0;i<Y_height;i++){
	//		for(int j=0;j<Y_width;j++){
	//			Yidx = i*Y_width + j;
	//			R = RGB_frames[frame][Yidx];
	//			G = RGB_frames[frame][Yidx + Ysize];
	//			B = RGB_frames[frame][Yidx + 2*Ysize];



	//			Yidx = i*Y_width + j;
	//			Cbidx = Ysize + (i/2)*(Y_width/2) + (j/2);
	//			Cridx = Ysize + Cbsize + (i/2)*(Y_width/2) + (j/2);


	//			Y = ( (  66 * R + 129 * G +  25 * B + 128) >> 8) +  16;
	//			Cb = ( ( -38 * R -  74 * G + 112 * B + 128) >> 8) + 128;
	//			Cr = ( ( 112 * R -  94 * G -  18 * B + 128) >> 8) + 128;



	//			YUV_frames[frame][Yidx] = (unsigned char)Y;

	//			YUV_frames[frame][Cbidx] = (unsigned char)Cb;

	//			YUV_frames[frame][Cridx] =  (unsigned char)Cr;
	//		}
	//	}
	//}

	/*char f_name[1000];
	char num[150];
	strcpy(f_name, "coming Sizes are: ");
	sprintf(num, "<%03ix%03i, %03ix%03i, %03ix%03i>", Y_height, Y_width, Cbidx/Cb_width, Cbidx%Cb_width, Cr_height, Cr_width);
	strcat(f_name, num);
	
	_tcscpy(message2, A2T(f_name));
	_trace(message2);
	OutputDebugString(_T("\n"));*/

	RGBconverter<<<BLOCKS, THREADS>>>(d_YUV_frames,d_RGB_frames,Y_height,Y_width,Cb_height,Cb_width,Cr_height,Cr_width,n_frames,pix_dist);
	//printf("Error: %s\n",cudaGetErrorString(cudaGetLastError()));
	//_tcscpy(message2, A2T(cudaGetErrorString(cudaGetLastError())));
	//_trace(message2);
	//OutputDebugString(_T("\n"));



	for(int frame=0;frame<n_frames; frame++){
		cudaMemcpy(YUV_frames[frame], d_YUV_frames + YUVframe_size*frame, sizeof(unsigned char)*YUVframe_size, cudaMemcpyDeviceToHost);
	}
	////printf("Error: %s\n",cudaGetErrorString(cudaGetLastError()));
	//_tcscpy(message3, A2T(cudaGetErrorString(cudaGetLastError())));
	//_trace(message3);
	//OutputDebugString(_T("\n"));



	//for(int frame=0;frame<n_frames; frame++){
		cudaFree(d_RGB_frames);
		cudaFree(d_YUV_frames);
	//}
		//printf("Error: %s\n",cudaGetErrorString(cudaGetLastError()));
	//OutputDebugString(_T("=======>Finished conversion ....!!!!!\n"));
	return 0;//if no problem found
}


__global__ void RGBconverter(unsigned char *YUV_frames, unsigned char *RGB_frames, int Y_height, int Y_width, int Cb_height, int Cb_width, int Cr_height, int Cr_width, int n_frames, 
	const int pix_dist){

	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int Ysize = (Y_height*Y_width);//frame size on a single plane
	int Cbsize = (Cb_height*Cb_width);//frame size on a single plane
	int Crsize = (Cr_height*Cr_width);//frame size on a single plane
	int YUV_frame_size = Ysize+Cbsize+Crsize;//size of the YUV container

	if(idx>=Ysize) return;//inactive threads

	int Y, Cr, Cb, R, G, B, Yidx, Cridx, Cbidx;
	int factor = ceil((float)Y_height/Cb_height);

	int i = idx/Y_width, j = idx%Y_width;

	Yidx = idx;
	Cbidx = Ysize + (i/factor)*(Y_width/factor) + (j/factor);
	Cridx = Ysize + Cbsize + (i/factor)*(Y_width/factor) + (j/factor);

	for(int frame=0;frame<n_frames; frame++){
		if(pix_dist == 1){//interlace    idx*n_channels + cha + frame*n_pixels*n_channels
			R = RGB_frames[Yidx*3 + 0 + frame*Ysize*3];
			G = RGB_frames[Yidx*3 + 1 + frame*Ysize*3];
			B = RGB_frames[Yidx*3 + 2 + frame*Ysize*3];


			Y = ( (  66 * R + 129 * G +  25 * B + 128) >> 8) +  16;//0.299*R + 0.587*G + 0.114*B;
			Cb = ( ( -38 * R -  74 * G + 112 * B + 128) >> 8) + 128;// 0.492*(B-Y);
			Cr = ( ( 112 * R -  94 * G -  18 * B + 128) >> 8) + 128;//0.877*(R-Y);


			YUV_frames[Yidx + YUV_frame_size*frame] = (unsigned char)Y;
			YUV_frames[Cbidx + YUV_frame_size*frame] = (unsigned char)Cb;
			YUV_frames[Cridx + YUV_frame_size*frame] =  (unsigned char)Cr;
		}
		else{//planar
			R = RGB_frames[Yidx + frame*Ysize*3];
			G = RGB_frames[Yidx + Ysize + frame*Ysize*3];
			B = RGB_frames[Yidx + 2*Ysize + frame*Ysize*3];


			Y = ( (  66 * R + 129 * G +  25 * B + 128) >> 8) +  16;//0.299*R + 0.587*G + 0.114*B;
			Cb = ( ( -38 * R -  74 * G + 112 * B + 128) >> 8) + 128;// 0.492*(B-Y);
			Cr = ( ( 112 * R -  94 * G -  18 * B + 128) >> 8) + 128;//0.877*(R-Y);


			YUV_frames[Yidx + YUV_frame_size*frame] = (unsigned char)Y;
			YUV_frames[Cbidx + YUV_frame_size*frame] = (unsigned char)Cb;
			YUV_frames[Cridx + YUV_frame_size*frame] =  (unsigned char)Cr;
		}
	}
}
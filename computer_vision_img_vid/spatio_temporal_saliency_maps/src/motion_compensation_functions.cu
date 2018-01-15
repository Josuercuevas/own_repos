#include "cpp_includes.h"
#include "motion_compensation_header.cuh"
#include "math_functions.h"

texture<unsigned char,1,cudaReadModeElementType> frames_texture;
#define macroBlock (7)//size of the block to perform the motion estimation for the minimum MAE (13x13 pixels)
#define sub_block (3)//size of the block to be used for the prediction error (7x7 pixels)


int compensation_main(unsigned char *frames[], const int width, const int height, const int n_frames, const int channels, bool first_frame){

	/********************* ALLOCATING MEMORY ******************************/
	unsigned char *d_frames;
	const int n_pixels = height*width;
	const int BLOCKS = (THREADS+n_pixels-1)/THREADS;
	size_t frame_size = sizeof(unsigned char)*height*width*channels;
	cudaMalloc((void**)&d_frames,frame_size*n_frames);
	for(int frame=0;frame<n_frames;frame++)//copy frames
		cudaMemcpyAsync(d_frames+(n_pixels*frame*channels),frames[frame],frame_size,cudaMemcpyHostToDevice);

	cudaBindTexture(NULL,frames_texture,d_frames,frame_size*n_frames);
	motion_estimation_kernel<<<BLOCKS,THREADS>>>(d_frames,width,height,n_frames,channels,first_frame);

	for(int frame=0;frame<n_frames;frame++)//copy frames back from GPU
		cudaMemcpyAsync(frames[frame],d_frames+(n_pixels*frame*channels),frame_size,cudaMemcpyDeviceToHost);

	cudaFree(d_frames);
	cudaUnbindTexture(frames_texture);
	printf("%s\n",cudaGetErrorString(cudaGetLastError()));
	return 0;//returns 0 if there isnt any problem with the function
}

__global__ void motion_estimation_kernel(unsigned char *frames, const int width, const int height, const int n_frames, const int channels, const bool first_frame){
	const int n_pixels = width*height;
	unsigned int tid = threadIdx.x;
	unsigned int idx = tid + blockIdx.x*blockDim.x;
	const int x = idx%width, y=idx/width; //to tell the location of the pixel in the x,y plane
	int n_channels = channels;


	const int win_x = macroBlock/2, win_y = macroBlock/2;//neighbors of the center pixel at the macroblock
	const int b_x = sub_block/2, b_y = sub_block/2;//neighbors of the center pixel at the subblock

	float Macro_block[macroBlock][macroBlock];//contains intensities in the macroblock

	if(idx>=n_pixels) return;//inactive threads

	int lower_bound = 0;

	if(!first_frame)
		lower_bound = n_frames-2;//previous to the last frame since the only one to be motion compensated is the last frame

	int macro_y, macro_x;

	for(int frame=lower_bound;frame<(n_frames-1);frame++){//all the frames to perform motion compensation
		//estimation of the minimum MAE in the macroblock, sliding window to avoid artifacts in the macroblock borders
		if(x>=(win_x) && x<(width-win_x) && y>=(win_y) && y<(height-win_y)){//taking care of the borders in the image
			
			//working in the macroblock a sliding window in the image
			macro_y = 0;
			for(int mac_y=-win_y;mac_y<=win_y;mac_y++){//size y of macroblock				
				macro_x = 0;
				for(int mac_x=-win_x;mac_x<=win_x;mac_x++){//size x of macroblock
					Macro_block[macro_y][macro_x] += (abs(tex1Dfetch(frames_texture,idx*n_channels + mac_x + mac_y*width + frame*n_pixels*channels) -
															tex1Dfetch(frames_texture,idx*n_channels + mac_x + mac_y*width + (frame+1)*n_pixels*channels)) +//R
														abs(tex1Dfetch(frames_texture,idx*n_channels + 1 + mac_x + mac_y*width + frame*n_pixels*channels) -
															tex1Dfetch(frames_texture,idx*n_channels + 1 + mac_x + mac_y*width + (frame+1)*n_pixels*channels)) +//G
														abs(tex1Dfetch(frames_texture,idx*n_channels + 2 + mac_x + mac_y*width + frame*n_pixels*channels) -
															tex1Dfetch(frames_texture,idx*n_channels + 2 + mac_x + mac_y*width + (frame+1)*n_pixels*channels)))/3;//B
					macro_x++;
				}
				macro_y++;
			}

			//working in the sub-block, a sliding window in the macroblock
			float max_va = 1E120;
			int locations;
			int flag = 0;
			//int wx, wy;
			
			macro_y = 0;
			for(int mac_y=-win_y;mac_y<=win_y;mac_y++){//size y of macroblock				
				macro_x = 0;
				for(int mac_x=-win_x;mac_x<=win_x;mac_x++){//size x of macroblock
					if(macro_y>=b_y && macro_y<(win_y-b_y) && macro_x>=b_x && macro_x<(win_x-b_x)){
						float subBlock = 0;//contains the MAE per sublock in the sub-block
						for(int i=-b_y;i<=b_y;i++){
							for(int j=-b_x;j<=b_x;j++){
								subBlock += Macro_block[macro_y+i][macro_x+j];
							}
						}
						subBlock /= (sub_block*sub_block);
						//if(tid==0)
							//printf("subblock: %4.2f\t",subBlock);

						if(max_va>subBlock){
							max_va = subBlock;
							locations = idx + mac_x + mac_y*width;
							//wx = mac_x;wy = mac_y;
							flag++;
						}
					}
					macro_x++;
				}
				macro_y++;
			}


			/*if(tid==0)
				printf("maximums: %4.2f\t",max_va);*/
			//if(tid==0)
				//printf("locations: %i\t%i\t%i\t",wx,wy,flag);
			
			//now is time to update the center pixel in the macroblock for the frame to be predicted or compensated, which is frame t and not t-1
			if(flag>1){//means that we really updated the values
				//printf("YES!!...");
				frames[(frame+1)*n_pixels*n_channels + idx*n_channels] = frames[(frame+1)*n_pixels*n_channels + locations*n_channels];//R
				frames[(frame+1)*n_pixels*n_channels + idx*n_channels + 1] = frames[(frame+1)*n_pixels*n_channels + locations*n_channels + 1];//G
				frames[(frame+1)*n_pixels*n_channels + idx*n_channels + 2] = frames[(frame+1)*n_pixels*n_channels + locations*n_channels + 2];//B
			}
		}
	}
}
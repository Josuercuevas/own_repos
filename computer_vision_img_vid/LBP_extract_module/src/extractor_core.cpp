/*
 * extractor_core.cpp
 *
 *  Created on: Dec 14, 2016
 *      Author: josue

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

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <CImg.h>

#include "../inc/lbp_extract.h"

/*
 * How the classes are to be mapped
 * */
static char *mapping[10] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

using namespace cimg_library;

static uint8_t LBP_MASK[3][3] = {
		{8, 4, 2},
		{16, 0, 1},
		{32, 64, 128}
};

uint32_t super_counter=0;
static uint8_t DEBUG_LEVEL = LBP_VERBOSE;


void set_debug_level(uint8_t level){
	DEBUG_LEVEL = level;
}


uint8_t integral_image(unsigned char *img, uint32_t width, uint32_t height, uint32_t *ii) {
	uint32_t *temp_sum = (uint32_t*)malloc(width*height*sizeof(uint32_t));
	int x, y;

	for(y = 0; y < height; y++){
		for(x = 0; x < width; x++){
			if(x == 0){
				temp_sum[(y*width)+x] = img[(y*width)+x];
			}else{
				temp_sum[(y*width)+x] = temp_sum[(y*width)+x-1] + img[(y*width)+x];
			}

			if(y == 0){
				ii[(y*width)+x] = temp_sum[(y*width)+x];
			}else{
				ii[(y*width)+x] = ii[((y-1)*width)+x] + temp_sum[(y*width)+x];
			}
		}
	}
	free(temp_sum);
	return FALSE;
}

#define phase (4)
uint8_t process_picture(LBP *feature_extrator, unsigned char* pix_data_int8,
		uint32_t channels_in){
	uint32_t cell_w = feature_extrator->w_cell;
	uint32_t cell_h = feature_extrator->h_cell;
	uint32_t img_h = feature_extrator->img_h;
	uint32_t img_w = feature_extrator->img_w;
	uint32_t ncell_h = feature_extrator->ncell_h;
	uint32_t ncell_w = feature_extrator->ncell_w;
	int32_t hist_id = 0;
	int32_t old_x = -1;
	int32_t old_y = -1;
	uint8_t IN_CHANNELS=1;

	if(DEBUG_LEVEL >= LBP_DEBUG){
		LOGD("Processing picture from user: %d, %d, %d", img_w, img_h, IN_CHANNELS);
	}

	uint32_t *Integral_Image = (uint32_t*)malloc(img_h*img_w*sizeof(uint32_t));
	if(integral_image(pix_data_int8, img_w, img_h, Integral_Image)){
		return LBP_CANT_EXTRACT_FEATURE;
	}


	uint8_t *temp_image = (uint8_t*)calloc(img_h*img_w, sizeof(uint8_t));

	for(int ch=0; ch<IN_CHANNELS; ch++){
		hist_id = 0;
		for(int h=0; h<img_h; h+=phase){
			for(int w=0; w<img_w; w+=phase){
				uint8_t accumulator=0;
				if(w<(img_w-phase) && h<(img_h-phase) && w>(phase-1) && h>(phase-1)){
					/*
					 * Cimage reads the input as PLANAR arranged
					 * -> checking box now
					 * */
					if(Integral_Image[h*img_w + w + ch*(img_w*img_h)] < Integral_Image[(h-phase)*img_w + (w-phase) + ch*(img_w*img_h)]){
						accumulator += LBP_MASK[0][0];
					}
					if(Integral_Image[h*img_w + w + ch*(img_w*img_h)] < Integral_Image[(h-phase)*img_w + (w) + ch*(img_w*img_h)]){
						accumulator += LBP_MASK[0][1];
					}
					if(Integral_Image[h*img_w + w + ch*(img_w*img_h)] < Integral_Image[(h-phase)*img_w + (w+phase) + ch*(img_w*img_h)]){
						accumulator += LBP_MASK[0][2];
					}
					if(Integral_Image[h*img_w + w + ch*(img_w*img_h)] < Integral_Image[(h)*img_w + (w-phase) + ch*(img_w*img_h)]){
						accumulator += LBP_MASK[1][0];
					}
					if(Integral_Image[h*img_w + w + ch*(img_w*img_h)] < Integral_Image[(h)*img_w + (w) + ch*(img_w*img_h)]){
						accumulator += LBP_MASK[1][1];
					}
					if(Integral_Image[h*img_w + w + ch*(img_w*img_h)] < Integral_Image[(h)*img_w + (w+phase) + ch*(img_w*img_h)]){
						accumulator += LBP_MASK[1][2];
					}
					if(Integral_Image[h*img_w + w + ch*(img_w*img_h)] < Integral_Image[(h+phase)*img_w + (w-phase) + ch*(img_w*img_h)]){
						accumulator += LBP_MASK[2][0];
					}
					if(Integral_Image[h*img_w + w + ch*(img_w*img_h)] < Integral_Image[(h+phase)*img_w + (w) + ch*(img_w*img_h)]){
						accumulator += LBP_MASK[2][1];
					}
					if(Integral_Image[h*img_w + w + ch*(img_w*img_h)] < Integral_Image[(h+phase)*img_w + (w+phase) + ch*(img_w*img_h)]){
						accumulator += LBP_MASK[2][2];
					}

					/*in case we want to debug images LBPs*/
					temp_image[h*img_w + w] = accumulator;

                    if((w%(cell_w) == 0) && w > 0){
                        if(old_x != w){
                        	if(DEBUG_LEVEL >= LBP_VERBOSE){
                        		printf("==> Changed at (X: %d, Y: %d, h: %d)\n", w, h, hist_id);
                        	}
                            old_x = w;
                            hist_id += 1;
                        }
                    }

                    if((h%(cell_h)==0) && h>0){
                        if(old_y != h){
                        	if(DEBUG_LEVEL >= LBP_VERBOSE){
                        		printf("---->>> Changed at (X: %d, Y: %d, h: %d)\n", w, h, hist_id);
                        	}
                            old_y = h;
                            hist_id += (img_w/cell_w); //restart since we advance in X
                        }
                    }

                    /*restarting counter for histogram if Y still the same*/
                    if(w >= img_w-2*phase){
                    	hist_id -= (img_w/cell_w - 1);
                    }

					feature_extrator->lbp_hist[hist_id*HIST_SIZE + temp_image[h*img_w + w]] += 1;
				}
			}
		}
	}

	/*normalizing everything*/
	for(int i=0; i<ncell_h*ncell_w; i++){
		for(int j=0; j<HIST_SIZE; j++){
			feature_extrator->lbp_hist[i*HIST_SIZE + j] /= (float)(img_w*img_h*IN_CHANNELS);
		}
	}

	if(DEBUG_LEVEL >= LBP_VERBOSE){
		char *debug_dir = "debug_images";
		struct stat check_dir;
		if (stat(debug_dir, &check_dir) == 0 && S_ISDIR(check_dir.st_mode)){
			/*pass no need to create dir*/
		}else{
			/*create debug dir*/
			mkdir(debug_dir, 0777);
		}

		/*LBP is always 1-channel*/
		CImg<unsigned char> LBP_image(temp_image, img_w, img_h);
		char name[128]={0};
		char name_histo[128]={0};
		sprintf(name, "%s/lbp_%d.jpeg", debug_dir, super_counter);
		sprintf(name_histo, "%s/lbp_%d.txt", debug_dir, super_counter);
		LBP_image.normalize(0, 255);
		LBP_image.save_jpeg(name);

		CImg<unsigned int> II_container(Integral_Image, img_w, img_h);
		char name2[128]={0};
		sprintf(name2, "%s/lbp_ii_%d.jpeg", debug_dir, super_counter);
		II_container.normalize(0, 255);
		II_container.save_jpeg(name2);
		super_counter++;

		FILE *hi = fopen(name_histo, "w+");
		for(int i=0; i<ncell_h*ncell_w; i++){
			for(int j=0; j<HIST_SIZE; j++){
				fprintf(hi, "%4.6f\n", feature_extrator->lbp_hist[i*HIST_SIZE + j]);
			}
		}
		fflush(hi);
		fclose(hi);
	}

	if(temp_image){
		free(temp_image);
		temp_image=NULL;
	}

	if(Integral_Image){
		free(Integral_Image);
	}
	return LBP_SUCCESS;
}

/*opens the binary files and creates images from it*/
void debug_histogram(uint8_t n_classes, char *path_save, LBP *feature_extrator,
		uint32_t channels){
	uint32_t ncell_h = feature_extrator->ncell_h;
	uint32_t ncell_w = feature_extrator->ncell_w;

	char *debug_dir = "debug_histograms";
	struct stat check_dir;
	if (stat(debug_dir, &check_dir) == 0 && S_ISDIR(check_dir.st_mode)){
		/*pass no need to create dir*/
	}else{
		/*create debug dir*/
		mkdir(debug_dir, 0777);
	}

	/*creating headers for the split binary files to be used during training*/
	int *binary_files=NULL;/*file where features is going to be saved*/
	binary_files = (int*)malloc(sizeof(int)*n_classes);

	for(int i=0; i<n_classes;i++){
		/*opening the binary file where LBP histograms are to be saved*/
		char name_feat[256]={0};
		sprintf(name_feat, "%s/%s.class_%i_train.lbp", path_save, "faces", i);
		binary_files[i] = DEFAULT_FD;
		binary_files[i] = open(name_feat, O_CREAT|O_NONBLOCK|O_RDWR, 0666);
		if(binary_files[i] == DEFAULT_FD){
			LOGE("Cannot open header file: %s, quitting ...", name_feat);
			return;
		}else{
			LOGV("File %s, opened", name_feat);
		}
	}


	feature_extrator->lbp_hist=NULL;
	feature_extrator->lbp_hist = (float*)calloc(sizeof(float), ncell_w*ncell_h*HIST_SIZE);
	if(!feature_extrator->lbp_hist){
		LOGE("Cannot create memory for LBP histogram, quitting application ....");
		return;
	}


	super_counter=0;

	for(int i=0; i<n_classes;i++){
		uint32_t n_header[2];
		read(binary_files[i], n_header, sizeof(uint32_t)*2);
		LOGV("Header_%d found of: %d, %d", i, n_header[0], n_header[1]);

		LOGV("READING ALL HISTOGRAMS.....");
		for(int h=0; h<n_header[0]; h++){
			read(binary_files[i], feature_extrator->lbp_hist, sizeof(float)*n_header[1]);
			char name_histo[128]={0};
			sprintf(name_histo, "%s/lbp_%d.txt", debug_dir, super_counter);
			FILE *hi = fopen(name_histo, "w+");
			for(int ii=0; ii<ncell_h*ncell_w; ii++){
				for(int j=0; j<HIST_SIZE; j++){
					fprintf(hi, "%4.6f\n", feature_extrator->lbp_hist[ii*HIST_SIZE + j]);
				}
			}
			fflush(hi);
			fclose(hi);
			super_counter++;
		}

		close(binary_files[i]);
	}

	if(feature_extrator->lbp_hist){
		free(feature_extrator->lbp_hist);
	}
}


void scramble_samples(uint32_t *negatives, uint32_t *positives){
	uint32_t sample_index=0;
	/*indexing first all the dataset*/
	for(sample_index=0; sample_index<MAX_POSITIVES; sample_index++){
		positives[sample_index] = sample_index;
	}
	for(sample_index=0; sample_index<MAX_NEGATIVES; sample_index++){
		negatives[sample_index] = sample_index;
	}

	/*now we want to scramble it*/
	for(sample_index=0; sample_index<MAX_POSITIVES; sample_index++){
		uint32_t swap_with = (uint32_t)(((float)rand()/(float)RAND_MAX)*MAX_POSITIVES);
		uint32_t temp = positives[swap_with];
		positives[swap_with] = positives[sample_index];
		positives[sample_index] = temp;
	}
	for(sample_index=0; sample_index<MAX_NEGATIVES; sample_index++){
		uint32_t swap_with = (uint32_t)(((float)rand()/(float)RAND_MAX)*MAX_NEGATIVES);
		uint32_t temp = negatives[swap_with];
		negatives[swap_with] = negatives[sample_index];
		negatives[sample_index] = temp;
	}
	return;
}

uint8_t extract_features(User_Option *options){
	LBP feature_extrator;
	char *path_save = options->bin_path;
	uint8_t n_classes = options->n_classes;
	uint32_t cell_w=options->w_cell, cell_h=options->h_cell;
	uint32_t img_h=options->img_h, img_w=options->img_w;
	uint32_t ncell_h=ceil((float)(options->img_h)/(float)(cell_h));
	uint32_t ncell_w=ceil((float)(options->img_w)/(float)(cell_w));

	uint32_t *positives = (uint32_t*)malloc(sizeof(uint32_t)*MAX_POSITIVES);
	uint32_t *negatives = (uint32_t*)malloc(sizeof(uint32_t)*MAX_NEGATIVES);

	/*main information to be needed*/
	feature_extrator.histogram_size = HIST_SIZE;
	feature_extrator.w_cell = cell_w;
	feature_extrator.h_cell = cell_h;
	feature_extrator.img_h = img_h;
	feature_extrator.img_w = img_w;
	feature_extrator.ncell_h = ncell_h;
	feature_extrator.ncell_w = ncell_w;

	/*the lbp size is the size of the grid wxh times the size of the histogram*/
	if(DEBUG_LEVEL>=LBP_DEBUG){
		LOGD("LBP features are going to be of %d dimensions (w: %d x h: %d x d: %d)",
				ncell_w*ncell_h*HIST_SIZE, ncell_w, ncell_h, HIST_SIZE);
	}

	feature_extrator.lbp_hist=NULL;
	feature_extrator.lbp_hist = (float*)calloc(sizeof(float), ncell_w*ncell_h*HIST_SIZE);
	if(!feature_extrator.lbp_hist){
		LOGE("Cannot create memory for LBP histogram, quitting application ....");
		return LBP_CANT_EXTRACT_FEATURE;
	}

	/*creating headers for the split binary files to be used during training*/
	int *binary_files=NULL;/*file where features is going to be saved*/
	binary_files = (int*)malloc(sizeof(int)*n_classes);


	for(int i=0; i<n_classes;i++){
		FILE *header_files=NULL;/*for trainer mainly*/
		char name_head[256]={0};
		sprintf(name_head, "%s/%s.class_%i.header", path_save, "faces", i);
		header_files = fopen(name_head, "w+");
		if(!header_files){
			LOGE("Cannot open header file: %s, quitting ...", name_head);
			return LBP_CANT_FIND_PATH;
		}else{
			if(DEBUG_LEVEL>=LBP_DEBUG){
				LOGD("File %s, opened", name_head);
			}
		}

		fprintf(header_files, "%d\n", 1);/*is binary file?*/
		fprintf(header_files, "%d\n", 0);/*indices supplied?*/
		fprintf(header_files, "%d\n", 0);/*labels supplied?*/
		fflush(header_files);
		fclose(header_files);

		/*opening the binary file where LBP histograms are to be saved*/
		char name_feat[256]={0};
		sprintf(name_feat, "%s/%s.class_%i_train.lbp", path_save, "faces", i);
		binary_files[i] = DEFAULT_FD;
		binary_files[i] = open(name_feat, O_CREAT|O_NONBLOCK|O_RDWR, 0666);
		if(binary_files[i] == DEFAULT_FD){
			LOGE("Cannot open header file: %s, quitting ...", name_feat);
			return LBP_CANT_FIND_PATH;
		}else{
			if(DEBUG_LEVEL>=LBP_DEBUG){
				LOGD("File %s, opened", name_feat);
			}
			/*write dim header*/
			uint32_t dummy_header[2]={10, 10};/*n_vectors, n_dimensions*/
			write(binary_files[i], dummy_header, sizeof(uint32_t)*2);
		}
	}

	/*scamble data*/
	scramble_samples(negatives, positives);

	uint32_t n_positives=0, n_negatives=0;
	uint32_t channels_tmp=0;
	uint8_t finished_pos=FALSE, finished_neg=FALSE;
	while(TRUE){
		char pic_name[256]={0};

		if(n_positives<MAX_FILES_POS_CLASS && !finished_pos){
			/*================================================================================*/
			/*positive samples*/
			/*check image to be processed*/
			uint32_t rand_num_pos = positives[n_positives];
			sprintf(pic_name, "%s/%s/%d.jpeg", options->database_path, mapping[1], rand_num_pos);
			if(access(pic_name, F_OK)){
				LOGE("Cannot access picture %s, finishing loop", pic_name);
				return LBP_CANT_OPEN_IMAGE;
			}
			CImg<unsigned char> positive_img(pic_name);
			if(DEBUG_LEVEL>=LBP_INFO){
				LOGI("Processing file: %s (w: %d, h: %d, c: %d)", pic_name,
						positive_img._width, positive_img._height, positive_img.spectrum());
			}

			/*call lbp extraction main*/
			if(process_picture(&feature_extrator, (unsigned char*)positive_img.data(),
					positive_img.spectrum())!=LBP_SUCCESS){
				return LBP_CANT_EXTRACT_FEATURE;
			}

			/*write the lbp histogram in the binary file*/
			/*write data*/
			write(binary_files[atoi(mapping[1])], feature_extrator.lbp_hist,
					sizeof(float)*ncell_w*ncell_h*HIST_SIZE);

			/*reset memory to zero bits*/
			memset(feature_extrator.lbp_hist, 0, ncell_w*ncell_h*HIST_SIZE*sizeof(float));
			n_positives++;

			channels_tmp = positive_img.spectrum();
			/*================================================================================*/
		}else{
			finished_pos=TRUE;
		}

		if(n_negatives<MAX_FILES_NEG_CLASS && !finished_neg){
			/*================================================================================*/
			/*negative samples*/
			/*check image to be processed*/
			memset(pic_name, 0, 256);
			uint32_t rand_num_neg = negatives[n_negatives];
			sprintf(pic_name, "%s/%s/%d.jpeg", options->database_path, mapping[0], rand_num_neg);
			if(access(pic_name, F_OK)){
				LOGE("Cannot access picture %s, finishing loop", pic_name);
				return LBP_CANT_OPEN_IMAGE;
			}
			CImg<unsigned char> negative_img(pic_name);
			if(DEBUG_LEVEL>=LBP_INFO){
				LOGI("Processing file: %s (w: %d, h: %d, c: %d)", pic_name,
						negative_img._width, negative_img._height, negative_img.spectrum());
			}

			/*call lbp extraction main*/
			if(process_picture(&feature_extrator, (unsigned char*)negative_img.data(),
					negative_img.spectrum())!=LBP_SUCCESS){
				return LBP_CANT_EXTRACT_FEATURE;
			}

			/*write the lbp histogram in the binary file*/
			/*write data*/
			write(binary_files[atoi(mapping[0])], feature_extrator.lbp_hist,
							sizeof(float)*ncell_w*ncell_h*HIST_SIZE);

			/*reset memory to zero bits*/
			memset(feature_extrator.lbp_hist, 0, ncell_w*ncell_h*HIST_SIZE*sizeof(float));
			n_negatives++;

			if(channels_tmp<negative_img.spectrum()){
				channels_tmp = negative_img.spectrum();
			}
			/*================================================================================*/
		}else{
			finished_neg=TRUE;
		}

		if(finished_neg && finished_pos){
			if(DEBUG_LEVEL>=LBP_DEBUG){
				LOGD("Finished extracting LBP for (Pos: %d - Neg: %d) files from each class",
						MAX_FILES_POS_CLASS, MAX_FILES_NEG_CLASS);
			}
			break;
		}
	}

	for(int i=0; i<n_classes;i++){
		if(i==0){//NO FACES
			lseek(binary_files[i], 0, SEEK_SET);
			/*modify header*/
			uint32_t header[2];
			header[0] = n_negatives;
			header[1] = ncell_w*ncell_h*HIST_SIZE;
			if(DEBUG_LEVEL>=LBP_DEBUG){
				LOGD("Modify header NOFACE to:\n"
						"\tn_vects: %d\n"
						"\tn_dims: %d\n", header[0], header[1]);
			}
			write(binary_files[i], header, sizeof(uint32_t)*2);
		}else if(i==1){//FACES
			lseek(binary_files[i], 0, SEEK_SET);
			/*modify header*/
			uint32_t header[2];
			header[0] = n_positives;
			header[1] = ncell_w*ncell_h*HIST_SIZE;
			if(DEBUG_LEVEL>=LBP_DEBUG){
				LOGD("Modify header FACE to:\n"
						"\tn_vects: %d\n"
						"\tn_dims: %d\n", header[0], header[1]);
			}
			write(binary_files[i], header, sizeof(uint32_t)*2);
		}
		close(binary_files[i]);
	}
	if(binary_files){
		free(binary_files);
		binary_files=NULL;
	}

	if(feature_extrator.lbp_hist){
		free(feature_extrator.lbp_hist);
		feature_extrator.lbp_hist=NULL;
	}

	if(DEBUG_LEVEL>=LBP_VERBOSE){
		debug_histogram(n_classes, path_save, &feature_extrator, channels_tmp);
	}

	if(positives){
		free(negatives);
		free(positives);
	}

	return LBP_SUCCESS;
}

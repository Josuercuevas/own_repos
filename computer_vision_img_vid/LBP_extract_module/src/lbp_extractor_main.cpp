/*
 * lbp_extractor_main.cpp
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
#include <stdint.h>
#include <string.h>

#include "../inc/lbp_extract.h"

static uint8_t DEBUG_LEVEL = LBP_VERBOSE;

void helper_print(){
	LOGV("./extract_lbp [debug_level warning/error/none/debug/verbose/info ] [ db_path /PATH/TO/DB ] [ cell_w (1-64) ] [ cell_h (1-64) ] "
			"[ save_folder /PATH/SAVE/LBP ] [ n_classes #-of-classes ] [img_w value] [img_h value]\n"
			""
			"db_path: Path where the database RAW_JPEG files are, default NONE\n"
			"cell_w: Cell width to extract the local BP, default NONE\n"
			"cell_h: Cell height to extract the local BP, default NONE\n"
			"img_w: Input image W, default NONE\n"
			"img_h: Input image H, default NONE\n"
			"save_folder: Folder where the binary LBP features are going to be saved, default NONE\n"
			"n_classes: Number of classes contained in the folder, default NONE\n"
			"debug_level: Debug level, default verbose"
			""
			"NOTE: Same size in the training images is assumed ....\n\n");
}

uint8_t parse_options(User_Option *options, int argc, char *argv[]){
	int i;
	memset(options->bin_path, 0, 256);
	memset(options->database_path, 0, 256);
	options->h_cell=0;
	options->w_cell=0;
	options->n_classes=0;
	options->debug_level=LBP_VERBOSE;
	uint8_t counter_check=0;

	for(i=1; i<argc; i+=2){
		if(strcmp(argv[i], "db_path")==0){
			strcpy(options->database_path, argv[i+1]);
			counter_check++;
		}else if(strcmp(argv[i], "cell_w")==0){
			options->w_cell = atoi(argv[i+1]);
			counter_check++;
		}else if(strcmp(argv[i], "cell_h")==0){
			options->h_cell = atoi(argv[i+1]);
			counter_check++;
		}else if(strcmp(argv[i], "img_h")==0){
			options->img_h = atoi(argv[i+1]);
			counter_check++;
		}else if(strcmp(argv[i], "img_w")==0){
			options->img_w = atoi(argv[i+1]);
			counter_check++;
		}else if(strcmp(argv[i], "save_folder")==0){
			strcpy(options->bin_path, argv[i+1]);
			counter_check++;
		}else if(strcmp(argv[i], "n_classes")==0){
			options->n_classes = atoi(argv[i+1]);
			counter_check++;
		}else if(strcmp(argv[i], "debug_level")==0){
			if(strcmp(argv[i+1], "none")==0){
				options->debug_level = LBP_NONE;
			}else if(strcmp(argv[i+1], "errors")==0){
				options->debug_level = LBP_ERRORS;
			}else if(strcmp(argv[i+1], "warnings")==0){
				options->debug_level = LBP_WARNINGS;
			}else if(strcmp(argv[i+1], "debug")==0){
				options->debug_level = LBP_DEBUG;
			}else if(strcmp(argv[i+1], "info")==0){
				options->debug_level = LBP_INFO;
			}else if(strcmp(argv[i+1], "verbose")==0){
				options->debug_level = LBP_VERBOSE;
			}else{
				LOGW("Cannot parse this debug level: %s, setting VERBOSE");
			}
		}else{
			LOGE("Cannot parse this option: %s, quitting....", argv[i]);
			return LBP_WRONG_OPTIONS;
		}
	}

	DEBUG_LEVEL = options->debug_level;
	set_debug_level(DEBUG_LEVEL);

	if(counter_check<7){
		LOGE("Not all options have been input, quitting ...");
		return LBP_WRONG_OPTIONS;
	}else{
		if(DEBUG_LEVEL>=LBP_DEBUG){
			LOGD("Options are as follow:"
					"db_path: %s\n"
					"cell_w: %d\n"
					"cell_h: %d\n"
					"img_w: %d\n"
					"img_h: %d\n"
					"save_folder: %s\n"
					"n_classes: %d\n"
					"debug_level: %d\n\n",
					options->database_path,
					options->w_cell,
					options->h_cell,
					options->img_w,
					options->img_h,
					options->bin_path,
					options->n_classes,
					options->debug_level);
		}
	}

	return LBP_SUCCESS;
}

int main(int argc, char *argv[]){
	/*options structure*/
	User_Option options;

	if(DEBUG_LEVEL>=LBP_WARNINGS){
		LOGW("Parsing user input options ...");
	}
	if(parse_options(&options, argc, argv)!=LBP_SUCCESS){
		helper_print();
		return LBP_PARSER_FAILED;
	}

	if(DEBUG_LEVEL>=LBP_WARNINGS){
		LOGW("Extracting features ...");
	}
	if(extract_features(&options)!=LBP_SUCCESS){
		return LBP_CANT_EXTRACT_FEATURE;
	}

	return LBP_SUCCESS;
}

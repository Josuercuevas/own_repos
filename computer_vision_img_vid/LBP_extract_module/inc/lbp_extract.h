/*
 * lbp_extract.h
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

#ifndef MASTER_LBP_EXTRACT_MODULE_INC_LBP_EXTRACT_H_
#define MASTER_LBP_EXTRACT_MODULE_INC_LBP_EXTRACT_H_

#define HIST_SIZE (256)
#define MAX_FILES_POS_CLASS (300)
#define MAX_FILES_NEG_CLASS (300)
#define TRUE (1)
#define FALSE (!TRUE)
#define DEFAULT_FD (-100000)

#define MAX_POSITIVES (4703)
#define MAX_NEGATIVES (16869)

enum _err_codes_lbp{
	LBP_SUCCESS=0,
	LBP_CANT_FIND_PATH,
	LBP_CANT_OPEN_IMAGE,
	LBP_CANT_EXTRACT_FEATURE,
	LBP_CANT_PARSE_OPTION,
	LBP_WRONG_OPTIONS,
	LBP_PARSER_FAILED,
	LBP_UNKNOWN_ERROR
};

enum _debug_lbp_levels{
	LBP_NONE=0,
	LBP_ERRORS,
	LBP_WARNINGS,
	LBP_DEBUG,
	LBP_INFO,
	LBP_VERBOSE
};

typedef struct _user_option{
	/*
	 * Path of the dataset
	 * */
	char database_path[256];
	/*
	 * Path to save the binary features
	 * */
	char bin_path[256];
	/*
	 * Number of classes in this dataset
	 * */
	uint8_t n_classes;
	/*
	 * Width and height of the cell to be used
	 * to extract the lbp histogram, default
	 * */
	uint32_t w_cell, h_cell;
	/*
	 * Image dimensions
	 * */
	uint32_t img_h, img_w;

	/*
	 * debug level
	 * */
	uint8_t debug_level;
}User_Option;

typedef struct _lbp_main_structure{
	/*
	 * Width and height of the cell to be used
	 * to extract the lbp histogram, default
	 * */
	uint32_t w_cell, h_cell;
	/*
	 * Image dimensions
	 * */
	uint32_t img_h, img_w;
	/*
	 * number of cells in the whole image
	 * */
	uint32_t ncell_h, ncell_w;
	/*
	 * Histogram dimensions
	 * */
	uint32_t histogram_size;
	/*
	 * Image to be used
	 * */
	char *image_path;
	/*
	 * lbp features to be written in binary file
	 * we will use floating points since we will
	 * train a model with float point precision
	 * but in extraction phase is done uint8_t
	 * */
	float *lbp_hist;
}LBP;

/*Color code for linux*/
#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KMAG  "\x1B[35m"
#define BOLD_TEXT "\e[1m\e[4m"
#define NORM_TEXT "\e[0m"


/*debug logs needed for determining problems during execution*/
#define LOGW(fmt, arg...) printf("%s%sWARNING:%s%s %s %d: " fmt "%s\n", \
		KYEL, BOLD_TEXT, NORM_TEXT, KYEL, __FUNCTION__, __LINE__, ##arg, KNRM)

#define LOGI(fmt, arg...) printf("%s%sINFO%s%s %s %d: " fmt "%s\n", \
		KMAG, BOLD_TEXT, NORM_TEXT, KMAG, __FUNCTION__, __LINE__, ##arg, KNRM)

#define LOGE(fmt, arg...) printf("%s%sERROR:%s%s %s %d: " fmt "%s\n", \
		KRED, BOLD_TEXT, NORM_TEXT, KRED, __FUNCTION__, __LINE__, ##arg, KNRM)

#define LOGD(fmt, arg...) printf("%s%sDEBUG:%s%s %s %d: " fmt "%s\n", \
		KGRN, BOLD_TEXT, NORM_TEXT, KGRN, __FUNCTION__, __LINE__, ##arg, KNRM)

#define LOGV(fmt, arg...) printf("%s%sVERBOSE:%s%s %s %d: " fmt "%s\n", \
		KNRM, BOLD_TEXT, NORM_TEXT, KNRM, __FUNCTION__, __LINE__, ##arg, KNRM)

#define LOGCRYDET(fmt, arg...) printf("%s%sDETECTOR:%s%s %s %d: " fmt "%s\n", \
		KCYAN, BOLD_TEXT, NORM_TEXT, KCYAN, __FUNCTION__, __LINE__, ##arg, KNRM)

#define LOGCRYREC(fmt, arg...) printf("%s%sRECOGNIZER:%s%s %s %d: " fmt "%s\n", \
		KCYAN, BOLD_TEXT, NORM_TEXT, KCYAN, __FUNCTION__, __LINE__, ##arg, KNRM)
/***************************************************************/

/*main extractor functions*/
uint8_t extract_features(User_Option *options);
void set_debug_level(uint8_t level);













#endif /* MASTER_LBP_EXTRACT_MODULE_INC_LBP_EXTRACT_H_ */

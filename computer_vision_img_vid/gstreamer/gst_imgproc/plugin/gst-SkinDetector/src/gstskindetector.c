/* GStreamer
 * Copyright (C) 2015 Josue Cuevas
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Suite 500,
 * Boston, MA 02110-1335, USA.
 */
/**
 * SECTION:element-gstskindetector
 *
 * The skindetector element has the power to binarize the images so the only part left is what is appear
 * to be skins, however this is done by using a heuristic threshold, which is known to be particularly good
 * for RGB, as well to other threshold when using HSV or Normalize RGB. Work only for 32 bits
 *
 * <Pipeline example>
 * |[
 * gst-launch -v v4l2src ! videoconvert ! video/x-raw,format=RGBx/BGRx/RGBA/BGRA ! skindetector normalize=1 ! videoconvert ! ximagesink
 * ]|
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/video/gstvideofilter.h>
#include <gst/gstutils.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include "gstskindetector.h"
#include "../../common/metainfo.h"

/*
 * Extra defines and macros to be used in this element
 * */
GST_DEBUG_CATEGORY_STATIC (gst_skindetector_debug_category);
#define GST_CAT_DEFAULT gst_skindetector_debug_category
#define NORMALIZE_DEFAULT (0) //default value for this property
#define TEXTURE_DEFAULT (0.1) //default value for this property
#define MERGING_DEFAULT (0.02) //default merging factor

/*
 * path from which the Histograms files are to be monitored for
 * uploading or updating
 * */

#if (defined(__arm__) || defined(__arm64__))
 #ifndef DATABASE_MODEL_PATH
  #define DATABASE_MODEL_PATH ("/home/root/skinphoto/")
 #endif	// DATABASE_MODEL_PATH not defined
#else // Josue's pc / pc
 #define DATABASE_MODEL_PATH ("/home/josue/Desktop/plugins/skin_dictionary/")
#endif
#define ADAPTIVE_DEFAULT (TRUE) //adaptive skin detection default value


/*
 * Prototypes Functions for the skin detection API
 * */

/* sets all the predetermined properties of the element */
static void gst_skindetector_set_property(GObject * object,
    guint property_id, const GValue * value, GParamSpec * pspec);

/* gets the property values in case we want to have more than one */
static void gst_skindetector_get_property(GObject * object,
    guint property_id, GValue * value, GParamSpec * pspec);

/* dispose the frame for the filter */
static void gst_skindetector_dispose(GObject * object);

/* finishes up the process done on the video frame */
static void gst_skindetector_finalize(GObject * object);

/* starts the detector, setting all the required parameters */
static gboolean gst_skindetector_start(GstBaseTransform * trans);

/* stops and flushes all the extra information collected during the
 * filtering proces */
static gboolean gst_skindetector_stop(GstBaseTransform * trans);

/* sets the values of properties if is left open to the user */
static gboolean gst_skindetector_set_info(GstVideoFilter * filter, GstCaps * incaps,
		GstVideoInfo * in_info, GstCaps * outcaps, GstVideoInfo * out_info);

/* here is the heart of the filter in which the skin detection is to be performed */
static GstFlowReturn gst_skindetector_transform_frame(GstVideoFilter * filter, GstVideoFrame * inframe,
		GstVideoFrame * outframe);

//for caps negotiation
static GstCaps *gst_skindetector_transform_caps(GstBaseTransform * trans, GstPadDirection direction,
    GstCaps * caps, GstCaps * filter);


/* Local skin detection functions using raw RGB space */
static gint get_frame_masks(GstVideoFrame* src_frame, GstSkinDetector *skindetector, gint srcformat);
static gint raw_detection(GstVideoFrame *src_frame, GstVideoFrame *out_frame, guint32 height,
		GstSkinDetector *skindetector);

/* Local skin detection functions using hsv space */
static gint hsv_detection(GstVideoFrame* src_frame, GstVideoFrame* out_frame, guint32 height,
		GstSkinDetector *skindetector);

/* Local skin detection functions using raw YUV space */
static gint raw_YUV_detection(GstVideoFrame *src_frame, GstVideoFrame *out_frame, guint32 height,
		gint normalize_frame);

/* space conversion from RGB to HSV */
static void RGBtoHSV(guint8 *r, guint8 *g, guint8 *b, gfloat *Hnew, gfloat *Snew, gfloat *Vnew, const gint n_pixels);

/* Texture extraction rutines */
static void mask_maker(GstSkinDetector *skindetector, guint32 height, guint32 width);
static gint texture_extraction(GstVideoFrame *inframe, GstVideoFrame *outframe, guint32 height, GstSkinDetector *skindetector);

/* Histogram estimation for adaptive thresholds */
static void update_thresholds(GstSkinDetector *skindetector);


/*
 * API's supported properties, which are to be used when performing
 * the video filtering or skin detection in our case
 * */
enum _props{
	PROP_0,
	NORMALIZE,
	TEXTURE_THRES,
	MERGING,
	ADAPTIVE_THRESHOLDS,
	PROP_LOAD_TEMPLATE,
	PROP_INTERNAL_PT_SKIN,
	PROP_INTERNAL_PT_TEXTURE,
	SPACE
};

/*
 * Color spaces that can be used when detection
 * skin
 * */
enum _skin_detection_color_space{
	hsv=0,
	rgb_raw,
	rgb_normalized,
	yuv_raw
};

#define GST_TYPE_COLOR_SPACE (gst_skindetector_colorspace_get_type ())
static GType gst_skindetector_colorspace_get_type(void){
  static GType qtype = 0;

  if (qtype == 0) {
	  static const GEnumValue values[] = {
	  {hsv, "HSV space", "hsv"},
	  {rgb_raw, "Raw RGB space", "rgb_raw"},
	  {rgb_normalized, "rgb_norm", "rgb_norm"},
	  {yuv_raw, "YUV=I420 space", "yuv"},
      {0, NULL, NULL}
    };

    qtype = g_enum_register_static ("Skin_detector_color_space", values);
  }
  return qtype;
}

/*
 * Capabilities of this element's source pad
 * */
#define VIDEO_SRC_CAPS GST_VIDEO_CAPS_MAKE("{ GRAY8 }")

/*
 * Capabilities of this element's sink pad
 * */
#define VIDEO_SINK_CAPS GST_VIDEO_CAPS_MAKE("{ RGBx, BGRx, xRGB, xBGR, RGBA, BGRA, ARGB, ABGR, I420}")


/*
 * class initialization, where all the function pointers and
 * required classes as well as meta-info are to be initialized
 * */
G_DEFINE_TYPE_WITH_CODE (GstSkinDetector, gst_skindetector, GST_TYPE_VIDEO_FILTER,
		GST_DEBUG_CATEGORY_INIT (gst_skindetector_debug_category, "skindetector", 0,
				"debug category for skindetector element"));

static void gst_skindetector_class_init(GstSkinDetectorClass * skindetector){
	GObjectClass *gobject_class = G_OBJECT_CLASS(skindetector);
	GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS(skindetector);
	GstVideoFilterClass *video_filter_class = GST_VIDEO_FILTER_CLASS(skindetector);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(gst_skindetector_set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(gst_skindetector_get_property);
	gobject_class->dispose = GST_DEBUG_FUNCPTR(gst_skindetector_dispose);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(gst_skindetector_finalize);
	base_transform_class->start = GST_DEBUG_FUNCPTR(gst_skindetector_start);
	base_transform_class->stop = GST_DEBUG_FUNCPTR(gst_skindetector_stop);
	base_transform_class->transform_caps = GST_DEBUG_FUNCPTR(gst_skindetector_transform_caps);
	video_filter_class->set_info = GST_DEBUG_FUNCPTR(gst_skindetector_set_info);
	video_filter_class->transform_frame = GST_DEBUG_FUNCPTR(gst_skindetector_transform_frame);

	/*
	* Instalation of the normalization property for this element
	* */
	g_object_class_install_property(gobject_class, NORMALIZE, g_param_spec_int("normalize", "normalize",
			"Normalization to 255 for black and white output visualization: normalize=0, default", 0,
			1, NORMALIZE_DEFAULT, G_PARAM_READABLE | G_PARAM_WRITABLE));

	g_object_class_install_property(gobject_class, TEXTURE_THRES, g_param_spec_float("texture_thres", "Texture Threshold",
				"Value to be used to extract texture: range[0-1], default=0.1", 0,
				1, TEXTURE_DEFAULT, G_PARAM_READABLE | G_PARAM_WRITABLE));

	g_object_class_install_property(gobject_class, MERGING, g_param_spec_float("merge", "Merging Factor",
					"Value used to merge Prior and frame histograms: range[0, 1], default=0.02", 0,
					1, MERGING_DEFAULT, G_PARAM_READABLE | G_PARAM_WRITABLE));

	g_object_class_install_property(gobject_class, ADAPTIVE_THRESHOLDS, g_param_spec_boolean("adaptive", "Activate Adaptive thresholds",
					"Activates adaptive thresholds for HSV: {true, false}, default=true",
					ADAPTIVE_DEFAULT, G_PARAM_READABLE | G_PARAM_WRITABLE));

	g_object_class_install_property(gobject_class, SPACE, g_param_spec_enum("space", "Color space",
		  "Color space to be used to detect skin", GST_TYPE_COLOR_SPACE, hsv,
		  G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

	g_object_class_install_property(gobject_class, PROP_LOAD_TEMPLATE,
			g_param_spec_boolean("loadtemplate", "Load Template", "Loads the template for skin detection, which is always done the"
					"first time the pipeline is launched",
					ADAPTIVE_DEFAULT, (G_PARAM_READWRITE|(GST_PARAM_CONTROLLABLE|G_PARAM_STATIC_NAME))));

	g_object_class_install_property(gobject_class, PROP_INTERNAL_PT_SKIN, g_param_spec_float("internal-pt-skin", "Internal processing time",
			"Internal processing time taken for a frame when performing the skin detection", 0,
			9999999, 0, G_PARAM_READABLE));
	g_object_class_install_property(gobject_class, PROP_INTERNAL_PT_TEXTURE, g_param_spec_float("internal-pt-texture", "Internal processing time",
			"Internal processing time taken for a frame when performing the texture extraction", 0,
			9999999, 0, G_PARAM_READABLE));

	/*
	* Metadata info for this particular element
	* */
	gst_element_class_set_static_metadata (GST_ELEMENT_CLASS(skindetector), "Skin detector", "Video Filter",
		  "Detects skin patches in a video frame, where the output image is a binary one, in case the user"
		  " wants to visualize black and white it needs to normalize it", "somecpmpnay <josuercuevas@gmail.com>");


	/*
	* Setting up pads and setting metadata should be moved to
	* base_class_init if you intend to subclass this class.
	* */
	gst_element_class_add_pad_template(GST_ELEMENT_CLASS(skindetector), gst_pad_template_new("src", GST_PAD_SRC,
		  GST_PAD_ALWAYS, gst_caps_from_string(VIDEO_SRC_CAPS)));
	gst_element_class_add_pad_template(GST_ELEMENT_CLASS(skindetector), gst_pad_template_new("sink", GST_PAD_SINK,
		  GST_PAD_ALWAYS, gst_caps_from_string(VIDEO_SINK_CAPS)));
}


/*
 * In charge of finding histogram files inside the folder set by the user in
 * "DATABASE_MODEL_PATH" which is defined in this element, contains the folder
 * where the templates are going to be saved
 * */
static gchar *H_file=NULL, *S_file=NULL, *V_file=NULL;
static gint get_files(GstControlSource *Controller, gchar *dir);

static gint get_files(GstControlSource *Controller, gchar *dir){
	DIR *dp;
	struct dirent *entry;
	struct stat statbuf;
	gint depth=0;
	gboolean H_found=FALSE, S_found=FALSE, V_found=FALSE;
	gint ret=0;
	gchar template[128]={0};
	long int pos=0;


	if((dp = opendir(dir)) == NULL) {
		/*error*/
		GST_ERROR_OBJECT(GST_OBJECT_CAST(Controller), "cannot open directory where templates are located: %s\n", dir);
		return -1;
	}
	chdir(dir);

	while((entry = readdir(dp)) != NULL){
		lstat(entry->d_name,&statbuf);
		pos = telldir(dp);
		if(S_ISDIR(statbuf.st_mode)){
			/*
			 * Found a directory, but we are going to ignore them we dont need them
			 * */
			if(strcmp(".",entry->d_name) == 0 ||
				strcmp("..",entry->d_name) == 0)
				continue;
			GST_DEBUG_OBJECT(GST_OBJECT_CAST(Controller), "A directory has been found we are not going to consider it: Level: %d %s Name: %s/",
					depth,"",entry->d_name);//folder only
		}else{
			/*
			 * Here we found a file, we have to determine how new it is, if is newer than the last one
			 * we can replace and update skin model
			 * */
			GST_DEBUG_OBJECT(GST_OBJECT_CAST(Controller), "File Found Checking if is a histogram file... %s", entry->d_name);
			if(!H_found && !S_found && !V_found){//first file to be found
				if(strncmp(entry->d_name+(strlen(entry->d_name)-6), "H.hist", 6)==0){
					if(!H_found){
						/*check creation time*/
						gchar *temp_name;
						struct stat first_file_stat;
						gint err = stat(entry->d_name, &first_file_stat);
						if(err!=0){
							GST_ERROR_OBJECT(GST_OBJECT_CAST(Controller),"Error reading the file containing the histogram for H space");
						}
						temp_name = g_strdup(entry->d_name);



						/* check if is really the newest one */
						struct dirent *entry2;
						struct stat statbuf2;
						while((entry2 = readdir(dp)) != NULL){
							lstat(entry2->d_name,&statbuf2);
							if(S_ISREG(statbuf2.st_mode)){
								/* checking the file is of the same extension */
								if(strncmp(entry2->d_name+(strlen(entry2->d_name)-6), "H.hist", 6)==0){
									GST_DEBUG_OBJECT(GST_OBJECT_CAST(Controller),"Found another H file: %s", entry2->d_name);
									struct stat next_file_stat;
									gint erre = stat(entry2->d_name, &next_file_stat);
									if(erre!=0){
										GST_ERROR_OBJECT(GST_OBJECT_CAST(Controller),"Error reading another file containing the histogram for H space");
									}else{
										/*check times*/
										if(difftime(next_file_stat.st_mtim.tv_sec, first_file_stat.st_mtim.tv_sec)>0){
											/* replacing older file name */
											gchar full_path[128]={0};
											sprintf(full_path, "%s%s", DATABASE_MODEL_PATH, temp_name);
											remove(full_path);
											g_free(temp_name);
											temp_name = g_strdup(entry2->d_name);

											/*update time*/
											err = stat(entry2->d_name, &first_file_stat);
											if(err!=0){
												GST_ERROR_OBJECT(GST_OBJECT_CAST(Controller),"Error reading the file containing the histogram for H space");
											}
										}
									}
								}
							}
						}

						/* repositioning pointer */
						seekdir(dp, pos);

						//we found a histogram for H
						GST_DEBUG_OBJECT(GST_OBJECT_CAST(Controller), "----> A H_hist has been found: Name: %s",temp_name);
						H_file = g_strdup(temp_name);
						H_found=TRUE;
						if(strlen(temp_name)-6>0){
							strncpy(template, temp_name, strlen(temp_name)-6);//dont consider the ending part
						}
						g_free(temp_name);
					}
				}

				if(!H_found){
					if(strncmp(entry->d_name+(strlen(entry->d_name)-6), "S.hist", 6)==0){
						if(!S_found){
							/*check creation time*/
							gchar *temp_name;
							struct stat first_file_stat;
							gint err = stat(entry->d_name, &first_file_stat);
							if(err!=0){
								GST_ERROR_OBJECT(GST_OBJECT_CAST(Controller),"Error reading the file containing the histogram for S space");
							}
							temp_name = g_strdup(entry->d_name);

							/* check if is really the newest one */
							struct dirent *entry2;
							struct stat statbuf2;
							while((entry2 = readdir(dp)) != NULL){
								lstat(entry2->d_name,&statbuf2);
								if(S_ISREG(statbuf2.st_mode)){
									/* checking the file is of the same extension */
									if(strncmp(entry2->d_name+(strlen(entry2->d_name)-6), "S.hist", 6)==0){
										GST_DEBUG_OBJECT(GST_OBJECT_CAST(Controller),"Found another S file: %s", entry2->d_name);
										struct stat next_file_stat;
										gint erre = stat(entry2->d_name, &next_file_stat);
										if(erre!=0){
											GST_ERROR_OBJECT(GST_OBJECT_CAST(Controller),"Error reading another file containing the histogram for S space");
										}else{
											/*check times*/
											if(difftime(next_file_stat.st_mtim.tv_sec, first_file_stat.st_mtim.tv_sec)>0){
												/* replacing older file name */
												gchar full_path[128]={0};
												sprintf(full_path, "%s%s", DATABASE_MODEL_PATH, temp_name);
												remove(full_path);
												g_free(temp_name);
												temp_name = g_strdup(entry2->d_name);

												/*update time*/
												err = stat(entry2->d_name, &first_file_stat);
												if(err!=0){
													GST_ERROR_OBJECT(GST_OBJECT_CAST(Controller),"Error reading the file containing the histogram for H space");
												}
											}
										}
									}
								}
							}

							/* repositioning pointer */
							seekdir(dp, pos);

							//we found a histogram for S
							GST_DEBUG_OBJECT(GST_OBJECT_CAST(Controller), "----> A S_hist has been found: Name: %s",temp_name);
							S_file = g_strdup(temp_name);
							S_found=TRUE;
							if(strlen(temp_name)-6>0){
								strncpy(template, temp_name, strlen(temp_name)-6);//dont consider the ending part
							}
							g_free(temp_name);
						}
					}
				}

				if(!H_found && !S_found){
					if(strncmp(entry->d_name+(strlen(entry->d_name)-6), "V.hist", 6)==0){
						if(!V_found){
							/*check creation time*/
							gchar *temp_name;
							struct stat first_file_stat;
							gint err = stat(entry->d_name, &first_file_stat);
							if(err!=0){
								GST_ERROR_OBJECT(GST_OBJECT_CAST(Controller),"Error reading the file containing the histogram for V space");
							}
							temp_name = g_strdup(entry->d_name);

							/* check if is really the newest one */
							struct dirent *entry2;
							struct stat statbuf2;
							while((entry2 = readdir(dp)) != NULL){
								lstat(entry2->d_name,&statbuf2);
								if(S_ISREG(statbuf2.st_mode)){
									/* checking the file is of the same extension */
									if(strncmp(entry2->d_name+(strlen(entry2->d_name)-6), "V.hist", 6)==0){
										GST_DEBUG_OBJECT(GST_OBJECT_CAST(Controller),"Found another V file: %s", entry2->d_name);
										struct stat next_file_stat;
										gint erre = stat(entry2->d_name, &next_file_stat);
										if(erre!=0){
											GST_ERROR_OBJECT(GST_OBJECT_CAST(Controller),"Error reading another file containing the histogram for V space");
										}else{
											/*check times*/
											if(difftime(next_file_stat.st_mtim.tv_sec, first_file_stat.st_mtim.tv_sec)>0){
												/* replacing older file name */
												gchar full_path[128]={0};
												sprintf(full_path, "%s%s", DATABASE_MODEL_PATH, temp_name);
												remove(full_path);
												g_free(temp_name);
												temp_name = g_strdup(entry2->d_name);

												/*update time*/
												err = stat(entry2->d_name, &first_file_stat);
												if(err!=0){
													GST_ERROR_OBJECT(GST_OBJECT_CAST(Controller),"Error reading the file containing the histogram for H space");
												}
											}
										}
									}
								}
							}

							/* repositioning pointer */
							seekdir(dp, pos);

							//we found a histogram for V
							GST_DEBUG_OBJECT(GST_OBJECT_CAST(Controller), "----> A V_hist has been found: Name: %s",temp_name);
							V_file = g_strdup(temp_name);
							V_found=TRUE;
							if(strlen(temp_name)-6>0){
								strncpy(template, temp_name, strlen(temp_name)-6);//dont consider the ending part
							}
							g_free(temp_name);
						}
					}
				}
			}else{//we have a template so we can match the right files if there are many
				GST_DEBUG_OBJECT(GST_OBJECT_CAST(Controller), "TEMPLATE: %s", template);
				if(strncmp(entry->d_name+(strlen(entry->d_name)-6), "H.hist", 6)==0){
					if(strlen(template) == (strlen(entry->d_name)-6)){
						if(!H_found && strncmp(entry->d_name, template, (strlen(entry->d_name)-6))==0){
							//we found a histogram for H
							GST_DEBUG_OBJECT(GST_OBJECT_CAST(Controller), "A H_hist has been found: Name: %s",entry->d_name);
							H_file = g_strdup(entry->d_name);
							H_found=TRUE;
						}else{
							if(strncmp(entry->d_name, template, (strlen(entry->d_name)-6))!=0){
								/* remove this doesn't match the template of the newest found */
								gchar full_path[128]={0};
								sprintf(full_path, "%s%s", DATABASE_MODEL_PATH, entry->d_name);
								remove(full_path);
							}
						}
					}else{
						/* remove this doesn't match the template of the newest found */
						gchar full_path[128]={0};
						sprintf(full_path, "%s%s", DATABASE_MODEL_PATH, entry->d_name);
						remove(full_path);
					}
				}

				if(strncmp(entry->d_name+(strlen(entry->d_name)-6), "S.hist", 6)==0){
					if(strlen(template) == (strlen(entry->d_name)-6)){
						if(!S_found && strncmp(entry->d_name, template, (strlen(entry->d_name)-6))==0){
							//we found a histogram for S
							GST_DEBUG_OBJECT(GST_OBJECT_CAST(Controller), "A S_hist has been found: Name: %s",entry->d_name);
							S_file = g_strdup(entry->d_name);
							S_found=TRUE;
						}else{
							if(strncmp(entry->d_name, template, (strlen(entry->d_name)-6))!=0){
								/* remove this doesn't match the template of the newest found */
								gchar full_path[128]={0};
								sprintf(full_path, "%s%s", DATABASE_MODEL_PATH, entry->d_name);
								remove(full_path);
							}
						}
					}else{
						/* remove this doesn't match the template of the newest found */
						gchar full_path[128]={0};
						sprintf(full_path, "%s%s", DATABASE_MODEL_PATH, entry->d_name);
						remove(full_path);
					}
				}

				if(strncmp(entry->d_name+(strlen(entry->d_name)-6), "V.hist", 6)==0){
					if(strlen(template) == (strlen(entry->d_name)-6)){
						if(!V_found && strncmp(entry->d_name, template, (strlen(entry->d_name)-6))==0){
							//we found a histogram for V
							GST_DEBUG_OBJECT(GST_OBJECT_CAST(Controller), "A V_hist has been found: Name: %s",entry->d_name);
							V_file = g_strdup(entry->d_name);
							V_found=TRUE;
						}else{
							if(strncmp(entry->d_name, template, (strlen(entry->d_name)-6))!=0){
								/* remove this doesn't match the template of the newest found */
								gchar full_path[128]={0};
								sprintf(full_path, "%s%s", DATABASE_MODEL_PATH, entry->d_name);
								remove(full_path);
							}
						}
					}else{
						/* remove this doesn't match the template of the newest found */
						gchar full_path[128]={0};
						sprintf(full_path, "%s%s", DATABASE_MODEL_PATH, entry->d_name);
						remove(full_path);
					}
				}
			}
		}
	}
	chdir("..");
	closedir(dp);

	if(H_found && S_found && V_found){
		/*we found the three histograms successfully*/
		ret = 1;
	}else{
		/*returned nothing no histograms found*/
		ret = 0;
	}

	return ret;
}


/*
 * This function helps us to determine if we need to load templates
 * which can be updated online.
 * */
static time_t mod_time=0;
static gboolean any_histogram(GstControlSource *Controller, GstClockTime timestamp, gdouble *new_histograms);

static gboolean any_histogram(GstControlSource *Controller, GstClockTime timestamp, gdouble *new_histograms){
	gint got_histograms=FALSE;
	struct stat file_stat;
	gchar path_H[1024];

	//Getting the file name accordingly
	if((got_histograms=get_files(Controller, DATABASE_MODEL_PATH))<0){
		/*error here*/
		*new_histograms = (gdouble)got_histograms;//error code
		return FALSE;
	}else if(got_histograms==0){
		/*no histograms found no need to load*/
		*new_histograms = (gdouble)got_histograms;
	}else{
		/*load new histograms files*/
		sprintf(path_H, "%s%s",DATABASE_MODEL_PATH, H_file);
		gint err = stat(path_H, &file_stat);
		gdouble difference = difftime((time_t)file_stat.st_mtime, mod_time);
		if(difference!=0){
			//new files
			GST_INFO_OBJECT(GST_OBJECT_CAST(Controller), "Checking modification time of %s (time-diff: %g) ..",
								H_file, difference);
			GST_INFO_OBJECT(GST_OBJECT_CAST(Controller), "NEW MODEL HAS TO BE LOADED..");
			mod_time=(time_t)file_stat.st_mtime;
			*new_histograms = (gdouble)got_histograms;
		}else{
			//same files
			GST_INFO_OBJECT(GST_OBJECT_CAST(Controller), "MODEL IS THE SAME IN MEMORY..");
			*new_histograms = (gdouble)0;
		}
	}

	return TRUE;
}




/*
 * This is the main function in charge of loading or updating the thresholds from the files
 * */
//path to the files used for skin model construction
static gchar *HFILE, *SFILE, *VFILE;
static gfloat *Gprior_H_histo, *Gprior_V_histo, *Gprior_S_histo;
static gfloat GHmin, GSmin, GVmin, GHmax, GSmax, GVmax;
static gfloat GoHmin, GoSmin, GoVmin, GoHmax, GoSmax, GoVmax;

static gboolean prior_histogram_extraction(GstControlSource *Controller, GstClockTime timestamp,
		GstClockTime interval, guint n_values, gdouble *values);

static gboolean prior_histogram_extraction(GstControlSource *Controller, GstClockTime timestamp,
		GstClockTime interval, guint n_values, gdouble *file_loaded){

	FILE *H=NULL, *S=NULL, *V=NULL;
	gfloat H_hist[256], S_hist[256], V_hist[256];
	guint i;
	guint H_count, S_count, V_count;
	gchar h_histo_path[256]={0}, s_histo_path[256]={0}, v_histo_path[256]={0};
	gint got_histograms;

	//Making concatenation of the file name and the path
	/*This part has to be modified accordingly*/
	sprintf(h_histo_path, "%s%s",DATABASE_MODEL_PATH, H_file);//file_%05d_%10d_H.hist
	sprintf(s_histo_path, "%s%s",DATABASE_MODEL_PATH, S_file);//file_%05d_%10d_S.hist
	sprintf(v_histo_path, "%s%s",DATABASE_MODEL_PATH, V_file);//file_%05d_%10d_V.hist

	/*we dont need these files anymore*/
	g_free(H_file); g_free(S_file); g_free(V_file);

	H = fopen(h_histo_path, "r+b");
	S = fopen(s_histo_path, "r+b");
	V = fopen(v_histo_path, "r+b");

	if(!H || !S || !V){
		GST_ERROR_OBJECT(GST_OBJECT_CAST(Controller), "Problem opening the Prior histogram files ..!!\n");
		*file_loaded = 0;
		return FALSE;
	}

	/******************** READING PIXEL COUNT ************************/
	/* reading pixel count for H channel */
	if(fread((void*)&H_count, sizeof(guint), 1, H) != 1){
		GST_ERROR_OBJECT(GST_OBJECT_CAST(Controller), "Could not read H count ..!!\n");
		*file_loaded = 0;
		return FALSE;
	}

	/* reading pixel count for S channel */
	if(fread((void*)&S_count, sizeof(guint), 1, S) != 1){
		GST_ERROR_OBJECT(GST_OBJECT_CAST(Controller), "Could not read S count ..!!\n");
		*file_loaded = 0;
		return FALSE;
	}

	/* reading pixel count for V channel */
	if(fread((void*)&V_count, sizeof(guint), 1, V) != 1){
		GST_ERROR_OBJECT(GST_OBJECT_CAST(Controller), "Could not read V count ..!!\n");
		*file_loaded = 0;
		return FALSE;
	}


	/********************* READING INITIAL THRESHOLDS *********************/
	if(fread((void*)&GHmin, sizeof(gfloat), 1, H) != 1 ||
			fread((void*)&GHmax, sizeof(gfloat), 1, H) != 1){
		GST_ERROR_OBJECT(GST_OBJECT_CAST(Controller), "Could not read H thresholds ..!!\n");
		*file_loaded = 0;
		return FALSE;
	}else{
		GoHmin = GHmin;
		GoHmax = GHmax;
	}

	if(fread((void*)&GSmin, sizeof(gfloat), 1, S) != 1 ||
			fread((void*)&GSmax, sizeof(gfloat), 1, S) != 1){
		GST_ERROR_OBJECT(GST_OBJECT_CAST(Controller), "Could not read S thresholds ..!!\n");
		*file_loaded = 0;
		return FALSE;
	}else{
		GoSmin = GSmin;
		GoSmax = GSmax;
	}

	if(fread((void*)&GVmin, sizeof(gfloat), 1, V) != 1 ||
			fread((void*)&GVmax, sizeof(gfloat), 1, V) != 1){
		GST_ERROR_OBJECT(GST_OBJECT_CAST(Controller), "Could not read V thresholds ..!!\n");
		*file_loaded = 0;
		return FALSE;
	}else{
		GoVmin = GVmin;
		GoVmax = GVmax;
	}


	/*********************** READING HISTOGRAMS ***********************/
	/* reading histogram for H channel */
	if(fread((void*)H_hist, sizeof(gfloat), 256, H) != 256){
		GST_ERROR_OBJECT(GST_OBJECT_CAST(Controller), "Could not read H histogram from template ..!!\n");
		*file_loaded = 0;
		return FALSE;
	}

	/* reading histogram for S channel */
	if(fread((void*)S_hist, sizeof(gfloat), 256, S) != 256){
		GST_ERROR_OBJECT(GST_OBJECT_CAST(Controller), "Could not read S histogram from template ..!!\n");
		*file_loaded = 0;
		return FALSE;
	}

	/* reading histogram for V channel */
	if(fread((void*)V_hist, sizeof(gfloat), 256, V) != 256){
		GST_ERROR_OBJECT(GST_OBJECT_CAST(Controller), "Could not read V histogram from template ..!!\n");
		*file_loaded = 0;
		return FALSE;
	}

	if(fclose(H) || fclose(S) || fclose(V)){
		GST_ERROR_OBJECT(GST_OBJECT_CAST(Controller), "Error closing one of the histogram files...!!");
		*file_loaded = 0;
		return FALSE;
	}



	GST_DEBUG_OBJECT(GST_OBJECT_CAST(Controller), "LOADING NEW SKIN TEMPLATE ...!!");

	/*normalizing the extracted histograms*/
	for(i=0; i<256; i++){
		if(H_hist[i]){//we have a value > 0
			*(Gprior_H_histo+i) = H_hist[i]/H_count;
		}

		if(S_hist[i]){//we have a value > 0
			*(Gprior_S_histo+i) = S_hist[i]/S_count;
		}

		if(V_hist[i]){//we have a value > 0
			*(Gprior_V_histo+i) = V_hist[i]/V_count;
		}
	}

	GST_DEBUG_OBJECT(GST_OBJECT_CAST(Controller), "The prior histograms have been extracted successfully: "
			"\nCOUNTS: <H_count: %d, S_count: %d, V_count: %d>\nTHRESHOLDS: <H: [%f, %f], S: [%f, %f], V: [%f, %f]>",
			H_count, S_count, V_count, GHmin, GHmax, GSmin, GSmax, GVmin, GVmax);

	HFILE = g_strdup(h_histo_path);
	SFILE = g_strdup(s_histo_path);
	VFILE = g_strdup(v_histo_path);

	*file_loaded = 1;

	return TRUE;
}


/*
 * This function is in oder to initialize the element before starts
 * processing the incoming frames, the idea is that structures, variables
 * or global information is to be set from here, therefore ensuring that
 * it is done only once during the lifetime of the pipeline
 * */
static void gst_skindetector_init (GstSkinDetector *skindetector){
	gint i;
	/*
	 * parameters to set that can't be initialize
	 * from the property set function
	 * */
	skindetector->normalize_frame=0;//always initialize to this value
	skindetector->base_skindetector.negotiated = FALSE;
	skindetector->make_masks = TRUE;//to build the masks only the first time
	skindetector->LittleE = FALSE;//big endian
	skindetector->Construct_Frame_masks = TRUE;//only once
	skindetector->text_thres = TEXTURE_DEFAULT;// default texture threshold
	skindetector->merging_factor = MERGING_DEFAULT; //default merging factor
	skindetector->adaptive = ADAPTIVE_DEFAULT;// activates adaptive thresholds
	skindetector->internal_PT_skin=0.f;//processing time for a frame for skin
	skindetector->internal_PT_texture=0.f;//processing time for a frame for texture
	skindetector->firs_time = TRUE;//we haven't load any file

	/*histograms*/
	skindetector->H_histo = NULL;
	skindetector->S_histo = NULL;
	skindetector->V_histo = NULL;
	skindetector->R_histo = NULL;
	skindetector->G_histo = NULL;
	skindetector->B_histo = NULL;
	skindetector->prior_H_histo = NULL;
	skindetector->prior_S_histo = NULL;
	skindetector->prior_V_histo = NULL;
	skindetector->R_min = 0;
	skindetector->R_max = 0;
	skindetector->G_min = 0;
	skindetector->G_max = 0;
	skindetector->B_min = 0;
	skindetector->B_max = 0;
	skindetector->template_load = TRUE;//first time always load
	skindetector->skin_control_source = NULL;
	skindetector->empty_frame = FALSE;//we assume the first frame is good
}


/*
 * Sets the property for the filter, which is basically to
 * know if he wants to output image or not (black and white)
 * */
void gst_skindetector_set_property(GObject * object, guint property_id, const GValue * value, GParamSpec * pspec){
	GstSkinDetector *skindetector = GST_SKINDETECTOR (object);

	GST_DEBUG_OBJECT(skindetector, "set_property");

	switch (property_id){
		case NORMALIZE://to see if the user wants to normalize the output frame to 255
			skindetector->normalize_frame = g_value_get_int(value);
			break;
		case SPACE://detection space to be used in the skin detector
			skindetector->space = g_value_get_enum(value);
			break;
		case TEXTURE_THRES://texture threshold to consider the block as having texture or not
			skindetector->text_thres = g_value_get_float(value);
			break;
		case MERGING://mergin factor for the histograms in the skin model
			skindetector->merging_factor = g_value_get_float(value);
			break;
		case ADAPTIVE_THRESHOLDS://to activate or deactivate skin threshold adaptation
			skindetector->adaptive = g_value_get_boolean(value);
			break;
		case PROP_LOAD_TEMPLATE://to load the templates or model extracted offline
			skindetector->template_load = g_value_get_boolean(value);
			break;
		default:
			G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
			break;
	}
}


/*
 * gets the property for the filter, which is basically to
 * know if he wants to output image or not (black and white)
 * */
void gst_skindetector_get_property(GObject * object, guint property_id, GValue * value, GParamSpec * pspec){
	GstSkinDetector *skindetector = GST_SKINDETECTOR (object);

	GST_DEBUG_OBJECT (skindetector, "get_property");

	switch (property_id) {
		case NORMALIZE://to see if the user wants to normalize the output frame to 255
			g_value_set_int(value, skindetector->normalize_frame);
			break;
		case SPACE://
			g_value_set_enum(value, skindetector->space);
			break;
		case TEXTURE_THRES://
			g_value_set_float(value, skindetector->text_thres);
			break;
		case MERGING://
			g_value_set_float(value, skindetector->merging_factor);
			break;
		case ADAPTIVE_THRESHOLDS://
			g_value_set_boolean(value, skindetector->adaptive);
			break;
		case PROP_LOAD_TEMPLATE://
			g_value_set_boolean(value, skindetector->template_load);
			break;
		case PROP_INTERNAL_PT_SKIN://
			g_value_set_float(value, skindetector->internal_PT_skin);
			break;
		case PROP_INTERNAL_PT_TEXTURE://
			g_value_set_float(value, skindetector->internal_PT_texture);
			break;
		default:
			G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
			break;
	}
}

/*
 * Cleans all the garbage information collected ... in our case
 * there is none for now
 * */
void gst_skindetector_dispose(GObject * object){
	GstSkinDetector *skindetector = GST_SKINDETECTOR (object);

	GST_DEBUG_OBJECT (skindetector, "Disposing the data used in the SkinDetector element");

	/* clean up as possible.  may be called multiple times */
	G_OBJECT_CLASS (gst_skindetector_parent_class)->dispose(object);
}

/*
 * Cleans all the garbage information collected ... in our case
 * there is none for now
 * */
void gst_skindetector_finalize(GObject * object){
	GstSkinDetector *skindetector = GST_SKINDETECTOR (object);

	GST_DEBUG_OBJECT (skindetector, "Finalizing SkinDetector element");

	/* clean up object here */
	G_OBJECT_CLASS (gst_skindetector_parent_class)->finalize (object);
}

static gboolean gst_skindetector_start(GstBaseTransform * trans){
	GstSkinDetector *skindetector = GST_SKINDETECTOR (trans);

	/*
	 * Just puts some debugging information for the comfortability
	 * of the user
	 * */
	if(skindetector->normalize_frame != NORMALIZE_DEFAULT)
	  GST_DEBUG_OBJECT (skindetector, "Output image will be normalized to 255");
	else
	  GST_DEBUG_OBJECT (skindetector, "Output image is going to be binary (0,1)");


	/*creating memory for histograms*/
	if(skindetector->R_histo==NULL)
		skindetector->R_histo = (gfloat*)g_malloc0(sizeof(gfloat)*256);

	if(skindetector->G_histo==NULL)
		skindetector->G_histo = (gfloat*)g_malloc0(sizeof(gfloat)*256);

	if(skindetector->B_histo==NULL)
		skindetector->B_histo = (gfloat*)g_malloc0(sizeof(gfloat)*256);

	/*local use*/
	if(skindetector->H_histo==NULL)
		skindetector->H_histo = (gfloat*)g_malloc0(sizeof(gfloat)*256);

	if(skindetector->S_histo==NULL)
		skindetector->S_histo = (gfloat*)g_malloc0(sizeof(gfloat)*256);

	if(skindetector->V_histo==NULL)
		skindetector->V_histo = (gfloat*)g_malloc0(sizeof(gfloat)*256);

	/*offline estimated*/
	if(skindetector->prior_H_histo==NULL)
		Gprior_H_histo = skindetector->prior_H_histo = (gfloat*)g_malloc0(sizeof(gfloat)*256);

	if(skindetector->prior_S_histo==NULL)
		Gprior_S_histo = skindetector->prior_S_histo = (gfloat*)g_malloc0(sizeof(gfloat)*256);

	if(skindetector->prior_V_histo==NULL)
		Gprior_V_histo = skindetector->prior_V_histo = (gfloat*)g_malloc0(sizeof(gfloat)*256);

	if(skindetector->H_histo==NULL || skindetector->S_histo==NULL || skindetector->V_histo==NULL ||
			skindetector->prior_H_histo==NULL || skindetector->prior_S_histo==NULL ||
			skindetector->prior_V_histo==NULL){
		GST_ERROR_OBJECT(skindetector, "Memory to store histograms was not allocated..!!\n");
		return FALSE;
	}


	/*Making the binding of the controller*/
	skindetector->skin_control_source = gst_trigger_control_source_new();
	skindetector->skin_control_source->get_value = (GstControlSourceGetValue)any_histogram;
	skindetector->skin_control_source->get_value_array = (GstControlSourceGetValueArray)prior_histogram_extraction;
	gst_object_add_control_binding(GST_OBJECT_CAST (skindetector),
		gst_direct_control_binding_new(GST_OBJECT_CAST (skindetector), "loadtemplate",
				skindetector->skin_control_source));

	/*to load the histogram the first time always*/
	skindetector->template_load=TRUE;

	return TRUE;
}

static gboolean gst_skindetector_stop(GstBaseTransform * trans){
	gint i;
	GstSkinDetector *skindetector = GST_SKINDETECTOR(trans);

	/*
	 * Just puts some debugging information for the comfortability
	 * of the user
	 * */
	GST_DEBUG_OBJECT(skindetector, "stopping pipeline and freeing histograms");

	/*offline estimated*/
	if(skindetector->adaptive){
		if(skindetector->prior_H_histo){
			g_free(skindetector->prior_H_histo);
		}
		if(skindetector->prior_S_histo){
			g_free(skindetector->prior_S_histo);
		}
		if(skindetector->prior_V_histo){
			g_free(skindetector->prior_V_histo);
		}

		/*locally estimated*/
		if(skindetector->H_histo){
			g_free(skindetector->H_histo);
		}
		if(skindetector->S_histo){
			g_free(skindetector->S_histo);
		}
		if(skindetector->V_histo){
			g_free(skindetector->V_histo);
		}

		/*locally estimated*/
		if(skindetector->R_histo){
			g_free(skindetector->R_histo);
		}
		if(skindetector->G_histo){
			g_free(skindetector->G_histo);
		}
		if(skindetector->B_histo){
			g_free(skindetector->B_histo);
		}

		/*GstControllers for skin detector*/
		if(skindetector->skin_control_source){
			g_object_unref(skindetector->skin_control_source);
			skindetector->skin_control_source = NULL;
		}
	}

	return TRUE;
}


static GstCaps *gst_skindetector_transform_caps(GstBaseTransform * trans, GstPadDirection direction,
    GstCaps * caps, GstCaps * filter){
	GstSkinDetector *skindetector=NULL;
	GstCaps *to=NULL, *ret=NULL;
	GstCaps *templ=NULL;
	GstStructure *structure=NULL;
	GstPad *other=NULL;
	gint i;
	GstVideoInfo info;

	skindetector = GST_SKINDETECTOR(trans);
	GST_LOG_OBJECT(skindetector, "transforming caps %" GST_PTR_FORMAT, caps);

	to = gst_caps_new_empty();

	for (i = 0; i < gst_caps_get_size(caps); i++){
		const GValue *v;
		GValue list = { 0, };
		GValue val = { 0, };

		structure = gst_structure_copy(gst_caps_get_structure (caps, i));

		g_value_init(&list, GST_TYPE_LIST);

		g_value_init(&val, G_TYPE_STRING);
		g_value_set_string(&val, "GRAY8");
		gst_value_list_append_value(&list, &val);
		g_value_unset(&val);

		g_value_init(&val, G_TYPE_STRING);
		g_value_set_string(&val, "RGBx");
		gst_value_list_append_value(&list, &val);
		g_value_unset(&val);

		g_value_init(&val, G_TYPE_STRING);
		g_value_set_string(&val, "RGBx");
		gst_value_list_append_value(&list, &val);
		g_value_unset(&val);

		g_value_init(&val, G_TYPE_STRING);
		g_value_set_string(&val, "RGBA");
		gst_value_list_append_value(&list, &val);
		g_value_unset(&val);

		g_value_init(&val, G_TYPE_STRING);
		g_value_set_string(&val, "xRGB");
		gst_value_list_append_value(&list, &val);
		g_value_unset(&val);

		g_value_init(&val, G_TYPE_STRING);
		g_value_set_string(&val, "ARGB");
		gst_value_list_append_value(&list, &val);
		g_value_unset(&val);

		g_value_init(&val, G_TYPE_STRING);
		g_value_set_string(&val, "BGRx");
		gst_value_list_append_value(&list, &val);
		g_value_unset(&val);

		g_value_init(&val, G_TYPE_STRING);
		g_value_set_string(&val, "BGRA");
		gst_value_list_append_value(&list, &val);
		g_value_unset(&val);

		g_value_init(&val, G_TYPE_STRING);
		g_value_set_string(&val, "xBGR");
		gst_value_list_append_value(&list, &val);
		g_value_unset(&val);

		g_value_init(&val, G_TYPE_STRING);
		g_value_set_string(&val, "ABGR");
		gst_value_list_append_value(&list, &val);
		g_value_unset(&val);

		v = gst_structure_get_value(structure, "format");

		if(direction == GST_PAD_SRC){
			gst_structure_remove_field(structure, "colorimetry");
			gst_structure_remove_field(structure, "chroma-site");

			gst_value_list_merge(&val, v, &list);
			gst_structure_set_value(structure, "format", &val);
			g_value_unset(&val);
			g_value_unset(&list);
			gst_caps_append_structure(to, structure);
		}else{
			gst_value_list_merge(&val, v, &list);
			gst_structure_set_value(structure, "format", &val);
			g_value_unset(&val);
			g_value_unset(&list);
			gst_caps_append_structure(to, structure);
		}


	}
	/* filter against set allowed caps on the pad */
	other = (direction == GST_PAD_SINK) ? trans->srcpad : trans->sinkpad;
	templ = gst_pad_get_pad_template_caps(other);
	ret = gst_caps_intersect(to, templ);
	gst_caps_unref(to);
	gst_caps_unref(templ);

	if (ret && filter){
		GstCaps *intersection;
		intersection = gst_caps_intersect_full(filter, ret, GST_CAPS_INTERSECT_FIRST);
		gst_caps_unref(ret);
		ret = intersection;
	}

	if(ret != NULL){
		GST_DEBUG_OBJECT(skindetector, "Caps negotiated successfully..!!");
		skindetector->base_skindetector.negotiated = TRUE;
	}else{
		GST_ERROR_OBJECT(skindetector, "Can't negotiate caps..!!");
		skindetector->base_skindetector.negotiated = FALSE;
	}

	return ret;
}


static gboolean gst_skindetector_set_info(GstVideoFilter * filter, GstCaps * incaps,
		GstVideoInfo * in_info, GstCaps * outcaps, GstVideoInfo * out_info){
	GstSkinDetector *skindetector = GST_SKINDETECTOR(filter);
	GST_DEBUG_OBJECT(skindetector, "setting info");
	return TRUE;
}

/*
 * Main function of this filter, where the skin detector is to be invoked and tested
 * to see its performance and accuracy
 *  */
static GstFlowReturn gst_skindetector_transform_frame(GstVideoFilter * filter, GstVideoFrame * inframe,
		GstVideoFrame * outframe){

	GstSkinDetector *skindetector = GST_SKINDETECTOR(filter);
	gint width, height, framesize;
	guint8 ierror;
	gint sstride[3], srcformat;

	if(!filter->negotiated){
		GST_DEBUG_OBJECT(skindetector, "Caps have NOT been negotiated, proceeding to negotiation phase..!!");
		GstBaseTransform *skindetectorBaseTransform = GST_BASE_TRANSFORM(filter);
		GstVideoFilterClass *skindetectorclass = GST_SKINDETECTOR_CLASS(filter);
		GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS(skindetectorclass);
		if(!base_transform_class->transform_caps){
			GST_ERROR_OBJECT(skindetector, "The caps negotiation have failed, closing application");
			return GST_FLOW_ERROR;
		}
	}


	//gets the frame to be processed
	skindetector->inframe = inframe;
	skindetector->outframe = outframe;

	GstClockTime timestamp = GST_BUFFER_TIMESTAMP(skindetector->inframe->buffer);
	if (GST_CLOCK_TIME_IS_VALID(timestamp) && (skindetector->adaptive || skindetector->template_load)){
		/*
		 * check the current model descriptor, to determine
		 * if the template has changed or we still using the same
		 * in the case that the template changes we will load the new one
		 * even if the adaptive option in the detector is deactivated
		 * */
		GST_DEBUG_OBJECT(skindetector, "Synch values for controller..");
		gdouble load_file;
		gst_control_source_get_value(skindetector->skin_control_source, timestamp, &load_file);
		GST_DEBUG_OBJECT(skindetector, "Done Synch values for controller..");


		if(load_file<0){
			//Problem updating the controller value
			return GST_FLOW_ERROR;
		}else if(load_file==1){
			gdouble file_loaded;
			gst_control_source_get_value_array(skindetector->skin_control_source, timestamp, NULL, 1, &file_loaded);

			if(file_loaded){
				//we loaded new template from GstController
				if(HFILE && SFILE && VFILE){
					GST_DEBUG_OBJECT(skindetector, "Removing files used for model construction.. \t%s\t%s\t%s\n\n",
							HFILE, SFILE, VFILE);
					/*freeing pointers since we used g_strdup*/
					g_free(HFILE);
					g_free(SFILE);
					g_free(VFILE);
					HFILE=NULL;
					SFILE=NULL;
					VFILE=NULL;
				}

				/*
				 * updating thresholds since we did extracted
				 * */
				skindetector->Hmax = GHmax;
				skindetector->Hmin = GHmin;
				skindetector->Smax = GSmax;
				skindetector->Smin = GSmin;
				skindetector->Vmax = GVmax;
				skindetector->Vmin = GVmin;

				/*
				 * to keep in memory in case we need to
				 * reset the detector
				 * */
				skindetector->oHmax = GoHmax;
				skindetector->oHmin = GoHmin;
				skindetector->oSmax = GoSmax;
				skindetector->oSmin = GoSmin;
				skindetector->oVmax = GoVmax;
				skindetector->oVmin = GoVmin;

				/* we don't need to worry about not having models in the folder */
				skindetector->firs_time = FALSE;
			}else{
				//Problem loading files
				GST_ERROR_OBJECT(skindetector, "There was a problem loading the model file ...!");
				return GST_FLOW_ERROR;
			}
		}else{
			if(skindetector->firs_time){
				GST_ERROR_OBJECT(skindetector, "There is no model at the default folder location and no model loaded in memory ... quitting!");
				return GST_FLOW_ERROR;
			}
		}
	}

	//strides to be used per channel
	sstride[0] = GST_VIDEO_FRAME_PLANE_STRIDE (skindetector->inframe, 0);
	sstride[1] = GST_VIDEO_FRAME_PLANE_STRIDE (skindetector->inframe, 1);
	sstride[2] = GST_VIDEO_FRAME_PLANE_STRIDE (skindetector->inframe, 2);

	//setting output frame to nothing, since we manipulated th buffers bits last time
	memset(GST_VIDEO_FRAME_PLANE_DATA(skindetector->outframe,0), 0, GST_VIDEO_FRAME_HEIGHT(outframe)*GST_VIDEO_FRAME_WIDTH(outframe));

	srcformat = GST_VIDEO_FRAME_FORMAT(inframe);
	width = sstride[0];
	height = GST_VIDEO_FRAME_HEIGHT(inframe);

	//-----------------------SKIN_DETECTION PART--------------------------//
	time_t start = clock(), end;
	time_t start1=0, end1=0, start2=0, end2=0;
	GST_DEBUG_OBJECT (skindetector, "Performing Skin Detection:\n\t<OutImage <s: %i, w: %i, h: %i, C: %i> ; InImage <s: %i, w: %i, H: %i, C: %i> >",
			GST_VIDEO_FRAME_PLANE_STRIDE (skindetector->outframe, 0), GST_VIDEO_FRAME_WIDTH(skindetector->outframe),
			GST_VIDEO_FRAME_HEIGHT(skindetector->outframe), GST_VIDEO_FRAME_N_COMPONENTS(skindetector->outframe),
			GST_VIDEO_FRAME_PLANE_STRIDE(skindetector->inframe, 0), GST_VIDEO_FRAME_WIDTH(skindetector->inframe),
			GST_VIDEO_FRAME_HEIGHT(skindetector->inframe), GST_VIDEO_FRAME_N_COMPONENTS(skindetector->inframe));

	if(srcformat==GST_VIDEO_FORMAT_I420){//Dealing with YUV frames of format 4:2:0
		/*
		 * We need to lock-mutex the access for this frame so we do not
		 * let another process to mess up with the data we are extracting
		 * and modifying
		 * */
		GST_OBJECT_LOCK(skindetector);

		/*
		 * Texture extraction part (from 1~7 bit)
		 * Implementing binary patterns to determine pixel texture
		 * which could be later used when using blob detection, in order
		 * to discriminate between skin and non-skin blobs
		 * */
		/*building only when the pipeline is launched*/
		if(skindetector->make_masks){
			mask_maker(skindetector, GST_VIDEO_FRAME_HEIGHT(inframe), GST_VIDEO_FRAME_WIDTH(inframe));
			skindetector->make_masks = FALSE;
		}

		if(!skindetector->normalize_frame){
			/*since if we normalize it there is no sense of doing this */
			texture_extraction(inframe, outframe, height, skindetector);
		}

		ierror = raw_YUV_detection(inframe, outframe, height, skindetector->normalize_frame);

		if(ierror != skin_success){
			GST_ERROR_OBJECT(skindetector, "Problem performing skin detection ...\n");
			return GST_FLOW_ERROR;
		}


		/*
		 * Unlocking the data for releasing the changes and output
		 * the modified frame
		 * */
		GST_OBJECT_UNLOCK(skindetector);
	}else if(srcformat==GST_VIDEO_FORMAT_RGBA || srcformat==GST_VIDEO_FORMAT_ARGB || srcformat==GST_VIDEO_FORMAT_BGRA
			|| srcformat==GST_VIDEO_FORMAT_ABGR || srcformat==GST_VIDEO_FORMAT_RGBx || srcformat==GST_VIDEO_FORMAT_xRGB
			|| srcformat==GST_VIDEO_FORMAT_BGRx || srcformat==GST_VIDEO_FORMAT_xBGR){

		//for sure it should be RGB since no other format is accepted in the CAPS
		/*
		 * We need to lock-mutex the access for this frame so we do not
		 * let another process to mess up with the data we are extracting
		 * and modifying
		 * */
		GST_OBJECT_LOCK(skindetector);

		/*
		 * Construct the masks for speed up computation in detection
		 * */
		if(skindetector->Construct_Frame_masks){
			GST_DEBUG_OBJECT(skindetector, "Constructing BitMasks for speeding skin detection");
			if(get_frame_masks(inframe, skindetector, srcformat)!=skin_success){
				return GST_FLOW_ERROR;
			}
		}


		/*
		 * Texture extraction part (from 1~7 bit)
		 * Implementing binary patterns to determine pixel texture
		 * which could be later used when using blob detection, in order
		 * to discriminate between skin and non-skin blobs
		 * */
		/*building only when the pipeline is launched*/
		if(skindetector->make_masks){
			mask_maker(skindetector, GST_VIDEO_FRAME_HEIGHT(inframe), GST_VIDEO_FRAME_WIDTH(inframe));
			skindetector->make_masks = FALSE;
		}

		if(!skindetector->normalize_frame){
			/*since if we normalize it there is no sense of doing this */
			start1=clock();
			texture_extraction(inframe, outframe, height, skindetector);
			end1=clock();
		}

		switch(skindetector->space) {
			case (hsv):
				start2=clock();
				ierror = hsv_detection(inframe, outframe, height, skindetector);
				end2=clock();
				break;
			case (rgb_raw):
				ierror = raw_detection(inframe, outframe, height, skindetector);
				break;
			default:
				GST_ERROR_OBJECT(skindetector, "This space is not supported by the API");
				return GST_FLOW_ERROR;
		}

		if(ierror != skin_success){
			GST_ERROR_OBJECT(skindetector, "Problem performing skin detection ...\n");
			return GST_FLOW_ERROR;
		}

		/*
		 * Unlocking the data for releasing the changes and output
		 * the modified frame
		 * */
		GST_OBJECT_UNLOCK(skindetector);
	}else{
		GST_ERROR_OBJECT (skindetector, "The input format of the frame is not supported by this element SKINDETECTOR: %s",
				gst_video_format_to_string(skindetector->base_skindetector.in_info.finfo->format));
		return GST_FLOW_ERROR;
	}
	end = clock();
	GST_DEBUG_OBJECT(skindetector, "++++++++++++++++++++++++++++++++++++++++++++++\n"
			" Total time in element Skin_Detector: %f ms. <Texture: %f, Skin:%f>\n"
			"++++++++++++++++++++++++++++++++++++++++++++++\n",1000*(((gfloat) end - start)/CLOCKS_PER_SEC),
			1000*(((gfloat) end1 - start1)/CLOCKS_PER_SEC), 1000*(((gfloat) end2 - start2)/CLOCKS_PER_SEC));
	skindetector->internal_PT_skin = 1000*(((gfloat) end2 - start2)/CLOCKS_PER_SEC);
	skindetector->internal_PT_texture = 1000*(((gfloat) end1 - start1)/CLOCKS_PER_SEC);
	//-----------------------SKIN_DETECTION PART--------------------------//

	return GST_FLOW_OK;
}




#define max(a, b) ({a>b?a:b;})
#define min(a, b) ({a<b?a:b;})


/*
 * FIXME
 * Not optimized ... dont use this space (we can think how to speed it up)
 * */
static gint raw_YUV_detection(GstVideoFrame *src_frame, GstVideoFrame *out_frame, guint32 height, gint normalize_frame){
	guint8 R, G, B, Y, Cb, Cr;
	guint8 *Y_data = GST_VIDEO_FRAME_COMP_DATA(src_frame, 0);
	guint8 *U_data = GST_VIDEO_FRAME_COMP_DATA(src_frame, 1);
	guint8 *V_data = GST_VIDEO_FRAME_COMP_DATA(src_frame, 2);
	guint8 *GRAY_data = GST_VIDEO_FRAME_COMP_DATA(out_frame, 0);
	gint Y_stride = GST_VIDEO_FRAME_COMP_STRIDE(src_frame, 0);
	gint U_stride = GST_VIDEO_FRAME_COMP_STRIDE(src_frame, 1);
	gint V_stride = GST_VIDEO_FRAME_COMP_STRIDE(src_frame, 2);
	gint GRAY_stride = GST_VIDEO_FRAME_COMP_STRIDE(out_frame, 0);
	gint i,j, k, l;
	gint img_size = height*Y_stride;
	gint u_size = U_stride*(height>>1);
	gboolean res = FALSE;

	for(i=0;i<height;i++){
		for(j=0;j<Y_stride;j++){
			//WE ASSUME THAT YUV IS SAMPLED AS 4:2:0 OTHERWISE WE HAVE A PROBLEM
			k = i/2;
			l = j/2;
			Y = Y_data[i*Y_stride + j];
			Cr = U_data[k*U_stride + l + img_size];//half resolution of the Y channel
			Cb = V_data[k*V_stride + l + img_size + u_size];//half resolution of the Y channel + U channel

			R = 1.164*(Y - 16) + 1.596*(Cb - 128);
			G = 1.164*(Y - 16) - 0.813*(Cb - 128) - 0.391*(Cr - 128);
			B = 1.164*(Y - 16) + 2.018*(Cr - 128);

			res = ( ( R > 95) && ( G > 40 ) && ( B > 20 ) && (max(R, max( G, B) ) - min(R, min(G, B) ) > 15) &&
					(abs(R - G) > 15) && (R > G) && (R > B) );

			//REMEMBER THE DESTINATION IS GRAY8
			if(normalize_frame){
				GRAY_data[i*GRAY_stride + j] = res ? 255 : 0;//skin detected R
			}else{
				GRAY_data[i*GRAY_stride + j] = res ? 1 : 0;//skin detected R
			}
		}
	}
	return skin_success;
}


static gint get_frame_masks(GstVideoFrame* src_frame, GstSkinDetector *skindetector, gint srcformat){
	/*
	 * Assuming 32 bits RGBA (or any combination)
	 * */

	if(src_frame->info.finfo->flags & GST_VIDEO_FORMAT_FLAG_LE){//LITTLE ENDIAN
		/* IMPORTANT (LITTLE ENDIAN) --->> FORMATS ARE NOT CHANGED:
		 * 		1. RGBA -> RGBA
		 * 		2. RGBx -> RGBx
		 * 		3. BGRA -> BGRA
		 * 		4. BGRx -> BGRx
		 * 		5. ARGB -> ARGB
		 * 		6. xRGB -> xRGB
		 * 		7. ABGR -> ABGR
		 * 		8. xBGR -> xBGR
		 * 		*/
		skindetector->LittleE = TRUE;
		if(srcformat==GST_VIDEO_FORMAT_RGBA || srcformat==GST_VIDEO_FORMAT_RGBx){
			skindetector->RedMask = 0xff000000;//R-mask
			skindetector->GreenMask = 0x00ff0000;//G-mask
			skindetector->BlueMask = 0x0000ff00;//B-mask
			skindetector->RedShift=24;
			skindetector->GreenShift=16;
			skindetector->BlueShift=8;
			GST_INFO_OBJECT(skindetector, "Little Endinan, RGBA(x)");
		}else if(srcformat==GST_VIDEO_FORMAT_ARGB || srcformat==GST_VIDEO_FORMAT_xRGB){
			skindetector->RedMask = 0x00ff0000;//R-mask
			skindetector->GreenMask = 0x0000ff00;//G-mask
			skindetector->BlueMask = 0x000000ff;//B-mask
			skindetector->RedShift=16;
			skindetector->GreenShift=8;
			skindetector->BlueShift=0;
			GST_INFO_OBJECT(skindetector, "Little Endinan, (x)ARGB");
		}else if(srcformat==GST_VIDEO_FORMAT_BGRA || srcformat==GST_VIDEO_FORMAT_BGRx){
			skindetector->RedMask = 0x0000ff00;//R-mask
			skindetector->GreenMask = 0x00ff0000;//G-mask
			skindetector->BlueMask = 0xff000000;//B-mask
			skindetector->RedShift=8;
			skindetector->GreenShift=16;
			skindetector->BlueShift=24;
			GST_INFO_OBJECT(skindetector, "Little Endinan, BGRA(x)");
		}else if(srcformat==GST_VIDEO_FORMAT_ABGR || srcformat==GST_VIDEO_FORMAT_xBGR){
			skindetector->RedMask = 0x000000ff;//R-mask
			skindetector->GreenMask = 0x0000ff00;//G-mask
			skindetector->BlueMask = 0x00ff0000;//B-mask
			skindetector->RedShift=0;
			skindetector->GreenShift=8;
			skindetector->BlueShift=16;
			GST_INFO_OBJECT(skindetector, "Little Endinan, (x)ABGR");
		} else{
			//something wrong
			GST_ERROR("This format of image is not supported by the API..!!");
			return skin_wrong_img_type;
		}
	}else{//BIG ENDIAN
		/* IMPORTANT (BIG ENDIAN) --->> FORMATS ARE CHANGED AS:
		 * 		1. RGBA -> ABGR
		 * 		2. RGBx -> xBGR
		 * 		3. BGRA -> ARGB
		 * 		4. BGRx -> xRGB
		 * 		5. ARGB -> BGRA
		 * 		6. xRGB -> BGRx
		 * 		7. ABGR -> RGBA
		 * 		8. xBGR -> RGBx
		 * 		*/
		skindetector->LittleE = FALSE;
		if(srcformat==GST_VIDEO_FORMAT_RGBA || srcformat==GST_VIDEO_FORMAT_RGBx){
			skindetector->RedMask = 0x000000ff;//R-mask
			skindetector->GreenMask = 0x0000ff00;//G-mask
			skindetector->BlueMask = 0x00ff0000;//B-mask
			skindetector->RedShift=0;
			skindetector->GreenShift=8;
			skindetector->BlueShift=16;
			GST_INFO_OBJECT(skindetector, "Big Endinan, RGBA(x)");
		}else if(srcformat==GST_VIDEO_FORMAT_ARGB || srcformat==GST_VIDEO_FORMAT_xRGB){
			skindetector->RedMask = 0x0000ff00;//R-mask
			skindetector->GreenMask = 0x00ff0000;//G-mask
			skindetector->BlueMask = 0xff000000;//B-mask
			skindetector->RedShift=8;
			skindetector->GreenShift=16;
			skindetector->BlueShift=24;
			GST_INFO_OBJECT(skindetector, "Big Endinan, (x)ARGB");
		}else if(srcformat==GST_VIDEO_FORMAT_BGRA || srcformat==GST_VIDEO_FORMAT_BGRx){
			skindetector->RedMask = 0x00ff0000;//R-mask
			skindetector->GreenMask = 0x0000ff00;//G-mask
			skindetector->BlueMask = 0x000000ff;//B-mask
			skindetector->RedShift=16;
			skindetector->GreenShift=8;
			skindetector->BlueShift=0;
			GST_INFO_OBJECT(skindetector, "Big Endinan, BGRA(x)");
		}else if(srcformat==GST_VIDEO_FORMAT_ABGR || srcformat==GST_VIDEO_FORMAT_xBGR){
			skindetector->RedMask = 0xff000000;//R-mask
			skindetector->GreenMask = 0x00ff0000;//G-mask
			skindetector->BlueMask = 0x0000ff00;//B-mask
			skindetector->RedShift=24;
			skindetector->GreenShift=16;
			skindetector->BlueShift=8;
			GST_INFO_OBJECT(skindetector, "Big Endinan, (x)ABGR");
		} else{
			//something wrong
			GST_ERROR("This format of image is not supported by the API..!!");
			return skin_wrong_img_type;
		}
	}

	skindetector->Construct_Frame_masks = FALSE;
	return skin_success;
}


static gint raw_detection(GstVideoFrame* src_frame, GstVideoFrame* out_frame, guint32 height, GstSkinDetector *skindetector){
	guint32 pos = 0, x, y, color;
	guint8 R = 0, G = 0, B = 0;
	gboolean res=FALSE;
	guint8 *src_data = NULL, *GRAY_data = NULL;
	gint src_stride = 0, GRAY_stride = 0, GRAY_comp = 0;
	gint pix_stride=4;//IMPORTANT: ASUMING 32 bits
	guint32 RedMask=skindetector->RedMask, GreenMask=skindetector->GreenMask, BlueMask=skindetector->BlueMask;
	guint8 RedShift=skindetector->RedShift, GreenShift=skindetector->GreenShift, BlueShift=skindetector->BlueShift;



	/* Beginning of the pointer, no matter endianess */
	src_data = GST_VIDEO_FRAME_PLANE_DATA(src_frame, 0);
	src_stride = GST_VIDEO_FRAME_COMP_STRIDE(src_frame, 0);

	/* Beginning of the pointer, no worry about endianess */
	GRAY_data = GST_VIDEO_FRAME_PLANE_DATA(out_frame, GRAY_comp);
	GRAY_stride = GST_VIDEO_FRAME_COMP_STRIDE(out_frame, GRAY_comp);


	while (pos < height*src_stride){

		/*
		 *  According to LE or BE given above, and channels distribution given by user
		 *
		 * Idea:
		 * 		R = ( RGB_ptr & RedMask ) >> RedShift
		 * 		G = ( RGB_ptr & GreenMask ) >> GreenShift
		 * 		B = ( RGB_ptr & BlueMask ) >> BlueShift
		 * */
		color = (*((guint32*)(src_data+pos)));
		R = (color&RedMask)>>RedShift;
		G = (color&GreenMask)>>GreenShift;
		B = (color&BlueMask)>>BlueShift;

		/* ( R > 95) && ( G > 40 ) && ( B > 20 ) && (max(R, max( G, B) ) - min(R, min(G, B) ) > 15) &&
				(abs(R - G) > 15) && (R > G) && (R > B) */

		res = ( ( R > 95) && ( G > 40 ) && ( B > 20 ) && (max(R, max( G, B) ) - min(R, min(G, B) ) > 15) &&
				(abs(R - G) > 15) && (R > G) && (R > B) ); //skin detection rule (fixed global thresholds now)


		/*
		 * REMEMBER THE DESTINATION IS GRAY8
		 * labeling pixel (according to classification)
		 *
		 * */
		if(skindetector->normalize_frame){
			*(GRAY_data+pos/4)  = res ? 255 : 0;
		}else{
			*(GRAY_data+pos/4)  = res ? 1 : 0;
		}

		pos+=pix_stride;
	}

	return skin_success;
}



static gint hsv_detection(GstVideoFrame* src_frame, GstVideoFrame* out_frame, guint32 height, GstSkinDetector *skindetector){
	const gint line_size=4;
	const gint pix_stride=4;//IMPORTANT: ASUMING 32 bits
	gint rgb_stride[line_size];
	gint single_stride[line_size];
	guint32 pos = 0, gpos, gpos2, x, y, color[line_size];
	guint8 R[line_size], G[line_size], B[line_size];
	gfloat H[line_size], S[line_size], V[line_size];
	gfloat R_factor;
	gfloat G_factor;
	gfloat B_factor;
	guint8 *src_data = NULL, *GRAY_data = NULL;
	gint src_stride = 0, GRAY_stride = 0, GRAY_comp = 0;
	guint32 RedMask=skindetector->RedMask, GreenMask=skindetector->GreenMask, BlueMask=skindetector->BlueMask;
	guint8 RedShift=skindetector->RedShift, GreenShift=skindetector->GreenShift, BlueShift=skindetector->BlueShift;
	gboolean res[line_size];




	if(skindetector->adaptive){
		R_factor = ((skindetector->R_max-skindetector->R_min)>0 ? 255.0/(gfloat)(skindetector->R_max-skindetector->R_min) : 0);
		G_factor = ((skindetector->G_max-skindetector->G_min)>0 ? 255.0/(gfloat)(skindetector->G_max-skindetector->G_min) : 0);
		B_factor = ((skindetector->B_max-skindetector->B_min)>0 ? 255.0/(gfloat)(skindetector->B_max-skindetector->B_min) : 0);
		skindetector->tot_count=0;
		memset(skindetector->H_histo, 0, sizeof(gfloat)*256);
		memset(skindetector->S_histo, 0, sizeof(gfloat)*256);
		memset(skindetector->V_histo, 0, sizeof(gfloat)*256);
		memset(skindetector->R_histo, 0, sizeof(gfloat)*256);
		memset(skindetector->G_histo, 0, sizeof(gfloat)*256);
		memset(skindetector->B_histo, 0, sizeof(gfloat)*256);
	}

	/* Beginning of the pointer, no matter endianess */
	src_data = GST_VIDEO_FRAME_PLANE_DATA(src_frame, 0);
	src_stride = GST_VIDEO_FRAME_COMP_STRIDE(src_frame, 0);

	/* Beginning of the pointer, no worry about endianess */
	GRAY_data = GST_VIDEO_FRAME_PLANE_DATA(out_frame, GRAY_comp);
	GRAY_stride = GST_VIDEO_FRAME_COMP_STRIDE(out_frame, GRAY_comp);

	/************ processing 16 pixels at a time ************/
	gint idx=0;
	while (pos < ((height*src_stride)-(pix_stride*line_size))){

		/*
		 *  According to LE or BE given above, and channels distribution given by user
		 *
		 * Idea:
		 * 		R = ( RGB_ptr & RedMask ) >> RedShift
		 * 		G = ( RGB_ptr & GreenMask ) >> GreenShift
		 * 		B = ( RGB_ptr & BlueMask ) >> BlueShift
		 * */
		memcpy(color, (guint32*)(src_data+pos), sizeof(guint32)*line_size);//reading RGBA at the same time


		idx=0;
		{ //processing all the pixels unrolling this part
			/*pixel 1*/
			(*(R+idx)) = ((*(color+idx))&RedMask)>>RedShift;
			(*(G+idx)) = ((*(color+idx))&GreenMask)>>GreenShift;
			(*(B+idx)) = ((*(color+idx))&BlueMask)>>BlueShift;
			idx++;

			/*pixel 2*/
			(*(R+idx)) = ((*(color+idx))&RedMask)>>RedShift;
			(*(G+idx)) = ((*(color+idx))&GreenMask)>>GreenShift;
			(*(B+idx)) = ((*(color+idx))&BlueMask)>>BlueShift;
			idx++;

			/*pixel 3*/
			(*(R+idx)) = ((*(color+idx))&RedMask)>>RedShift;
			(*(G+idx)) = ((*(color+idx))&GreenMask)>>GreenShift;
			(*(B+idx)) = ((*(color+idx))&BlueMask)>>BlueShift;
			idx++;

			/*pixel 4*/
			(*(R+idx)) = ((*(color+idx))&RedMask)>>RedShift;
			(*(G+idx)) = ((*(color+idx))&GreenMask)>>GreenShift;
			(*(B+idx)) = ((*(color+idx))&BlueMask)>>BlueShift;
		}//unrolling this part


		idx=0;
		/* histogram accumulation */
		if(skindetector->adaptive){
			/*pixel 1*/
			*(skindetector->R_histo+(*(R+idx))) += 1;
			*(skindetector->G_histo+(*(G+idx))) += 1;
			*(skindetector->B_histo+(*(B+idx))) += 1;
			if(R_factor>0 && G_factor>0 && B_factor>0){
				/*saturated*/
				if((*(R+idx))<skindetector->R_min){
					(*(R+idx)) = skindetector->R_min;
				}
				if((*(R+idx))>skindetector->R_max){
					(*(R+idx)) = skindetector->R_max;
				}

				if((*(G+idx))<skindetector->G_min){
					(*(G+idx)) = skindetector->G_min;
				}
				if((*(G+idx))>skindetector->G_max){
					(*(G+idx)) = skindetector->G_max;
				}

				if((*(B+idx))<skindetector->B_min){
					(*(B+idx)) = skindetector->B_min;
				}
				if((*(B+idx))>skindetector->B_max){
					(*(B+idx)) = skindetector->B_max;
				}

				(*(R+idx)) = (guint8)((gfloat)((*(R+idx))-skindetector->R_min)*R_factor);
				(*(G+idx)) = (guint8)((gfloat)((*(G+idx))-skindetector->G_min)*G_factor);
				(*(B+idx)) = (guint8)((gfloat)((*(B+idx))-skindetector->B_min)*B_factor);
			}
			idx++;

			/*pixel 2*/
			*(skindetector->R_histo+(*(R+idx))) += 1;
			*(skindetector->G_histo+(*(G+idx))) += 1;
			*(skindetector->B_histo+(*(B+idx))) += 1;
			if(R_factor>0 && G_factor>0 && B_factor>0){
				/*saturated*/
				if((*(R+idx))<skindetector->R_min){
					(*(R+idx)) = skindetector->R_min;
				}
				if((*(R+idx))>skindetector->R_max){
					(*(R+idx)) = skindetector->R_max;
				}

				if((*(G+idx))<skindetector->G_min){
					(*(G+idx)) = skindetector->G_min;
				}
				if((*(G+idx))>skindetector->G_max){
					(*(G+idx)) = skindetector->G_max;
				}

				if((*(B+idx))<skindetector->B_min){
					(*(B+idx)) = skindetector->B_min;
				}
				if((*(B+idx))>skindetector->B_max){
					(*(B+idx)) = skindetector->B_max;
				}

				(*(R+idx)) = (guint8)((gfloat)((*(R+idx))-skindetector->R_min)*R_factor);
				(*(G+idx)) = (guint8)((gfloat)((*(G+idx))-skindetector->G_min)*G_factor);
				(*(B+idx)) = (guint8)((gfloat)((*(B+idx))-skindetector->B_min)*B_factor);
			}
			idx++;

			/*pixel 3*/
			*(skindetector->R_histo+(*(R+idx))) += 1;
			*(skindetector->G_histo+(*(G+idx))) += 1;
			*(skindetector->B_histo+(*(B+idx))) += 1;
			if(R_factor>0 && G_factor>0 && B_factor>0){
				/*saturated*/
				if((*(R+idx))<skindetector->R_min){
					(*(R+idx)) = skindetector->R_min;
				}
				if((*(R+idx))>skindetector->R_max){
					(*(R+idx)) = skindetector->R_max;
				}

				if((*(G+idx))<skindetector->G_min){
					(*(G+idx)) = skindetector->G_min;
				}
				if((*(G+idx))>skindetector->G_max){
					(*(G+idx)) = skindetector->G_max;
				}

				if((*(B+idx))<skindetector->B_min){
					(*(B+idx)) = skindetector->B_min;
				}
				if((*(B+idx))>skindetector->B_max){
					(*(B+idx)) = skindetector->B_max;
				}

				(*(R+idx)) = (guint8)((gfloat)((*(R+idx))-skindetector->R_min)*R_factor);
				(*(G+idx)) = (guint8)((gfloat)((*(G+idx))-skindetector->G_min)*G_factor);
				(*(B+idx)) = (guint8)((gfloat)((*(B+idx))-skindetector->B_min)*B_factor);
			}
			idx++;

			/*pixel 4*/
			*(skindetector->R_histo+(*(R+idx))) += 1;
			*(skindetector->G_histo+(*(G+idx))) += 1;
			*(skindetector->B_histo+(*(B+idx))) += 1;
			if(R_factor>0 && G_factor>0 && B_factor>0){
				/*saturated*/
				if((*(R+idx))<skindetector->R_min){
					(*(R+idx)) = skindetector->R_min;
				}
				if((*(R+idx))>skindetector->R_max){
					(*(R+idx)) = skindetector->R_max;
				}

				if((*(G+idx))<skindetector->G_min){
					(*(G+idx)) = skindetector->G_min;
				}
				if((*(G+idx))>skindetector->G_max){
					(*(G+idx)) = skindetector->G_max;
				}

				if((*(B+idx))<skindetector->B_min){
					(*(B+idx)) = skindetector->B_min;
				}
				if((*(B+idx))>skindetector->B_max){
					(*(B+idx)) = skindetector->B_max;
				}

				(*(R+idx)) = (guint8)((gfloat)((*(R+idx))-skindetector->R_min)*R_factor);
				(*(G+idx)) = (guint8)((gfloat)((*(G+idx))-skindetector->G_min)*G_factor);
				(*(B+idx)) = (guint8)((gfloat)((*(B+idx))-skindetector->B_min)*B_factor);
			}
		}





		/* tranforming to hsv space */
		RGBtoHSV(R, G, B, H, S, V, line_size);


		idx=0;
		/*
		 * Adaptation of the cake used to determine the thresholds for the HSV space
		 * using histogram and AWB normalization
		 */
		{
			/*pixel 1*/
			(*(res+idx)) = ( ((*(H+idx)) > skindetector->Hmin) && ((*(H+idx)) < skindetector->Hmax) && ((*(S+idx)) > skindetector->Smin) &&
					((*(V+idx)) > skindetector->Vmin));//
			idx++;

			/*pixel 2*/
			(*(res+idx)) = ( ((*(H+idx)) > skindetector->Hmin) && ((*(H+idx)) < skindetector->Hmax) && ((*(S+idx)) > skindetector->Smin) &&
					((*(V+idx)) > skindetector->Vmin));//
			idx++;

			/*pixel 3*/
			(*(res+idx)) = ( ((*(H+idx)) > skindetector->Hmin) && ((*(H+idx)) < skindetector->Hmax) && ((*(S+idx)) > skindetector->Smin) &&
					((*(V+idx)) > skindetector->Vmin));//
			idx++;

			/*pixel 4*/
			(*(res+idx)) = ( ((*(H+idx)) > skindetector->Hmin) && ((*(H+idx)) < skindetector->Hmax) && ((*(S+idx)) > skindetector->Smin) &&
					((*(V+idx)) > skindetector->Vmin));//
		}


		idx=0;
		/* Performing skin detection on the Gray8 Frame */
		if(skindetector->normalize_frame){//**************************NORMALIZATION
			/* done only once */
			gpos = ((pos+idx*pix_stride)%src_stride)/4;//x
			gpos2 = (pos+idx*pix_stride)/src_stride;//y
			gpos = gpos2*GRAY_stride + gpos;

			/*pixel 1*/
			if((*(res+idx))){
				*(GRAY_data+gpos+idx)  |= ((1<<8) -1);
				if(skindetector->adaptive){
					*(skindetector->H_histo+(gint)(*(H+idx))) += 1;
					*(skindetector->S_histo+(gint)(*(S+idx))) += 1;
					*(skindetector->V_histo+(gint)(*(V+idx))) += 1;
					skindetector->tot_count++;
				}
			}else{
				*(GRAY_data+gpos+idx)  |= 0;
			}
			idx++;

			/*pixel 2*/
			if((*(res+idx))){
				*(GRAY_data+gpos+idx)  |= ((1<<8) -1);
				if(skindetector->adaptive){
					*(skindetector->H_histo+(gint)(*(H+idx))) += 1;
					*(skindetector->S_histo+(gint)(*(S+idx))) += 1;
					*(skindetector->V_histo+(gint)(*(V+idx))) += 1;
					skindetector->tot_count++;
				}
			}else{
				*(GRAY_data+gpos+idx)  |= 0;
			}
			idx++;

			/*pixel 3*/
			if((*(res+idx))){
				*(GRAY_data+gpos+idx)  |= ((1<<8) -1);
				if(skindetector->adaptive){
					*(skindetector->H_histo+(gint)(*(H+idx))) += 1;
					*(skindetector->S_histo+(gint)(*(S+idx))) += 1;
					*(skindetector->V_histo+(gint)(*(V+idx))) += 1;
					skindetector->tot_count++;
				}
			}else{
				*(GRAY_data+gpos+idx)  |= 0;
			}
			idx++;

			/*pixel 4*/
			if((*(res+idx))){
				*(GRAY_data+gpos+idx)  |= ((1<<8) -1);
				if(skindetector->adaptive){
					*(skindetector->H_histo+(gint)(*(H+idx))) += 1;
					*(skindetector->S_histo+(gint)(*(S+idx))) += 1;
					*(skindetector->V_histo+(gint)(*(V+idx))) += 1;
					skindetector->tot_count++;
				}
			}else{
				*(GRAY_data+gpos+idx)  |= 0;
			}
		}else{//****************************************************************NO NORMALIZATION
			/* done only once */
			gpos = ((pos+idx*pix_stride)%src_stride)/4;//x
			gpos2 = (pos+idx*pix_stride)/src_stride;//y
			gpos = gpos2*GRAY_stride + gpos;

			/*pixel 1*/
			if((*(res+idx))){
				*(GRAY_data+gpos)  |= 1;
				if(skindetector->adaptive){
					*(skindetector->H_histo+(gint)(*(H+idx))) += 1;
					*(skindetector->S_histo+(gint)(*(S+idx))) += 1;
					*(skindetector->V_histo+(gint)(*(V+idx))) += 1;
					skindetector->tot_count++;
				}
			}else{
				*(GRAY_data+gpos)  |= 0;
			}
			idx++;

			/*pixel 2*/
			if((*(res+idx))){
				*(GRAY_data+gpos+idx)  |= 1;
				if(skindetector->adaptive){
					*(skindetector->H_histo+(gint)(*(H+idx))) += 1;
					*(skindetector->S_histo+(gint)(*(S+idx))) += 1;
					*(skindetector->V_histo+(gint)(*(V+idx))) += 1;
					skindetector->tot_count++;
				}
			}else{
				*(GRAY_data+gpos+idx)  |= 0;
			}
			idx++;

			/*pixel 3*/
			if((*(res+idx))){
				*(GRAY_data+gpos+idx)  |= 1;
				if(skindetector->adaptive){
					*(skindetector->H_histo+(gint)(*(H+idx))) += 1;
					*(skindetector->S_histo+(gint)(*(S+idx))) += 1;
					*(skindetector->V_histo+(gint)(*(V+idx))) += 1;
					skindetector->tot_count++;
				}
			}else{
				*(GRAY_data+gpos+idx)  |= 0;
			}
			idx++;

			/*pixel 4*/
			if((*(res+idx))){
				*(GRAY_data+gpos+idx)  |= 1;
				if(skindetector->adaptive){
					*(skindetector->H_histo+(gint)(*(H+idx))) += 1;
					*(skindetector->S_histo+(gint)(*(S+idx))) += 1;
					*(skindetector->V_histo+(gint)(*(V+idx))) += 1;
					skindetector->tot_count++;
				}
			}else{
				*(GRAY_data+gpos+idx)  |= 0;
			}
		}


		/*increasing by the pixel stride and the number of pixels read*/
		pos+=pix_stride*line_size;
		idx=0;
	}

	/*********************** UPDATING THRESHOLDS *********************/
	if(skindetector->adaptive){
		update_thresholds(skindetector);
	}


	return skin_success;
}

#undef max
#undef min




static void update_thresholds(GstSkinDetector *skindetector){
	guint i;
	gfloat Old_weight = skindetector->merging_factor, New_weight = 1-Old_weight, norm;
	gfloat mean_H=0, mean_S=0, mean_V=0;//where the peak is
	gfloat max_H_val=0, max_S_val=0, max_V_val=0;
	gfloat percentage = 0.01;
	gfloat total_skin_pix=0;
	gint fsize = GST_VIDEO_FRAME_WIDTH(skindetector->inframe)*GST_VIDEO_FRAME_HEIGHT(skindetector->inframe);

	/*
	 * template histograms fetching the whole chunk of
	 * memory
	 * */
	gfloat template_H_histo[256], template_S_histo[256], template_V_histo[256];
	memcpy(template_H_histo, skindetector->prior_H_histo, sizeof(gfloat)*256);
	memcpy(template_S_histo, skindetector->prior_S_histo, sizeof(gfloat)*256);
	memcpy(template_V_histo, skindetector->prior_V_histo, sizeof(gfloat)*256);


	/*Chi-Squared*/
	for(i=0;i<256;i++){
		(*(skindetector->H_histo+i)) /= skindetector->tot_count;
		(*(skindetector->S_histo+i)) /= skindetector->tot_count;
		(*(skindetector->V_histo+i)) /= skindetector->tot_count;

		//H
		if(((*(skindetector->H_histo+i)) + (*(template_H_histo+i))) > 0){
			(*(skindetector->H_histo+i)) = 0.5*( ( ((*(skindetector->H_histo+i)) - (*(template_H_histo+i))) *
					((*(skindetector->H_histo+i)) - (*(template_H_histo+i))) ) /
					((*(skindetector->H_histo+i)) + (*(template_H_histo+i))) );
		}else{
			(*(skindetector->H_histo+i)) = 0;
		}

		//S
		if(((*(skindetector->S_histo+i)) + (*(template_S_histo+i))) > 0){
			(*(skindetector->S_histo+i)) = 0.5*( ( ((*(skindetector->S_histo+i)) - (*(template_S_histo+i))) *
					((*(skindetector->S_histo+i)) - (*(template_S_histo+i))) )/
							((*(skindetector->S_histo+i)) + (*(template_S_histo+i))) );
		}else{
			(*(skindetector->S_histo+i)) = 0;
		}

		//V
		if(((*(skindetector->V_histo+i)) + (*(template_V_histo+i))) > 0){
			(*(skindetector->V_histo+i)) = 0.5*( ( ((*(skindetector->V_histo+i)) - (*(template_V_histo+i))) *
					((*(skindetector->V_histo+i)) - (*(template_V_histo+i))) )/
							((*(skindetector->V_histo+i)) + (*(template_V_histo+i))) );
		}else{
			(*(skindetector->V_histo+i)) = 0;
		}
	}




	/*
	 * merging histograms using the factor given in the initialization part
	 * and normalizing at the same time using the maxes and mins as AWB
	 * */
	for(i=0;i<256;i++){
		/* H channel */
		if(i>0){
			(*(skindetector->R_histo+i)) += (*(skindetector->R_histo+i-1));
		}

		mean_H += (*(skindetector->H_histo+i));//estimating mean

		if(max_H_val<(*(skindetector->H_histo+i))){
			max_H_val = (*(skindetector->H_histo+i));
		}


		/* S channel */
		if(i>0){
			(*(skindetector->G_histo+i)) += (*(skindetector->G_histo+i-1));
		}

		mean_S += (*(skindetector->S_histo+i));//estimating mean

		if(max_S_val<(*(skindetector->S_histo+i))){
			max_S_val = (*(skindetector->S_histo+i));
		}

		/* V channel */
		if(i>0){
			(*(skindetector->B_histo+i)) += (*(skindetector->B_histo+i-1));
		}

		mean_V += (*(skindetector->V_histo+i));//estimating mean

		if(max_V_val<(*(skindetector->V_histo+i))){
			max_V_val = (*(skindetector->V_histo+i));
		}
	}

	mean_H /= 256; mean_S /= 256; mean_V /= 256;

	if(mean_H>0 && mean_S>0 && mean_V>0){
		/* Covering 90% of the histograms area with new thresholds (sort of confidence interval idea)*/
		gfloat H_N = mean_H*max_H_val*256;
		gfloat S_N = mean_S*max_S_val*256;
		gfloat V_N = mean_V*max_V_val*256;


		gfloat H_maxth = (1-percentage)*H_N;

		gfloat H_minth = percentage*H_N;
		gfloat S_minth = percentage*S_N;
		gfloat V_minth = percentage*V_N;

		GST_DEBUG_OBJECT(skindetector, "UPDATING: means: (%f, %f, %f); mins: (%f, %f, %f); maxes: (%f)", mean_H, mean_S, mean_V,
				H_minth, S_minth, V_minth, H_maxth);


		gint H_mini=0, H_maxi=0;
		gint S_mini=0;
		gint V_mini=0;

		gfloat H_cmini=0, H_cmaxi=H_N;
		gfloat S_cmini=0;
		gfloat V_cmini=0;


		gboolean H_lower=TRUE, H_higher=TRUE;
		gboolean S_lower=TRUE;
		gboolean V_lower=TRUE;


		/*moving boundaries accordingly*/
		for (i = 0; i < 256; i++){
			if((*(skindetector->R_histo+i)) <= ((fsize)*0.015)){
				skindetector->R_min = i;
			}
			if((*(skindetector->G_histo+i)) <= ((fsize)*0.015)){
				skindetector->G_min = i;
			}
			if((*(skindetector->B_histo+i)) <= ((fsize)*0.015)){
				skindetector->B_min = i;
			}


			if((*(skindetector->R_histo+(255-i))) > ((fsize*0.985)-1)){
				skindetector->R_max = (255-i);
			}
			if((*(skindetector->G_histo+(255-i))) > ((fsize*0.985)-1)){
				skindetector->G_max = (255-i);
			}
			if((*(skindetector->B_histo+(255-i))) > ((fsize*0.985)-1)){
				skindetector->B_max = (255-i);
			}


			/* Moving lower bound for the thresholds*/
			H_cmini += (*(skindetector->H_histo+i));
			S_cmini += (*(skindetector->S_histo+i));
			V_cmini += (*(skindetector->V_histo+i));

			//H
			if(H_cmini >= H_minth && H_lower==TRUE){
				H_mini = i;
				H_lower=FALSE;
			}

			//S
			if(S_cmini >= S_minth && S_lower==TRUE){
				S_mini = i;
				S_lower=FALSE;
			}

			//V
			if(V_cmini >= V_minth && V_lower==TRUE){
				V_mini = i;
				V_lower=FALSE;
			}


			/* Moving upper bound for the thresholds*/
			H_cmaxi -= (*(skindetector->H_histo+(256-i)));

			//H
			if(H_cmaxi <= H_maxth && H_higher==TRUE){
				H_maxi = 255-i;
				H_higher=FALSE;
			}

			if(H_lower==FALSE && H_higher ==FALSE && S_lower==FALSE && V_lower==FALSE){
				/* just to make sure the arc is not too tiny at least 20 degrees*/
				if((H_maxi-H_mini)<20){
					H_maxi = H_mini+20;
				}
				break;
			}
		}


		if(skindetector->R_max < 255 - 1){
			skindetector->R_max += 1;
		}
		if(skindetector->G_max < 255 - 1){
			skindetector->G_max += 1;
		}
		if(skindetector->B_max < 255 - 1){
			skindetector->B_max += 1;
		}


		GST_LOG_OBJECT(skindetector, "UPDATING: New thresholds will be: < H: [%d, %d]; S: [%d]; V: [%d]>",
				H_mini, H_maxi, S_mini, V_mini);
		GST_LOG_OBJECT(skindetector, "UPDATING: space has been shifted to: < R: [%d, %d]; G: [%d, %d]; B: [%d, %d]>",
				skindetector->R_min, skindetector->R_max, skindetector->G_min, skindetector->G_max, skindetector->B_min, skindetector->B_max);


		total_skin_pix = (gfloat)skindetector->tot_count/
							(gfloat)((GST_VIDEO_FRAME_WIDTH(skindetector->inframe)*GST_VIDEO_FRAME_HEIGHT(skindetector->inframe)));

		if(total_skin_pix<0.01){
			/*
			 * The thresholds are not good enough probably we adapted too much
			 * sort of recovery in case sudden change, coming back to reference
			 * values.
			 *
			 * Another idea:
			 * Another mechanism can be implemented here, like face detection
			 * then cropping for HSV histogram extraction
			 * */
			GST_WARNING_OBJECT(skindetector, "Resetting the threshold since no significant number of pixels are found to be skin (%f) ...!!", total_skin_pix);

			/*
			 * coming back to initial guesses given by the template made by the user
			 * */
			skindetector->Hmin = skindetector->oHmin;
			skindetector->Hmax = skindetector->oHmax;
			skindetector->Smin = skindetector->oSmin;
			skindetector->Smax = skindetector->oSmax;
			skindetector->Vmin = skindetector->oVmin;
			skindetector->Vmax = skindetector->oVmax;
//			skindetector->empty_frame=TRUE;
		}else{
			/*final updating*/
			skindetector->Hmin = H_mini;
			skindetector->Hmax = H_maxi;
			skindetector->Smin = S_mini;
			skindetector->Vmin = V_mini;
//			skindetector->empty_frame=FALSE;
		}
	}else{
		/*
		 * This means that the frame changes so much that we are able to see anything now
		 * */
//		skindetector->empty_frame=TRUE;
		GST_LOG_OBJECT(skindetector, "NOTHING IN THIS FRAME..!!");
	}
}


#define hscale ( 180.f/360.f );
static void RGBtoHSV(guint8 *r, guint8 *g, guint8 *b, gfloat *Hnew, gfloat *Snew, gfloat *Vnew,const gint n_pixels){

	//HSV
	gfloat h[n_pixels], s[n_pixels], v[n_pixels];
	gfloat vmin[n_pixels], diff[n_pixels];
	gint idx=0;

	{//checking v values
		//pixel 1
		v[idx+1] = v[idx] = vmin[idx+1] = vmin[idx] = r[idx];
		idx++;

		//pixel 2
//		v[idx] = vmin[idx] = r[idx];
		idx++;

		//pixel 3
		v[idx+1] = v[idx] = vmin[idx+1] = vmin[idx] = r[idx];
//		idx++;

		//pixel 4
//		v[idx] = vmin[idx] = r[idx];
	}



	idx=0;
	{//maximum of the v values
		//pixel 1
		v[idx+1] = v[idx] = ((v[idx]<g[idx])?((g[idx]<b[idx])?b[idx]:g[idx]):(v[idx]<b[idx])?b[idx]:v[idx]);
		idx++;

		//pixel 2
//		v[idx] = ((v[idx]<g[idx])?((g[idx]<b[idx])?b[idx]:g[idx]):(v[idx]<b[idx])?b[idx]:v[idx]);
		idx++;

		//pixel 3
		v[idx+1] = v[idx] = ((v[idx]<g[idx])?((g[idx]<b[idx])?b[idx]:g[idx]):(v[idx]<b[idx])?b[idx]:v[idx]);
//			idx++;

		//pixel 4
//		v[idx] = ((v[idx]<g[idx])?((g[idx]<b[idx])?b[idx]:g[idx]):(v[idx]<b[idx])?b[idx]:v[idx]);
	}

	idx=0;
	{//minimum of the v values
		//pixel 1
		vmin[idx+1] = vmin[idx] = ((vmin[idx]>g[idx])?((g[idx]>b[idx])?b[idx]:g[idx]):(vmin[idx]>b[idx])?b[idx]:vmin[idx]);
		idx++;

		//pixel 2
//		vmin[idx] = ((vmin[idx]>g[idx])?((g[idx]>b[idx])?b[idx]:g[idx]):(vmin[idx]>b[idx])?b[idx]:vmin[idx]);
		idx++;

		//pixel 3
		vmin[idx+1] = vmin[idx] = ((vmin[idx]>g[idx])?((g[idx]>b[idx])?b[idx]:g[idx]):(vmin[idx]>b[idx])?b[idx]:vmin[idx]);
//		idx++;

		//pixel 4
//		vmin[idx] = ((vmin[idx]>g[idx])?((g[idx]>b[idx])?b[idx]:g[idx]):(vmin[idx]>b[idx])?b[idx]:vmin[idx]);
	}


	idx=0;
	{//estimating differences and s values
		//pixel 1
		diff[idx+1] = diff[idx] = v[idx] - vmin[idx];
		s[idx+1] = s[idx] = diff[idx]/(v[idx] + FLT_EPSILON);
		diff[idx+1] = diff[idx] = (60.f/(diff[idx] + FLT_EPSILON));
		idx++;

		//pixel 2
//		diff[idx] = v[idx] - vmin[idx];
//		s[idx] = diff[idx]/(v[idx] + FLT_EPSILON);
//		diff[idx] = (60.f/(diff[idx] + FLT_EPSILON));
		idx++;

		//pixel 3
		diff[idx+1] = diff[idx] = v[idx] - vmin[idx];
		s[idx+1] = s[idx] = diff[idx]/(v[idx] + FLT_EPSILON);
		diff[idx+1] = diff[idx] = (60.f/(diff[idx] + FLT_EPSILON));
//		idx++;

		//pixel 4
//		diff[idx] = v[idx] - vmin[idx];
//		s[idx] = diff[idx]/(v[idx] + FLT_EPSILON);
//		diff[idx] = (60.f/(diff[idx] + FLT_EPSILON));
	}


	idx=0;
	{//estimating h values
		//pixel 1
		h[idx] = (v[idx]==r[idx]?((gfloat)(g[idx] - b[idx])*diff[idx]):(v[idx]==g[idx]?((gfloat)(b[idx] -
				r[idx])*diff[idx] + 120.f):((gfloat)(r[idx] - g[idx])*diff[idx] + 240.f)));
		h[idx] += (h[idx]<0?360.f:0.f);
		Hnew[idx] = h[idx]*hscale;
		Snew[idx] = s[idx]*255.f;
		Vnew[idx] = v[idx];

		Hnew[idx+1] = Hnew[idx];
		Snew[idx+1] = Snew[idx];
		Vnew[idx+1] = Vnew[idx];


		idx++;
//
//		//pixel 2
//		h[idx] = (v[idx]==r[idx]?((gfloat)(g[idx] - b[idx])*diff[idx]):(v[idx]==g[idx]?((gfloat)(b[idx] -
//				r[idx])*diff[idx] + 120.f):((gfloat)(r[idx] - g[idx])*diff[idx] + 240.f)));
//		h[idx] += (h[idx]<0?360.f:0.f);
//		Hnew[idx] = h[idx]*hscale;
//		Snew[idx] = s[idx]*255.f;
//		Vnew[idx] = v[idx];
		idx++;
//
		//pixel 3
		h[idx] = (v[idx]==r[idx]?((gfloat)(g[idx] - b[idx])*diff[idx]):(v[idx]==g[idx]?((gfloat)(b[idx] -
				r[idx])*diff[idx] + 120.f):((gfloat)(r[idx] - g[idx])*diff[idx] + 240.f)));
		h[idx] += (h[idx]<0?360.f:0.f);
		Hnew[idx] = h[idx]*hscale;
		Snew[idx] = s[idx]*255.f;
		Vnew[idx] = v[idx];
//		idx++;

		Hnew[idx+1] = Hnew[idx];
		Snew[idx+1] = Snew[idx];
		Vnew[idx+1] = Vnew[idx];
//
//		//pixel 4
//		h[idx] = (v[idx]==r[idx]?((gfloat)(g[idx] - b[idx])*diff[idx]):(v[idx]==g[idx]?((gfloat)(b[idx] -
//				r[idx])*diff[idx] + 120.f):((gfloat)(r[idx] - g[idx])*diff[idx] + 240.f)));
//		h[idx] += (h[idx]<0?360.f:0.f);
//		Hnew[idx] = h[idx]*hscale;
//		Snew[idx] = s[idx]*255.f;
//		Vnew[idx] = v[idx];
	}
}


/*
 * Mask build for the texture extraction rutine
 * */
static void mask_maker(GstSkinDetector *skindetector, guint32 height, guint32 width){
	guint i;
	for(i=0;i<NMASKS;i++){
		skindetector->masks[i].Mask_Hsize = height >> (i+1);
		skindetector->masks[i].Mask_Wsize = width >> (i+1);
		skindetector->masks[i].Cx = skindetector->masks[i].Mask_Wsize >> 1;
		skindetector->masks[i].Cy = skindetector->masks[i].Mask_Hsize >> 1;
		skindetector->masks[i].mask = 1<<(i+1);

		GST_INFO_OBJECT(skindetector, "Mask size: <W: %d, H: %d>\tCenter: <X: %d, Y: %d>\tMask: %d\n\n",
				skindetector->masks[i].Mask_Wsize, skindetector->masks[i].Mask_Hsize,
				skindetector->masks[i].Cx, skindetector->masks[i].Cy, skindetector->masks[i].mask);
	}
}



















//888888888888888888888 enable for now only delete when compiling
//	#define HAVE_NEON
//888888888888888888888 enable for now only delete when compiling

#ifdef HAVE_NEON
#include <NE10.h>

	static gint texture_extraction(GstVideoFrame *inframe, GstVideoFrame *outframe, guint32 height, GstSkinDetector *skindetector){
		guint32 mask_id, color, color_patch[8];
		guint8 *src_data = NULL, *GRAY_data = NULL, Pel_R, Pel_G, Pel_B;
		guint32 meanR, meanG, meanB;
		gint denominator, checker;
		guint32 diff[2*3]={0}, Pel_diff;//current and previous pixel (RGB)
		guint8 RGB_pix_stride=4;//packed pixels
		guint32 Img_pos, Patch_pos, x, y, width, x_check, y_check;
		gint px, py;
		gint src_stride = 0, GRAY_stride = 0, GRAY_comp = 0;
		guint32 patch_counter;
		gfloat mask_size, patch_factor=1.f;
		guint32 RedMask=skindetector->RedMask, GreenMask=skindetector->GreenMask, BlueMask=skindetector->BlueMask;
		guint8 RedShift=skindetector->RedShift, GreenShift=skindetector->GreenShift, BlueShift=skindetector->BlueShift;
		gboolean entered=FALSE;
		guint8 chunk_stride;
		guint MASK_W;


		//==== NE10 types
		ne10_float32_t temp_x[1], temp_y[1], temp_maskx, temp_masky;

		GST_INFO_OBJECT(skindetector, "HAS NEON SUPPORT ...!!!");

		/* Beginning of the pointer, no matter endianess */
		src_data = GST_VIDEO_FRAME_PLANE_DATA(inframe, 0);
		src_stride = GST_VIDEO_FRAME_COMP_STRIDE(inframe, 0);
		width = GST_VIDEO_FRAME_WIDTH(inframe);

		/* Beginning of the pointer, no worry about endianess */
		GRAY_data = GST_VIDEO_FRAME_PLANE_DATA(outframe, GRAY_comp);
		GRAY_stride = GST_VIDEO_FRAME_COMP_STRIDE(outframe, GRAY_comp);

		//for all the masks, NON-OVERLAPPING PATCHES
		for(mask_id=0; mask_id<NMASKS;mask_id++){
			Img_pos=x=y=0;//position of the mask center basically
			mask_size = skindetector->masks[mask_id].Mask_Hsize*skindetector->masks[mask_id].Mask_Wsize;
			mask_size += (mask_size + mask_size);
			patch_factor = 1.f/mask_size;

			const gint MASK_CX = skindetector->masks[mask_id].Cx;
			const gint MASK_CY = skindetector->masks[mask_id].Cy;
			MASK_W = skindetector->masks[mask_id].Mask_Wsize;
			const guint8 MASK_MASK = skindetector->masks[mask_id].mask;



			/*
			 * To unroll the loop below since it is too expensive
			 * and it is the same value for the whole patch, so up to
			 * 8 pixels at a time
			 * */
			if(MASK_W>8){
				chunk_stride = 8;
			}else if(MASK_W>4){
				chunk_stride = 4;
			}else if(MASK_W>2){
				chunk_stride = 2;
			}else{
				chunk_stride = 1;
			}

			//for checking if we still inside the patch
			checker = src_stride-chunk_stride*RGB_pix_stride;


			//==================NE10 types
			ne10_float32_t twotimesy = MASK_CY+MASK_CY, twotimesx = MASK_CX+MASK_CX;


			while(TRUE){//image loop
				/*positioning mask*/
				if(y==0 && x==0){
					temp_x[0] = x; temp_y[0] = y; temp_maskx = MASK_CX; temp_masky = MASK_CY;
					ne10_addc_float_neon(temp_x, temp_x, temp_maskx, 1);
					ne10_addc_float_neon(temp_y, temp_y, temp_masky, 1);
					x = temp_x[0]; y = temp_y[0];
				}else{
					temp_x[0] = x;

					ne10_addc_float_neon(temp_x, temp_x, twotimesx, 1);
					x = temp_x[0];
				}

				if(x>=width){
					x = MASK_CX;
					temp_y[0] = y;

					ne10_addc_float_neon(temp_y, temp_y, twotimesy, 1);
					y = temp_y[0];
					if(y>=height){
						//we finished the frame
						break;
					}
				}

				Img_pos = y*src_stride + x*RGB_pix_stride;//location in the source && destination images
				color = (*(guint32*)(src_data+Img_pos));
				meanR = ((color&RedMask)>>RedShift);//pivot pixel R
				meanG = ((color&GreenMask)>>GreenShift);//pivot pixel G
				meanB = ((color&BlueMask)>>BlueShift);//pivot pixel B

				patch_counter=0;
				px = -MASK_CX;//pixel position in the patch, and counter to determine pattern significance
				py = -MASK_CY;//pixel position in the patch, and counter to determine pattern significance

				/*shifting the locations with respect to center*/
				denominator=1;
				Pel_diff = 0;

				while(TRUE){
					/*working with the relative position in the patch*/
					Patch_pos = ((y+py)*src_stride + (x+px)*RGB_pix_stride);

					/*safety*/
					x_check = Patch_pos%src_stride;
					y_check = Patch_pos/src_stride;

					if((x_check<0 || x_check>=src_stride) && (y_check<0 || y_check>=height)){
						break;
					}else if((x_check<0 || x_check>=src_stride) && (y_check>0 || y_check<height)){
						temp_x[0]=px; temp_y[0]=py; temp_maskx=MASK_CX;
						ne10_addc_float_neon(temp_y, temp_y, 1, 1);
						py=temp_y[0];
						if(py>=MASK_CY){
							//we finished the patch
							break;
						}
						px=-MASK_CX;
						continue;//we havent finished the height
					}else if((y_check<0 || y_check>=height) && (x_check>0 || x_check<checker)){
						temp_x[0]=px; temp_y[0]=py; temp_maskx=MASK_CX; temp_masky=chunk_stride;
						ne10_addc_float_neon(temp_x, temp_x, temp_masky, 1);
						px=temp_x[0];
						if(px>=MASK_CX){
							ne10_addc_float_neon(temp_y, temp_y, 1, 1);
							py=temp_y[0];
							if(py>=MASK_CY){
								//we finished the patch
								break;
							}
							px=-MASK_CX;
						}
						continue;//we havent finished the width
					}


					/************************* UNROLLING ****************************/
					//accessing the whole chunk at once, for processing
					memcpy(color_patch, (guint32*)(src_data+Patch_pos), sizeof(guint32)*chunk_stride);

					/******************* PIXEL 1 *************************/
					if(chunk_stride>=1){
						Pel_R = (color_patch[0]&RedMask)>>RedShift;//pel pixel R
						Pel_G = (color_patch[0]&GreenMask)>>GreenShift;//pel pixel G
						Pel_B = (color_patch[0]&BlueMask)>>BlueShift;//pel pixel B

						/*checking differences*/
						//R
						if(Pel_R>meanR || (Pel_diff>10)){
							if(Patch_pos==0){
								diff[0] = Pel_R;
							}else if(Patch_pos==1){
								diff[1] = Pel_R;
							}else{//shifting
								diff[0] = diff[1];
								diff[1] = Pel_R;
							}

							entered = TRUE;
							patch_counter++;
						}

						//G
						if(Pel_G>meanG || (Pel_diff>10)){
							if(Patch_pos==0){
								diff[2] = Pel_G;
							}else if(Patch_pos==1){
								diff[3] = Pel_G;
							}else{//shifting
								diff[2] = diff[3];
								diff[3] = Pel_G;
							}

							entered = TRUE;
							patch_counter++;
						}

						//B
						if(Pel_B>meanB || (Pel_diff>10)){
							if(Patch_pos==0){
								diff[4] = Pel_B;
							}else if(Patch_pos==1){
								diff[5] = Pel_B;
							}else{//shifting
								diff[4] = diff[5];
								diff[5] = Pel_B;
							}

							entered = TRUE;
							patch_counter++;
						}


						if(entered){
							/* update the difference */
							Pel_diff *= (denominator+denominator+denominator);
							Pel_diff += (*(diff+1))>(*diff) ? ((*(diff+1))-(*diff)) : ((*diff)-(*(diff+1)));
							Pel_diff += (*(diff+3))>(*(diff+2)) ? ((*(diff+3))-(*(diff+2))) : ((*(diff+2))-(*(diff+3)));
							Pel_diff += (*(diff+5))>(*(diff+4)) ? ((*(diff+5))-(*(diff+4))) : ((*(diff+4))-(*(diff+5)));
							Pel_diff *= 1.f/(denominator+denominator+denominator);

							/*update mean*/
							meanB *= denominator;
							meanB += Pel_B;
							meanB *= 1.f/denominator;

							meanG *= denominator;
							meanG += Pel_G;
							meanG *= 1.f/denominator;

							meanR *= denominator;
							meanR += Pel_R;
							meanR *= 1.f/denominator;

							//update denominator
							denominator++;

							entered = FALSE;
						}

						if(chunk_stride>=2){
							/******************* PIXEL 2 *************************/
							Pel_R = (color_patch[1]&RedMask)>>RedShift;//pel pixel R
							Pel_G = (color_patch[1]&GreenMask)>>GreenShift;//pel pixel G
							Pel_B = (color_patch[1]&BlueMask)>>BlueShift;//pel pixel B

							/*checking differences*/
							//R
							if(Pel_R>meanR || (Pel_diff>10)){
								if(Patch_pos==0){
									diff[0] = Pel_R;
								}else if(Patch_pos==1){
									diff[1] = Pel_R;
								}else{//shifting
									diff[0] = diff[1];
									diff[1] = Pel_R;
								}

								entered = TRUE;
								patch_counter++;
							}

							//G
							if(Pel_G>meanG || (Pel_diff>10)){
								if(Patch_pos==0){
									diff[2] = Pel_G;
								}else if(Patch_pos==1){
									diff[3] = Pel_G;
								}else{//shifting
									diff[2] = diff[3];
									diff[3] = Pel_G;
								}

								entered = TRUE;
								patch_counter++;
							}

							//B
							if(Pel_B>meanB || (Pel_diff>10)){
								if(Patch_pos==0){
									diff[4] = Pel_B;
								}else if(Patch_pos==1){
									diff[5] = Pel_B;
								}else{//shifting
									diff[4] = diff[5];
									diff[5] = Pel_B;
								}

								entered = TRUE;
								patch_counter++;
							}


							if(entered){
								/* update the difference */
								Pel_diff *= (denominator+denominator+denominator);
								Pel_diff += (*(diff+1))>(*diff) ? ((*(diff+1))-(*diff)) : ((*diff)-(*(diff+1)));
								Pel_diff += (*(diff+3))>(*(diff+2)) ? ((*(diff+3))-(*(diff+2))) : ((*(diff+2))-(*(diff+3)));
								Pel_diff += (*(diff+5))>(*(diff+4)) ? ((*(diff+5))-(*(diff+4))) : ((*(diff+4))-(*(diff+5)));
								Pel_diff *= 1.f/(denominator+denominator+denominator);

								/*update mean*/
								meanB *= denominator;
								meanB += Pel_B;
								meanB *= 1.f/denominator;

								meanG *= denominator;
								meanG += Pel_G;
								meanG *= 1.f/denominator;

								meanR *= denominator;
								meanR += Pel_R;
								meanR *= 1.f/denominator;

								//update denominator
								denominator++;

								entered = FALSE;
							}

							if(chunk_stride>=4){
								/******************* PIXEL 3 *************************/
								Pel_R = (color_patch[2]&RedMask)>>RedShift;//pel pixel R
								Pel_G = (color_patch[2]&GreenMask)>>GreenShift;//pel pixel G
								Pel_B = (color_patch[2]&BlueMask)>>BlueShift;//pel pixel B

								/*checking differences*/
								//R
								if(Pel_R>meanR || (Pel_diff>10)){
									if(Patch_pos==0){
										diff[0] = Pel_R;
									}else if(Patch_pos==1){
										diff[1] = Pel_R;
									}else{//shifting
										diff[0] = diff[1];
										diff[1] = Pel_R;
									}

									entered = TRUE;
									patch_counter++;
								}

								//G
								if(Pel_G>meanG || (Pel_diff>10)){
									if(Patch_pos==0){
										diff[2] = Pel_G;
									}else if(Patch_pos==1){
										diff[3] = Pel_G;
									}else{//shifting
										diff[2] = diff[3];
										diff[3] = Pel_G;
									}

									entered = TRUE;
									patch_counter++;
								}

								//B
								if(Pel_B>meanB || (Pel_diff>10)){
									if(Patch_pos==0){
										diff[4] = Pel_B;
									}else if(Patch_pos==1){
										diff[5] = Pel_B;
									}else{//shifting
										diff[4] = diff[5];
										diff[5] = Pel_B;
									}

									entered = TRUE;
									patch_counter++;
								}


								if(entered){
									/* update the difference */
									Pel_diff *= (denominator+denominator+denominator);
									Pel_diff += (*(diff+1))>(*diff) ? ((*(diff+1))-(*diff)) : ((*diff)-(*(diff+1)));
									Pel_diff += (*(diff+3))>(*(diff+2)) ? ((*(diff+3))-(*(diff+2))) : ((*(diff+2))-(*(diff+3)));
									Pel_diff += (*(diff+5))>(*(diff+4)) ? ((*(diff+5))-(*(diff+4))) : ((*(diff+4))-(*(diff+5)));
									Pel_diff *= 1.f/(denominator+denominator+denominator);

									/*update mean*/
									meanB *= denominator;
									meanB += Pel_B;
									meanB *= 1.f/denominator;

									meanG *= denominator;
									meanG += Pel_G;
									meanG *= 1.f/denominator;

									meanR *= denominator;
									meanR += Pel_R;
									meanR *= 1.f/denominator;

									//update denominator
									denominator++;

									entered = FALSE;
								}

								/******************* PIXEL 4 *************************/
								Pel_R = (color_patch[3]&RedMask)>>RedShift;//pel pixel R
								Pel_G = (color_patch[3]&GreenMask)>>GreenShift;//pel pixel G
								Pel_B = (color_patch[3]&BlueMask)>>BlueShift;//pel pixel B

								/*checking differences*/
								//R
								if(Pel_R>meanR || (Pel_diff>10)){
									if(Patch_pos==0){
										diff[0] = Pel_R;
									}else if(Patch_pos==1){
										diff[1] = Pel_R;
									}else{//shifting
										diff[0] = diff[1];
										diff[1] = Pel_R;
									}

									entered = TRUE;
									patch_counter++;
								}

								//G
								if(Pel_G>meanG || (Pel_diff>10)){
									if(Patch_pos==0){
										diff[2] = Pel_G;
									}else if(Patch_pos==1){
										diff[3] = Pel_G;
									}else{//shifting
										diff[2] = diff[3];
										diff[3] = Pel_G;
									}

									entered = TRUE;
									patch_counter++;
								}

								//B
								if(Pel_B>meanB || (Pel_diff>10)){
									if(Patch_pos==0){
										diff[4] = Pel_B;
									}else if(Patch_pos==1){
										diff[5] = Pel_B;
									}else{//shifting
										diff[4] = diff[5];
										diff[5] = Pel_B;
									}

									entered = TRUE;
									patch_counter++;
								}


								if(entered){
									/* update the difference */
									Pel_diff *= (denominator+denominator+denominator);
									Pel_diff += (*(diff+1))>(*diff) ? ((*(diff+1))-(*diff)) : ((*diff)-(*(diff+1)));
									Pel_diff += (*(diff+3))>(*(diff+2)) ? ((*(diff+3))-(*(diff+2))) : ((*(diff+2))-(*(diff+3)));
									Pel_diff += (*(diff+5))>(*(diff+4)) ? ((*(diff+5))-(*(diff+4))) : ((*(diff+4))-(*(diff+5)));
									Pel_diff *= 1.f/(denominator+denominator+denominator);

									/*update mean*/
									meanB *= denominator;
									meanB += Pel_B;
									meanB *= 1.f/denominator;

									meanG *= denominator;
									meanG += Pel_G;
									meanG *= 1.f/denominator;

									meanR *= denominator;
									meanR += Pel_R;
									meanR *= 1.f/denominator;

									//update denominator
									denominator++;

									entered = FALSE;
								}

								if(chunk_stride>=8){
									/******************* PIXEL 5 *************************/
									Pel_R = (color_patch[4]&RedMask)>>RedShift;//pel pixel R
									Pel_G = (color_patch[4]&GreenMask)>>GreenShift;//pel pixel G
									Pel_B = (color_patch[4]&BlueMask)>>BlueShift;//pel pixel B

									/*checking differences*/
									//R
									if(Pel_R>meanR || (Pel_diff>10)){
										if(Patch_pos==0){
											diff[0] = Pel_R;
										}else if(Patch_pos==1){
											diff[1] = Pel_R;
										}else{//shifting
											diff[0] = diff[1];
											diff[1] = Pel_R;
										}

										entered = TRUE;
										patch_counter++;
									}

									//G
									if(Pel_G>meanG || (Pel_diff>10)){
										if(Patch_pos==0){
											diff[2] = Pel_G;
										}else if(Patch_pos==1){
											diff[3] = Pel_G;
										}else{//shifting
											diff[2] = diff[3];
											diff[3] = Pel_G;
										}

										entered = TRUE;
										patch_counter++;
									}

									//B
									if(Pel_B>meanB || (Pel_diff>10)){
										if(Patch_pos==0){
											diff[4] = Pel_B;
										}else if(Patch_pos==1){
											diff[5] = Pel_B;
										}else{//shifting
											diff[4] = diff[5];
											diff[5] = Pel_B;
										}

										entered = TRUE;
										patch_counter++;
									}


									if(entered){
										/* update the difference */
										Pel_diff *= (denominator+denominator+denominator);
										Pel_diff += (*(diff+1))>(*diff) ? ((*(diff+1))-(*diff)) : ((*diff)-(*(diff+1)));
										Pel_diff += (*(diff+3))>(*(diff+2)) ? ((*(diff+3))-(*(diff+2))) : ((*(diff+2))-(*(diff+3)));
										Pel_diff += (*(diff+5))>(*(diff+4)) ? ((*(diff+5))-(*(diff+4))) : ((*(diff+4))-(*(diff+5)));
										Pel_diff *= 1.f/(denominator+denominator+denominator);

										/*update mean*/
										meanB *= denominator;
										meanB += Pel_B;
										meanB *= 1.f/denominator;

										meanG *= denominator;
										meanG += Pel_G;
										meanG *= 1.f/denominator;

										meanR *= denominator;
										meanR += Pel_R;
										meanR *= 1.f/denominator;

										//update denominator
										denominator++;

										entered = FALSE;
									}

									/******************* PIXEL 6 *************************/
									Pel_R = (color_patch[5]&RedMask)>>RedShift;//pel pixel R
									Pel_G = (color_patch[5]&GreenMask)>>GreenShift;//pel pixel G
									Pel_B = (color_patch[5]&BlueMask)>>BlueShift;//pel pixel B

									/*checking differences*/
									//R
									if(Pel_R>meanR || (Pel_diff>10)){
										if(Patch_pos==0){
											diff[0] = Pel_R;
										}else if(Patch_pos==1){
											diff[1] = Pel_R;
										}else{//shifting
											diff[0] = diff[1];
											diff[1] = Pel_R;
										}

										entered = TRUE;
										patch_counter++;
									}

									//G
									if(Pel_G>meanG || (Pel_diff>10)){
										if(Patch_pos==0){
											diff[2] = Pel_G;
										}else if(Patch_pos==1){
											diff[3] = Pel_G;
										}else{//shifting
											diff[2] = diff[3];
											diff[3] = Pel_G;
										}

										entered = TRUE;
										patch_counter++;
									}

									//B
									if(Pel_B>meanB || (Pel_diff>10)){
										if(Patch_pos==0){
											diff[4] = Pel_B;
										}else if(Patch_pos==1){
											diff[5] = Pel_B;
										}else{//shifting
											diff[4] = diff[5];
											diff[5] = Pel_B;
										}

										entered = TRUE;
										patch_counter++;
									}


									if(entered){
										/* update the difference */
										Pel_diff *= (denominator+denominator+denominator);
										Pel_diff += (*(diff+1))>(*diff) ? ((*(diff+1))-(*diff)) : ((*diff)-(*(diff+1)));
										Pel_diff += (*(diff+3))>(*(diff+2)) ? ((*(diff+3))-(*(diff+2))) : ((*(diff+2))-(*(diff+3)));
										Pel_diff += (*(diff+5))>(*(diff+4)) ? ((*(diff+5))-(*(diff+4))) : ((*(diff+4))-(*(diff+5)));
										Pel_diff *= 1.f/(denominator+denominator+denominator);

										/*update mean*/
										meanB *= denominator;
										meanB += Pel_B;
										meanB *= 1.f/denominator;

										meanG *= denominator;
										meanG += Pel_G;
										meanG *= 1.f/denominator;

										meanR *= denominator;
										meanR += Pel_R;
										meanR *= 1.f/denominator;

										//update denominator
										denominator++;

										entered = FALSE;
									}


									/******************* PIXEL 7 *************************/
									Pel_R = (color_patch[6]&RedMask)>>RedShift;//pel pixel R
									Pel_G = (color_patch[6]&GreenMask)>>GreenShift;//pel pixel G
									Pel_B = (color_patch[6]&BlueMask)>>BlueShift;//pel pixel B

									/*checking differences*/
									//R
									if(Pel_R>meanR || (Pel_diff>10)){
										if(Patch_pos==0){
											diff[0] = Pel_R;
										}else if(Patch_pos==1){
											diff[1] = Pel_R;
										}else{//shifting
											diff[0] = diff[1];
											diff[1] = Pel_R;
										}

										entered = TRUE;
										patch_counter++;
									}

									//G
									if(Pel_G>meanG || (Pel_diff>10)){
										if(Patch_pos==0){
											diff[2] = Pel_G;
										}else if(Patch_pos==1){
											diff[3] = Pel_G;
										}else{//shifting
											diff[2] = diff[3];
											diff[3] = Pel_G;
										}

										entered = TRUE;
										patch_counter++;
									}

									//B
									if(Pel_B>meanB || (Pel_diff>10)){
										if(Patch_pos==0){
											diff[4] = Pel_B;
										}else if(Patch_pos==1){
											diff[5] = Pel_B;
										}else{//shifting
											diff[4] = diff[5];
											diff[5] = Pel_B;
										}

										entered = TRUE;
										patch_counter++;
									}


									if(entered){
										/* update the difference */
										Pel_diff *= (denominator+denominator+denominator);
										Pel_diff += (*(diff+1))>(*diff) ? ((*(diff+1))-(*diff)) : ((*diff)-(*(diff+1)));
										Pel_diff += (*(diff+3))>(*(diff+2)) ? ((*(diff+3))-(*(diff+2))) : ((*(diff+2))-(*(diff+3)));
										Pel_diff += (*(diff+5))>(*(diff+4)) ? ((*(diff+5))-(*(diff+4))) : ((*(diff+4))-(*(diff+5)));
										Pel_diff *= 1.f/(denominator+denominator+denominator);

										/*update mean*/
										meanB *= denominator;
										meanB += Pel_B;
										meanB *= 1.f/denominator;

										meanG *= denominator;
										meanG += Pel_G;
										meanG *= 1.f/denominator;

										meanR *= denominator;
										meanR += Pel_R;
										meanR *= 1.f/denominator;

										//update denominator
										denominator++;

										entered = FALSE;
									}

									/******************* PIXEL 8 *************************/
									Pel_R = (color_patch[7]&RedMask)>>RedShift;//pel pixel R
									Pel_G = (color_patch[7]&GreenMask)>>GreenShift;//pel pixel G
									Pel_B = (color_patch[7]&BlueMask)>>BlueShift;//pel pixel B

									/*checking differences*/
									//R
									if(Pel_R>meanR || (Pel_diff>10)){
										if(Patch_pos==0){
											diff[0] = Pel_R;
										}else if(Patch_pos==1){
											diff[1] = Pel_R;
										}else{//shifting
											diff[0] = diff[1];
											diff[1] = Pel_R;
										}

										entered = TRUE;
										patch_counter++;
									}

									//G
									if(Pel_G>meanG || (Pel_diff>10)){
										if(Patch_pos==0){
											diff[2] = Pel_G;
										}else if(Patch_pos==1){
											diff[3] = Pel_G;
										}else{//shifting
											diff[2] = diff[3];
											diff[3] = Pel_G;
										}

										entered = TRUE;
										patch_counter++;
									}

									//B
									if(Pel_B>meanB || (Pel_diff>10)){
										if(Patch_pos==0){
											diff[4] = Pel_B;
										}else if(Patch_pos==1){
											diff[5] = Pel_B;
										}else{//shifting
											diff[4] = diff[5];
											diff[5] = Pel_B;
										}

										entered = TRUE;
										patch_counter++;
									}


									if(entered){
										/* update the difference */
										Pel_diff *= (denominator+denominator+denominator);
										Pel_diff += (*(diff+1))>(*diff) ? ((*(diff+1))-(*diff)) : ((*diff)-(*(diff+1)));
										Pel_diff += (*(diff+3))>(*(diff+2)) ? ((*(diff+3))-(*(diff+2))) : ((*(diff+2))-(*(diff+3)));
										Pel_diff += (*(diff+5))>(*(diff+4)) ? ((*(diff+5))-(*(diff+4))) : ((*(diff+4))-(*(diff+5)));
										Pel_diff *= 1.f/(denominator+denominator+denominator);

										/*update mean*/
										meanB *= denominator;
										meanB += Pel_B;
										meanB *= 1.f/denominator;

										meanG *= denominator;
										meanG += Pel_G;
										meanG *= 1.f/denominator;

										meanR *= denominator;
										meanR += Pel_R;
										meanR *= 1.f/denominator;

										//update denominator
										denominator++;
										entered = FALSE;
									}
								}// stride >= 8
							}// stride >= 4
						}// stride >= 2
					}// stride >= 1
					/************************* UNROLLING ****************************/


					temp_x[0]=px;
					temp_masky=chunk_stride;
					ne10_addc_float_neon(temp_x, temp_x, temp_masky, 1);
					px=temp_x[0];
					if(px>=MASK_CX){
						temp_y[0]=py;
						ne10_addc_float_neon(temp_y, temp_y, 1, 1);
						py=temp_y[0];
						if(py>=MASK_CY){
							//we finished the patch
							break;
						}

						temp_maskx=MASK_CX;
						ne10_subc_float_neon(temp_x, temp_x, temp_maskx, 1);
						px=temp_x[0];
					}
				}//loop patch



				/* "THRESHOLD %" of the pixels show difference pattern wrt the patch mean*/
				if(((gfloat)patch_counter)*patch_factor >= skindetector->text_thres){
					/*fill the corresponding pixel in gray frame*/
					px = -MASK_CX;//pixel position in the patch, and counter to determine pattern significance
					py = -MASK_CY;//pixel position in the patch, and counter to determine pattern significance

					while(TRUE){
						/*working with the relative position in the patch*/
						Patch_pos = ((y+py)*GRAY_stride + (x+px));//-----------

						/*safety*/
						x_check = Patch_pos%GRAY_stride;
						y_check = Patch_pos/GRAY_stride;
						if((x_check<0 || x_check>=src_stride) && (y_check<0 || y_check>=height)){
							break;
						}else if((x_check<0 || x_check>=src_stride) && (y_check>0 || y_check<height)){
							temp_x[0]=px; temp_y[0]=py; temp_maskx=MASK_CX;
							ne10_addc_float_neon(temp_y, temp_y, 1, 1);
							py=temp_y[0];
							if(py>=MASK_CY){
								//we finished the patch
								break;
							}
							px=-MASK_CX;
							continue;//we havent finished the height
						}else if((y_check<0 || y_check>=height) && (x_check>0 || x_check<GRAY_stride-chunk_stride)){
							temp_x[0]=px; temp_y[0]=py; temp_maskx=MASK_CX; temp_masky=chunk_stride;
							ne10_addc_float_neon(temp_x, temp_x, temp_masky, 1);
							px=temp_x[0];
							if(px>=MASK_CX){
								ne10_addc_float_neon(temp_y, temp_y, 1, 1);
								py=temp_y[0];
								if(py>=MASK_CY){
									//we finished the patch
									break;
								}
								px=-MASK_CX;
							}
							continue;//we havent finished the width
						}

						/************************* UNROLLING ****************************/
						if(chunk_stride>=1){
							(*(GRAY_data+Patch_pos)) |= MASK_MASK;//switching ON the corresponding bit
							if(chunk_stride>=2){
								(*(GRAY_data+Patch_pos+1)) |= MASK_MASK;//switching ON the corresponding bit
								if(chunk_stride>=4){
									(*(GRAY_data+Patch_pos+2)) |= MASK_MASK;//switching ON the corresponding bit
									(*(GRAY_data+Patch_pos+3)) |= MASK_MASK;//switching ON the corresponding bit
									if(chunk_stride>=8){
										(*(GRAY_data+Patch_pos+4)) |= MASK_MASK;//switching ON the corresponding bit
										(*(GRAY_data+Patch_pos+5)) |= MASK_MASK;//switching ON the corresponding bit
										(*(GRAY_data+Patch_pos+6)) |= MASK_MASK;//switching ON the corresponding bit
										(*(GRAY_data+Patch_pos+7)) |= MASK_MASK;//switching ON the corresponding bit
									}// stride >= 8
								}// stride >= 4
							}// stride >= 2
						}// stride >= 1
						/************************* UNROLLING ****************************/

						temp_x[0]=px;
						temp_masky=chunk_stride;
						ne10_addc_float_neon(temp_x, temp_x, temp_masky, 1);
						px=temp_x[0];
						if(px>=MASK_CX){
							temp_y[0]=py;
							ne10_addc_float_neon(temp_y, temp_y, 1, 1);
							py=temp_y[0];
							if(py>=MASK_CY){
								//we finished the patch
								break;
							}
							px=-MASK_CX;
						}

					}//loop patch
				}//Filling gray image
			}//finished image
		}//finished patches
		return TRUE;
	}
#else// HAVE_NEON
	static gint texture_extraction(GstVideoFrame *inframe, GstVideoFrame *outframe, guint32 height, GstSkinDetector *skindetector){
		guint32 mask_id, color;
		guint8 *src_data = NULL, *GRAY_data = NULL, Pel_R, Pel_G, Pel_B;
		guint32 meanR, meanG, meanB;
		gint denominator;
		guint32 diff[2*3]={0}, Pel_diff;//current and previous pixel (RGB)
		guint8 RGB_pix_stride=4;//packed pixels
		guint32 Img_pos, Patch_pos, x, y, width, x_check, y_check;
		gint px, py;
		gint src_stride = 0, GRAY_stride = 0, GRAY_comp = 0;
		guint32 patch_counter;
		gfloat mask_size, patch_factor=1.f;
		guint32 RedMask=skindetector->RedMask, GreenMask=skindetector->GreenMask, BlueMask=skindetector->BlueMask;
		guint8 RedShift=skindetector->RedShift, GreenShift=skindetector->GreenShift, BlueShift=skindetector->BlueShift;
		gboolean entered=FALSE;
		guint8 chunk_stride;
		guint MASK_W;

		/* Beginning of the pointer, no matter endianess */
		src_data = GST_VIDEO_FRAME_PLANE_DATA(inframe, 0);
		src_stride = GST_VIDEO_FRAME_COMP_STRIDE(inframe, 0);
		width = GST_VIDEO_FRAME_WIDTH(inframe);

		/* Beginning of the pointer, no worry about endianess */
		GRAY_data = GST_VIDEO_FRAME_PLANE_DATA(outframe, GRAY_comp);
		GRAY_stride = GST_VIDEO_FRAME_COMP_STRIDE(outframe, GRAY_comp);

		//for all the masks, NON-OVERLAPPING PATCHES
		for(mask_id=0; mask_id<NMASKS;mask_id++){
			Img_pos=x=y=0;//position of the mask center basically
			mask_size = skindetector->masks[mask_id].Mask_Hsize*skindetector->masks[mask_id].Mask_Wsize;
			mask_size += (mask_size + mask_size);
			patch_factor = 1.f/mask_size;

			const gint MASK_CX = skindetector->masks[mask_id].Cx;
			const gint MASK_CY = skindetector->masks[mask_id].Cy;
			MASK_W = skindetector->masks[mask_id].Mask_Wsize;
			const guint8 MASK_MASK = skindetector->masks[mask_id].mask;

			while(TRUE){//image loop
				/*positioning mask*/
				if(y==0 && x==0){
					x += MASK_CX;
					y += MASK_CY;
				}else{
					x += (MASK_CX+MASK_CX);//no overlap allowed
				}

				if(x>=width){
					x = MASK_CX;
					y += (MASK_CY+MASK_CY);//no overlaped allowed
					if(y>=height){
						//we finished the frame
						break;
					}
				}

				Img_pos = y*src_stride + x*RGB_pix_stride;//location in the source && destination images
				color = (*(guint32*)(src_data+Img_pos));
				meanR = ((color&RedMask)>>RedShift);//pivot pixel R
				meanG = ((color&GreenMask)>>GreenShift);//pivot pixel G
				meanB = ((color&BlueMask)>>BlueShift);//pivot pixel B

				patch_counter=0;
				px = -MASK_CX;//pixel position in the patch, and counter to determine pattern significance
				py = -MASK_CY;//pixel position in the patch, and counter to determine pattern significance

				/*shifting the locations with respect to center*/
				denominator=1;
				Pel_diff = 0;

				/*
				 * To unroll the loop below since it is too expensive
				 * and it is the same value for the whole patch, so up to
				 * 8 pixels at a time
				 * */
				if(MASK_W>8){
					chunk_stride = 8;
				}else if(MASK_W>4){
					chunk_stride = 4;
				}else if(MASK_W>2){
					chunk_stride = 2;
				}else{
					chunk_stride = 1;
				}

				while(TRUE){
					/*working with the relative position in the patch*/
					Patch_pos = ((y+py)*src_stride + (x+px)*RGB_pix_stride);

					/*safety*/
					x_check = Patch_pos%src_stride;
					y_check = Patch_pos/src_stride;
					if((x_check<0 || x_check>=src_stride) && (y_check<0 || y_check>=height)){
						break;
					}else if((x_check<0 || x_check>=src_stride) && (y_check>0 || y_check<height)){
						py++;
						if(py>=MASK_CY){
							//we finished the patch
							break;
						}
						px=-MASK_CX;
						continue;//we havent finished the height
					}else if((y_check<0 || y_check>=height) && (x_check>0 || x_check<src_stride-chunk_stride*RGB_pix_stride)){
						px+=chunk_stride;
						if(px>=MASK_CX){
							py++;
							if(py>=MASK_CY){
								//we finished the patch
								break;
							}
							px=-MASK_CX;
						}
						continue;//we havent finished the width
					}


					/************************* UNROLLING ****************************/
					/******************* PIXEL 1 *************************/
					if(chunk_stride>=1){
						color = (*(guint32*)(src_data+Patch_pos));
						Pel_R = (color&RedMask)>>RedShift;//pel pixel R
						Pel_G = (color&GreenMask)>>GreenShift;//pel pixel G
						Pel_B = (color&BlueMask)>>BlueShift;//pel pixel B

						/*checking differences*/
						//R
						if(Pel_R>meanR || (Pel_diff>10)){
							if(Patch_pos==0){
								diff[0] = Pel_R;
							}else if(Patch_pos==1){
								diff[1] = Pel_R;
							}else{//shifting
								diff[0] = diff[1];
								diff[1] = Pel_R;
							}

							entered = TRUE;
							patch_counter++;
						}

						//G
						if(Pel_G>meanG || (Pel_diff>10)){
							if(Patch_pos==0){
								diff[2] = Pel_G;
							}else if(Patch_pos==1){
								diff[3] = Pel_G;
							}else{//shifting
								diff[2] = diff[3];
								diff[3] = Pel_G;
							}

							entered = TRUE;
							patch_counter++;
						}

						//B
						if(Pel_B>meanB || (Pel_diff>10)){
							if(Patch_pos==0){
								diff[4] = Pel_B;
							}else if(Patch_pos==1){
								diff[5] = Pel_B;
							}else{//shifting
								diff[4] = diff[5];
								diff[5] = Pel_B;
							}

							entered = TRUE;
							patch_counter++;
						}


						if(entered){
							/* update the difference */
							Pel_diff *= (denominator+denominator+denominator);
							Pel_diff += (*(diff+1))>(*diff) ? ((*(diff+1))-(*diff)) : ((*diff)-(*(diff+1)));
							Pel_diff += (*(diff+3))>(*(diff+2)) ? ((*(diff+3))-(*(diff+2))) : ((*(diff+2))-(*(diff+3)));
							Pel_diff += (*(diff+5))>(*(diff+4)) ? ((*(diff+5))-(*(diff+4))) : ((*(diff+4))-(*(diff+5)));
							Pel_diff *= 1.f/(denominator+denominator+denominator);

							/*update mean*/
							meanB *= denominator;
							meanB += Pel_B;
							meanB *= 1.f/denominator;

							meanG *= denominator;
							meanG += Pel_G;
							meanG *= 1.f/denominator;

							meanR *= denominator;
							meanR += Pel_R;
							meanR *= 1.f/denominator;

							//update denominator
							denominator++;

							entered = FALSE;
						}

						if(chunk_stride>=2){
							/******************* PIXEL 2 *************************/
							Patch_pos+=RGB_pix_stride;
							color = (*(guint32*)(src_data+Patch_pos));
							Pel_R = (color&RedMask)>>RedShift;//pel pixel R
							Pel_G = (color&GreenMask)>>GreenShift;//pel pixel G
							Pel_B = (color&BlueMask)>>BlueShift;//pel pixel B

							/*checking differences*/
							//R
							if(Pel_R>meanR || (Pel_diff>10)){
								if(Patch_pos==0){
									diff[0] = Pel_R;
								}else if(Patch_pos==1){
									diff[1] = Pel_R;
								}else{//shifting
									diff[0] = diff[1];
									diff[1] = Pel_R;
								}

								entered = TRUE;
								patch_counter++;
							}

							//G
							if(Pel_G>meanG || (Pel_diff>10)){
								if(Patch_pos==0){
									diff[2] = Pel_G;
								}else if(Patch_pos==1){
									diff[3] = Pel_G;
								}else{//shifting
									diff[2] = diff[3];
									diff[3] = Pel_G;
								}

								entered = TRUE;
								patch_counter++;
							}

							//B
							if(Pel_B>meanB || (Pel_diff>10)){
								if(Patch_pos==0){
									diff[4] = Pel_B;
								}else if(Patch_pos==1){
									diff[5] = Pel_B;
								}else{//shifting
									diff[4] = diff[5];
									diff[5] = Pel_B;
								}

								entered = TRUE;
								patch_counter++;
							}


							if(entered){
								/* update the difference */
								Pel_diff *= (denominator+denominator+denominator);
								Pel_diff += (*(diff+1))>(*diff) ? ((*(diff+1))-(*diff)) : ((*diff)-(*(diff+1)));
								Pel_diff += (*(diff+3))>(*(diff+2)) ? ((*(diff+3))-(*(diff+2))) : ((*(diff+2))-(*(diff+3)));
								Pel_diff += (*(diff+5))>(*(diff+4)) ? ((*(diff+5))-(*(diff+4))) : ((*(diff+4))-(*(diff+5)));
								Pel_diff *= 1.f/(denominator+denominator+denominator);

								/*update mean*/
								meanB *= denominator;
								meanB += Pel_B;
								meanB *= 1.f/denominator;

								meanG *= denominator;
								meanG += Pel_G;
								meanG *= 1.f/denominator;

								meanR *= denominator;
								meanR += Pel_R;
								meanR *= 1.f/denominator;

								//update denominator
								denominator++;

								entered = FALSE;
							}

							if(chunk_stride>=4){
								/******************* PIXEL 3 *************************/
								Patch_pos+=RGB_pix_stride;
								color = (*(guint32*)(src_data+Patch_pos));
								Pel_R = (color&RedMask)>>RedShift;//pel pixel R
								Pel_G = (color&GreenMask)>>GreenShift;//pel pixel G
								Pel_B = (color&BlueMask)>>BlueShift;//pel pixel B

								/*checking differences*/
								//R
								if(Pel_R>meanR || (Pel_diff>10)){
									if(Patch_pos==0){
										diff[0] = Pel_R;
									}else if(Patch_pos==1){
										diff[1] = Pel_R;
									}else{//shifting
										diff[0] = diff[1];
										diff[1] = Pel_R;
									}

									entered = TRUE;
									patch_counter++;
								}

								//G
								if(Pel_G>meanG || (Pel_diff>10)){
									if(Patch_pos==0){
										diff[2] = Pel_G;
									}else if(Patch_pos==1){
										diff[3] = Pel_G;
									}else{//shifting
										diff[2] = diff[3];
										diff[3] = Pel_G;
									}

									entered = TRUE;
									patch_counter++;
								}

								//B
								if(Pel_B>meanB || (Pel_diff>10)){
									if(Patch_pos==0){
										diff[4] = Pel_B;
									}else if(Patch_pos==1){
										diff[5] = Pel_B;
									}else{//shifting
										diff[4] = diff[5];
										diff[5] = Pel_B;
									}

									entered = TRUE;
									patch_counter++;
								}


								if(entered){
									/* update the difference */
									Pel_diff *= (denominator+denominator+denominator);
									Pel_diff += (*(diff+1))>(*diff) ? ((*(diff+1))-(*diff)) : ((*diff)-(*(diff+1)));
									Pel_diff += (*(diff+3))>(*(diff+2)) ? ((*(diff+3))-(*(diff+2))) : ((*(diff+2))-(*(diff+3)));
									Pel_diff += (*(diff+5))>(*(diff+4)) ? ((*(diff+5))-(*(diff+4))) : ((*(diff+4))-(*(diff+5)));
									Pel_diff *= 1.f/(denominator+denominator+denominator);

									/*update mean*/
									meanB *= denominator;
									meanB += Pel_B;
									meanB *= 1.f/denominator;

									meanG *= denominator;
									meanG += Pel_G;
									meanG *= 1.f/denominator;

									meanR *= denominator;
									meanR += Pel_R;
									meanR *= 1.f/denominator;

									//update denominator
									denominator++;

									entered = FALSE;
								}

								/******************* PIXEL 4 *************************/
								Patch_pos+=RGB_pix_stride;
								color = (*(guint32*)(src_data+Patch_pos));
								Pel_R = (color&RedMask)>>RedShift;//pel pixel R
								Pel_G = (color&GreenMask)>>GreenShift;//pel pixel G
								Pel_B = (color&BlueMask)>>BlueShift;//pel pixel B

								/*checking differences*/
								//R
								if(Pel_R>meanR || (Pel_diff>10)){
									if(Patch_pos==0){
										diff[0] = Pel_R;
									}else if(Patch_pos==1){
										diff[1] = Pel_R;
									}else{//shifting
										diff[0] = diff[1];
										diff[1] = Pel_R;
									}

									entered = TRUE;
									patch_counter++;
								}

								//G
								if(Pel_G>meanG || (Pel_diff>10)){
									if(Patch_pos==0){
										diff[2] = Pel_G;
									}else if(Patch_pos==1){
										diff[3] = Pel_G;
									}else{//shifting
										diff[2] = diff[3];
										diff[3] = Pel_G;
									}

									entered = TRUE;
									patch_counter++;
								}

								//B
								if(Pel_B>meanB || (Pel_diff>10)){
									if(Patch_pos==0){
										diff[4] = Pel_B;
									}else if(Patch_pos==1){
										diff[5] = Pel_B;
									}else{//shifting
										diff[4] = diff[5];
										diff[5] = Pel_B;
									}

									entered = TRUE;
									patch_counter++;
								}


								if(entered){
									/* update the difference */
									Pel_diff *= (denominator+denominator+denominator);
									Pel_diff += (*(diff+1))>(*diff) ? ((*(diff+1))-(*diff)) : ((*diff)-(*(diff+1)));
									Pel_diff += (*(diff+3))>(*(diff+2)) ? ((*(diff+3))-(*(diff+2))) : ((*(diff+2))-(*(diff+3)));
									Pel_diff += (*(diff+5))>(*(diff+4)) ? ((*(diff+5))-(*(diff+4))) : ((*(diff+4))-(*(diff+5)));
									Pel_diff *= 1.f/(denominator+denominator+denominator);

									/*update mean*/
									meanB *= denominator;
									meanB += Pel_B;
									meanB *= 1.f/denominator;

									meanG *= denominator;
									meanG += Pel_G;
									meanG *= 1.f/denominator;

									meanR *= denominator;
									meanR += Pel_R;
									meanR *= 1.f/denominator;

									//update denominator
									denominator++;

									entered = FALSE;
								}

								if(chunk_stride>=8){
									/******************* PIXEL 5 *************************/
									Patch_pos+=RGB_pix_stride;
									color = (*(guint32*)(src_data+Patch_pos));
									Pel_R = (color&RedMask)>>RedShift;//pel pixel R
									Pel_G = (color&GreenMask)>>GreenShift;//pel pixel G
									Pel_B = (color&BlueMask)>>BlueShift;//pel pixel B

									/*checking differences*/
									//R
									if(Pel_R>meanR || (Pel_diff>10)){
										if(Patch_pos==0){
											diff[0] = Pel_R;
										}else if(Patch_pos==1){
											diff[1] = Pel_R;
										}else{//shifting
											diff[0] = diff[1];
											diff[1] = Pel_R;
										}

										entered = TRUE;
										patch_counter++;
									}

									//G
									if(Pel_G>meanG || (Pel_diff>10)){
										if(Patch_pos==0){
											diff[2] = Pel_G;
										}else if(Patch_pos==1){
											diff[3] = Pel_G;
										}else{//shifting
											diff[2] = diff[3];
											diff[3] = Pel_G;
										}

										entered = TRUE;
										patch_counter++;
									}

									//B
									if(Pel_B>meanB || (Pel_diff>10)){
										if(Patch_pos==0){
											diff[4] = Pel_B;
										}else if(Patch_pos==1){
											diff[5] = Pel_B;
										}else{//shifting
											diff[4] = diff[5];
											diff[5] = Pel_B;
										}

										entered = TRUE;
										patch_counter++;
									}


									if(entered){
										/* update the difference */
										Pel_diff *= (denominator+denominator+denominator);
										Pel_diff += (*(diff+1))>(*diff) ? ((*(diff+1))-(*diff)) : ((*diff)-(*(diff+1)));
										Pel_diff += (*(diff+3))>(*(diff+2)) ? ((*(diff+3))-(*(diff+2))) : ((*(diff+2))-(*(diff+3)));
										Pel_diff += (*(diff+5))>(*(diff+4)) ? ((*(diff+5))-(*(diff+4))) : ((*(diff+4))-(*(diff+5)));
										Pel_diff *= 1.f/(denominator+denominator+denominator);

										/*update mean*/
										meanB *= denominator;
										meanB += Pel_B;
										meanB *= 1.f/denominator;

										meanG *= denominator;
										meanG += Pel_G;
										meanG *= 1.f/denominator;

										meanR *= denominator;
										meanR += Pel_R;
										meanR *= 1.f/denominator;

										//update denominator
										denominator++;

										entered = FALSE;
									}

									/******************* PIXEL 6 *************************/
									Patch_pos+=RGB_pix_stride;
									color = (*(guint32*)(src_data+Patch_pos));
									Pel_R = (color&RedMask)>>RedShift;//pel pixel R
									Pel_G = (color&GreenMask)>>GreenShift;//pel pixel G
									Pel_B = (color&BlueMask)>>BlueShift;//pel pixel B

									/*checking differences*/
									//R
									if(Pel_R>meanR || (Pel_diff>10)){
										if(Patch_pos==0){
											diff[0] = Pel_R;
										}else if(Patch_pos==1){
											diff[1] = Pel_R;
										}else{//shifting
											diff[0] = diff[1];
											diff[1] = Pel_R;
										}

										entered = TRUE;
										patch_counter++;
									}

									//G
									if(Pel_G>meanG || (Pel_diff>10)){
										if(Patch_pos==0){
											diff[2] = Pel_G;
										}else if(Patch_pos==1){
											diff[3] = Pel_G;
										}else{//shifting
											diff[2] = diff[3];
											diff[3] = Pel_G;
										}

										entered = TRUE;
										patch_counter++;
									}

									//B
									if(Pel_B>meanB || (Pel_diff>10)){
										if(Patch_pos==0){
											diff[4] = Pel_B;
										}else if(Patch_pos==1){
											diff[5] = Pel_B;
										}else{//shifting
											diff[4] = diff[5];
											diff[5] = Pel_B;
										}

										entered = TRUE;
										patch_counter++;
									}


									if(entered){
										/* update the difference */
										Pel_diff *= (denominator+denominator+denominator);
										Pel_diff += (*(diff+1))>(*diff) ? ((*(diff+1))-(*diff)) : ((*diff)-(*(diff+1)));
										Pel_diff += (*(diff+3))>(*(diff+2)) ? ((*(diff+3))-(*(diff+2))) : ((*(diff+2))-(*(diff+3)));
										Pel_diff += (*(diff+5))>(*(diff+4)) ? ((*(diff+5))-(*(diff+4))) : ((*(diff+4))-(*(diff+5)));
										Pel_diff *= 1.f/(denominator+denominator+denominator);

										/*update mean*/
										meanB *= denominator;
										meanB += Pel_B;
										meanB *= 1.f/denominator;

										meanG *= denominator;
										meanG += Pel_G;
										meanG *= 1.f/denominator;

										meanR *= denominator;
										meanR += Pel_R;
										meanR *= 1.f/denominator;

										//update denominator
										denominator++;

										entered = FALSE;
									}


									/******************* PIXEL 7 *************************/
									Patch_pos+=RGB_pix_stride;
									color = (*(guint32*)(src_data+Patch_pos));
									Pel_R = (color&RedMask)>>RedShift;//pel pixel R
									Pel_G = (color&GreenMask)>>GreenShift;//pel pixel G
									Pel_B = (color&BlueMask)>>BlueShift;//pel pixel B

									/*checking differences*/
									//R
									if(Pel_R>meanR || (Pel_diff>10)){
										if(Patch_pos==0){
											diff[0] = Pel_R;
										}else if(Patch_pos==1){
											diff[1] = Pel_R;
										}else{//shifting
											diff[0] = diff[1];
											diff[1] = Pel_R;
										}

										entered = TRUE;
										patch_counter++;
									}

									//G
									if(Pel_G>meanG || (Pel_diff>10)){
										if(Patch_pos==0){
											diff[2] = Pel_G;
										}else if(Patch_pos==1){
											diff[3] = Pel_G;
										}else{//shifting
											diff[2] = diff[3];
											diff[3] = Pel_G;
										}

										entered = TRUE;
										patch_counter++;
									}

									//B
									if(Pel_B>meanB || (Pel_diff>10)){
										if(Patch_pos==0){
											diff[4] = Pel_B;
										}else if(Patch_pos==1){
											diff[5] = Pel_B;
										}else{//shifting
											diff[4] = diff[5];
											diff[5] = Pel_B;
										}

										entered = TRUE;
										patch_counter++;
									}


									if(entered){
										/* update the difference */
										Pel_diff *= (denominator+denominator+denominator);
										Pel_diff += (*(diff+1))>(*diff) ? ((*(diff+1))-(*diff)) : ((*diff)-(*(diff+1)));
										Pel_diff += (*(diff+3))>(*(diff+2)) ? ((*(diff+3))-(*(diff+2))) : ((*(diff+2))-(*(diff+3)));
										Pel_diff += (*(diff+5))>(*(diff+4)) ? ((*(diff+5))-(*(diff+4))) : ((*(diff+4))-(*(diff+5)));
										Pel_diff *= 1.f/(denominator+denominator+denominator);

										/*update mean*/
										meanB *= denominator;
										meanB += Pel_B;
										meanB *= 1.f/denominator;

										meanG *= denominator;
										meanG += Pel_G;
										meanG *= 1.f/denominator;

										meanR *= denominator;
										meanR += Pel_R;
										meanR *= 1.f/denominator;

										//update denominator
										denominator++;

										entered = FALSE;
									}

									/******************* PIXEL 8 *************************/
									Patch_pos+=RGB_pix_stride;
									color = (*(guint32*)(src_data+Patch_pos));
									Pel_R = (color&RedMask)>>RedShift;//pel pixel R
									Pel_G = (color&GreenMask)>>GreenShift;//pel pixel G
									Pel_B = (color&BlueMask)>>BlueShift;//pel pixel B

									/*checking differences*/
									//R
									if(Pel_R>meanR || (Pel_diff>10)){
										if(Patch_pos==0){
											diff[0] = Pel_R;
										}else if(Patch_pos==1){
											diff[1] = Pel_R;
										}else{//shifting
											diff[0] = diff[1];
											diff[1] = Pel_R;
										}

										entered = TRUE;
										patch_counter++;
									}

									//G
									if(Pel_G>meanG || (Pel_diff>10)){
										if(Patch_pos==0){
											diff[2] = Pel_G;
										}else if(Patch_pos==1){
											diff[3] = Pel_G;
										}else{//shifting
											diff[2] = diff[3];
											diff[3] = Pel_G;
										}

										entered = TRUE;
										patch_counter++;
									}

									//B
									if(Pel_B>meanB || (Pel_diff>10)){
										if(Patch_pos==0){
											diff[4] = Pel_B;
										}else if(Patch_pos==1){
											diff[5] = Pel_B;
										}else{//shifting
											diff[4] = diff[5];
											diff[5] = Pel_B;
										}

										entered = TRUE;
										patch_counter++;
									}


									if(entered){
										/* update the difference */
										Pel_diff *= (denominator+denominator+denominator);
										Pel_diff += (*(diff+1))>(*diff) ? ((*(diff+1))-(*diff)) : ((*diff)-(*(diff+1)));
										Pel_diff += (*(diff+3))>(*(diff+2)) ? ((*(diff+3))-(*(diff+2))) : ((*(diff+2))-(*(diff+3)));
										Pel_diff += (*(diff+5))>(*(diff+4)) ? ((*(diff+5))-(*(diff+4))) : ((*(diff+4))-(*(diff+5)));
										Pel_diff *= 1.f/(denominator+denominator+denominator);

										/*update mean*/
										meanB *= denominator;
										meanB += Pel_B;
										meanB *= 1.f/denominator;

										meanG *= denominator;
										meanG += Pel_G;
										meanG *= 1.f/denominator;

										meanR *= denominator;
										meanR += Pel_R;
										meanR *= 1.f/denominator;

										//update denominator
										denominator++;
										entered = FALSE;
									}
								}// stride >= 8
							}// stride >= 4
						}// stride >= 2
					}// stride >= 1
					/************************* UNROLLING ****************************/

					px+=chunk_stride;
					if(px>=MASK_CX){
						py++;
						if(py>=MASK_CY){
							//we finished the patch
							break;
						}
						px=-MASK_CX;
					}
				}//loop patch


				/* "THRESHOLD %" of the pixels show difference pattern wrt the patch mean*/
				if(((gfloat)patch_counter)*patch_factor >= skindetector->text_thres){
					/*fill the corresponding pixel in gray frame*/
					px = -MASK_CX;//pixel position in the patch, and counter to determine pattern significance
					py = -MASK_CY;//pixel position in the patch, and counter to determine pattern significance

					while(TRUE){
						/*working with the relative position in the patch*/
						Patch_pos = ((y+py)*GRAY_stride + (x+px));//-----------

						/*safety*/
						x_check = Patch_pos%GRAY_stride;
						y_check = Patch_pos/GRAY_stride;
						if((x_check<0 || x_check>=GRAY_stride) && (y_check<0 || y_check>=height)){
							break;
						}else if((x_check<0 || x_check>=GRAY_stride) && (y_check>0 || y_check<height)){
							py++;
							if(py>=MASK_CY){
								//we finished the patch
								break;
							}
							px=-MASK_CX;
							continue;//we havent finished the height
						}else if((y_check<0 || y_check>=height) && (x_check>0 || x_check<GRAY_stride-chunk_stride)){
							px+=chunk_stride;
							if(px>=MASK_CX){
								py++;
								if(py>=MASK_CY){
									//we finished the patch
									break;
								}
								px=-MASK_CX;
							}
							continue;//we havent finished the width
						}

						/************************* UNROLLING ****************************/
						if(chunk_stride>=1){
							(*(GRAY_data+Patch_pos)) |= MASK_MASK;//switching ON the corresponding bit
							if(chunk_stride>=2){
								(*(GRAY_data+Patch_pos+1)) |= MASK_MASK;//switching ON the corresponding bit
								if(chunk_stride>=4){
									(*(GRAY_data+Patch_pos+2)) |= MASK_MASK;//switching ON the corresponding bit
									(*(GRAY_data+Patch_pos+3)) |= MASK_MASK;//switching ON the corresponding bit
									if(chunk_stride>=8){
										(*(GRAY_data+Patch_pos+4)) |= MASK_MASK;//switching ON the corresponding bit
										(*(GRAY_data+Patch_pos+5)) |= MASK_MASK;//switching ON the corresponding bit
										(*(GRAY_data+Patch_pos+6)) |= MASK_MASK;//switching ON the corresponding bit
										(*(GRAY_data+Patch_pos+7)) |= MASK_MASK;//switching ON the corresponding bit
									}// stride >= 8
								}// stride >= 4
							}// stride >= 2
						}// stride >= 1
						/************************* UNROLLING ****************************/

						px+=chunk_stride;//
						if(px>=MASK_CX){
							py++;
							if(py>=MASK_CY){
								//we finished the patch
								break;
							}
							px=-MASK_CX;
						}


					}//loop patch
				}//Filling gray image
			}//finished image
		}//finished patches
		return TRUE;
	}
#endif













#if 0
static gboolean plugin_init (GstPlugin * plugin){

	return gst_element_register (plugin, "skindetector", GST_RANK_NONE, GST_TYPE_SKINDETECTOR);
}

/*
 * Just general information of this element.
 *  */
#ifndef VERSION
	#define VERSION "0.0.1"
#endif
#ifndef PACKAGE
	#define PACKAGE "Skin_Detector_API"
#endif
#ifndef PACKAGE_NAME
	#define PACKAGE_NAME "Image_proc_API"
#endif
#ifndef GST_PACKAGE_ORIGIN
	#define GST_PACKAGE_ORIGIN "None"
#endif

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR, GST_VERSION_MINOR, skindetector,
		"Skin detector plugin to be used, to detect patches of skin a video frame, RGB or I402, "
		"it outputs a binary image (0,1) or a normalized (0,255) in order to visualize it", plugin_init, VERSION, "LGPL",
		PACKAGE_NAME, GST_PACKAGE_ORIGIN)
#endif

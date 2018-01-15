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
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA 02110-1301, USA.
 */

#ifndef _GST_BLOBDETECTOR_H_
#define _GST_BLOBDETECTOR_H_

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/video/gstvideofilter.h>
#include "../../common/metainfo.h"

G_BEGIN_DECLS

#define GST_TYPE_BLOBDETECTOR   (gst_blobdetector_get_type())
#define GST_BLOBDETECTOR(obj)   (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_BLOBDETECTOR,GstBlobDetector))
#define GST_BLOBDETECTOR_CLASS(klass)   (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_BLOBDETECTOR,GstBlobDetectorClass))
#define GST_IS_BLOBDETECTOR(obj)   (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_BLOBDETECTOR))
#define GST_IS_BLOBDETECTOR_CLASS(obj)   (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_BLOBDETECTOR))

typedef struct _GstBlobDetector GstBlobDetector;
typedef struct _GstBlobDetectorClass GstBlobDetectorClass;
typedef struct _tracker_data tracker_data;


/*
 * Constains the information regarding a particular
 * blob
 * */
struct _blobdims{
	guint32 xmin, xmax, ymin, ymax;//for BBs
	guint32 x, y;//coordinates in the entire image
	guint32 width,height;//size of the blob
	gfloat Blob_Pel_Density;//pixel density of the blob
	gboolean passed;//flag to signal if blob has to be considered
};

typedef struct _blobdims BLOB;

struct _GstBlobDetector
{
	GstVideoFilter base_blobdetector;

	//to check if the user wants to normalize the output frame
	gint normalize_frame;

	/*
	* this will point to the original data (or copy), where the user
	* needs to decide the best way to handle this
	* */
	GstVideoFrame * inframe;

	/*
	 * This will contain the resulting detection,
	 * size is the same as the original image, but the values
	 * are just 0 or 1 for detection or no detection
	 */
	GstVideoFrame * outframe;
	guint32 *labels;

	/*
	 * General information of the image that has to be provided by the
	 * user, or could be set by Gstreamer
	 * */
	guint32 height;
	guint32 width;
	guint32 channels_format;//plannar or interlace
	guint32 detect_space;//space to be used in the skin detection
	guint32 image_type;//space of the image, RGB, YUV, normal
	guint32 channels;//how many channels


	/*
	 * This will contain the resulting detection,
	 * size is the same as the original image, but the values
	 * are just 0 or 1 for detection or no detection
	 */
	BLOB *blobs;//blob information
	guint8 n_blobs;//how many blobs we have
	gfloat pixel_density;//to estimage the pixel density of each blob
	gfloat text_thres;//used to filter out the blobs with high texture value

	/*
	 * Minimum area size of blob detector
	 * */
	guint minarea, maxarea, minwidth, minheight, maxwidth, maxheight;

	/* shadow variables */
	GList *seedShadow;  /* current shadow */
	GList *rowHead;     /* current row shadows */
	GList *pendHead;    /* other pending shadows */
	GList *freeHead;    /* unused shadow nodes */

	/*
	 * Allow overlap or not , by default is yes so two
	 * connecting blobs are combined to form a single one
	 * */
	gint Allowoverlap;

	/* Processing time for every frame */
	gfloat internal_pt;//internal processing time
};


/* line shadow */
typedef struct _tracker_data{
   int lft, rgt;           /* endpoints */
   int row, par;           /* row and parent row */
   gboolean ok;             /* valid flag */
};


struct _GstBlobDetectorClass
{
  GstVideoFilterClass base_blobdetector_class;
};

GType gst_blobdetector_get_type (void);
void* gst_buffer_add_blob_meta(GstBuffer *buffer, GstBlobDetector *blobdetector);


G_END_DECLS

#endif

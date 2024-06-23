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

#ifndef _GST_MOMENTNORMALIZATION_H_
#define _GST_MOMENTNORMALIZATION_H_

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/video/gstvideofilter.h>
#include <time.h>
#include "../../common/metainfo.h"

G_BEGIN_DECLS

#define GST_TYPE_MOMENTNORMALIZATION   (gst_momentnormalization_get_type())
#define GST_MOMENTNORMALIZATION(obj)   (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_MOMENTNORMALIZATION,GstMomentNormalization))
#define GST_MOMENTNORMALIZATION_CLASS(klass)   (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_MOMENTNORMALIZATION,GstMomentNormalizationClass))
#define GST_IS_MOMENTNORMALIZATION(obj)   (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_MOMENTNORMALIZATION))
#define GST_IS_MOMENTNORMALIZATION_CLASS(obj)   (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_MOMENTNORMALIZATION))

typedef struct _GstMomentNormalization GstMomentNormalization;
typedef struct _GstMomentNormalizationClass GstMomentNormalizationClass;
typedef struct _blobdimsmoment BLOBMOMENT;
typedef struct _PreviousBlobs PreviousBlobs;



/*
* Contains the information regarding a particular
* blob
* */
struct _blobdimsmoment{
	guint32 xmin, xmax, ymin, ymax;//for BBs
	guint32 x, y;//coordinates in the entire image
	guint32 width,height;//size of the blob
	guint8 *patch;//pixels for this patch or blob
	guint8 *transformed;//pixels for this patch or blob normalized

	/*private data for patch processing*/
	gdouble *moments;//moments needed for processing
	gdouble eigen_vals[2], eigen_vec[2][2];//eigen values and vectors
	gdouble A[2][2];//compactification
	gdouble compact_moments[4];//compact image moments
	gdouble Tensors[3];//three normalization tensors
	gdouble angle;//rotation angle


};

struct _PreviousBlobs{
	/*
	 * will contain the blobs from previous frame
	 * so we can perform the comparison to determine
	 * the similarity between blobs and make a more
	 * accurate estimation of the blob motion and then
	 * output the log "img_proc" or "No img_proc"*/
	guint8 *NormalizedBlob;//data contained in the blob
	guint32 width,height;//size of the blob with PADDING
	guint32 paddinx,paddiny;//corresponding paddings
};

struct _GstMomentNormalization
{
	GstVideoFilter base_momentnormalization;

	//info needed for the element only
	gint patch_normalization;
	gint padding_x, padding_y;

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

	/*
	 * Previous blobs to be used for comparison and determine
	 * motion in the incoming frame
	 * */
	PreviousBlobs *Prev_Norm_Blobs;
	gboolean any_blob, any_buffer;
	guint n_prev_blobs;
	guint64 diff_count, diff_count_nonskin;
	gfloat diff_acumm_skin, diff_acumm_nonskin;
	GstBuffer *prevBuffer;

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
	BLOBMOMENT *blobs;
	guint8 n_blobs;
	guint32 BBh;
	guint32 BBw;
	guint32 BBx;
	guint32 BBy;

	/*
	 * In charge of tell us how many blobs were used for difference estimation
	 * */
	gint n_diff_blobs;


	/*
	 * File in charge of keeping the log
	 * for this particular streaming
	 * Format -> Time : Status
	 * */
	FILE *img_proc_Log;
	gint output_logfile;
	gchar *location;

	/* Processing time for every frame */
	gfloat internal_pt;//internal processing time
};

struct _GstMomentNormalizationClass
{
  GstVideoFilterClass base_momentnormalization_class;
};

GType gst_momentnormalization_get_type (void);

G_END_DECLS

#endif

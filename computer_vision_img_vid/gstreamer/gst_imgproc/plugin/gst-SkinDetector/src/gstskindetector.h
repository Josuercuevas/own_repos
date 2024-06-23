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

#ifndef _GST_SKINDETECTOR_H_
#define _GST_SKINDETECTOR_H_

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/video/gstvideofilter.h>
#include <gst/gstutils.h>

G_BEGIN_DECLS

#define GST_TYPE_SKINDETECTOR   (gst_skindetector_get_type())
#define GST_SKINDETECTOR(obj)   (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_SKINDETECTOR,GstSkinDetector))
#define GST_SKINDETECTOR_CLASS(klass)   (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_SKINDETECTOR,GstSkinDetectorClass))
#define GST_IS_SKINDETECTOR(obj)   (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_SKINDETECTOR))
#define GST_IS_SKINDETECTOR_CLASS(obj)   (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_SKINDETECTOR))

#define NMASKS (7)//check filter mask for details

typedef struct _GstSkinDetector GstSkinDetector;
typedef struct _GstSkinDetectorClass GstSkinDetectorClass;
typedef struct _FilterMask FilterMask;

struct _FilterMask{
	/*
	 * Every mask will have a different resolution
	 * therefore we need to signal this in a different
	 * bit in the pixel, the higher the resolution,
	 * the larger the bit to be activated:
	 *
	 * Idea:
	 * skin-detector (bit 0)
	 * 1/2 resolution (bit 1) 2
	 * 1/4 resolution (bit 2) 4
	 * 1/8 resolution (bit 3) 8
	 * 1/16 resolution (bit 4) 16
	 * 1/32 resolution (bit 5) 32
	 * 1/64 resolution (bit 6) 64
	 * 1/128 resolution (bit 7) 128
	 * */
	guint Mask_Hsize, Mask_Wsize;//window size to be used
	guint Cx, Cy;//centers from which the extraction has to take place
	guint8 mask;//mask to be used when setting the values of the pixels
};

struct _GstSkinDetector
{
	GstVideoFilter base_skindetector;

	//to check if the user wants to normalize the output frame
	gint normalize_frame;

	/*
	* this will point to the original data to be used for the
	* skin detection part
	* */
	GstVideoFrame * inframe;

	/*
	 * This will contain the resulting detection,
	 * size is the same as the original image, but the values
	 * are just 0 or 1 for detection or no detection
	 */
	GstVideoFrame * outframe;

	/*
	 * General information of the image that has to be provided by the
	 * user, or could be set by Gstreamer
	 * */
	guint space;

	/*
	 * Masks and shifts for fast implementation
	 * Endianness detector
	 * */
	guint32 RedMask, GreenMask, BlueMask;
	guint8 RedShift, GreenShift, BlueShift;
	gboolean LittleE, Construct_Frame_masks;

	/*
	 * Masks to be used when detecting texture, we will use the available
	 * bits in the output buffer
	 * */
	FilterMask masks[NMASKS];//since we have 7 available bits in the skin image
	gboolean make_masks;//just a flag to signal if the mask info has to be built
	gfloat text_thres;

	/*
	 * For histogram extraction which is later going to be used for threshold
	 * adaptation (all normalized)
	 * */
	gfloat *H_histo, *S_histo, *V_histo;
	gfloat *R_histo, *G_histo, *B_histo;
	guint tot_count;
	gfloat *prior_H_histo, *prior_S_histo, *prior_V_histo;
	gint R_min, R_max, G_min, G_max, B_min, B_max;

	/*
	 * Threshold values that should be updated periodically,
	 * according the HSV coordinate and AWB histogram merge
	 * approach (current frame + prior patch)
	 * */
	gfloat Hmin, Hmax;
	gfloat Smin, Smax;
	gfloat Vmin, Vmax;
	gboolean empty_frame;

	/*in order to restart thresholds*/
	gfloat oHmin, oHmax;
	gfloat oSmin, oSmax;
	gfloat oVmin, oVmax;

	/*merging factor to be used*/
	gfloat merging_factor;
	gboolean adaptive;

	/* to check if is the first time we load a model */
	gboolean firs_time;

	/*
	 * To load the model "template" histograms without restarting
	 * the pipeline. Specially when external applications do update
	 * histograms and HSV thresholds
	 * */
	GstControlSource *skin_control_source;//to bind the source files
	gboolean template_load;//signals the need to load new histograms and template

	/* property to be queried from outside the element */
	gfloat internal_PT_skin;//skin processing time for a frame
	gfloat internal_PT_texture;//texture processing time for a frame
};

struct _GstSkinDetectorClass
{
  GstVideoFilterClass base_skindetector_class;
};


GType gst_skindetector_get_type (void);

G_END_DECLS

#endif

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

#ifndef _GST_DISTANCETRANSFORM_H_
#define _GST_DISTANCETRANSFORM_H_

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/video/gstvideofilter.h>

G_BEGIN_DECLS

#define GST_TYPE_DISTANCETRANSFORM   (gst_distancetransform_get_type())
#define GST_DISTANCETRANSFORM(obj)   (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_DISTANCETRANSFORM,GstDistanceTransform))
#define GST_DISTANCETRANSFORM_CLASS(klass)   (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_DISTANCETRANSFORM,GstDistanceTransformClass))
#define GST_IS_DISTANCETRANSFORM(obj)   (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_DISTANCETRANSFORM))
#define GST_IS_DISTANCETRANSFORM_CLASS(obj)   (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_DISTANCETRANSFORM))

typedef struct _GstDistanceTransform GstDistanceTransform;
typedef struct _GstDistanceTransformClass GstDistanceTransformClass;

struct _GstDistanceTransform{
	GstVideoFilter DT_base;

	/* incoming frame after normalization */
	GstVideoFrame *inframe;

	/* ongoing frame after transformation */
	GstVideoFrame *outframe;

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
	 * For the distance transform estimation, we dont want to get
	 * skeleton beyond the BBs
	 * */
	guint32 BBh;
	guint32 BBw;
	guint32 BBx;
	guint32 BBy;

	gint normalize;
};

struct _GstDistanceTransformClass{
	GstVideoFilterClass DT_class;
};

GType gst_distancetransform_get_type (void);

G_END_DECLS

#endif

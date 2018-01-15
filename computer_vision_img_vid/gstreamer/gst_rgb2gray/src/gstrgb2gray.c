/* GStreamer
 * Copyright (C) 2015 FIXME <fixme@example.com>
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/video/gstvideofilter.h>
#include "gstrgb2gray.h"

GST_DEBUG_CATEGORY_STATIC (gst_rgb2gray_debug_category);
#define GST_CAT_DEFAULT gst_rgb2gray_debug_category

/* prototypes */


static void gst_rgb2gray_set_property (GObject * object,
    guint property_id, const GValue * value, GParamSpec * pspec);
static void gst_rgb2gray_get_property (GObject * object,
    guint property_id, GValue * value, GParamSpec * pspec);
static void gst_rgb2gray_dispose (GObject * object);
static void gst_rgb2gray_finalize (GObject * object);

static gboolean gst_rgb2gray_start (GstBaseTransform * trans);
static gboolean gst_rgb2gray_stop (GstBaseTransform * trans);
static gboolean gst_rgb2gray_set_info (GstVideoFilter * filter, GstCaps * incaps,
    GstVideoInfo * in_info, GstCaps * outcaps, GstVideoInfo * out_info);
static GstFlowReturn gst_rgb2gray_transform_frame (GstVideoFilter * filter,
    GstVideoFrame * inframe, GstVideoFrame * outframe);
static GstFlowReturn gst_rgb2gray_transform_frame_ip (GstVideoFilter * filter,
    GstVideoFrame * frame);

enum
{
  PROP_0
};

/* pad templates */

/* FIXME: add/remove formats you can handle */
#define VIDEO_SRC_CAPS \
    GST_VIDEO_CAPS_MAKE("{ I420, Y444, Y42B, UYVY, RGBA }")

/* FIXME: add/remove formats you can handle */
#define VIDEO_SINK_CAPS \
    GST_VIDEO_CAPS_MAKE("{ I420, Y444, Y42B, UYVY, RGBA }")


/* class initialization */

G_DEFINE_TYPE_WITH_CODE (GstRgb2gray, gst_rgb2gray, GST_TYPE_VIDEO_FILTER,
  GST_DEBUG_CATEGORY_INIT (gst_rgb2gray_debug_category, "rgb2gray", 0,
  "debug category for rgb2gray element"));

static void
gst_rgb2gray_class_init (GstRgb2grayClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS (klass);
  GstVideoFilterClass *video_filter_class = GST_VIDEO_FILTER_CLASS (klass);

  /* Setting up pads and setting metadata should be moved to
     base_class_init if you intend to subclass this class. */
  gst_element_class_add_pad_template (GST_ELEMENT_CLASS(klass),
      gst_pad_template_new ("src", GST_PAD_SRC, GST_PAD_ALWAYS,
        gst_caps_from_string (VIDEO_SRC_CAPS)));
  gst_element_class_add_pad_template (GST_ELEMENT_CLASS(klass),
      gst_pad_template_new ("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
        gst_caps_from_string (VIDEO_SINK_CAPS)));

  gst_element_class_set_static_metadata (GST_ELEMENT_CLASS(klass),
      "FIXME Long name", "Generic", "FIXME Description",
      "FIXME <fixme@example.com>");

  gobject_class->set_property = gst_rgb2gray_set_property;
  gobject_class->get_property = gst_rgb2gray_get_property;
  gobject_class->dispose = gst_rgb2gray_dispose;
  gobject_class->finalize = gst_rgb2gray_finalize;
  base_transform_class->start = GST_DEBUG_FUNCPTR (gst_rgb2gray_start);
  base_transform_class->stop = GST_DEBUG_FUNCPTR (gst_rgb2gray_stop);
  video_filter_class->set_info = GST_DEBUG_FUNCPTR (gst_rgb2gray_set_info);
  video_filter_class->transform_frame = GST_DEBUG_FUNCPTR (gst_rgb2gray_transform_frame);
  video_filter_class->transform_frame_ip = GST_DEBUG_FUNCPTR (gst_rgb2gray_transform_frame_ip);

}

static void
gst_rgb2gray_init (GstRgb2gray *rgb2gray)
{
}

void
gst_rgb2gray_set_property (GObject * object, guint property_id,
    const GValue * value, GParamSpec * pspec)
{
  GstRgb2gray *rgb2gray = GST_RGB2GRAY (object);

  GST_DEBUG_OBJECT (rgb2gray, "set_property");

  switch (property_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
      break;
  }
}

void
gst_rgb2gray_get_property (GObject * object, guint property_id,
    GValue * value, GParamSpec * pspec)
{
  GstRgb2gray *rgb2gray = GST_RGB2GRAY (object);

  GST_DEBUG_OBJECT (rgb2gray, "get_property");

  switch (property_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
      break;
  }
}

void
gst_rgb2gray_dispose (GObject * object)
{
  GstRgb2gray *rgb2gray = GST_RGB2GRAY (object);

  GST_DEBUG_OBJECT (rgb2gray, "dispose");

  /* clean up as possible.  may be called multiple times */

  G_OBJECT_CLASS (gst_rgb2gray_parent_class)->dispose (object);
}

void
gst_rgb2gray_finalize (GObject * object)
{
  GstRgb2gray *rgb2gray = GST_RGB2GRAY (object);

  GST_DEBUG_OBJECT (rgb2gray, "finalize");

  /* clean up object here */

  G_OBJECT_CLASS (gst_rgb2gray_parent_class)->finalize (object);
}

static gboolean
gst_rgb2gray_start (GstBaseTransform * trans)
{
  GstRgb2gray *rgb2gray = GST_RGB2GRAY (trans);

  GST_DEBUG_OBJECT (rgb2gray, "start");

  return TRUE;
}

static gboolean
gst_rgb2gray_stop (GstBaseTransform * trans)
{
  GstRgb2gray *rgb2gray = GST_RGB2GRAY (trans);

  GST_DEBUG_OBJECT (rgb2gray, "stop");

  return TRUE;
}

static gboolean
gst_rgb2gray_set_info (GstVideoFilter * filter, GstCaps * incaps,
    GstVideoInfo * in_info, GstCaps * outcaps, GstVideoInfo * out_info)
{
  GstRgb2gray *rgb2gray = GST_RGB2GRAY (filter);

  GST_DEBUG_OBJECT (rgb2gray, "set_info");

  return TRUE;
}

/* transform */
static GstFlowReturn
gst_rgb2gray_transform_frame (GstVideoFilter * filter, GstVideoFrame * inframe,
    GstVideoFrame * outframe)
{
  GstRgb2gray *rgb2gray = GST_RGB2GRAY (filter);

  gint src_stride, dst_stride, height;
  guint8 *src_data=NULL, *dst_data=NULL;

  height=GST_VIDEO_FRAME_HEIGHT(inframe);

  GST_DEBUG_OBJECT (rgb2gray, "transform_frame");

  /*RGB frame*/
  src_data = GST_VIDEO_FRAME_PLANE_DATA(inframe, 0);
  src_stride = GST_VIDEO_FRAME_COMP_STRIDE(inframe, 0);

  /*GRAY-intensity*/
  dst_data = GST_VIDEO_FRAME_PLANE_DATA(outframe, 0);
  dst_stride = GST_VIDEO_FRAME_COMP_STRIDE(outframe, 0);

  if(src_stride!=dst_stride){
	  GST_ERROR_OBJECT(GST_OBJECT_CAST(rgb2gray), "Strides don't match check your source and sink negotiation..!!");
	  return GST_FLOW_ERROR;
  }

  gint position=0, frame_size=height*src_stride;
  guint8 intensity;
  while(position<frame_size){
	  intensity = (guint8)fmin( ((float)(*(src_data+position))*0.2989 + (float)(*(src_data+position+1))*0.5870 + (float)(*(src_data+position+2))*0.1140),
			  255.0);

	  // if((*(src_data+position))>=128 && (*(src_data+position))<=154){
		//   (*(dst_data+position)) = 255;
		//   (*(dst_data+position+1)) = 0;
		//   (*(dst_data+position+2)) = 0;
	  // }else{
		//   (*(dst_data+position)) = 0;
		//   (*(dst_data+position+1)) = 0;
		//   (*(dst_data+position+2)) = 0;
	  // }
	  position+=4;
  }

  return GST_FLOW_OK;
}

static GstFlowReturn
gst_rgb2gray_transform_frame_ip (GstVideoFilter * filter, GstVideoFrame * frame)
{
  GstRgb2gray *rgb2gray = GST_RGB2GRAY (filter);

  GST_DEBUG_OBJECT (rgb2gray, "transform_frame_ip");

  return GST_FLOW_OK;
}

static gboolean
plugin_init (GstPlugin * plugin)
{

  /* FIXME Remember to set the rank if it's an element that is meant
     to be autoplugged by decodebin. */
  return gst_element_register (plugin, "rgb2gray", GST_RANK_NONE,
      GST_TYPE_RGB2GRAY);
}

/* FIXME: these are normally defined by the GStreamer build system.
   If you are creating an element to be included in gst-plugins-*,
   remove these, as they're always defined.  Otherwise, edit as
   appropriate for your external plugin package. */
#ifndef VERSION
#define VERSION "0.0.FIXME"
#endif
#ifndef PACKAGE
#define PACKAGE "FIXME_package"
#endif
#ifndef PACKAGE_NAME
#define PACKAGE_NAME "FIXME_package_name"
#endif
#ifndef GST_PACKAGE_ORIGIN
#define GST_PACKAGE_ORIGIN "http://FIXME.org/"
#endif

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    rgb2gray,
    "FIXME plugin description",
    plugin_init, VERSION, "LGPL", PACKAGE_NAME, GST_PACKAGE_ORIGIN)

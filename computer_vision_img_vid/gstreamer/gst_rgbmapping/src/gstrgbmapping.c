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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/video/gstvideofilter.h>
#include <stdio.h>
#include <gst/gstutils.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include "gstrgbmapping.h"

GST_DEBUG_CATEGORY_STATIC (gst_rgbmapping_debug_category);
#define GST_CAT_DEFAULT gst_rgbmapping_debug_category

/* prototypes */
#define SAMPLES_EXTRACTED (256)


static void gst_rgbmapping_set_property (GObject * object,
    guint property_id, const GValue * value, GParamSpec * pspec);
static void gst_rgbmapping_get_property (GObject * object,
    guint property_id, GValue * value, GParamSpec * pspec);
static void gst_rgbmapping_dispose (GObject * object);
static void gst_rgbmapping_finalize (GObject * object);

static gboolean gst_rgbmapping_start (GstBaseTransform * trans);
static gboolean gst_rgbmapping_stop (GstBaseTransform * trans);
static gboolean gst_rgbmapping_set_info (GstVideoFilter * filter, GstCaps * incaps,
    GstVideoInfo * in_info, GstCaps * outcaps, GstVideoInfo * out_info);
static GstFlowReturn gst_rgbmapping_transform_frame (GstVideoFilter * filter,
    GstVideoFrame * inframe, GstVideoFrame * outframe);
static GstFlowReturn gst_rgbmapping_transform_frame_ip (GstVideoFilter * filter,
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

G_DEFINE_TYPE_WITH_CODE (GstRgbmapping, gst_rgbmapping, GST_TYPE_VIDEO_FILTER,
  GST_DEBUG_CATEGORY_INIT (gst_rgbmapping_debug_category, "rgbmapping", 0,
  "debug category for rgbmapping element"));

static void
gst_rgbmapping_class_init (GstRgbmappingClass * klass)
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

  gobject_class->set_property = gst_rgbmapping_set_property;
  gobject_class->get_property = gst_rgbmapping_get_property;
  gobject_class->dispose = gst_rgbmapping_dispose;
  gobject_class->finalize = gst_rgbmapping_finalize;
  base_transform_class->start = GST_DEBUG_FUNCPTR (gst_rgbmapping_start);
  base_transform_class->stop = GST_DEBUG_FUNCPTR (gst_rgbmapping_stop);
  video_filter_class->set_info = GST_DEBUG_FUNCPTR (gst_rgbmapping_set_info);
  video_filter_class->transform_frame = GST_DEBUG_FUNCPTR (gst_rgbmapping_transform_frame);
  video_filter_class->transform_frame_ip = GST_DEBUG_FUNCPTR (gst_rgbmapping_transform_frame_ip);

}

static void
gst_rgbmapping_init (GstRgbmapping *rgbmapping)
{
	GST_DEBUG_OBJECT(GST_OBJECT_CAST(rgbmapping), "Creating Mapping table..!!");
	rgbmapping->mapping_table=NULL;
	rgbmapping->mapping_table = (guint8*)g_malloc(sizeof(guint8)*(SAMPLES_EXTRACTED*3));
	if(!rgbmapping->mapping_table){
		GST_ERROR_OBJECT(GST_OBJECT_CAST(rgbmapping), "CANNOT CREATE MEMORY FOR MAPPING TABLE..!!");
	}
}

void
gst_rgbmapping_set_property (GObject * object, guint property_id,
    const GValue * value, GParamSpec * pspec)
{
  GstRgbmapping *rgbmapping = GST_RGBMAPPING (object);

  GST_DEBUG_OBJECT (rgbmapping, "set_property");

  switch (property_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
      break;
  }
}

void
gst_rgbmapping_get_property (GObject * object, guint property_id,
    GValue * value, GParamSpec * pspec)
{
  GstRgbmapping *rgbmapping = GST_RGBMAPPING (object);

  GST_DEBUG_OBJECT (rgbmapping, "get_property");

  switch (property_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
      break;
  }
}

void
gst_rgbmapping_dispose (GObject * object)
{
  GstRgbmapping *rgbmapping = GST_RGBMAPPING (object);

  GST_DEBUG_OBJECT (rgbmapping, "dispose");

  /* clean up as possible.  may be called multiple times */

  G_OBJECT_CLASS (gst_rgbmapping_parent_class)->dispose (object);
}

void
gst_rgbmapping_finalize (GObject * object)
{
  GstRgbmapping *rgbmapping = GST_RGBMAPPING (object);

  GST_DEBUG_OBJECT (rgbmapping, "finalize");

  /* clean up object here */
  if(rgbmapping->mapping_table){
	  g_free(rgbmapping->mapping_table);
	  rgbmapping->mapping_table=NULL;
  }

  G_OBJECT_CLASS (gst_rgbmapping_parent_class)->finalize (object);
}

static gboolean
gst_rgbmapping_start (GstBaseTransform * trans)
{
  GstRgbmapping *rgbmapping = GST_RGBMAPPING (trans);

  GST_DEBUG_OBJECT (rgbmapping, "start");

  if(rgbmapping->mapping_table){
	  /*load mapping table*/
		FILE *hsv_table=NULL;
		hsv_table=fopen("pseudocolor_table", "r+b");
		if(!hsv_table){
			GST_ERROR_OBJECT(GST_OBJECT_CAST(rgbmapping), "Cannot open pseudo-color file: %s",
					"pseudocolor_table");
			return FALSE;
		}else{
			/*reading the whole file at once*/
			gint count = fread((void*)rgbmapping->mapping_table, sizeof(guint8), SAMPLES_EXTRACTED*3, hsv_table);

			fclose(hsv_table);
		}
  }

  return TRUE;
}

static gboolean
gst_rgbmapping_stop (GstBaseTransform * trans)
{
  GstRgbmapping *rgbmapping = GST_RGBMAPPING (trans);

  GST_DEBUG_OBJECT (rgbmapping, "stop");

  return TRUE;
}

static gboolean
gst_rgbmapping_set_info (GstVideoFilter * filter, GstCaps * incaps,
    GstVideoInfo * in_info, GstCaps * outcaps, GstVideoInfo * out_info)
{
  GstRgbmapping *rgbmapping = GST_RGBMAPPING (filter);

  GST_DEBUG_OBJECT (rgbmapping, "set_info");

  return TRUE;
}

/* transform */
static GstFlowReturn
gst_rgbmapping_transform_frame (GstVideoFilter * filter, GstVideoFrame * inframe,
    GstVideoFrame * outframe)
{
  GstRgbmapping *rgbmapping = GST_RGBMAPPING (filter);

  GST_DEBUG_OBJECT (rgbmapping, "transform_frame");


  if(rgbmapping->mapping_table){
	  /*we can do mapping*/
		gint src_stride, dst_stride, height;
		guint8 *src_data=NULL, *dst_data=NULL;

		height=GST_VIDEO_FRAME_HEIGHT(inframe);

		/*RGB frame*/
		src_data = GST_VIDEO_FRAME_PLANE_DATA(inframe, 0);
		src_stride = GST_VIDEO_FRAME_COMP_STRIDE(inframe, 0);

		/*GRAY-intensity*/
		dst_data = GST_VIDEO_FRAME_PLANE_DATA(outframe, 0);
		dst_stride = GST_VIDEO_FRAME_COMP_STRIDE(outframe, 0);

		if(src_stride!=dst_stride){
		  GST_ERROR_OBJECT(GST_OBJECT_CAST(rgbmapping), "Strides don't match check your source and sink negotiation..!!");
		  return GST_FLOW_ERROR;
		}

		gint position=0, frame_size=height*src_stride;
		gfloat intensity;
		gint pos;
		while(position<frame_size){
		  intensity = fmin( ((float)(*(src_data+position))*0.2989 + (float)(*(src_data+position+1))*0.5870 + (float)(*(src_data+position+2))*0.1140),
				  255.0);

		  if(SAMPLES_EXTRACTED>256){
			  /*half pixel*/
			  if((intensity - (gint)(intensity))<0.5){
				  pos = ((gint)(intensity))*2;
			  }else{
				  pos = ((gint)(intensity))*2 + 1;
			  }
		  }else{
			  pos = ((gint)(intensity));
		  }

		  (*(dst_data+position)) = rgbmapping->mapping_table[pos*3];/*R*/
		  (*(dst_data+position+1)) = rgbmapping->mapping_table[pos*3+1];/*G*/
		  (*(dst_data+position+2)) = rgbmapping->mapping_table[pos*3+2];/*B*/

		  if((*(dst_data+position))>=0 && (*(dst_data+position))<=50 && (*(dst_data+position+1))>=48 && (*(dst_data+position+2))){
			  (*(dst_data+position)) = 255;
			  (*(dst_data+position+1)) = 255;
			  (*(dst_data+position+2)) = 255;
		  }else{
			  (*(dst_data+position)) = 0;
			  (*(dst_data+position+1)) = 0;
			  (*(dst_data+position+2)) = 0;
		  }

		  position+=4;
		}
  }else{
	  GST_ERROR_OBJECT(GST_OBJECT_CAST(rgbmapping), "No Mapping table..!!");
	  return GST_FLOW_ERROR;
  }

  return GST_FLOW_OK;
}

static GstFlowReturn
gst_rgbmapping_transform_frame_ip (GstVideoFilter * filter, GstVideoFrame * frame)
{
  GstRgbmapping *rgbmapping = GST_RGBMAPPING (filter);

  GST_DEBUG_OBJECT (rgbmapping, "transform_frame_ip");

  return GST_FLOW_OK;
}

static gboolean
plugin_init (GstPlugin * plugin)
{

  /* FIXME Remember to set the rank if it's an element that is meant
     to be autoplugged by decodebin. */
  return gst_element_register (plugin, "rgbmapping", GST_RANK_NONE,
      GST_TYPE_RGBMAPPING);
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
    rgbmapping,
    "FIXME plugin description",
    plugin_init, VERSION, "LGPL", PACKAGE_NAME, GST_PACKAGE_ORIGIN)

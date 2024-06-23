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
 * SECTION:element-gstcolorretinex
 *
 * The colorretinex element does FIXME stuff.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v fakesrc ! colorretinex ! FIXME ! fakesink
 * ]|
 * FIXME Describe what the pipeline does.
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/video/gstvideofilter.h>
#include "gstcolorretinex.h"
#include "includes/RetinexLib.h"
#include "includes/norm.h"

GST_DEBUG_CATEGORY_STATIC (gst_colorretinex_debug_category);
#define GST_CAT_DEFAULT gst_colorretinex_debug_category

#define RETINEX_THRES_DEFAULT (4)

/* prototypes */
static void gst_colorretinex_set_property (GObject * object, guint property_id,
		const GValue * value, GParamSpec * pspec);
static void gst_colorretinex_get_property (GObject * object, guint property_id,
		GValue * value, GParamSpec * pspec);
static void gst_colorretinex_dispose (GObject * object);
static void gst_colorretinex_finalize (GObject * object);

static gboolean gst_colorretinex_start (GstBaseTransform * trans);
static gboolean gst_colorretinex_stop (GstBaseTransform * trans);
static gboolean gst_colorretinex_set_info (GstVideoFilter * filter, GstCaps * incaps,
		GstVideoInfo * in_info, GstCaps * outcaps, GstVideoInfo * out_info);
static GstFlowReturn gst_colorretinex_transform_frame (GstVideoFilter * filter,
		GstVideoFrame * inframe, GstVideoFrame * outframe);

//for caps negotiation
static gboolean gst_colorretinex_set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps);

enum
{
	PROP_0,
	RETINEX_THRESHOLD
};


/*
 * Supported image types for the retinex routine
 *  */
#define VIDEO_SRC_CAPS GST_VIDEO_CAPS_MAKE("{ RGBx, BGRx, xRGB, xBGR, RGBA, BGRA, ARGB, ABGR }")

/*
 * Supported image types by the retinex routine
 *  */
#define VIDEO_SINK_CAPS GST_VIDEO_CAPS_MAKE("{ RGBx, BGRx, xRGB, xBGR, RGBA, BGRA, ARGB, ABGR }")


/*
 * class initialization to define everything to be used in this
 * element for Gstreamer implementation
 * */

G_DEFINE_TYPE_WITH_CODE (GstColorRetinex, gst_colorretinex, GST_TYPE_VIDEO_FILTER,
		GST_DEBUG_CATEGORY_INIT (gst_colorretinex_debug_category, "colorretinex", 0,
				"debug category for colorretinex element"));

static void gst_colorretinex_class_init (GstColorRetinexClass * Retinex){
  GObjectClass *gobject_class = G_OBJECT_CLASS (Retinex);
  GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS (Retinex);
  GstVideoFilterClass *video_filter_class = GST_VIDEO_FILTER_CLASS (Retinex);


  gobject_class->set_property = GST_DEBUG_FUNCPTR (gst_colorretinex_set_property);
  gobject_class->get_property = GST_DEBUG_FUNCPTR (gst_colorretinex_get_property);
  gobject_class->dispose = GST_DEBUG_FUNCPTR (gst_colorretinex_dispose);
  gobject_class->finalize = GST_DEBUG_FUNCPTR (gst_colorretinex_finalize);
  base_transform_class->start = GST_DEBUG_FUNCPTR (gst_colorretinex_start);
  base_transform_class->stop = GST_DEBUG_FUNCPTR (gst_colorretinex_stop);
  base_transform_class->set_caps = GST_DEBUG_FUNCPTR(gst_colorretinex_set_caps);
  video_filter_class->set_info = GST_DEBUG_FUNCPTR (gst_colorretinex_set_info);
  video_filter_class->transform_frame = GST_DEBUG_FUNCPTR (gst_colorretinex_transform_frame);



  gst_element_class_set_static_metadata (GST_ELEMENT_CLASS(Retinex),
      "Color Constancy routine to enhance frame for better detection when using color space",
	  "VideoFilter", "Using a predefined threshold [0..255] the Retinex routine has the main "
	  "purpose of enhancing the frames details in the color space RGB or YUV",
      "somecpmpnay <josuercuevas@gmail.com>");


  /*
   * Only parameter to be used in the retinex routine
   * */
  g_object_class_install_property (GST_ELEMENT_CLASS(Retinex), RETINEX_THRESHOLD, g_param_spec_int("thres", "thres",
		  "Retinex Threshold to be used in the computation, default=4, range [0...255]",
		  0, 100, RETINEX_THRES_DEFAULT, G_PARAM_READWRITE));


  /*
   * Setting up pads and setting metadata should be moved to
   * base_class_init if you intend to subclass this class.
   * */
	gst_element_class_add_pad_template (GST_ELEMENT_CLASS(Retinex), gst_pad_template_new ("src", GST_PAD_SRC, GST_PAD_ALWAYS,
		  gst_caps_from_string (VIDEO_SRC_CAPS)));
	gst_element_class_add_pad_template (GST_ELEMENT_CLASS(Retinex),  gst_pad_template_new ("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
		  gst_caps_from_string (VIDEO_SINK_CAPS)));

}

/*
 * Default parameters used for the Color Retinex routine
 * */
static void gst_colorretinex_init (GstColorRetinex *colorretinex){
	/*
	 * Setting the default parameter value for the retinex
	 * threshold
	 * */
	colorretinex->indata = NULL;
	colorretinex->outdata = NULL;
	colorretinex->RetinexThreshold = RETINEX_THRES_DEFAULT;
	colorretinex->base_colorretinex.negotiated = FALSE;
}

/*
 * Setting the property value for the parameters of the
 * Retinex algorithm
 * */
void gst_colorretinex_set_property (GObject * object, guint property_id,
    const GValue * value, GParamSpec * pspec){

	GstColorRetinex *colorretinex = GST_COLORRETINEX (object);

	GST_DEBUG_OBJECT (colorretinex, "set_property");

	switch (property_id) {
		case RETINEX_THRESHOLD://RETINEX properties
			colorretinex->RetinexThreshold = g_value_get_int(value);
			break;
		default:
			G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
			break;
	}
}

/*
 * Getting the property value for the parameters of the
 * Retinex algorithm
 * */
void gst_colorretinex_get_property (GObject * object, guint property_id,
    GValue * value, GParamSpec * pspec){

	GstColorRetinex *colorretinex = GST_COLORRETINEX (object);

	GST_DEBUG_OBJECT (colorretinex, "get_property");

	switch (property_id) {
		case RETINEX_THRESHOLD://RETINEX properties
			g_value_set_int(value, colorretinex->RetinexThreshold);
			break;
		default:
			G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
			break;
	}
}

void gst_colorretinex_dispose (GObject * object){
	GstColorRetinex *colorretinex = GST_COLORRETINEX (object);

	GST_DEBUG_OBJECT (colorretinex, "dispose");

	G_OBJECT_CLASS (gst_colorretinex_parent_class)->dispose (object);
}

void gst_colorretinex_finalize (GObject * object){
	GstColorRetinex *colorretinex = GST_COLORRETINEX (object);

	GST_DEBUG_OBJECT (colorretinex, "finalize");

	G_OBJECT_CLASS (gst_colorretinex_parent_class)->finalize (object);
}

static gboolean gst_colorretinex_start (GstBaseTransform * trans)
{
  GstColorRetinex *colorretinex = GST_COLORRETINEX (trans);

  GST_DEBUG_OBJECT (colorretinex, "start");

  return TRUE;
}

static gboolean gst_colorretinex_stop (GstBaseTransform * trans){
	GstColorRetinex *colorretinex = GST_COLORRETINEX (trans);

	GST_DEBUG_OBJECT (colorretinex, "stop: freeing used memory");


	/* clean up object here */
	if(colorretinex->indata)
		g_free(colorretinex->indata);

	if(colorretinex->outdata)
		g_free(colorretinex->outdata);

	colorretinex->indata = NULL;
	colorretinex->outdata = NULL;


	return TRUE;
}


static gboolean gst_colorretinex_set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps){
	GstVideoFilter *filter = GST_VIDEO_FILTER_CAST (trans);
	GstVideoFilterClass *fclass;
	GstVideoInfo in_info, out_info;
	gboolean res;
	const gchar *src_format, *sink_format;
	gchar *src_color, *sink_color;

	GST_DEBUG_OBJECT(filter, "Caps negotiation Retinex..!!");

	GST_OBJECT_LOCK(filter);
	incaps = gst_caps_make_writable(incaps);
	outcaps = gst_caps_make_writable(outcaps);

	/* input caps */
	if (!gst_video_info_from_caps(&in_info, incaps))
	goto invalid_caps;

	/* output caps */
	if (!gst_video_info_from_caps(&out_info, outcaps))
	goto invalid_caps;

	fclass = GST_VIDEO_FILTER_GET_CLASS(filter);
	if(fclass->set_info)
		res = fclass->set_info(filter, incaps, &in_info, outcaps, &out_info);
	else
		res = TRUE;


	sink_format = gst_video_format_to_string (in_info.finfo->format);
	incaps = gst_caps_new_simple ("video/x-raw",
	      "format", G_TYPE_STRING, sink_format,
	      "width", G_TYPE_INT, in_info.width,
	      "height", G_TYPE_INT, in_info.height,
	      "pixel-aspect-ratio", GST_TYPE_FRACTION, in_info.par_n, in_info.par_d, NULL);

	src_format = gst_video_format_to_string (out_info.finfo->format);
	outcaps = gst_caps_new_simple ("video/x-raw",
	      "format", G_TYPE_STRING, src_format,
	      "width", G_TYPE_INT, out_info.width,
	      "height", G_TYPE_INT, out_info.height,
	      "pixel-aspect-ratio", GST_TYPE_FRACTION, out_info.par_n, out_info.par_d, NULL);

	if (res) {
		filter->in_info = in_info;
		filter->out_info = out_info;
		if (fclass->transform_frame == NULL)
			gst_base_transform_set_in_place(trans, TRUE);
		if (fclass->transform_frame_ip == NULL)
			GST_BASE_TRANSFORM_CLASS (fclass)->transform_ip_on_passthrough = FALSE;
	}

	//This is the part that will determine the negotiation result
	filter->negotiated = res;

	if(filter->negotiated)
		GST_DEBUG_OBJECT(filter, "Retinex VideoFilter has negotiated caps successfully..!!");

	GST_OBJECT_UNLOCK(filter);

	return res;

	/* ERRORS */
invalid_caps:
	{
		GST_ERROR_OBJECT(filter, "Caps couldn't be negotiated");
		filter->negotiated = FALSE;
		GST_OBJECT_UNLOCK(filter);
		return FALSE;
	}
}


static gboolean gst_colorretinex_set_info (GstVideoFilter * filter, GstCaps * incaps,
    GstVideoInfo * in_info, GstCaps * outcaps, GstVideoInfo * out_info){
	GstColorRetinex *colorretinex = GST_COLORRETINEX (filter);
	GST_DEBUG_OBJECT(colorretinex, "Setting info..!!");
	return TRUE;
}





guint8 min_val(guint8 a, guint8 b){
	guint8 c = a<b ? a : b;
	return c;
}

guint8 max_val(gfloat a, gfloat b){
	gfloat c = a>b ? a : b;
	return min_val((guint8)c, 255);
}


/*
 * Color Consistency routine (RETINEX) main function where the color space is to be enhanced for
 * better detection when performing further processes */
static GstFlowReturn gst_colorretinex_transform_frame (GstVideoFilter * filter, GstVideoFrame * inframe,
    GstVideoFrame * outframe){

	GstColorRetinex *colorretinex = GST_COLORRETINEX(filter);
	size_t width, height, framesize, cwidth, cheight,chromasize;
	guint8 *src[3], *dest[3];
	gint sstride[3], dstride[3], srcformat;
	guint x, y, pos=0;
	guint Ysrc, Usrc, Vsrc;//source image
	guint Ydest, Udest, Vdest;//destination image

	/*
	 * If we dont consider alpha we are going to loose it
	 * */
	guint32 RedMask, GreenMask, BlueMask, AlphaMask, temp_val[4];
	guint8 RedShift, GreenShift, BlueShift, AlphaShift;
	gint pix_stride=4;//IMPORTANT: ASUMING 32 bits


	if(!filter->negotiated){
		GST_DEBUG_OBJECT (colorretinex, "Caps have NOT been negotiated, proceeding to negotiation phase..!!");
		GstBaseTransform *colorretinexBaseTransform = GST_BASE_TRANSFORM(filter);
		GstVideoFilterClass *colorretinexclass = GST_COLORRETINEX_CLASS(filter);
		GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS(colorretinexclass);
		if(!base_transform_class->set_caps){
			GST_ERROR_OBJECT (colorretinex, "The caps negotiation have failed, closing application");
			return GST_FLOW_ERROR;
		}
	}

	//gets info of the frame to be processed
	src[0] = GST_VIDEO_FRAME_PLANE_DATA(inframe, 0);// R or Y
	src[1] = GST_VIDEO_FRAME_PLANE_DATA(inframe, 1);// G or U
	src[2] = GST_VIDEO_FRAME_PLANE_DATA(inframe, 2);// B or V
	dest[0] = GST_VIDEO_FRAME_PLANE_DATA(outframe, 0);
	dest[1] = GST_VIDEO_FRAME_PLANE_DATA(outframe, 1);
	dest[2] = GST_VIDEO_FRAME_PLANE_DATA(outframe, 2);

	//strides to be used per channel
	sstride[0] = GST_VIDEO_FRAME_PLANE_STRIDE (inframe, 0);
	sstride[1] = GST_VIDEO_FRAME_PLANE_STRIDE (inframe, 1);
	sstride[2] = GST_VIDEO_FRAME_PLANE_STRIDE (inframe, 2);
	dstride[0] = GST_VIDEO_FRAME_PLANE_STRIDE (outframe, 0);
	dstride[1] = GST_VIDEO_FRAME_PLANE_STRIDE (outframe, 1);
	dstride[2] = GST_VIDEO_FRAME_PLANE_STRIDE (outframe, 2);

	srcformat = GST_VIDEO_FRAME_FORMAT(inframe);

	width = sstride[0];
	height = GST_VIDEO_FRAME_HEIGHT(inframe);

	//============================ Retinex Implementation for this element =======================================//

	GST_DEBUG_OBJECT (colorretinex, "Implementing Retinex algorithm");
	if(srcformat==GST_VIDEO_FORMAT_I420){//Dealing with YUV frames of format 4
		/*
		 * We need to lock-mutex the access for this frame so we do not
		 * let another process to mess up with the data we are extracting
		 * and modifying (DONT USE THIS NOW)
		 * */
		GST_OBJECT_LOCK(colorretinex);

		cwidth = width>>1; cheight = height>>1;
		framesize = width*height; chromasize = cwidth*cheight;

		//Retinex asks for floating points, so we have to create this
		//just in case, enter just once
		if(colorretinex->indata==NULL){
			GST_DEBUG_OBJECT(colorretinex, "No memory has been created for the retinex, creating memory..!!");
			//remember is a binary image therefore is black and white
			colorretinex->indata = (gfloat*)g_malloc(sizeof(gfloat)*(framesize + 2*chromasize));//size of the frame YUV
			if(colorretinex->indata==NULL){
				GST_ERROR_OBJECT(colorretinex, "Frame memory was not created successfully");
				return GST_FLOW_ERROR;
			}
		}
		if(colorretinex->outdata==NULL){
			GST_DEBUG_OBJECT(colorretinex, "No memory has been created for the retinex, creating memory..!!");
			//remember is a binary image therefore is black and white
			colorretinex->outdata = (gfloat*)g_malloc(sizeof(gfloat)*(framesize + 2*chromasize));//size of the frame YUV
			if(colorretinex->outdata==NULL){
				GST_ERROR_OBJECT(colorretinex, "Frame memory was not created successfully");
				return GST_FLOW_ERROR;
			}
		}


		//cast the values from 8bits to float one-one correspondence
		for (y = 0; y < height; y++) {
			for (x = 0; x < width; x++) {
				//locations estimations
				Ysrc = y*sstride[0] + x;
				Usrc = (y/2)*sstride[1] + (x/2);
				Vsrc = (y/2)*sstride[2] + (x/2);
				colorretinex->indata[Ysrc] = (gfloat)src[0][Ysrc];//Y
				colorretinex->indata[framesize + Usrc] = (gfloat)src[1][Usrc];//U
				colorretinex->indata[framesize + chromasize + Vsrc] = (gfloat)src[2][Vsrc];//V
			}
		}

		//copy data
		memcpy(colorretinex->outdata, colorretinex->indata, sizeof(gfloat)*(framesize + 2*chromasize));


		//CALL RETINEX ROUTINE PER CHANNEL
		//Y
		time_t start=clock(), end;
		if (NULL == retinex_pde(colorretinex->outdata, width, height, (gfloat)colorretinex->RetinexThreshold)) {
			GST_ERROR_OBJECT(colorretinex, "the retinex PDE failed..!!\n");
			return GST_FLOW_ERROR;
		}
		normalize_mean_dt(colorretinex->outdata, colorretinex->indata, framesize);
		end=clock();
		GST_DEBUG_OBJECT(colorretinex, "time in Y: %f ms.\n", 1000*(((float)end-start)/CLOCKS_PER_SEC));

		//U
		start=clock();
		if (NULL == retinex_pde(colorretinex->outdata + framesize, cwidth, cheight, (gfloat)colorretinex->RetinexThreshold)) {
			GST_ERROR_OBJECT(colorretinex, "the retinex PDE failed..!!\n");
			return GST_FLOW_ERROR;
		}
		normalize_mean_dt(colorretinex->outdata + framesize, colorretinex->indata + framesize, chromasize);
		end=clock();
		GST_DEBUG_OBJECT(colorretinex, "time in U: %f ms.\n", 1000*(((float)end-start)/CLOCKS_PER_SEC));

		//V
		start=clock();
		if (NULL == retinex_pde(colorretinex->outdata + framesize + chromasize, cwidth, cheight, (gfloat)colorretinex->RetinexThreshold)) {
			GST_ERROR_OBJECT(colorretinex, "the retinex PDE failed..!!\n");
			return GST_FLOW_ERROR;
		}
		normalize_mean_dt(colorretinex->outdata + framesize + chromasize, colorretinex->indata + framesize + chromasize, chromasize);
		end=clock();
		GST_DEBUG_OBJECT(colorretinex, "time in V: %f ms.\n", 1000*(((float)end-start)/CLOCKS_PER_SEC));

		//casting back the values from float to 8bits
		for (y = 0; y < height; y++) {
			for (x = 0; x < width; x++) {
				//locations estimations
				Ysrc = y*sstride[0] + x;
				Ydest = y*dstride[0] + x;
				Usrc = (y/2)*sstride[1] + (x/2);
				Udest = (y/2)*dstride[1] + (x/2);
				Vsrc = (y/2)*sstride[2] + (x/2);
				Vdest = (y/2)*dstride[2] + (x/2);

				if(colorretinex->outdata[Ysrc]<0 || colorretinex->outdata[Ysrc]>255){
					colorretinex->outdata[Ysrc] = colorretinex->indata[Ysrc];
				}

				if(colorretinex->outdata[framesize + Usrc]<0 || colorretinex->outdata[framesize + Usrc]>255){
					colorretinex->outdata[framesize + Usrc] = colorretinex->indata[framesize + Usrc];
				}

				if(colorretinex->outdata[framesize + chromasize + Vsrc]<0 || colorretinex->outdata[framesize + chromasize + Vsrc]>255){
					colorretinex->outdata[framesize + chromasize + Vsrc] = colorretinex->indata[framesize + chromasize + Vsrc];
				}

				dest[0][Ydest] = min_val(max_val(colorretinex->outdata[Ysrc],0.0), 255);//Y
				dest[1][Udest] = min_val(max_val(colorretinex->outdata[framesize + Usrc],0.0), 255);//U
				dest[2][Vdest] = min_val(max_val(colorretinex->outdata[framesize + chromasize + Vsrc],0.0), 255);//V
			}
		}

		/*
		 * Unlocking the data for releasing the changes and output
		 * the modified frame
		 * */
		GST_OBJECT_UNLOCK(colorretinex);
	}else if(srcformat==GST_VIDEO_FORMAT_RGBx || srcformat==GST_VIDEO_FORMAT_RGBA || srcformat==GST_VIDEO_FORMAT_BGRx
			|| srcformat==GST_VIDEO_FORMAT_BGRA || srcformat==GST_VIDEO_FORMAT_xRGB || srcformat==GST_VIDEO_FORMAT_ARGB ||
			srcformat==GST_VIDEO_FORMAT_xBGR || srcformat==GST_VIDEO_FORMAT_ABGR){
		/*
		 * We need to lock-mutex the access for this frame so we do not
		 * let another process to mess up with the data we are extracting
		 * and modifying
		 * */
		GST_OBJECT_LOCK(colorretinex);

		framesize = width*height;

		/*
		 * Retinex asks for floating points, so we have to create this
		 * just in case, enter just once
		 * */
		if(colorretinex->indata==NULL){
			GST_DEBUG_OBJECT(colorretinex, "No memory has been created for the retinex, creating memory..!!");
			//remember is a binary image therefore is black and white
			colorretinex->indata = (gfloat*)g_malloc(sizeof(gfloat)*(framesize));//size of the frame YUV
			if(colorretinex->indata==NULL){
				GST_ERROR_OBJECT(colorretinex, "Frame memory was not created successfully");
				return GST_FLOW_ERROR;
			}
		}
		if(colorretinex->outdata==NULL){
			GST_DEBUG_OBJECT(colorretinex, "No memory has been created for the retinex, creating memory..!!");
			//remember is a binary image therefore is black and white
			colorretinex->outdata = (gfloat*)g_malloc(sizeof(gfloat)*(framesize));//size of the frame YUV
			if(colorretinex->outdata==NULL){
				GST_ERROR_OBJECT(colorretinex, "Frame memory was not created successfully");
				return GST_FLOW_ERROR;
			}
		}


		/*
		 * Assuming 32 bits RGBA (or any combination)
		 * */
		if(inframe->info.finfo->flags & GST_VIDEO_FORMAT_FLAG_LE){//LITTLE ENDIAN
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
			if(srcformat==GST_VIDEO_FORMAT_RGBA || srcformat==GST_VIDEO_FORMAT_RGBx){
				RedMask = 0xff000000;//R-mask
				GreenMask = 0x00ff0000;//G-mask
				BlueMask = 0x0000ff00;//B-mask
				AlphaMask = 0x000000ff;//A-mask
				RedShift=24, GreenShift=16, BlueShift=8, AlphaShift=0;
			}else if(srcformat==GST_VIDEO_FORMAT_ARGB || srcformat==GST_VIDEO_FORMAT_xRGB){
				RedMask = 0x00ff0000;//R-mask
				GreenMask = 0x0000ff00;//G-mask
				BlueMask = 0x000000ff;//B-mask
				AlphaMask = 0xff000000;//A-mask
				RedShift=16, GreenShift=8, BlueShift=0, AlphaShift=24;
			}else if(srcformat==GST_VIDEO_FORMAT_BGRA || srcformat==GST_VIDEO_FORMAT_BGRx){
				RedMask = 0x0000ff00;//R-mask
				GreenMask = 0x00ff0000;//G-mask
				BlueMask = 0xff000000;//B-mask
				AlphaMask = 0x000000ff;//A-mask
				RedShift=8, GreenShift=16, BlueShift=24, AlphaShift=0;
			}else if(srcformat==GST_VIDEO_FORMAT_ABGR || srcformat==GST_VIDEO_FORMAT_xBGR){
				RedMask = 0x000000ff;//R-mask
				GreenMask = 0x0000ff00;//G-mask
				BlueMask = 0x00ff0000;//B-mask
				AlphaMask = 0xff000000;//A-mask
				RedShift=0, GreenShift=8, BlueShift=16, AlphaShift=24;
			} else{
				//something wrong
				GST_ERROR("This format of image is not supported by the API..!!");
				return GST_FLOW_ERROR;
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
			if(srcformat==GST_VIDEO_FORMAT_RGBA || srcformat==GST_VIDEO_FORMAT_RGBx){
				RedMask = 0x000000ff;//R-mask
				GreenMask = 0x0000ff00;//G-mask
				BlueMask = 0x00ff0000;//B-mask
				AlphaMask = 0xff000000;//A-mask
				RedShift=0, GreenShift=8, BlueShift=16, AlphaShift=24;
			}else if(srcformat==GST_VIDEO_FORMAT_ARGB || srcformat==GST_VIDEO_FORMAT_xRGB){
				RedMask = 0x0000ff00;//R-mask
				GreenMask = 0x00ff0000;//G-mask
				BlueMask = 0xff000000;//B-mask
				AlphaMask = 0x000000ff;//A-mask
				RedShift=8, GreenShift=16, BlueShift=24, AlphaShift=0;
			}else if(srcformat==GST_VIDEO_FORMAT_BGRA || srcformat==GST_VIDEO_FORMAT_BGRx){
				RedMask = 0x00ff0000;//R-mask
				GreenMask = 0x0000ff00;//G-mask
				BlueMask = 0x000000ff;//B-mask
				AlphaMask = 0xff000000;//A-mask
				RedShift=16, GreenShift=8, BlueShift=0, AlphaShift=24;
			}else if(srcformat==GST_VIDEO_FORMAT_ABGR || srcformat==GST_VIDEO_FORMAT_xBGR){
				RedMask = 0xff000000;//R-mask
				GreenMask = 0x00ff0000;//G-mask
				BlueMask = 0x0000ff00;//B-mask
				AlphaMask = 0x000000ff;//A-mask
				RedShift=24, GreenShift=16, BlueShift=8, AlphaShift=0;
			} else{
				//something wrong
				GST_ERROR("This format of image is not supported by the API..!!");
				return GST_FLOW_ERROR;
			}
		}



		//cast the values from 8bits to float one-one correspondence, planar
		pos=0;
		while(pos<framesize){
			//locations estimations
			colorretinex->indata[pos/4] = (gfloat)(((*((guint32*)(src[0]+pos)))&RedMask)>>RedShift);//R
			colorretinex->indata[(pos + framesize)/4] = (gfloat)(((*((guint32*)(src[0]+pos)))&GreenMask)>>GreenShift);//G
			colorretinex->indata[(pos + 2*framesize)/4] = (gfloat)(((*((guint32*)(src[0]+pos)))&BlueMask)>>BlueShift);//B
			colorretinex->indata[(pos + 3*framesize)/4] = (gfloat)(((*((guint32*)(src[0]+pos)))&AlphaMask)>>AlphaShift);//A
			pos+=pix_stride;
		}

		//copy data
		memcpy(colorretinex->outdata, colorretinex->indata, sizeof(gfloat)*(framesize));


		//CALL RETINEX ROUTINE PER CHANNEL
		//R
		time_t start=clock(), end;
		if (NULL == retinex_pde(colorretinex->outdata, width/4, height, (gfloat)colorretinex->RetinexThreshold)) {
			GST_ERROR_OBJECT(colorretinex, "the retinex PDE failed..!!\n");
			return GST_FLOW_ERROR;
		}
		normalize_mean_dt(colorretinex->outdata, colorretinex->indata, framesize/4);
		end=clock();
		GST_DEBUG_OBJECT(colorretinex, "time in R: %f ms.\n", 1000*(((float)end-start)/CLOCKS_PER_SEC));

		//G
		start=clock();
		if (NULL == retinex_pde(colorretinex->outdata + framesize/4, width/4, height, (gfloat)colorretinex->RetinexThreshold)) {
			GST_ERROR_OBJECT(colorretinex, "the retinex PDE failed..!!\n");
			return GST_FLOW_ERROR;
		}
		normalize_mean_dt(colorretinex->outdata + framesize/4, colorretinex->indata + framesize/4, framesize/4);
		end=clock();
		GST_DEBUG_OBJECT(colorretinex, "time in G: %f ms.\n", 1000*(((float)end-start)/CLOCKS_PER_SEC));

		//B
		start=clock();
		if (NULL == retinex_pde(colorretinex->outdata + 2*framesize/4, width/4, height, (gfloat)colorretinex->RetinexThreshold)) {
			GST_ERROR_OBJECT(colorretinex, "the retinex PDE failed..!!\n");
			return GST_FLOW_ERROR;
		}
		normalize_mean_dt(colorretinex->outdata + 2*framesize/4, colorretinex->indata + 2*framesize/4, framesize/4);
		end=clock();
		GST_DEBUG_OBJECT(colorretinex, "time in B: %f ms.\n", 1000*(((float)end-start)/CLOCKS_PER_SEC));

		//casting back the values from float to 8bits, 32 bits including alpha channel
		pos=0;
		while(pos<framesize){
			//locations estimations
			temp_val[0] = min_val(max_val(*(colorretinex->outdata+(pos/4)),0.0), (guint8)255);//R
			temp_val[1] = min_val(max_val(*(colorretinex->outdata+((framesize + pos)/4)),0.0), (guint8)255);//G
			temp_val[2] = min_val(max_val(*(colorretinex->outdata+((2*framesize + pos)/4)),0.0), (guint8)255);//B
			temp_val[3] = min_val(max_val(*(colorretinex->outdata+((3*framesize + pos)/4)),0.0), (guint8)255);//A

			/*
			 * bit OR operator to joint RGBA according to the shift value found when detecting LE or BE
			 * */
			*((guint32*)(dest[0]+pos)) = (temp_val[0]<<RedShift | temp_val[1]<<GreenShift | temp_val[2]<<BlueShift | temp_val[3]<<AlphaShift);
//			g_print("<%x>\t", *((guint32*)(dest[0]+pos)));
			pos+=pix_stride;
		}




		/*
		 * Unlocking the data for releasing the changes and output
		 * the modified frame
		 * */
		GST_OBJECT_UNLOCK(colorretinex);
	}else{
		GST_ERROR_OBJECT (colorretinex, "The input format of the frame is not supported by this element RETINEX: %s",
				gst_video_format_to_string(colorretinex->base_colorretinex.in_info.finfo->format));
		return GST_FLOW_ERROR;
	}

	//============================ Retinex Implementation for this element =======================================//

	return GST_FLOW_OK;
}









#if 0
static gboolean plugin_init (GstPlugin * plugin){
  return gst_element_register (plugin, "colorretinex", GST_RANK_NONE, GST_TYPE_COLORRETINEX);
}


#ifndef VERSION
#define VERSION "0.0.1"
#endif
#ifndef PACKAGE
#define PACKAGE "Color_Consistency_RETINEX_API"
#endif
#ifndef PACKAGE_NAME
#define PACKAGE_NAME "Image_proc_API"
#endif
#ifndef GST_PACKAGE_ORIGIN
#define GST_PACKAGE_ORIGIN "none"
#endif

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR, GST_VERSION_MINOR,  colorretinex,
    "Color consistency routine for frame color space enhancement, using the Retinex algorithm",
    plugin_init, VERSION, "LGPL", PACKAGE_NAME, GST_PACKAGE_ORIGIN)

#endif

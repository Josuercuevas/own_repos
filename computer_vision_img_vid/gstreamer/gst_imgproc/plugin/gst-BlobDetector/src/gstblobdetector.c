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
 * SECTION:element-gstblobdetector
 *
 * The blobdetector element does FIXME stuff.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v fakesrc ! blobdetector ! FIXME ! fakesink
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
#include "gstblobdetector.h"
#include "includes/BlobDetection.h"
#include "../../common/metainfo.h"


GST_DEBUG_CATEGORY_STATIC (gst_blobdetector_debug_category);
#define GST_CAT_DEFAULT gst_blobdetector_debug_category

#define NORMALIZE_DEFAULT (0) //default value for this property
#define MINX (32) //default value for this property
#define MINY (32) //default value for this property
#define MAXX (512) //default value for this property
#define MAXY (512) //default value for this property
#define MAXAREA (MAXX*MAXY*0.8) //default value for this property
#define MINAREA (MINX*MINY*1.2) //default value for this property
#define ALLOWOVERLAP (TRUE) //overlap property in the blobs if more than one
#define TEXTURE_DEFAULT (0.25) //default value for this property

/*
 * Detector function prototypes
 * */
static void gst_blobdetector_set_property (GObject * object, guint property_id,
		const GValue * value, GParamSpec * pspec);
static void gst_blobdetector_get_property (GObject * object, guint property_id,
		GValue * value, GParamSpec * pspec);
static void gst_blobdetector_dispose (GObject * object);
static void gst_blobdetector_finalize (GObject * object);
static gboolean gst_blobdetector_start (GstBaseTransform * trans);
static gboolean gst_blobdetector_stop (GstBaseTransform * trans);
static gboolean gst_blobdetector_set_info (GstVideoFilter * filter, GstCaps * incaps,
		GstVideoInfo * in_info, GstCaps * outcaps, GstVideoInfo * out_info);
static GstFlowReturn gst_blobdetector_transform_frame (GstVideoFilter * filter,
		GstVideoFrame * inframe, GstVideoFrame * outframe);

//for caps negotiation
static gboolean gst_blobdetector_set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps);


enum _blobprops{
  PROP_0,
  NORMALIZE_blob,
  MINX_BLOB,
  MINY_BLOB,
  MAXX_BLOB,
  MAXY_BLOB,
  MIN_AREA,
  MAX_AREA,
  OVERLAP_BLOBS,
  PROP_INTERNAL_PT,
  TEXTURE_THRES
};

/*
 * Supported format for this element, in reality this image
 * should be binary, that is to say there is only
 * one channel with information
 *  */
#define VIDEO_SRC_CAPS GST_VIDEO_CAPS_MAKE("{ GRAY8, RGBx, RGBA, xRGB, ARGB, BGRx, BGRA, xBGR, ABGR }")

/*
 * This is the output format of the frame, in case the
 * user wants to output the result
 *  */
#define VIDEO_SINK_CAPS GST_VIDEO_CAPS_MAKE("{ GRAY8, RGBx, RGBA, xRGB, ARGB, BGRx, BGRA, xBGR, ABGR }")


/*
 * class initialization, where all the function pointers and
 * required classes as well as meta-info are to be initialized
 * */
G_DEFINE_TYPE_WITH_CODE (GstBlobDetector, gst_blobdetector, GST_TYPE_VIDEO_FILTER,
		GST_DEBUG_CATEGORY_INIT (gst_blobdetector_debug_category, "blobdetector", 0, "debug category for blobdetector element"));

static void gst_blobdetector_class_init (GstBlobDetectorClass * blobdetector){
  GObjectClass *gobject_class = G_OBJECT_CLASS (blobdetector);
  GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS (blobdetector);
  GstVideoFilterClass *video_filter_class = GST_VIDEO_FILTER_CLASS (blobdetector);

  /*
   * Inner debugging of the detector (Detection and Extraction)
   * */
  debug_ini_detection();

  /*
   * Linking element functions to the corresponding
   * pointers and debugging the necessary information
   * in case the user requests it
   * */
  gobject_class->set_property = GST_DEBUG_FUNCPTR(gst_blobdetector_set_property);
  gobject_class->get_property = GST_DEBUG_FUNCPTR(gst_blobdetector_get_property);
  gobject_class->dispose = GST_DEBUG_FUNCPTR(gst_blobdetector_dispose);
  gobject_class->finalize = GST_DEBUG_FUNCPTR(gst_blobdetector_finalize);
  base_transform_class->start = GST_DEBUG_FUNCPTR(gst_blobdetector_start);
  base_transform_class->stop = GST_DEBUG_FUNCPTR(gst_blobdetector_stop);
  base_transform_class->set_caps = GST_DEBUG_FUNCPTR(gst_blobdetector_set_caps);
  video_filter_class->set_info = GST_DEBUG_FUNCPTR(gst_blobdetector_set_info);
  video_filter_class->transform_frame = GST_DEBUG_FUNCPTR(gst_blobdetector_transform_frame);


  /*
	* Instalation of the normalization property for this element
	* */
	g_object_class_install_property (gobject_class, NORMALIZE_blob, g_param_spec_int("normalize", "normalize",
	"Gray level image output: 1: normalize 0:not normalize, default=0", 0,
	1, NORMALIZE_DEFAULT, G_PARAM_READABLE | G_PARAM_WRITABLE));

	/*
	* Instalation of the area property for this element
	* */
	g_object_class_install_property (gobject_class, MINX_BLOB, g_param_spec_int("minx", "Minimum Width for BB",
		"minimum width of the Bounding box, default=32", 5,
		INT_MAX, MINX, G_PARAM_READABLE | G_PARAM_WRITABLE));
	g_object_class_install_property (gobject_class, MINY_BLOB, g_param_spec_int("miny", "Minimum Height for BB",
		"minimum height of the Bounding box, default=32", 5,
		INT_MAX, MINY, G_PARAM_READABLE | G_PARAM_WRITABLE));
	g_object_class_install_property (gobject_class, MAXX_BLOB, g_param_spec_int("maxx", "Maximum Width for BB",
		"minimum width of the Bounding box, default=512", 5,
		INT_MAX, MAXX, G_PARAM_READABLE | G_PARAM_WRITABLE));
	g_object_class_install_property (gobject_class, MAXY_BLOB, g_param_spec_int("maxy", "Maximum Height for BB",
		"maximum height of the Bounding box, default=512", 5,
		INT_MAX, MAXY, G_PARAM_READABLE | G_PARAM_WRITABLE));
	g_object_class_install_property (gobject_class, MIN_AREA, g_param_spec_int("minArea", "Minimum Area of BB",
		"minimum area of the Bounding box, default=32^2 x 1.2", 5,
		INT_MAX, MINAREA, G_PARAM_READABLE | G_PARAM_WRITABLE));
	g_object_class_install_property (gobject_class, MAX_AREA, g_param_spec_int("maxArea", "Maximum Area of BB",
		"maximum area of the Bounding box, default=512^2 x 0.8", 5,
		INT_MAX, MAXAREA, G_PARAM_READABLE | G_PARAM_WRITABLE));
	g_object_class_install_property (gobject_class, OVERLAP_BLOBS, g_param_spec_int("overlap", "Overlap Blobs",
			"Allow overlapping the blobs, if this is true TWO overlapping blobs will create ONE new bigger blob; 1: ALLOW OVERLAP 0:DON'T OVERLAP, default=1", 0,
			1, ALLOWOVERLAP, G_PARAM_READABLE | G_PARAM_WRITABLE));
	g_object_class_install_property (gobject_class, TEXTURE_THRES, g_param_spec_float("texture_thres", "Texture Threshold",
			"Value to be used to extract texture: range[0-1], default=0.25", 0,
			1, TEXTURE_DEFAULT, G_PARAM_READABLE | G_PARAM_WRITABLE));


	g_object_class_install_property (gobject_class, PROP_INTERNAL_PT, g_param_spec_float("internal-pt", "Internal processing time",
				"Time taken to processing a frame", 0,
				9999999, 0, G_PARAM_READABLE));

	gst_element_class_set_static_metadata (GST_ELEMENT_CLASS(blobdetector), "Blob detector", "VideoFilter",
	  "Extracts the blobs from a binary image", "somecpmpnay <josuercuevas@gmail.com>");



	/* Setting up pads and setting metadata should be moved to
	* base_class_init if you intend to subclass this class. */
	gst_element_class_add_pad_template (GST_ELEMENT_CLASS(blobdetector), gst_pad_template_new ("src", GST_PAD_SRC, GST_PAD_ALWAYS,
		  gst_caps_from_string (VIDEO_SRC_CAPS)));
	gst_element_class_add_pad_template (GST_ELEMENT_CLASS(blobdetector), gst_pad_template_new ("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
		  gst_caps_from_string (VIDEO_SINK_CAPS)));
}

/*
 * This function is in oder to initialize the element before starts
 * processing the incoming frames, the idea is that structures, variables
 * or global information is to be set from here, therefore ensuring that
 * it is done only once during the lifetime of the pipeline
 * */
static void gst_blobdetector_init (GstBlobDetector *blobdetector){
	/*
	 * parameters to set that cant be initialize
	 * from the property set function
	 * */
	blobdetector->blobs = NULL;
	blobdetector->labels = NULL;
	blobdetector->normalize_frame = NORMALIZE_DEFAULT;
	blobdetector->minheight = MINY;
	blobdetector->minwidth = MINX;
	blobdetector->maxheight = MAXY;
	blobdetector->maxwidth = MAXX;
	blobdetector->minarea = MINAREA;
	blobdetector->maxarea = MAXAREA;
	blobdetector->base_blobdetector.negotiated = FALSE;
	blobdetector->Allowoverlap = ALLOWOVERLAP;
	blobdetector->text_thres = TEXTURE_DEFAULT;
	blobdetector->internal_pt=0.f;
}

/*
 * Sets the property for the filter, which is basically to
 * know if he wants to output image or not (black and white)
 * */
void gst_blobdetector_set_property (GObject * object, guint property_id, const GValue * value, GParamSpec * pspec){
  GstBlobDetector *blobdetector = GST_BLOBDETECTOR (object);

  GST_DEBUG_OBJECT (blobdetector, "set_property");

  switch (property_id) {
	case NORMALIZE_blob://to see if the user wants to normalize the output frame to 255
		blobdetector->normalize_frame = g_value_get_int(value);
		break;
	case MINX_BLOB:
		blobdetector->minwidth = g_value_get_int(value);
		break;
	case MINY_BLOB:
		blobdetector->minheight = g_value_get_int(value);
		break;
	case MAXX_BLOB:
		blobdetector->maxwidth = g_value_get_int(value);
		break;
	case MAXY_BLOB:
		blobdetector->maxheight = g_value_get_int(value);
		break;
	case MIN_AREA:
		blobdetector->minarea = g_value_get_int(value);
		break;
	case MAX_AREA:
		blobdetector->maxarea = g_value_get_int(value);
		break;
	case OVERLAP_BLOBS:
		blobdetector->Allowoverlap= g_value_get_int(value);
		break;
	case TEXTURE_THRES://
		blobdetector->text_thres = g_value_get_float(value);
		break;
	default:
	  G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
	  break;
  }
}


/*
 * gets the property for the filter, which is basically to
 * know if he wants to output image or not (black and white)
 * */
void gst_blobdetector_get_property (GObject * object, guint property_id, GValue * value, GParamSpec * pspec){
  GstBlobDetector *blobdetector = GST_BLOBDETECTOR (object);

  GST_DEBUG_OBJECT (blobdetector, "get_property");

  switch (property_id) {
	case NORMALIZE_blob://to see if the user wants to normalize the output frame to 255
		g_value_set_int(value, blobdetector->normalize_frame);
		break;
	case MINX_BLOB:
		g_value_set_int(value, blobdetector->minwidth);
		break;
	case MINY_BLOB:
		g_value_set_int(value, blobdetector->minheight);
		break;
	case MAXX_BLOB:
		g_value_set_int(value, blobdetector->maxwidth);
		break;
	case MAXY_BLOB:
		g_value_set_int(value, blobdetector->maxheight);
		break;
	case MAX_AREA:
		g_value_set_int(value, blobdetector->maxarea);
		break;
	case MIN_AREA:
		g_value_set_int(value, blobdetector->minarea);
		break;
	case OVERLAP_BLOBS:
		g_value_set_int(value, blobdetector->Allowoverlap);
		break;
	case TEXTURE_THRES://
		g_value_set_float(value, blobdetector->text_thres);
		break;
	case PROP_INTERNAL_PT://
		g_value_set_float(value, blobdetector->internal_pt);
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
void gst_blobdetector_dispose (GObject * object){
	GstBlobDetector *blobdetector = GST_BLOBDETECTOR (object);

	GST_DEBUG_OBJECT (blobdetector, "dispose");

	/* clean up as possible.  may be called multiple times */
	if(blobdetector->labels!=NULL){
		g_free(blobdetector->labels);
		blobdetector->labels=NULL;
	}
	G_OBJECT_CLASS (gst_blobdetector_parent_class)->dispose (object);
}

/*
 * Cleans all the garbage information collected ... in our case
 * there is none for now
 * */
void gst_blobdetector_finalize (GObject * object){
	GstBlobDetector *blobdetector = GST_BLOBDETECTOR (object);

	GST_DEBUG_OBJECT (blobdetector, "finalize");

	/* clean up object here */
	if(blobdetector->labels!=NULL){
		g_free(blobdetector->labels);
		blobdetector->labels=NULL;
	}
	G_OBJECT_CLASS (gst_blobdetector_parent_class)->finalize (object);
}


static gboolean gst_blobdetector_start (GstBaseTransform * trans){
	GstBlobDetector *blobdetector = GST_BLOBDETECTOR (trans);

	/*
	* Just puts some debugging information for the comfortability
	* of the user
	* */
	if(blobdetector->normalize_frame != NORMALIZE_DEFAULT)
		GST_DEBUG_OBJECT (blobdetector, "Output image will be normalized to 255");
	else
		GST_DEBUG_OBJECT (blobdetector, "Output image is going to be binary (0,1)");

	GST_DEBUG_OBJECT (blobdetector, "start");

	return TRUE;
}

static gboolean gst_blobdetector_stop (GstBaseTransform * trans){
	GstBlobDetector *blobdetector = GST_BLOBDETECTOR (trans);
	/*
	* Just puts some debugging information for the comfortability
	* of the user
	* */
	GST_DEBUG_OBJECT (blobdetector, "stop");

	return TRUE;
}

//caps negotiation part
static gboolean gst_blobdetector_set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps){
	GstVideoFilter *filter = GST_VIDEO_FILTER_CAST (trans);
	GstVideoFilterClass *fclass;
	GstVideoInfo in_info, out_info;
	gboolean res;
	const gchar *src_format, *sink_format;
	gchar *src_color, *sink_color;

	GST_DEBUG_OBJECT(filter, "Caps negotiation BlobDetector..!!");
	GST_OBJECT_LOCK(filter);

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
		GST_DEBUG_OBJECT(filter, "Blob Detector VideoFilter has negotiated caps successfully..!!");

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

static gboolean gst_blobdetector_set_info (GstVideoFilter * filter, GstCaps * incaps,
		GstVideoInfo * in_info, GstCaps * outcaps, GstVideoInfo * out_info){
  GstBlobDetector *blobdetector = GST_BLOBDETECTOR (filter);

  //in case we need to set extra info for the element for processing purposes

  GST_DEBUG_OBJECT (blobdetector, "set_info");

  return TRUE;
}


/*
 * Function to add the metainformation referring to the blobs
 * to be processed in the Normalization element in the pipeline
 * */
void* gst_buffer_add_blob_meta(GstBuffer *buffer, GstBlobDetector *blobdetector){
	guint i, index, boxes;
	GstMetaBLOB *meta = NULL;
	//GstVideoRegionOfInterestMeta *meta=NULL;

	//check is the incoming buffer is truly one
	g_return_val_if_fail(GST_IS_BUFFER(buffer), NULL);

	//making the buffer writable for adding the metadata
	buffer = gst_buffer_make_writable(buffer);

	//adding the metadata structure to the buffer
	meta = (GstMetaBLOB*)gst_buffer_add_meta(buffer,BLOB_META_INFO, NULL);
	//meta = gst_buffer_add_video_region_of_interest_meta (buffer, "BBs", 0, 0, 0, 0);

	//everything is fine now we proceed to insert the information
	if(blobdetector->n_blobs>0){
		//we have some blobs to process later
		boxes = blobdetector->n_blobs;//amount of BBs found meta->n_blobs =
		GST_DEBUG_OBJECT(blobdetector, "Meta Blobs found: %i\n", blobdetector->n_blobs);
		//GST_DEBUG_OBJECT(blobdetector, "Blobs found: %i\n", meta->n_blobs);
		//g_print("Blobs found: %i\n", meta->n_blobs);

		index=0;
		for(i=0;i<boxes;i++){
			if(blobdetector->blobs[i].passed){
//				meta->x = blobdetector->blobs[i].xmin;//starting x location
//				meta->y  = blobdetector->blobs[i].ymin;//starting y location
//				meta->w = blobdetector->blobs[i].width;//width of blob
//				meta->h = blobdetector->blobs[i].height;//height of blob
//				GST_DEBUG_OBJECT(blobdetector, "Meta Blob %i info: <%i, %i, %i, %i>", i, meta->x, meta->y, meta->w, meta->h);

				meta->height[index] = blobdetector->blobs[i].height;//height of blob
				meta->width[index] = blobdetector->blobs[i].width;//width of blob
				meta->x[index] = blobdetector->blobs[i].xmin;//starting x location
				meta->y[index] = blobdetector->blobs[i].ymin;//starting y location
				GST_DEBUG_OBJECT(blobdetector, "Blob %i info: <%i, %i, %i, %i>", index, meta->x[index], meta->y[index], meta->width[index], meta->height[index]);
				index++;
			}
		}
		meta->n_blobs = index;//patches found
	}else{
		//no blobs to process so flag this frame
		meta->n_blobs = blobdetector->n_blobs;//the value would be zero
		GST_DEBUG_OBJECT(blobdetector, "Meta Blobs found: %i\n",blobdetector->n_blobs);
		//GST_DEBUG_OBJECT(blobdetector, "Blobs found: %i\n",meta->n_blobs);
		//g_print("Blobs found: %i", meta->n_blobs);
	}

	return NULL;
}




/*
 * Main function of this filter, where the blob detector is to be invoked and tested
 * to see its performance and accuracy
 *  */
static GstFlowReturn gst_blobdetector_transform_frame (GstVideoFilter * filter,
		GstVideoFrame * inframe, GstVideoFrame * outframe){

	GstBlobDetector *blobdetector = GST_BLOBDETECTOR(filter);
	gint width, height;
	guint8 ierror;
	gint sstride, dstride[3], srcformat;
	GstMapInfo outinfo;

	if(!filter->negotiated){
		GST_DEBUG_OBJECT (blobdetector, "Caps have NOT been negotiated, proceeding to negotiation phase..!!");
		GstBaseTransform *blobdetectorBaseTransform = GST_BASE_TRANSFORM(filter);
		GstVideoFilterClass *blobdetectorclass = GST_BLOBDETECTOR_CLASS(filter);
		GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS(blobdetectorclass);
		if(!base_transform_class->set_caps){
			GST_ERROR_OBJECT (blobdetector, "The caps negotiation have failed, closing application");
			return GST_FLOW_ERROR;
		}
	}


	//gets info of the frame to be processed
	blobdetector->inframe = inframe;
	blobdetector->outframe = outframe;

	//strides to be used per channel
	sstride = GST_VIDEO_FRAME_PLANE_STRIDE (inframe, 0);
	dstride[0] = GST_VIDEO_FRAME_PLANE_STRIDE (outframe, 0);
	dstride[1] = GST_VIDEO_FRAME_PLANE_STRIDE (outframe, 1);
	dstride[2] = GST_VIDEO_FRAME_PLANE_STRIDE (outframe, 2);

	srcformat = GST_VIDEO_FRAME_FORMAT(inframe);

	width = sstride;
	height = GST_VIDEO_FRAME_HEIGHT(inframe);

	GST_DEBUG_OBJECT (blobdetector, "Threshold for blob sizes is: MIN ->  <%ix%i> or %i pixels area\n"
			" MAX ->  <%ix%i> or %i pixels area", blobdetector->minwidth,
			blobdetector->minheight, blobdetector->minarea, blobdetector->maxwidth,
			blobdetector->maxheight, blobdetector->maxarea);

	GST_DEBUG("outframe->buffer: %p", outframe->buffer);
	//-----------------------BLOB EXTRACTION PART--------------------------//
	time_t start = clock(), end;
	GST_DEBUG_OBJECT (blobdetector, "Performing Blob Detection:\n\t<OutImage <s: %i, w: %i, h: %i, C: %i> ; InImage <s: %i, w: %i, H: %i, C: %i> >",
			GST_VIDEO_FRAME_PLANE_STRIDE (blobdetector->outframe, 0), GST_VIDEO_FRAME_WIDTH(blobdetector->outframe),
			GST_VIDEO_FRAME_HEIGHT(blobdetector->outframe), GST_VIDEO_FRAME_N_COMPONENTS(blobdetector->outframe),
			GST_VIDEO_FRAME_PLANE_STRIDE(blobdetector->inframe, 0), GST_VIDEO_FRAME_WIDTH(blobdetector->inframe),
			GST_VIDEO_FRAME_HEIGHT(blobdetector->inframe), GST_VIDEO_FRAME_N_COMPONENTS(blobdetector->inframe));

	/*
	 * We need to lock-mutex the access for this frame so we do not
	 * let another process to mess up with the data we are extracting
	 * and modifying
	 * */

	if(srcformat==GST_VIDEO_FORMAT_I420){
		GST_OBJECT_LOCK(blobdetector);

		blobdetector->channels = 1;//always this number since we support only binary
		blobdetector->height = height;
		blobdetector->width = width;
		blobdetector->channels_format = planar;
		blobdetector->image_type = srcformat;

		ierror = Blob_error_handler(perform_blobdetection(blobdetector));
		if(ierror != blob_success){
			GST_ERROR_OBJECT(blobdetector, "Problem performing blob detection ...\n");
		}

		/*
		 * Setting up the metadata to be used in the Normalization for processing purposes
		 * its important to mention that the whole structure is coming and is to be used
		 * when processing the frame in the normalization element
		 * */

		blobdetector->outframe->buffer =  gst_buffer_make_writable(blobdetector->outframe->buffer);
		gst_buffer_map(blobdetector->outframe->buffer, &outinfo, GST_MAP_WRITE);
		if(blobdetector->outframe->buffer != NULL && GST_IS_BUFFER (blobdetector->outframe->buffer)){
			GST_DEBUG_OBJECT (blobdetector, "Adding metadata to buffer in blob detector");
			gst_buffer_add_blob_meta(blobdetector->outframe->buffer, blobdetector);
			GST_DEBUG_OBJECT (blobdetector, "Finished adding metadata to buffer in blob detector");
		}
		gst_buffer_unmap(blobdetector->outframe->buffer, &outinfo);


		/*
		 * Unlocking the data for releasing the changes and output
		 * the modified frame
		 * */
		g_free(blobdetector->blobs);
		blobdetector->blobs=NULL;
		GST_OBJECT_UNLOCK(blobdetector);

	}else if(srcformat==GST_VIDEO_FORMAT_RGBA || srcformat==GST_VIDEO_FORMAT_ARGB || srcformat==GST_VIDEO_FORMAT_BGRA
			|| srcformat==GST_VIDEO_FORMAT_ABGR || srcformat==GST_VIDEO_FORMAT_RGBx || srcformat==GST_VIDEO_FORMAT_xRGB
			|| srcformat==GST_VIDEO_FORMAT_BGRx || srcformat==GST_VIDEO_FORMAT_xBGR){
		GST_OBJECT_LOCK(blobdetector);

		blobdetector->channels = 1;//always this number since we support only binary
		blobdetector->height = height;
		blobdetector->width = width;
		blobdetector->channels_format = interleaved;
		blobdetector->image_type = srcformat;

		ierror = Blob_error_handler(perform_blobdetection(blobdetector));
		if(ierror != blob_success){
			GST_ERROR_OBJECT(blobdetector, "Problem performing blob detection ...\n");
		}

		/*
		 * Setting up the metadata to be used in the Normalization for processing purposes
		 * its important to mention that the whole structure is coming and is to be used
		 * when processing the frame in the normalization element
		 * */

		gst_buffer_map(blobdetector->outframe->buffer, &outinfo, GST_MAP_WRITE);
		if(blobdetector->outframe->buffer != NULL && GST_IS_BUFFER (blobdetector->outframe->buffer)){
			GST_DEBUG_OBJECT (blobdetector, "Adding metadata to buffer in blob detector");
			gst_buffer_add_blob_meta(blobdetector->outframe->buffer, blobdetector);
			GST_DEBUG_OBJECT (blobdetector, "Finished adding metadata to buffer in blob detector");
		}
		gst_buffer_unmap(blobdetector->outframe->buffer, &outinfo);


		/*
		 * Unlocking the data for releasing the changes and output
		 * the modified frame
		 * */
		g_free(blobdetector->blobs);
		blobdetector->blobs=NULL;
		GST_OBJECT_UNLOCK(blobdetector);

	}else if(srcformat==GST_VIDEO_FORMAT_GRAY8){
		GstVideoRegionOfInterestMeta *ROImeta=NULL, *tmpROIMeta = NULL;
		GstVideoMeta* meta = NULL;
		GstMapInfo map;
		gpointer data;
		gint stride;
		GstBuffer* tmpBuffer = outframe->buffer;

		GST_OBJECT_LOCK(blobdetector);

		blobdetector->channels = 1;//always this number since we support only binary
		blobdetector->height = height;
		blobdetector->width = width;
		blobdetector->channels_format = planar;
		blobdetector->image_type = srcformat;

		ierror = Blob_error_handler(perform_blobdetection(blobdetector));
		if(ierror != blob_success){
			GST_ERROR_OBJECT(blobdetector, "Problem performing blob detection ...\n");
		}

		outframe->buffer =  gst_buffer_make_writable(outframe->buffer);
		gst_buffer_map(outframe->buffer, &outinfo, GST_MAP_WRITE);
		if(outframe->buffer != NULL && GST_IS_BUFFER (outframe->buffer)){
			GST_DEBUG_OBJECT (blobdetector, "Adding metadata to buffer in blob detector");
			gst_buffer_add_blob_meta(outframe->buffer, blobdetector);
			GST_DEBUG_OBJECT (blobdetector, "Finished adding metadata to buffer in blob detector");
		}
		gst_buffer_unmap(outframe->buffer, &outinfo);


		/*
		 * Unlocking the data for releasing the changes and output
		 * the modified frame
		 * */
		g_free(blobdetector->blobs);
		blobdetector->blobs=NULL;
		GST_OBJECT_UNLOCK(blobdetector);
	}else{
		GST_ERROR_OBJECT (blobdetector, "The input format of the frame is not supported by this element BLOBDETECTOR: %s",
				gst_video_format_to_string(blobdetector->base_blobdetector.in_info.finfo->format));
		return GST_FLOW_ERROR;
	}
	end = clock();
	GST_DEBUG_OBJECT(blobdetector, "++++++++++++++++++++++++++++++++++++++++++++++\n"
			"++++++++++++++++++++++++++++++++++++++++++++++\n"
			" Total time in element Blob Detector: %f ms.\n"
			"++++++++++++++++++++++++++++++++++++++++++++++\n"
			"++++++++++++++++++++++++++++++++++++++++++++++\n",1000 * (((float) end - start) / CLOCKS_PER_SEC));
	blobdetector->internal_pt = 1000*(((float)(end-start))/CLOCKS_PER_SEC);

	return GST_FLOW_OK;
}

#if 0
static gboolean plugin_init (GstPlugin * plugin){

  return gst_element_register (plugin, "blobdetector", GST_RANK_NONE, GST_TYPE_BLOBDETECTOR);
}


#ifndef VERSION
#define VERSION "0.0.1"
#endif
#ifndef PACKAGE
#define PACKAGE "Blob_Detector_API"
#endif
#ifndef PACKAGE_NAME
#define PACKAGE_NAME "Image_proc_API"
#endif
#ifndef GST_PACKAGE_ORIGIN
#define GST_PACKAGE_ORIGIN "None"
#endif

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR, GST_VERSION_MINOR, blobdetector,
		"Blob detection API for binary images, it outoputs a gray-scaled image",
		plugin_init, VERSION, "LGPL", PACKAGE_NAME, GST_PACKAGE_ORIGIN)

#endif

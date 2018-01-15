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
 * SECTION:element-gstdistancetransform
 *
 * The distancetransform element does FIXME stuff.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v fakesrc ! distancetransform ! FIXME ! fakesink
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
#include "gstdistancetransform.h"

GST_DEBUG_CATEGORY_STATIC (gst_distancetransform_debug_category);
#define GST_CAT_DEFAULT gst_distancetransform_debug_category

#define NORMALIZE_DEFAULT (0)
#define INF (10000000.0f)
#define max_val(a,b) \
  ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
    _a > _b ? _a : _b; })

#define min_val(a,b) \
  ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
    _a < _b ? _a : _b; })

#define square(a) \
  ({ __typeof__ (a) _a = (a); \
    _a * _a; })



/*
 * Prototypes neeeded when performing transformation
 * */
static void gst_distancetransform_set_property (GObject *DT, guint property_id,
		const GValue * value, GParamSpec * pspec);
static void gst_distancetransform_get_property (GObject * DT, guint property_id,
		GValue * value, GParamSpec * pspec);
static void gst_distancetransform_dispose (GObject  *DT);
static void gst_distancetransform_finalize (GObject *DT);

static gboolean gst_distancetransform_start (GstBaseTransform *DT_base);
static gboolean gst_distancetransform_stop (GstBaseTransform *DT_base);
static gboolean gst_distancetransform_set_info (GstVideoFilter *DT, GstCaps * incaps,
		GstVideoInfo * in_info, GstCaps * outcaps, GstVideoInfo * out_info);
static GstFlowReturn gst_distancetransform_transform_frame (GstVideoFilter *DT,
		GstVideoFrame * inframe, GstVideoFrame * outframe);

static gboolean gst_distancetransform_set_caps(GstBaseTransform *DT, GstCaps *incaps,
		GstCaps *outcaps);

/*
 * Functions neeed to perform the transformation
 * */
static gboolean gst_2D_dt_I420(GstDistanceTransform *DT);
static gboolean gst_2D_dt_RGB(GstDistanceTransform *DT);
static gfloat* gst_1D_dt(gfloat *pixels, guint n);

enum
{
  PROP_0,
  NORMALIZE
};

/* pad templates */
/*
 * Supported formats by the source pad
 *  */
#define VIDEO_SRC_CAPS GST_VIDEO_CAPS_MAKE("{ I420, RGB }")

/*
 * Supported formats by the sink pad
 *  */
#define VIDEO_SINK_CAPS GST_VIDEO_CAPS_MAKE("{ I420, RGB }")


/* class initialization */
G_DEFINE_TYPE_WITH_CODE (GstDistanceTransform, gst_distancetransform, GST_TYPE_VIDEO_FILTER,
  GST_DEBUG_CATEGORY_INIT (gst_distancetransform_debug_category, "distancetransform", 0,
  "debug category for distancetransform element"));

static void gst_distancetransform_class_init (GstDistanceTransformClass * DT_class){
	GObjectClass *gobject_class = G_OBJECT_CLASS(DT_class);
	GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS(DT_class);
	GstVideoFilterClass *video_filter_class = GST_VIDEO_FILTER_CLASS(DT_class);

	gobject_class->set_property = GST_DEBUG_FUNCPTR(gst_distancetransform_set_property);
	gobject_class->get_property = GST_DEBUG_FUNCPTR(gst_distancetransform_get_property);
	gobject_class->dispose = GST_DEBUG_FUNCPTR(gst_distancetransform_dispose);
	gobject_class->finalize = GST_DEBUG_FUNCPTR(gst_distancetransform_finalize);
	base_transform_class->start = GST_DEBUG_FUNCPTR(gst_distancetransform_start);
	base_transform_class->stop = GST_DEBUG_FUNCPTR(gst_distancetransform_stop);
	base_transform_class->set_caps = GST_DEBUG_FUNCPTR(gst_distancetransform_set_caps);
	video_filter_class->set_info = GST_DEBUG_FUNCPTR(gst_distancetransform_set_info);
	video_filter_class->transform_frame = GST_DEBUG_FUNCPTR(gst_distancetransform_transform_frame);


	gst_element_class_set_static_metadata (GST_ELEMENT_CLASS(DT_class),
			"Distance Transform", "Video Filter", "Calculates the distance transform of a binary image",
			"Josue R. Cuevas <josuercuevas@gmail.com>");

	  /*
		* Installation of the normalization property for this element
		* */
		g_object_class_install_property (gobject_class, NORMALIZE, g_param_spec_int("normalize", "normalize",
				"Black and White output image: 1: normalize 0:not normalize, default=0", 0,
				1, NORMALIZE_DEFAULT, G_PARAM_READABLE | G_PARAM_WRITABLE));

	/* Setting up pads and setting metadata should be moved to
	 base_class_init if you intend to subclass this class. */
	gst_element_class_add_pad_template (GST_ELEMENT_CLASS(DT_class),
			gst_pad_template_new ("src", GST_PAD_SRC, GST_PAD_ALWAYS, gst_caps_from_string (VIDEO_SRC_CAPS)));
	gst_element_class_add_pad_template (GST_ELEMENT_CLASS(DT_class),
			gst_pad_template_new ("sink", GST_PAD_SINK, GST_PAD_ALWAYS, gst_caps_from_string (VIDEO_SINK_CAPS)));

}

static void gst_distancetransform_init (GstDistanceTransform *DT){
	/*
	 * Initialization part of the default values
	 * None for now
	 * */
	DT->inframe = NULL;
	DT->outframe = NULL;
}


/* Property setter */
void gst_distancetransform_set_property (GObject *DT, guint property_id,
		const GValue *value, GParamSpec * pspec){
	GstDistanceTransform *DistanceTransform = GST_DISTANCETRANSFORM(DT);

	GST_DEBUG_OBJECT (DistanceTransform, "Setting property");

	switch (property_id){
		case NORMALIZE:
			DistanceTransform->normalize = g_value_get_int(value);
			break;
		default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID(DT, property_id, pspec);
		break;
	}
}

/* Property grabber */
void gst_distancetransform_get_property (GObject *DT, guint property_id,
		GValue * value, GParamSpec * pspec){
	GstDistanceTransform *DistanceTransform = GST_DISTANCETRANSFORM (DT);

	GST_DEBUG_OBJECT (DistanceTransform, "Getting the properties of this element");

	switch (property_id) {
		case NORMALIZE://to see if the user wants to normalize the output frame to 255
			g_value_set_int(value, DistanceTransform->normalize);
			break;
		default:
			G_OBJECT_WARN_INVALID_PROPERTY_ID(DT, property_id, pspec);
			break;
	}
}

/*
 * In case we need to dispose of any memory we don't need anymore
 * during or at the end of the execution
 * */
void gst_distancetransform_dispose (GObject *DT){
	GstDistanceTransform *DistanceTransform = GST_DISTANCETRANSFORM(DT);

	GST_DEBUG_OBJECT (DistanceTransform, "Disposing any memory used during execution");

	/* clean up as possible.  may be called multiple times */
	G_OBJECT_CLASS (gst_distancetransform_parent_class)->dispose(DT);
}

/*
 * Called just once when the pipeline stops or it is called
 * */
void gst_distancetransform_finalize (GObject *DT){
	GstDistanceTransform *DistanceTransform = GST_DISTANCETRANSFORM (DT);

	GST_DEBUG_OBJECT (DistanceTransform, "Finalizing the Distance Transformation Element");

	/* clean up object here */
	gst_distancetransform_dispose(DT);
	G_OBJECT_CLASS(gst_distancetransform_parent_class)->finalize(DT);
}

static gboolean gst_distancetransform_start (GstBaseTransform *DT_base){
	GstDistanceTransform *DistanceTransform = GST_DISTANCETRANSFORM(DT_base);

	GST_DEBUG_OBJECT (DistanceTransform, "starting the transformation process");

	return TRUE;
}

static gboolean gst_distancetransform_stop (GstBaseTransform *DT_base){
	GstDistanceTransform *DistanceTransform = GST_DISTANCETRANSFORM(DT_base);

	GST_DEBUG_OBJECT (DistanceTransform, "stopping the plugin");

	return TRUE;
}

static gboolean gst_distancetransform_set_info (GstVideoFilter *DT, GstCaps * incaps,
    GstVideoInfo * in_info, GstCaps * outcaps, GstVideoInfo * out_info){
	GstDistanceTransform *DistanceTransform = GST_DISTANCETRANSFORM(DT);

	GST_DEBUG_OBJECT (DistanceTransform, "Setting info in the incoming and outgoing frames");

	return TRUE;
}

static gboolean gst_distancetransform_set_caps(GstBaseTransform *DT, GstCaps *incaps, GstCaps *outcaps){
	GstVideoFilter *DistanceTransform = GST_VIDEO_FILTER_CAST (DT);
	GstVideoFilterClass *fclass;
	GstVideoInfo in_info, out_info;
	gboolean res;
	const gchar *src_format, *sink_format;
	gchar *src_color, *sink_color;

	GST_DEBUG_OBJECT(DistanceTransform, "Caps negotiation Distance Transform..!!");

	/* input caps */
	if (!gst_video_info_from_caps(&in_info, incaps))
	goto invalid_caps;

	/* output caps */
	if (!gst_video_info_from_caps(&out_info, outcaps))
	goto invalid_caps;

	fclass = GST_VIDEO_FILTER_GET_CLASS(DistanceTransform);
	if(fclass->set_info)
		res = fclass->set_info(DistanceTransform, incaps, &in_info, outcaps, &out_info);
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
		DistanceTransform->in_info = in_info;
		DistanceTransform->out_info = out_info;
		if (fclass->transform_frame == NULL)
			gst_base_transform_set_in_place(DistanceTransform, TRUE);
		if (fclass->transform_frame_ip == NULL)
			GST_BASE_TRANSFORM_CLASS (fclass)->transform_ip_on_passthrough = FALSE;
	}

	//This is the part that will determine the negotiation result
	DistanceTransform->negotiated = res;

	if(DistanceTransform->negotiated)
		GST_DEBUG_OBJECT(DistanceTransform, "Distance Transform VideoFilter has negotiated caps successfully..!!");

	return res;

	/* ERRORS */
invalid_caps:
	{
		GST_ERROR_OBJECT(DistanceTransform, "Caps couldn't be negotiated");
		DistanceTransform->negotiated = FALSE;
		return FALSE;
	}
}


/*
 * Implementing the distance Transform
 * using envelope parabolic approach
 * */
static GstFlowReturn gst_distancetransform_transform_frame (GstVideoFilter *DT, GstVideoFrame * inframe,
    GstVideoFrame * outframe){
	GstDistanceTransform *DistanceTransform = GST_DISTANCETRANSFORM(DT);
	guint8 ierror;
	gint sstride, dstride[3], srcformat;
	gint width, height, i, j;
	GstMapInfo ininfo;
	GstVideoRegionOfInterestMeta *ROImeta=NULL;

	if(!DT->negotiated){
		GST_DEBUG_OBJECT (DistanceTransform, "Caps have NOT been negotiated, proceeding to negotiation phase..!!");
		GstBaseTransform *DistanceTransformBaseTransform = GST_BASE_TRANSFORM(DT);
		GstVideoFilterClass *DTclass = GST_DISTANCETRANSFORM_CLASS(DT);
		GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS(DTclass);
		if(!base_transform_class->set_caps){
			GST_ERROR_OBJECT (DistanceTransform, "The caps negotiation have failed, closing application");
			return GST_FLOW_ERROR;
		}
	}

	/* gets info of the frame to be processed */
	DistanceTransform->inframe = inframe;
	DistanceTransform->outframe = outframe;

	/* strides to be used per channel */
	sstride = GST_VIDEO_FRAME_PLANE_STRIDE (inframe, 0);
	dstride[0] = GST_VIDEO_FRAME_PLANE_STRIDE (outframe, 0);
	dstride[1] = GST_VIDEO_FRAME_PLANE_STRIDE (outframe, 1);
	dstride[2] = GST_VIDEO_FRAME_PLANE_STRIDE (outframe, 2);


	srcformat = GST_VIDEO_FRAME_FORMAT(inframe);

	width = sstride;
	height = GST_VIDEO_FRAME_HEIGHT(inframe);


	//================================= FRAME TRANSFORMATION USING ENVELOP METHOD =====================================//
	time_t start = clock(), end;
	GST_DEBUG_OBJECT (DistanceTransform, "Implementing distance transformation");
	/*
	 * We need to lock-mutex the access for this frame so we do not
	 * let another process to mess up with the data we are extracting
	 * and modifying
	 * */
	GST_OBJECT_LOCK(DistanceTransform);

	if(srcformat==GST_VIDEO_FORMAT_I420){
		DistanceTransform->channels = 1;//always this number since we support only binary
		DistanceTransform->height = height;
		DistanceTransform->width = width;
		DistanceTransform->image_type = GST_VIDEO_FORMAT_I420;

		gst_buffer_map(DistanceTransform->inframe->buffer, &ininfo, GST_MAP_READ);
		ROImeta = gst_buffer_get_video_region_of_interest_meta(DistanceTransform->inframe->buffer);
		if(ROImeta!=NULL){
			//we have to process Bounding boxes
			GST_DEBUG_OBJECT(DistanceTransform, "%i PATCHES FOUND: \n", 1);
			for(i=0;i<1;i++){
				DistanceTransform->BBx = ROImeta->x;//BlobMeta->x[i];//
				DistanceTransform->BBy = ROImeta->y;//BlobMeta->y[i];//
				DistanceTransform->BBw = ROImeta->w;//BlobMeta->width[i];//
				DistanceTransform->BBh = ROImeta->h;//BlobMeta->height[i];//
				GST_DEBUG_OBJECT(DistanceTransform, "<%i, %i, %i, %i>\n", DistanceTransform->BBx,
						DistanceTransform->BBy, DistanceTransform->BBw, DistanceTransform->BBh);
			}
			gst_buffer_unmap(DistanceTransform->inframe->buffer, &ininfo);

			/*
			 * Performing the transformation for this type
			 * */
			if(gst_2D_dt_I420(DistanceTransform)){
				GST_ERROR_OBJECT(DistanceTransform, "Error during distance transformation routine..!!!");
				return GST_FLOW_ERROR;
			}else{
				GST_DEBUG_OBJECT(DistanceTransform, "Distance transformation performed correctly..!!!");
			}
		}else{
			gst_buffer_unmap(DistanceTransform->inframe->buffer, &ininfo);
			//no object to normalize nothing done for now
		}
	}else if(srcformat==GST_VIDEO_FORMAT_RGB){
		DistanceTransform->channels = 1;//always this number since we support only binary
		DistanceTransform->height = height;
		DistanceTransform->width = width;
		DistanceTransform->image_type = GST_VIDEO_FORMAT_RGB;

		gst_buffer_map(DistanceTransform->inframe->buffer, &ininfo, GST_MAP_READ);
		ROImeta = gst_buffer_get_video_region_of_interest_meta(DistanceTransform->inframe->buffer);
		if(ROImeta!=NULL){
			//we have to process Bounding boxes
			GST_DEBUG_OBJECT(DistanceTransform, "%i PATCHES FOUND: \n", 1);
			for(i=0;i<1;i++){
				DistanceTransform->BBx = ROImeta->x;//
				DistanceTransform->BBy = ROImeta->y;//
				DistanceTransform->BBw = ROImeta->w;//
				DistanceTransform->BBh = ROImeta->h;//
				GST_DEBUG_OBJECT(DistanceTransform, "<%i, %i, %i, %i>\n", DistanceTransform->BBx,
						DistanceTransform->BBy, DistanceTransform->BBw, DistanceTransform->BBh);
			}
			gst_buffer_unmap(DistanceTransform->inframe->buffer, &ininfo);

			/*
			 * Performing the transformation for this type
			 * */
			if(gst_2D_dt_RGB(DistanceTransform)){
				GST_ERROR_OBJECT(DistanceTransform, "Error during distance transformation routine..!!!");
				return GST_FLOW_ERROR;
			}else{
				GST_DEBUG_OBJECT(DistanceTransform, "Distance transformation performed correctly..!!!");
			}
		}else{
			gst_buffer_unmap(DistanceTransform->inframe->buffer, &ininfo);
			//no object to normalize nothing done for now
		}
	}else{
		GST_ERROR_OBJECT (DistanceTransform, "The input format of the frame is not supported by this element MOMENTNORMALIZATION: %s",
				gst_video_format_to_string(DistanceTransform->DT_base.in_info.finfo->format));
		return GST_FLOW_ERROR;
	}

	/*
	 * Unlocking the data for releasing the changes and output
	 * the modified frame
	 * */
	GST_OBJECT_UNLOCK(DistanceTransform);

	end = clock();
	GST_DEBUG_OBJECT(DistanceTransform, "time in GstDistanceTransform: %f ms.\n",1000 * (((float) end - start) / CLOCKS_PER_SEC));
	//================================= FRAME TRANSFORMATION USING ENVELOP METHOD =====================================//


	return GST_FLOW_OK;
}


static gfloat* gst_1D_dt(gfloat *pixels, guint n){
    gfloat *d;
    gint *v;
    gfloat *z, s;
    gint k = 0;
    guint q;

    /*
     * Preparing memory
     * */
    d = g_malloc0(sizeof(gfloat)*n);
    v = g_malloc0(sizeof(gint)*n);
    z = g_malloc0(sizeof(gfloat)*(n+1));

    v[0] = 0;
    z[0] = -INF;
    z[1] = INF;

    for(q=1; q <= (n-1); q++)    {
        s = ((pixels[q] + square(q)) - (pixels[v[k]] + square(v[k]))) / (2*q - 2*v[k]);
        while (s<=z[k])        {
            k--;
            s = ((pixels[q] + square(q)) - (pixels[v[k]] + square(v[k]))) / (2*q - 2*v[k]);
        }
        k++; v[k] = q; z[k] = s; z[k + 1] = INF;
    }

    k = 0;
    for(q = 0; q <= (n-1); q++)    {
        while (z[k+1] < q)        {
            k++;
        }
        d[q] = square(q - v[k]) + pixels[v[k]];
    }

    g_free(v);
    g_free(z);
    return d;
}

/*
 * Scanning the 2 dimensions of the incoming image
 * */
static gboolean gst_2D_dt_I420(GstDistanceTransform *DT){
	gboolean ierror = FALSE;//no error at all
	GstMapInfo inbuff, outbuff;
	guint32 width = DT->width;
	guint32 height = DT->height;
	gfloat *pixels, *distance, *outdist;
	guint8 *src_ptr;
	guint32 x, y;
	gfloat max_dist=0;



	GST_DEBUG_OBJECT(DT, "Implementing gst_2D_dt_I420: <%i, %i>", width, height);
	pixels = g_malloc0(sizeof(gfloat)*(max_val(height,width)));
	outdist = g_malloc0(sizeof(gfloat)*(height*width));

	/*
	 * Mapping data to be used from the frames
	 * */
	gst_buffer_map(DT->inframe->buffer, &inbuff, GST_MAP_READWRITE);
	DT->outframe->buffer = gst_buffer_make_writable(DT->outframe->buffer);
	gst_buffer_map(DT->outframe->buffer, &outbuff, GST_MAP_READWRITE);
	src_ptr = inbuff.data;

	for(y=0; y<height; y++){
		for(x = 0; x<width; x++){
			if(src_ptr[y*width + x]==0)
				outdist[y*width + x] = 0.0f;//Y
			else
				outdist[y*width + x] = INF;//Y
		}
	}



    /*
     * Columns transformation
     * */
	GST_DEBUG_OBJECT(DT, "Transforming columns");
    for(x=0; x<width; x++){
        for(y=0; y<height; y++){
        	//only one channels contains the info
        	pixels[y] = outdist[y*width + x];//Y
        }
        distance = gst_1D_dt(pixels, height);
        for(y=1; y<height-1; y++){
        	if(((x)>=DT->BBx) && ((x)<=(DT->BBx+DT->BBw)) && (y>=DT->BBy) && (y<=(DT->BBy+DT->BBh))){
				outdist[y*width/3 + x] = distance[x];//Y
				if(outdist[y*width/3 + x]>max_dist)
					max_dist = outdist[y*width/3 + x];
        	}else{
        		outdist[y*width/3 + x] = 0;//Y
        	}
        }
        g_free(distance);
    }

    /*
	 * Rows transformation
	 * */
    GST_DEBUG_OBJECT(DT, "Transforming rows");
    for(y=0; y<height; y++){
        for(x = 0; x < width; x++){
            pixels[x] = outdist[y*width + x];//Y
        }

        distance = gst_1D_dt(pixels, width);

        for(x=1; x<width-1; x++){
        	if(((x)>=DT->BBx) && ((x)<=(DT->BBx+DT->BBw)) && (y>=DT->BBy) && (y<=(DT->BBy+DT->BBh))){
				outdist[y*width/3 + x] = distance[x];//Y
				if(outdist[y*width/3 + x]>max_dist)
					max_dist = outdist[y*width/3 + x];
        	}else{
        		outdist[y*width/3 + x] = 0;//Y
        	}
        }
        g_free(distance);
    }

    if(DT->normalize){
    	//normalization of the frame
    	for(y=0; y<height; y++){
			for(x = 0; x<width; x++){
				if(((x)>=DT->BBx) && ((x)<=(DT->BBx+DT->BBw)) && (y>=DT->BBy) && (y<=(DT->BBy+DT->BBh))){
					//outdist[y*width + x] = min_val(outdist[y*width + x], (max_dist/10));
					gboolean temp = (gboolean)( ((outdist[y*width + x]-(max_dist*0.75)) > 0) != (src_ptr[y*width + x] >0) );
					//outdist[y*width + x] = max_val(outdist[y*width + x], (max_dist/100))-(max_dist/100);
					//src_ptr[y*width + x] = (guint8)((outdist[y*width + x]/(max_dist/10)) * 255);
					src_ptr[y*width + x] = temp ? 0 : 255 ;
				}

			}
    	}
    }else{
    	//normalization of the frame
    	for(y=0; y<height; y++){
			for(x = 0; x<width; x++){
				if(((x)>=DT->BBx) && ((x)<=(DT->BBx+DT->BBw)) && (y>=DT->BBy) && (y<=(DT->BBy+DT->BBh))){
					gboolean temp = (gboolean)( ((outdist[y*width + x]-(max_dist*0.75)) > 0) != (src_ptr[y*width + x] >0) );
					src_ptr[y*width + x] = temp ? 0 : 1;
				}
			}
    	}
    }

    /*
	 * Passing the data from one buffer to the other
	 * */
	GST_DEBUG_OBJECT(DT, "Getting data to output buffer: size = <%i> Max_inte: %f", inbuff.size, max_dist);
	memcpy(outbuff.data, inbuff.data, inbuff.size);

    /*
     * Unmapping the data used to the system can use/free it without
     * any problem
     * */
    gst_buffer_unmap(DT->inframe->buffer, &inbuff);
    gst_buffer_unmap(DT->outframe->buffer, &outbuff);

    g_free(pixels);
    g_free(outdist);

    /* returning error if any */
	return ierror;
}


/*
 * Scanning the 2 dimensions of the incoming image
 * */
static gboolean gst_2D_dt_RGB(GstDistanceTransform *DT){
	gboolean ierror = FALSE;//no error at all
	GstMapInfo inbuff, outbuff;
	guint32 width = DT->width;
	guint32 height = DT->height;
	gfloat *pixels, *outdist;
	guint8 *src_ptr;
	guint32 x, y;
	gfloat max_dist=0;
	gfloat factor = 0.01;//factor for normalization



	GST_DEBUG_OBJECT(DT, "Implementing gst_2D_dt_RGB: <%i, %i>, BB: <%i, %i, %i, %i>", width, height,
			DT->BBy, DT->BBx, DT->BBw, DT->BBh);
	pixels = g_malloc0(sizeof(gfloat)*(max_val(height,width/3)));
	outdist = g_malloc0(sizeof(gfloat)*(height*(width/3)));

	/*
	 * Mapping data to be used from the frames
	 * */
	//GST_OBJECT_LOCK(DT);
	gst_buffer_map(DT->inframe->buffer, &inbuff, GST_MAP_READ);
	DT->outframe->buffer = gst_buffer_make_writable(DT->outframe->buffer);
	gst_buffer_map(DT->outframe->buffer, &outbuff, GST_MAP_READWRITE);

	src_ptr = inbuff.data;

	for(y=0; y<height; y++){
		for(x = 0; x<width; x+=3){
			if(src_ptr[y*width + x]==0){//inside BB
				outdist[(y*width + x)/3] = 0.0f;//R
			}
			else{
				//Region of interest
				outdist[(y*width + x)/3] = INF;//R
			}
		}
	}

    /*
     * Columns transformation
     * */
	GST_DEBUG_OBJECT(DT, "Transforming columns");
    for(x=0; x<width/3; x++){
        for(y=0; y<height; y++){
        	//only one channels contains the info
        	pixels[y] = outdist[y*width/3 + x];//Y
        }
        gfloat *distance = gst_1D_dt(pixels, height);
        for(y=0; y<height; y++){
        	if(((x)>=DT->BBx) && ((x)<=(DT->BBx+DT->BBw)) && (y>=DT->BBy) && (y<=(DT->BBy+DT->BBh))){
				outdist[y*width/3 + x] = distance[y];//Y
				if(outdist[y*width/3 + x]>max_dist)
					max_dist = outdist[y*width/3 + x];
        	}else{
        		outdist[y*width/3 + x] = 0;//Y
        	}
        }
        g_free(distance);
    }

    /*
	 * Rows transformation
	 * */
    GST_DEBUG_OBJECT(DT, "Transforming rows");
    for(y=0; y<height; y++){
        for(x = 0; x<width/3; x++){
            pixels[x] = outdist[y*width/3 + x];//Y
        }

        gfloat *distance = gst_1D_dt(pixels, width/3);

        for(x=0; x<width/3; x++){
        	if(((x)>=DT->BBx) && ((x)<=(DT->BBx+DT->BBw)) && (y>=DT->BBy) && (y<=(DT->BBy+DT->BBh))){
				outdist[y*width/3 + x] = distance[x];//Y
				if(outdist[y*width/3 + x]>max_dist)
					max_dist = outdist[y*width/3 + x];
        	}else{
        		outdist[y*width/3 + x] = 0;//Y
        	}
        }
        g_free(distance);
    }

    if(DT->normalize){
    	//normalization of the frame
    	for(y=0; y<height; y++){
			for(x = 0; x<width; x+=3){
				if(((x/3)>=DT->BBx) && ((x/3)<=(DT->BBx+DT->BBw)) && (y>=DT->BBy) && (y<=(DT->BBy+DT->BBh))){
					//outdist[(y*width + x)/3] = min_val(outdist[(y*width + x)/3], (max_dist/factor));
					//outdist[y*width + x] = max_val(outdist[y*width + x], (max_dist/100))-(max_dist/100);
					gboolean temp = (gboolean)( ((sqrt((outdist[(y*width + x)/3])) - (/*sqrt(max_dist)*/ width*factor)) > 0) != (src_ptr[y*width + x] > 0) );
					src_ptr[y*width + x] = (guint8)(temp ? 0 : 255);//((outdist[(y*width + x)/3]/(max_dist/factor)) * 255);//R
					src_ptr[y*width + x + 1] = (guint8)(temp ? 0 : 255);//((outdist[(y*width + x)/3]/(max_dist/factor)) * 255);//G
					src_ptr[y*width + x + 2] = (guint8)(temp ? 0 : 255);//((outdist[(y*width + x)/3]/(max_dist/factor)) * 255);//B
				}else{
					src_ptr[y*width + x] = 0;
					src_ptr[y*width + x + 1] = 0;
					src_ptr[y*width + x + 2] = 0;
				}
			}
    	}
    }else{
    	//normalization of the frame
    	for(y=0; y<height; y++){
			for(x = 0; x<width; x+=3){
				if(((x/3)>=DT->BBx) && ((x/3)<=(DT->BBx+DT->BBw)) && (y>=DT->BBy) && (y<=(DT->BBy+DT->BBh))){
					gboolean temp = (gboolean)( ((outdist[(y*width + x)/3] - (max_dist/factor)) > 0) != (src_ptr[y*width + x] > 0) );
					src_ptr[y*width + x] = (guint8)(temp ? 0 : 1);//((outdist[(y*width + x)/3]/(max_dist/50)) * 255);//R
					src_ptr[y*width + x + 1] = (guint8)(temp ? 0 : 1);//((outdist[(y*width + x)/3]/(max_dist/50)) * 255);//G
					src_ptr[y*width + x + 2] = (guint8)(temp ? 0 : 1);//((outdist[(y*width + x)/3]/(max_dist/50)) * 255);//B
				}else{
					src_ptr[y*width + x] = 0;
					src_ptr[y*width + x + 1] = 0;
					src_ptr[y*width + x + 2] = 0;
				}

			}
    	}
    }

    /*
	 * Passing the data from one buffer to the other
	 * */
	GST_DEBUG_OBJECT(DT, "Getting data to output buffer: sizes = <%i, %i> Max_distance: %f", inbuff.size, outbuff.size, max_dist);
	memcpy(outbuff.data, inbuff.data, inbuff.size);

    /*
     * Unmapping the data used to the system can use/free it without
     * any problem
     * */
    gst_buffer_unmap(DT->inframe->buffer, &inbuff);
    gst_buffer_unmap(DT->outframe->buffer, &outbuff);
    //GST_OBJECT_UNLOCK(DT);

    g_free(pixels);
    g_free(outdist);

    /* returning error if any */
	return ierror;
}









#if 0
static gboolean plugin_init(GstPlugin * plugin){
  return gst_element_register (plugin, "distancetransform", GST_RANK_NONE, GST_TYPE_DISTANCETRANSFORM);
}


#ifndef VERSION
#define VERSION "0.0.1"
#endif
#ifndef PACKAGE
#define PACKAGE "Distance_Transform_API"
#endif
#ifndef PACKAGE_NAME
#define PACKAGE_NAME "Image_proc_API"
#endif
#ifndef GST_PACKAGE_ORIGIN
#define GST_PACKAGE_ORIGIN "None"
#endif

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR, GST_VERSION_MINOR, distancetransform,
		"Distance transformation of binary images, outputs a gray level image where the darker represent the larger distance "
		"from the non-object part of the image", plugin_init, VERSION, "LGPL", PACKAGE_NAME, GST_PACKAGE_ORIGIN)
#endif

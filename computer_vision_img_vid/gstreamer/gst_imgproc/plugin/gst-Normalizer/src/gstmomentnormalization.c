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
 * SECTION:element-gstmomentnormalization
 *
 * The momentnormalization element does FIXME stuff.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v fakesrc ! momentnormalization ! FIXME ! fakesink
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
#include <time.h>
#include "gstmomentnormalization.h"
#include "../../common/metainfo.h"
#include "includes/MomentNormalizer.h"

GST_DEBUG_CATEGORY_STATIC (gst_momentnormalization_debug_category);
#define GST_CAT_DEFAULT gst_momentnormalization_debug_category

#define NORMALIZE_DEFAULT (0)
#define OUTPUTLOG_DEFAULT (0)
#define DEFAULT_LOCATION ("loginfo.txt")
#define PAD_X_DEFAULT (MAX_PAD)
#define PAD_Y_DEFAULT (MAX_PAD)

static gint global_xpad, global_ypad;

/*
 * Functions prototypes
 * */
static void gst_momentnormalization_set_property (GObject * object,
    guint property_id, const GValue * value, GParamSpec * pspec);
static void gst_momentnormalization_get_property (GObject * object,
    guint property_id, GValue * value, GParamSpec * pspec);
static void gst_momentnormalization_dispose (GObject * object);
static void gst_momentnormalization_finalize (GObject * object);

static gboolean gst_momentnormalization_start (GstBaseTransform * trans);
static gboolean gst_momentnormalization_stop (GstBaseTransform * trans);
static gboolean gst_momentnormalization_set_info (GstVideoFilter * filter, GstCaps * incaps,
    GstVideoInfo * in_info, GstCaps * outcaps, GstVideoInfo * out_info);
static GstFlowReturn gst_momentnormalization_transform_frame (GstVideoFilter * filter,
    GstVideoFrame * inframe, GstVideoFrame * outframe);

//for caps negotiation
static gboolean gst_momentnormalization_set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps);

/* for fixing output window */
static GstCaps *gst_momentnormalization_fixate_caps (GstBaseTransform * base,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps);
static GstCaps * gst_momentnormalization_fixate_caps(GstBaseTransform * base, GstPadDirection direction,
    GstCaps * caps, GstCaps * othercaps);

enum
{
  PROP_0,
  NORMALIZE_PATCH,
  PAD_X,
  PAD_Y,
  OUTPUTLOG,
  PROP_LOCATION,
  PROP_INTERNAL_PT
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


/* class initialization */

G_DEFINE_TYPE_WITH_CODE (GstMomentNormalization, gst_momentnormalization, GST_TYPE_VIDEO_FILTER,
		GST_DEBUG_CATEGORY_INIT (gst_momentnormalization_debug_category, "momentnormalization", 0,
				"debug category for momentnormalization element"));

static void gst_momentnormalization_class_init (GstMomentNormalizationClass * normalization){
  GObjectClass *gobject_class = G_OBJECT_CLASS (normalization);
  GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS (normalization);
  GstVideoFilterClass *video_filter_class = GST_VIDEO_FILTER_CLASS (normalization);

  normalizer_debug_init();

  /*
   * Function pointers for creation of the element
   * */
  gobject_class->set_property = gst_momentnormalization_set_property;
  gobject_class->get_property = gst_momentnormalization_get_property;
  gobject_class->dispose = gst_momentnormalization_dispose;
  gobject_class->finalize = gst_momentnormalization_finalize;
  base_transform_class->start = GST_DEBUG_FUNCPTR (gst_momentnormalization_start);
  base_transform_class->stop = GST_DEBUG_FUNCPTR (gst_momentnormalization_stop);
  base_transform_class->set_caps = GST_DEBUG_FUNCPTR(gst_momentnormalization_set_caps);
  video_filter_class->set_info = GST_DEBUG_FUNCPTR (gst_momentnormalization_set_info);
  video_filter_class->transform_frame = GST_DEBUG_FUNCPTR (gst_momentnormalization_transform_frame);


  gst_element_class_set_static_metadata (GST_ELEMENT_CLASS(normalization), "PeiLin moment normalization", "VideoFilter",
  		  "Pei&Lin moment normalization implementation for binarized, where a bounding box of the object to normalized is given",
  		  "somecpmpnay <josuercuevas@gmail.com>");


  /*
	* Installation of the normalization property for this element
	* */
	g_object_class_install_property (gobject_class, NORMALIZE_PATCH, g_param_spec_int("normalize", "normalize",
			"Black and White output image: 1: normalize 0:not normalize, default=0", 0,
			1, NORMALIZE_DEFAULT, G_PARAM_READABLE | G_PARAM_WRITABLE));


	/*
	* Installation of the padding in X property for this element
	* */
	g_object_class_install_property (gobject_class, PAD_X, g_param_spec_int("padx", "padx",
			"Padding to be used when normalizing the patches, default= 512 pixels", 0,
			1280, PAD_X_DEFAULT, G_PARAM_READABLE | G_PARAM_WRITABLE));

	/*
	* Installation of the padding in Y property for this element
	* */
	g_object_class_install_property (gobject_class, PAD_Y, g_param_spec_int("pady", "pady",
			"Padding to be used when normalizing the patches, default= 512 pixels", 0,
			1280, PAD_Y_DEFAULT, G_PARAM_READABLE | G_PARAM_WRITABLE));

	/*
	* in charge of activating the log file output
	* */
	g_object_class_install_property (gobject_class, OUTPUTLOG, g_param_spec_int("output-log", "Output Log File",
			"Outputs the log file for the img_proc detector: 1:YES 0:NO, default=0", 0,
			1, OUTPUTLOG_DEFAULT, G_PARAM_READABLE | G_PARAM_WRITABLE));

	/*
	* path of the file to be used for output log information
	* */
	g_object_class_install_property (gobject_class, PROP_LOCATION, g_param_spec_string ("location", "File Location",
	          "Location of the file to write", NULL,  G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

	g_object_class_install_property (gobject_class, PROP_INTERNAL_PT, g_param_spec_float("internal-pt", "Internal processing time",
				"Time taken to processing a frame", 0,
				9999999, 0, G_PARAM_READABLE));


  /*
   * Setting up pads and setting metadata should be moved to
   * base_class_init if you intend to subclass this class.
   * */
	gst_element_class_add_pad_template (GST_ELEMENT_CLASS(normalization), gst_pad_template_new ("src", GST_PAD_SRC, GST_PAD_ALWAYS,
			gst_caps_from_string (VIDEO_SRC_CAPS)));
	gst_element_class_add_pad_template (GST_ELEMENT_CLASS(normalization), gst_pad_template_new ("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
			gst_caps_from_string (VIDEO_SINK_CAPS)));


	/*
	 * For caps transformation handling
	 * */
	base_transform_class->fixate_caps = GST_DEBUG_FUNCPTR(gst_momentnormalization_fixate_caps);

}

static void gst_momentnormalization_init (GstMomentNormalization *momentnormalization)
{
	//initialization of the parameters to be used to interact with
	//the normalization part of this VideoFilter
	momentnormalization->padding_x = PAD_X_DEFAULT;
	global_xpad = PAD_X_DEFAULT;
	momentnormalization->padding_y = PAD_Y_DEFAULT;
	global_ypad = PAD_Y_DEFAULT;
	momentnormalization->patch_normalization = NORMALIZE_DEFAULT;
	momentnormalization->base_momentnormalization.negotiated = FALSE;
	momentnormalization->output_logfile = OUTPUTLOG_DEFAULT;
	momentnormalization->location = g_strdup(DEFAULT_LOCATION);
	momentnormalization->img_proc_Log = NULL;
	momentnormalization->any_blob = FALSE;
	momentnormalization->any_buffer = FALSE;
	momentnormalization->Prev_Norm_Blobs = NULL;
	momentnormalization->internal_pt=0.f;
}


/*
 * Setting the values of this property for this element
 * */
void gst_momentnormalization_set_property (GObject * object, guint property_id, const GValue * value, GParamSpec * pspec){
  GstMomentNormalization *momentnormalization = GST_MOMENTNORMALIZATION (object);

  GST_DEBUG_OBJECT (momentnormalization, "set_property");

  switch (property_id) {
	case NORMALIZE_PATCH://to see if the user wants to normalize the output frame to 255
		momentnormalization->patch_normalization = g_value_get_int(value);
		break;
	case PAD_X://padding
		momentnormalization->padding_x = g_value_get_int(value);
		global_xpad = momentnormalization->padding_x;
		break;
	case PAD_Y://padding
		momentnormalization->padding_y = g_value_get_int(value);
		global_ypad = momentnormalization->padding_y;
		break;
	case OUTPUTLOG://output log file
		momentnormalization->output_logfile = g_value_get_int(value);
		break;
    case PROP_LOCATION://location of the file
    	if(momentnormalization->location){
    		g_free(momentnormalization->location);
    	}
      momentnormalization->location = g_value_dup_string(value);
      break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
		break;
  }
}


/*
 * Getting the values of this property for this element
 * */
void gst_momentnormalization_get_property (GObject * object, guint property_id, GValue * value, GParamSpec * pspec){
	GstMomentNormalization *momentnormalization = GST_MOMENTNORMALIZATION (object);

	GST_DEBUG_OBJECT (momentnormalization, "get_property");

	switch (property_id) {
		case NORMALIZE_PATCH://to see if the user wants to normalize the output frame to 255
			g_value_set_int(value, momentnormalization->patch_normalization);
			break;
		case PAD_X://padding
			g_value_set_int(value, momentnormalization->padding_x);
			break;
		case PAD_Y://padding
			g_value_set_int(value, momentnormalization->padding_y);
			break;
		case OUTPUTLOG://output log file
			g_value_set_int(value, momentnormalization->output_logfile);
			break;
		case PROP_LOCATION://location of the file
			g_value_set_string (value, momentnormalization->location);
			break;
		case PROP_INTERNAL_PT://
			g_value_set_float(value, momentnormalization->internal_pt);
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
void gst_momentnormalization_dispose (GObject * object){
  GstMomentNormalization *momentnormalization = GST_MOMENTNORMALIZATION (object);

  GST_DEBUG_OBJECT (momentnormalization, "dispose");

  /* clean up as possible.  may be called multiple times */

  G_OBJECT_CLASS (gst_momentnormalization_parent_class)->dispose (object);
}

/*
 * Cleans all the garbage information collected ... in our case
 * there is none for now
 * */
void gst_momentnormalization_finalize (GObject * object){
  GstMomentNormalization *momentnormalization = GST_MOMENTNORMALIZATION (object);

  GST_DEBUG_OBJECT (momentnormalization, "finalize");

  /* clean up object here */

  G_OBJECT_CLASS (gst_momentnormalization_parent_class)->finalize (object);
}

static gboolean gst_momentnormalization_start (GstBaseTransform * trans){
	GstMomentNormalization *momentnormalization = GST_MOMENTNORMALIZATION (trans);

	/*
	* Just puts some debugging information for the comfortability
	* of the user
	* */
	if(momentnormalization->patch_normalization != NORMALIZE_DEFAULT)
		GST_DEBUG_OBJECT (momentnormalization, "Output image will be normalized to 255");
	else
		GST_DEBUG_OBJECT (momentnormalization, "Output image is going to be binary (0,1)");

	GST_DEBUG_OBJECT (momentnormalization, "start");

  return TRUE;
}

static gboolean gst_momentnormalization_stop (GstBaseTransform * trans){
	GstMomentNormalization *momentnormalization = GST_MOMENTNORMALIZATION (trans);
	guint i;
	/*
	* Just puts some debugging information for the comfortability
	* of the user
	* */
	GST_DEBUG_OBJECT (momentnormalization, "stop");

	/*
	 * UNREF THE PREVIOUS BUFFER
	 * */
	if(momentnormalization->Prev_Norm_Blobs){
		for(i=0;i<momentnormalization->n_prev_blobs;i++){
			/*
			 * Freeing the blob's data
			 * */
			g_free(momentnormalization->Prev_Norm_Blobs[i].NormalizedBlob);
		}

		/*
		 * Free all the pointers to the blobs
		 * */
		g_free(momentnormalization->Prev_Norm_Blobs);
		momentnormalization->Prev_Norm_Blobs = NULL;
		momentnormalization->any_blob = FALSE;
	}

	/*
	 * Flushing all information to log file
	 * and closing if we created it
	 * */
	if(momentnormalization->output_logfile){
		fflush(momentnormalization->img_proc_Log);
		fclose(momentnormalization->img_proc_Log);
		g_free(momentnormalization->location);
		momentnormalization->img_proc_Log = NULL;
		momentnormalization->location = NULL;
		momentnormalization->output_logfile = OUTPUTLOG_DEFAULT;
	}

	if(momentnormalization->any_buffer){
		gst_buffer_unref(momentnormalization->prevBuffer);
		momentnormalization->prevBuffer = NULL;
		momentnormalization->any_buffer=FALSE;
	}

	return TRUE;
}

static GstCaps * gst_momentnormalization_fixate_caps(GstBaseTransform * base, GstPadDirection direction,
    GstCaps * caps, GstCaps * othercaps){
	GstStructure *ins, *outs;
	const GValue *from_par, *to_par;
	GValue fpar = { 0, }, tpar = {0,};
	gint wl = 0, hl = 0;

	othercaps = gst_caps_truncate(othercaps);
	othercaps = gst_caps_make_writable(othercaps);

	GST_DEBUG_OBJECT(base, "***** trying to fixate othercaps ***** NEW: %" GST_PTR_FORMAT
			" based on caps OLD: %" GST_PTR_FORMAT, othercaps, caps);

	ins = gst_caps_get_structure (caps, 0);
	outs = gst_caps_get_structure (othercaps, 0);

	from_par = gst_structure_get_value(ins, "pixel-aspect-ratio");
	to_par = gst_structure_get_value(outs, "pixel-aspect-ratio");

	/*
	 * If we're fixating from the sinkpad we always set the PAR and
	 * assume that missing PAR on the sinkpad means 1/1 and
	 * missing PAR on the srcpad means undefined
	*/
	if (direction == GST_PAD_SINK){
		if (!from_par){
			g_value_init (&fpar, GST_TYPE_FRACTION);
			gst_value_set_fraction (&fpar, 1, 1);
			from_par = &fpar;
		}
		if (!to_par){
			g_value_init(&tpar, GST_TYPE_FRACTION_RANGE);
			gst_value_set_fraction_range_full(&tpar, 1, G_MAXINT, G_MAXINT, 1);
			to_par = &tpar;
		}
	}else{
		if (!to_par){
			g_value_init (&tpar, GST_TYPE_FRACTION);
			gst_value_set_fraction(&tpar, 1, 1);
			to_par = &tpar;

			gst_structure_set(outs, "pixel-aspect-ratio", GST_TYPE_FRACTION, 1, 1, NULL);
		}
		if (!from_par){
			g_value_init(&fpar, GST_TYPE_FRACTION);
			gst_value_set_fraction(&fpar, 1, 1);
			from_par = &fpar;
		}
	}

	/*
	 * we have both PAR but they might not be fixated
	 * */
	{
		gint from_w, from_h, from_par_n, from_par_d, to_par_n, to_par_d;
		gint w = 0, h = 0;
		gint from_dar_n, from_dar_d;
		gint num, den;

		/*
		 * from_par should be fixed
		 * */
		g_return_val_if_fail(gst_value_is_fixed(from_par), othercaps);

		from_par_n = gst_value_get_fraction_numerator(from_par);
		from_par_d = gst_value_get_fraction_denominator(from_par);

		gst_structure_get_int(ins, "width", &from_w);
		gst_structure_get_int(ins, "height", &from_h);

		gst_structure_get_int(outs, "width", &w);
		gst_structure_get_int(outs, "height", &h);

		/*
		 * if both width and height are already fixed, we can't do anything
		 * about it anymore
		 * */
		if (w==global_xpad && h==global_ypad) {
			guint n, d;

			GST_DEBUG_OBJECT(base, "dimensions already set to %dx%d, not fixating", w, h);
			if (!gst_value_is_fixed (to_par)){
				if (gst_video_calculate_display_ratio(&n, &d, from_w, from_h,
						from_par_n, from_par_d, w, h)){
				  GST_DEBUG_OBJECT(base, "fixating to_par to %dx%d", n, d);
				  if (gst_structure_has_field (outs, "pixel-aspect-ratio"))
					  gst_structure_fixate_field_nearest_fraction(outs, "pixel-aspect-ratio", n, d);
				  else if (n != d)
					gst_structure_set(outs, "pixel-aspect-ratio", GST_TYPE_FRACTION, n, d, NULL);
				}
			}
			goto done;
		}

		/* Calculate input DAR */
		if (!gst_util_fraction_multiply(from_w, from_h, from_par_n, from_par_d, &from_dar_n, &from_dar_d)) {
			GST_ELEMENT_ERROR(base, CORE, NEGOTIATION, (NULL),
					("Error calculating the output scaled size - integer overflow"));
			goto done;
		}

		GST_DEBUG_OBJECT(base, "Input DAR is %d/%d", from_dar_n, from_dar_d);

		/* If either width or height are fixed there's not much we
		 * can do either except choosing a height or width and PAR
		 * that matches the DAR as good as possible
		 */
		if (h == global_ypad){
			GstStructure *tmp;
			gint set_w, set_par_n, set_par_d;

			GST_DEBUG_OBJECT(base, "height is fixed (%d)", h);

			/* If the PAR is fixed too, there's not much to do
			* except choosing the width that is nearest to the
			* width with the same DAR */
			if (gst_value_is_fixed (to_par)){
				to_par_n = gst_value_get_fraction_numerator(to_par);
				to_par_d = gst_value_get_fraction_denominator(to_par);

				GST_DEBUG_OBJECT (base, "PAR is fixed %d/%d", to_par_n, to_par_d);

				if (!gst_util_fraction_multiply(from_dar_n, from_dar_d, to_par_d, to_par_n, &num, &den)) {
				  GST_ELEMENT_ERROR (base, CORE, NEGOTIATION, (NULL),
						  ("Error calculating the output scaled size - integer overflow"));
				  goto done;
				}

				w = (guint) gst_util_uint64_scale_int(h, num, den);
				gst_structure_fixate_field_nearest_int(outs, "width", w);

				goto done;
			}

			/* The PAR is not fixed and it's quite likely that we can set
			* an arbitrary PAR. */

			/* Check if we can keep the input width */
			tmp = gst_structure_copy(outs);
			gst_structure_fixate_field_nearest_int(tmp, "width", from_w);
			gst_structure_get_int(tmp, "width", &set_w);

			/* Might have failed but try to keep the DAR nonetheless by
			* adjusting the PAR */
			if (!gst_util_fraction_multiply(from_dar_n, from_dar_d, h, set_w, &to_par_n, &to_par_d)){
				GST_ELEMENT_ERROR(base, CORE, NEGOTIATION, (NULL),
					("Error calculating the output scaled size - integer overflow"));
				gst_structure_free(tmp);
				goto done;
			}

			if (!gst_structure_has_field(tmp, "pixel-aspect-ratio"))
				gst_structure_set_value(tmp, "pixel-aspect-ratio", to_par);

			gst_structure_fixate_field_nearest_fraction(tmp, "pixel-aspect-ratio", to_par_n, to_par_d);
			gst_structure_get_fraction(tmp, "pixel-aspect-ratio", &set_par_n, &set_par_d);
			gst_structure_free(tmp);

			/*
			 * Check if the adjusted PAR is accepted
			 * */
			if (set_par_n == to_par_n && set_par_d == to_par_d){
				if (gst_structure_has_field(outs, "pixel-aspect-ratio") || set_par_n != set_par_d)
					gst_structure_set(outs, "width", G_TYPE_INT, set_w, "pixel-aspect-ratio",
							GST_TYPE_FRACTION, set_par_n, set_par_d, NULL);
				goto done;
			}

			/*
			 * Otherwise scale the width to the new PAR and check if the
			 * adjusted with is accepted. If all that fails we can't keep
			 * the DAR
			 * */
			if (!gst_util_fraction_multiply(from_dar_n, from_dar_d, set_par_d, set_par_n, &num, &den)){
				GST_ELEMENT_ERROR(base, CORE, NEGOTIATION, (NULL),
						("Error calculating the output scaled size - integer overflow"));
				goto done;
			}

			w = (guint) gst_util_uint64_scale_int(h, num, den);
			gst_structure_fixate_field_nearest_int(outs, "width", w);
			if (gst_structure_has_field(outs, "pixel-aspect-ratio") || set_par_n != set_par_d)
				gst_structure_set(outs, "pixel-aspect-ratio", GST_TYPE_FRACTION, set_par_n, set_par_d, NULL);

			goto done;
		}else if (w == global_xpad){
			GstStructure *tmp;
			gint set_h, set_par_n, set_par_d;

			GST_DEBUG_OBJECT(base, "width is fixed (%d)", w);

			/* If the PAR is fixed too, there's not much to do
			* except choosing the height that is nearest to the
			* height with the same DAR */
			if (gst_value_is_fixed (to_par)) {
				to_par_n = gst_value_get_fraction_numerator(to_par);
				to_par_d = gst_value_get_fraction_denominator(to_par);

				GST_DEBUG_OBJECT(base, "PAR is fixed %d/%d", to_par_n, to_par_d);

				if (!gst_util_fraction_multiply (from_dar_n, from_dar_d, to_par_d,
						to_par_n, &num, &den)){
					GST_ELEMENT_ERROR (base, CORE, NEGOTIATION, (NULL),
						  ("Error calculating the output scaled size - integer overflow"));
					goto done;
				}

				h = (guint) gst_util_uint64_scale_int(w, den, num);
				gst_structure_fixate_field_nearest_int(outs, "height", h);

				goto done;
			}

			/* The PAR is not fixed and it's quite likely that we can set
			* an arbitrary PAR. */

			/* Check if we can keep the input height */
			tmp = gst_structure_copy(outs);
			gst_structure_fixate_field_nearest_int(tmp, "height", from_h);
			gst_structure_get_int(tmp, "height", &set_h);

			/* Might have failed but try to keep the DAR nonetheless by
			* adjusting the PAR */
			if (!gst_util_fraction_multiply(from_dar_n, from_dar_d, set_h, w, &to_par_n, &to_par_d)){
				GST_ELEMENT_ERROR (base, CORE, NEGOTIATION, (NULL),
						("Error calculating the output scaled size - integer overflow"));
				gst_structure_free (tmp);
				goto done;
			}

			if (!gst_structure_has_field(tmp, "pixel-aspect-ratio"))
				gst_structure_set_value(tmp, "pixel-aspect-ratio", to_par);

			gst_structure_fixate_field_nearest_fraction(tmp, "pixel-aspect-ratio", to_par_n, to_par_d);
			gst_structure_get_fraction(tmp, "pixel-aspect-ratio", &set_par_n, &set_par_d);
			gst_structure_free(tmp);

			/* Check if the adjusted PAR is accepted */
			if (set_par_n == to_par_n && set_par_d == to_par_d) {
				if (gst_structure_has_field(outs, "pixel-aspect-ratio") || set_par_n != set_par_d)
					gst_structure_set (outs, "height", G_TYPE_INT, set_h,
							"pixel-aspect-ratio", GST_TYPE_FRACTION, set_par_n, set_par_d, NULL);
				goto done;
			}

			/* Otherwise scale the height to the new PAR and check if the
			* adjusted with is accepted. If all that fails we can't keep
			* the DAR */
			if (!gst_util_fraction_multiply(from_dar_n, from_dar_d, set_par_d,
				  set_par_n, &num, &den)){
				GST_ELEMENT_ERROR (base, CORE, NEGOTIATION, (NULL),
						("Error calculating the output scaled size - integer overflow"));
				goto done;
			}

			h = (guint) gst_util_uint64_scale_int(w, den, num);
			gst_structure_fixate_field_nearest_int(outs, "height", h);

			if (gst_structure_has_field(outs, "pixel-aspect-ratio") || set_par_n != set_par_d)
				gst_structure_set(outs, "pixel-aspect-ratio", GST_TYPE_FRACTION, set_par_n, set_par_d, NULL);

			goto done;
		}else if(gst_value_is_fixed(to_par)){
			GstStructure *tmp;
			gint set_h, set_w, f_h, f_w;

			GST_DEBUG_OBJECT(base, "FIXATING W and H at TOPAR");

			to_par_n = gst_value_get_fraction_numerator(to_par);
			to_par_d = gst_value_get_fraction_denominator(to_par);

			/* Calculate scale factor for the PAR change */
			if (!gst_util_fraction_multiply (from_dar_n, from_dar_d, to_par_n,
				  to_par_d, &num, &den)) {
				GST_ELEMENT_ERROR(base, CORE, NEGOTIATION, (NULL),
						("Error calculating the output scaled size - integer overflow"));
				goto done;
			}

			/* Try to keep the input height (because of interlacing) */
			tmp = gst_structure_copy (outs);
			gst_structure_fixate_field_nearest_int (tmp, "height", global_ypad);
			gst_structure_get_int (tmp, "height", &set_h);

			/* This might have failed but try to scale the width
			* to keep the DAR nonetheless */
			w = (guint) gst_util_uint64_scale_int (set_h, num, den);
			gst_structure_fixate_field_nearest_int (tmp, "width", global_xpad);
			gst_structure_get_int (tmp, "width", &set_w);
			gst_structure_free (tmp);

			/* We kept the DAR and the height is nearest to the original height */
			if (set_w == global_xpad) {
				gst_structure_set (outs, "width", G_TYPE_INT, set_w, "height",
						G_TYPE_INT, set_h, NULL);
				goto done;
			}

			f_h = set_h;
			f_w = set_w;

			/* If the former failed, try to keep the input width at least */
			tmp = gst_structure_copy (outs);
			gst_structure_fixate_field_nearest_int (tmp, "width", global_xpad);
			gst_structure_get_int (tmp, "width", &set_w);

			/* This might have failed but try to scale the width
			* to keep the DAR nonetheless */
			h = (guint) gst_util_uint64_scale_int (set_w, den, num);
			gst_structure_fixate_field_nearest_int (tmp, "height", global_ypad);
			gst_structure_get_int (tmp, "height", &set_h);
			gst_structure_free (tmp);

			/* We kept the DAR and the width is nearest to the original width */
			if (set_h == global_ypad) {
				gst_structure_set (outs, "width", G_TYPE_INT, set_w, "height",
						G_TYPE_INT, set_h, NULL);
				goto done;
			}

			/* If all this failed, keep the height that was nearest to the orignal
			* height and the nearest possible width. This changes the DAR but
			* there's not much else to do here.
			*/
			gst_structure_set (outs, "width", G_TYPE_INT, global_xpad, "height", G_TYPE_INT,
				  global_ypad, NULL);

			GST_DEBUG_OBJECT (base, "GOTTEN INFO ---->> W: %i, H: %i" , set_w, set_h);
			goto done;
		}else{
			GstStructure *tmp;
			gint set_h, set_w, set_par_n, set_par_d, tmp2;

			/* width, height and PAR are not fixed but passthrough is not possible */

			/* First try to keep the height and width as good as possible
			* and scale PAR */
			tmp = gst_structure_copy (outs);
			gst_structure_fixate_field_nearest_int (tmp, "height", from_h);
			gst_structure_get_int (tmp, "height", &set_h);
			gst_structure_fixate_field_nearest_int (tmp, "width", from_w);
			gst_structure_get_int (tmp, "width", &set_w);

			if (!gst_util_fraction_multiply (from_dar_n, from_dar_d, set_h, set_w,
				  &to_par_n, &to_par_d)) {
				GST_ELEMENT_ERROR (base, CORE, NEGOTIATION, (NULL),
						("Error calculating the output scaled size - integer overflow"));
				goto done;
			}

			if (!gst_structure_has_field (tmp, "pixel-aspect-ratio"))
				gst_structure_set_value (tmp, "pixel-aspect-ratio", to_par);

			gst_structure_fixate_field_nearest_fraction (tmp, "pixel-aspect-ratio",
			  to_par_n, to_par_d);

			gst_structure_get_fraction (tmp, "pixel-aspect-ratio", &set_par_n,
			  &set_par_d);

			gst_structure_free (tmp);

			if (set_par_n == to_par_n && set_par_d == to_par_d){
				gst_structure_set (outs, "width", G_TYPE_INT, set_w, "height",
					G_TYPE_INT, set_h, NULL);

				if (gst_structure_has_field (outs, "pixel-aspect-ratio") || set_par_n != set_par_d)
				  gst_structure_set (outs, "pixel-aspect-ratio", GST_TYPE_FRACTION, set_par_n, set_par_d, NULL);

				goto done;
			}

			/* Otherwise try to scale width to keep the DAR with the set
			* PAR and height */
			if (!gst_util_fraction_multiply (from_dar_n, from_dar_d, set_par_d,
				  set_par_n, &num, &den)){
				GST_ELEMENT_ERROR (base, CORE, NEGOTIATION, (NULL),
						("Error calculating the output scaled size - integer overflow"));
				goto done;
			}

			w = (guint) gst_util_uint64_scale_int (set_h, num, den);
			tmp = gst_structure_copy (outs);
			gst_structure_fixate_field_nearest_int (tmp, "width", w);
			gst_structure_get_int (tmp, "width", &tmp2);
			gst_structure_free (tmp);

			if (tmp2 == w){
				gst_structure_set (outs, "width", G_TYPE_INT, tmp2, "height", G_TYPE_INT, set_h, NULL);
				if (gst_structure_has_field (outs, "pixel-aspect-ratio") || set_par_n != set_par_d)
					gst_structure_set (outs, "pixel-aspect-ratio", GST_TYPE_FRACTION,
							set_par_n, set_par_d, NULL);
				goto done;
			}

			/* ... or try the same with the height */
			h = (guint) gst_util_uint64_scale_int (set_w, den, num);
			tmp = gst_structure_copy (outs);
			gst_structure_fixate_field_nearest_int (tmp, "height", h);
			gst_structure_get_int (tmp, "height", &tmp2);
			gst_structure_free (tmp);

			if (tmp2 == h){
				gst_structure_set (outs, "width", G_TYPE_INT, set_w, "height", G_TYPE_INT, tmp2, NULL);
				if (gst_structure_has_field (outs, "pixel-aspect-ratio") || set_par_n != set_par_d)
					gst_structure_set (outs, "pixel-aspect-ratio", GST_TYPE_FRACTION,
							set_par_n, set_par_d, NULL);
				goto done;
			}

			/* If all fails we can't keep the DAR and take the nearest values
			* for everything from the first try */
			gst_structure_set (outs, "width", G_TYPE_INT, set_w, "height", G_TYPE_INT, set_h, NULL);
			if (gst_structure_has_field (outs, "pixel-aspect-ratio") || set_par_n != set_par_d)
			gst_structure_set (outs, "pixel-aspect-ratio", GST_TYPE_FRACTION, set_par_n, set_par_d, NULL);
		}
	}

done:
	GST_DEBUG_OBJECT (base, "fixated othercaps to %" GST_PTR_FORMAT, othercaps);
	gst_structure_get_int (outs, "width", &wl);
	gst_structure_get_int (outs, "height", &hl);
	GST_DEBUG_OBJECT (base, "W: %i, H: %i" , wl, hl);

	if (from_par == &fpar)
		g_value_unset (&fpar);
	if (to_par == &tpar)
		g_value_unset (&tpar);

	return othercaps;
}

//Caps negotiation part
static gboolean gst_momentnormalization_set_caps(GstBaseTransform *trans, GstCaps *incaps, GstCaps *outcaps){
	GstVideoFilter *filter = GST_VIDEO_FILTER_CAST (trans);
	GstVideoFilterClass *fclass;
	GstVideoInfo in_info, out_info;
	gboolean res;
	const gchar *src_format, *sink_format;
	gchar *src_color, *sink_color;

	GST_DEBUG_OBJECT(filter, "Caps negotiation MomentNormalization..!!");
	GST_OBJECT_LOCK(filter);

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

	sink_format= gst_video_format_to_string (in_info.finfo->format);
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

	GST_DEBUG_OBJECT(filter, "INCOMING FRAMES: %" GST_PTR_FORMAT, incaps);
	GST_DEBUG_OBJECT(filter, "ONGOING FRAMES: %" GST_PTR_FORMAT, outcaps);


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
		GST_DEBUG_OBJECT(filter, "Moment Normalization VideoFilter has negotiated caps successfully..!!");

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

static gboolean gst_momentnormalization_set_info (GstVideoFilter * filter, GstCaps * incaps, GstVideoInfo * in_info,
		GstCaps * outcaps, GstVideoInfo * out_info){
	GstMomentNormalization *momentnormalization = GST_MOMENTNORMALIZATION (filter);
	//in case we need to set extra info for the element for processing purposes
	GST_DEBUG_OBJECT (momentnormalization, "setting info for the frames");

	return TRUE;
}


/*
 * Function to add the metainformation referring to the blobs
 * to be processed in the Normalization element in the pipeline
 * */
void* gst_buffer_add_momentnormalization_meta(GstBuffer *buffer, GstMomentNormalization *momentnormalization){
	guint i, boxes;
	GstVideoRegionOfInterestMeta *meta=NULL;

	//check is the incoming buffer is truly one
	g_return_val_if_fail(GST_IS_BUFFER(buffer), NULL);

	//making the buffer writable for adding the metadata
	buffer = gst_buffer_make_writable(buffer);

	//adding the metadata structure to the buffer
	meta = gst_buffer_add_video_region_of_interest_meta (buffer, "BBs", 0, 0, 0, 0);

	//everything is fine now we proceed to insert the information
	meta->x = momentnormalization->BBx;
	meta->y = momentnormalization->BBy;
	meta->h = momentnormalization->BBh;
	meta->w = momentnormalization->BBw;

	if(meta->x>=0 && meta->y>=0 && meta->w>=0 && meta->h>=0){
		GST_DEBUG_OBJECT(momentnormalization, "Meta normalization info: <%i, %i, %i, %i>", meta->x, meta->y, meta->w, meta->h);
	}else{
		GST_ERROR_OBJECT(momentnormalization, "******************************************************************************");
		GST_ERROR_OBJECT(momentnormalization, "PROBLEM WITH THE NORMALIZE PATCH GOING TO DISTANCE TRANSFORM");
		GST_ERROR_OBJECT(momentnormalization, "Meta normalization info: <%i, %i, %i, %i>", meta->x, meta->y, meta->w, meta->h);
		GST_ERROR_OBJECT(momentnormalization, "******************************************************************************");
	}

	return NULL;
}


/*
 * Main body of this element in charge of calling the functions or API
 * to perform PEILIN normalization
 *  */
static GstFlowReturn gst_momentnormalization_transform_frame(GstVideoFilter * filter, GstVideoFrame * inframe, GstVideoFrame * outframe){

	GstMomentNormalization *momentnormalization = GST_MOMENTNORMALIZATION (filter);
	const GstMetaInfo *info = gst_buffer_get_blob_meta(METAIMPL);
	GstMeta *meta;
	gpointer state = NULL;
	GstMetaBLOB *BlobMeta = NULL;
//	GstVideoRegionOfInterestMeta *ROImeta=NULL;
	gboolean process_it=FALSE;

	GstMapInfo ininfo, outinfo, tmpinfo;

	gint width, height, i, j;
	guint8 ierror;
	gint sstride, dstride[3], srcformat;


	if(!filter->negotiated){
		GST_DEBUG_OBJECT (momentnormalization, "Caps have NOT been negotiated, proceeding to negotiation phase..!!");
		GstBaseTransform *momentnormalizationBaseTransform = GST_BASE_TRANSFORM(filter);
		GstVideoFilterClass *momentnormalizationclass = GST_MOMENTNORMALIZATION_CLASS(filter);
		GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS(momentnormalizationclass);
		if(!base_transform_class->set_caps(momentnormalizationBaseTransform, momentnormalizationBaseTransform->srcpad,
				momentnormalizationBaseTransform->srcpad)){
			GST_ERROR_OBJECT (momentnormalization, "The caps negotiation have failed, closing application");
			return GST_FLOW_ERROR;
		}
	}



	/*
	 * we want log file output
	 * */
	if(momentnormalization->img_proc_Log == NULL){
		if(momentnormalization->output_logfile && momentnormalization->location){
			momentnormalization->img_proc_Log = fopen(momentnormalization->location, "w+b");
		}
	}

	//gets info of the frame to be processed
	momentnormalization->inframe = inframe;
	momentnormalization->outframe = outframe;

	//strides to be used per channel
	sstride = GST_VIDEO_FRAME_PLANE_STRIDE (inframe, 0);
	dstride[0] = GST_VIDEO_FRAME_PLANE_STRIDE (outframe, 0);
	dstride[1] = GST_VIDEO_FRAME_PLANE_STRIDE (outframe, 1);
	dstride[2] = GST_VIDEO_FRAME_PLANE_STRIDE (outframe, 2);


	srcformat = GST_VIDEO_FRAME_FORMAT(inframe);

	width = sstride;
	height = GST_VIDEO_FRAME_HEIGHT(inframe);

	GST_DEBUG_OBJECT(filter, "INCOMING FRAMES: <%i, %i, %i>", GST_VIDEO_FRAME_HEIGHT(inframe),
			GST_VIDEO_FRAME_WIDTH(inframe), GST_VIDEO_FRAME_PLANE_STRIDE (inframe, 0));
	GST_DEBUG_OBJECT(filter, "ONGOING FRAMES: <%i, %i, %i>", GST_VIDEO_FRAME_HEIGHT(outframe),
			GST_VIDEO_FRAME_WIDTH(outframe), GST_VIDEO_FRAME_PLANE_STRIDE (outframe, 0));

	//================================= FRAME NORMALIZATION USING PEILIN METHOD =====================================//
	time_t start = clock(), end;
	GST_DEBUG_OBJECT(momentnormalization, "Performing Normalization implementing PeiLin method");
	GST_DEBUG_OBJECT(momentnormalization, "srcformat: %d", srcformat);

	/*
	 * We need to lock-mutex the access for this frame so we do not
	 * let another process to mess up with the data we are extracting
	 * and modifying
	 * */
	GST_OBJECT_LOCK(momentnormalization);

	if(srcformat==GST_VIDEO_FORMAT_I420){
		momentnormalization->channels = 1;//always this number since we support only binary
		momentnormalization->height = height;
		momentnormalization->width = width;
		momentnormalization->channels_format = planar;
		momentnormalization->image_type = GST_VIDEO_FORMAT_I420;

		/*
		 * Extracts the metadata information making sure we are not
		 * accessing the wrong chunk of memory, this is done by using
		 * the "gst_meta_get_info" function from Gstreamer
		 * */
		gst_buffer_map(momentnormalization->inframe->buffer, &ininfo, GST_MAP_READ);
		BlobMeta = (GstMetaBLOB*)gst_buffer_get_meta(momentnormalization->inframe->buffer, info->api);
//		while((meta = gst_buffer_iterate_meta(inframe->buffer, &state))){
//			if (meta->info->api == info->api) {
//				GST_DEBUG_OBJECT(momentnormalization, "Metadata found with ID: %g \n", meta->info->api);
//				BlobMeta = (GstMetaBLOB*)meta;
//				break;//since we found the metainfo needed
//			}
//		}
//		gst_buffer_map(inframe->buffer, &ininfo, GST_MAP_READ);
//		ROImeta = gst_buffer_get_video_region_of_interest_meta(momentnormalization->inframe->buffer);

		//avoid copy the data, perform just assignation
		if(BlobMeta->n_blobs > 0){//ROImeta!=NULL
			GST_DEBUG_OBJECT(momentnormalization, "Metadata found..!! ID=%i", info->api);//ROImeta->id);
			//we have to process Bounding boxes
			momentnormalization->n_blobs = BlobMeta->n_blobs;//1;//
			momentnormalization->blobs = (BLOBMOMENT*)g_malloc(sizeof(BLOBMOMENT)*momentnormalization->n_blobs);
			GST_DEBUG_OBJECT(momentnormalization, "%i PATCHES FOUND: \n", momentnormalization->n_blobs);
			for(i=0;i<momentnormalization->n_blobs;i++){
				PATCH_PADDING_X[i] = momentnormalization->padding_x;
				PATCH_PADDING_Y[i] = momentnormalization->padding_y;
				GST_DEBUG_OBJECT(momentnormalization, "%i, %i\n", PATCH_PADDING_X[i], PATCH_PADDING_Y[i]);
				momentnormalization->blobs[i].x = BlobMeta->x[i];//ROImeta->x;//
				momentnormalization->blobs[i].y = BlobMeta->y[i];//ROImeta->y;//
				momentnormalization->blobs[i].width = BlobMeta->width[i];//ROImeta->w;//
				momentnormalization->blobs[i].height = BlobMeta->height[i];//ROImeta->h;//

				if(momentnormalization->blobs[i].x && momentnormalization->blobs[i].y && momentnormalization->blobs[i].width &&
						momentnormalization->blobs[i].height)
					process_it=TRUE;//at least one blob has to be processed


				GST_DEBUG_OBJECT(momentnormalization, "<%i, %i, %i, %i>\n", momentnormalization->blobs[i].x,
						momentnormalization->blobs[i].y, momentnormalization->blobs[i].width, momentnormalization->blobs[i].height);
			}
			gst_buffer_unmap(momentnormalization->inframe->buffer, &ininfo);

			//CALLING NORMALIZATION
			if(process_it){
				ierror = Normalization_error_handler(perform_normalization(momentnormalization));
				GST_DEBUG_OBJECT (momentnormalization, "Motion detected (skin) = %4.3f%%; "
						"Motion detected (non-skin) = %4.3f%%", momentnormalization->diff_acumm_skin*100.0f
						, momentnormalization->diff_acumm_nonskin*100.0f);

				//20% and 15% motion in skin and non-skin pixels respectively
				if(momentnormalization->diff_acumm_skin > 0.20 || momentnormalization->diff_acumm_nonskin > 0.15){
					GST_WARNING_OBJECT(momentnormalization, "CONDITION: %lld, AWAKE ..!!", (g_get_real_time()/1000));
					if(momentnormalization->output_logfile){
//						fprintf(momentnormalization->img_proc_Log, "%f %f\n", momentnormalization->diff_acumm_skin,
//								momentnormalization->diff_acumm_nonskin);

						fprintf(momentnormalization->img_proc_Log, "%lld : AWAKE\n", (g_get_real_time()/1000));
					}
				}else{
					GST_WARNING_OBJECT(momentnormalization, "CONDITION: %lld, SKIN_DETECTED ..!!", (g_get_real_time()/1000));
					if(momentnormalization->output_logfile){
//						fprintf(momentnormalization->img_proc_Log, "%f %f\n", momentnormalization->diff_acumm_skin,
//								momentnormalization->diff_acumm_nonskin);

						fprintf(momentnormalization->img_proc_Log, "%lld : SKIN_DETECTED\n", (g_get_real_time()/1000));
					}
				}
			}else{
				/*
				 * In case the blob detector fails in finding a blob we will not process this frame
				 * in the moment normalization filter, and we will just output the difference previously
				 * calculated
				 * */
				GST_WARNING_OBJECT(momentnormalization, "CONDITION: UNKNOWN ..!!");
				if(momentnormalization->any_buffer){
					gst_buffer_map(momentnormalization->outframe->buffer, &outinfo, GST_MAP_WRITE);
					gst_buffer_map(momentnormalization->prevBuffer, &ininfo, GST_MAP_READ);
					memcpy(outinfo.data, ininfo.data, ininfo.size);
					gst_buffer_unmap(momentnormalization->outframe->buffer, &outinfo);
					gst_buffer_unmap(momentnormalization->prevBuffer, &ininfo);
				}
				ierror = normalization_success;
			}

			if(ierror != normalization_success){
				GST_ERROR_OBJECT(momentnormalization, "Problem performing normalization of the blobs ...\n");
				return GST_FLOW_ERROR;
			}

			/*
			 * Setting up the metadata to be used in the Normalization for processing purposes
			 * its important to mention that the whole structure is coming and is to be used
			 * when processing the frame in the normalization element
			 * */
//			gst_buffer_map(momentnormalization->outframe->buffer, &outinfo, GST_MAP_WRITE);
//			if(momentnormalization->outframe->buffer != NULL && GST_IS_BUFFER (momentnormalization->outframe->buffer)){
//				GST_DEBUG_OBJECT (momentnormalization, "Adding metadata to buffer in moment normalization");
//				gst_buffer_add_momentnormalization_meta(momentnormalization->outframe->buffer, momentnormalization);
//				GST_DEBUG_OBJECT (momentnormalization, "Finished adding metadata to buffer in moment normalization");
//			}
//			gst_buffer_unmap(momentnormalization->outframe->buffer, &outinfo);
		}else{
			gst_buffer_unmap(momentnormalization->inframe->buffer, &ininfo);
			//no object to normalize nothing done for now
		}
	}else if(srcformat==GST_VIDEO_FORMAT_RGBA || srcformat==GST_VIDEO_FORMAT_ARGB || srcformat==GST_VIDEO_FORMAT_BGRA
			|| srcformat==GST_VIDEO_FORMAT_ABGR || srcformat==GST_VIDEO_FORMAT_RGBx || srcformat==GST_VIDEO_FORMAT_xRGB
			|| srcformat==GST_VIDEO_FORMAT_BGRx || srcformat==GST_VIDEO_FORMAT_xBGR){

		momentnormalization->channels = 1;//always this number since we support only binary
		momentnormalization->height = height;
		momentnormalization->width = width;
		momentnormalization->channels_format = interleaved;
		momentnormalization->image_type = srcformat;

		/*
		 * Extracts the metadata information making sure we are not
		 * accessing the wrong chunk of memory, this is done by using
		 * the "gst_meta_get_info" function from Gstreamer
		 * */
		gst_buffer_map(momentnormalization->inframe->buffer, &ininfo, GST_MAP_READ);
		BlobMeta = (GstMetaBLOB*)gst_buffer_get_meta(momentnormalization->inframe->buffer, info->api);
//		while((meta = gst_buffer_iterate_meta(inframe->buffer, &state))){
//			if (meta->info->api == info->api) {
//				GST_DEBUG_OBJECT(momentnormalization, "Metadata found with ID: %g \n", meta->info->api);
//				BlobMeta = (GstMetaBLOB*)meta;
//				break;//since we found the metainfo needed
//			}
//		}
//		gst_buffer_map(inframe->buffer, &ininfo, GST_MAP_READ);
//		ROImeta = gst_buffer_get_video_region_of_interest_meta(momentnormalization->inframe->buffer);

		//avoid copy the data, perform just assignation
		if(BlobMeta->n_blobs > 0){//ROImeta!=NULL
			GST_DEBUG_OBJECT(momentnormalization, "Metadata found..!! ID=%i", info->api);//ROImeta->id);
			//we have to process Bounding boxes
			momentnormalization->n_blobs = BlobMeta->n_blobs;//1;//
			momentnormalization->blobs = (BLOBMOMENT*)g_malloc(sizeof(BLOBMOMENT)*momentnormalization->n_blobs);
			GST_DEBUG_OBJECT(momentnormalization, "%i PATCHES FOUND: \n", momentnormalization->n_blobs);
			for(i=0;i<momentnormalization->n_blobs;i++){
				PATCH_PADDING_X[i] = momentnormalization->padding_x;
				PATCH_PADDING_Y[i] = momentnormalization->padding_y;
				GST_DEBUG_OBJECT(momentnormalization, "Padding of blob %i is ==> <%i, %i>\n", i,
						PATCH_PADDING_X[i], PATCH_PADDING_Y[i]);
				momentnormalization->blobs[i].x = BlobMeta->x[i];//ROImeta->x;//
				momentnormalization->blobs[i].y = BlobMeta->y[i];//ROImeta->y;//
				momentnormalization->blobs[i].width = BlobMeta->width[i];//ROImeta->w;//
				momentnormalization->blobs[i].height = BlobMeta->height[i];//ROImeta->h;//

				if(momentnormalization->blobs[i].x && momentnormalization->blobs[i].y && momentnormalization->blobs[i].width &&
						momentnormalization->blobs[i].height)
					process_it=TRUE;//at least one blob has to be processed


				GST_DEBUG_OBJECT(momentnormalization, "Blob %i has dimensions ==> <%i, %i, %i, %i>\n", i,
						momentnormalization->blobs[i].x, momentnormalization->blobs[i].y,
						momentnormalization->blobs[i].width, momentnormalization->blobs[i].height);
			}
			gst_buffer_unmap(momentnormalization->inframe->buffer, &ininfo);

			//CALLING NORMALIZATION
			if(process_it){
				ierror = Normalization_error_handler(perform_normalization(momentnormalization));
				GST_DEBUG_OBJECT (momentnormalization, "Motion detected (skin) = %4.3f%%; "
						"Motion detected (non-skin) = %4.3f%%", momentnormalization->diff_acumm_skin*100.0f
						, momentnormalization->diff_acumm_nonskin*100.0f);

				//20% and 15% motion in skin and non-skin pixels respectively
				if(momentnormalization->diff_acumm_skin > 0.20 || momentnormalization->diff_acumm_nonskin > 0.15){
					GST_WARNING_OBJECT(momentnormalization, "CONDITION: %lld, AWAKE ..!!", (g_get_real_time()/1000));
					if(momentnormalization->output_logfile){
//						fprintf(momentnormalization->img_proc_Log, "%f %f\n", momentnormalization->diff_acumm_skin,
//								momentnormalization->diff_acumm_nonskin);

						fprintf(momentnormalization->img_proc_Log, "%lld : AWAKE\n", (g_get_real_time()/1000));
					}
				}else{
					GST_WARNING_OBJECT(momentnormalization, "CONDITION: %lld, SKIN_DETECTED ..!!", (g_get_real_time()/1000));
					if(momentnormalization->output_logfile){
//						fprintf(momentnormalization->img_proc_Log, "%f %f\n", momentnormalization->diff_acumm_skin,
//								momentnormalization->diff_acumm_nonskin);

						fprintf(momentnormalization->img_proc_Log, "%lld : SKIN_DETECTED\n", (g_get_real_time()/1000));
					}
				}
			}else{
				/*
				 * In case the blob detector fails in finding a blob we will not process this frame
				 * in the moment normalization filter, and we will just output the difference previously
				 * calculated
				 * */
				GST_WARNING_OBJECT(momentnormalization, "CONDITION: UNKNOWN ..!!");
				if(momentnormalization->any_buffer){
					gst_buffer_map(momentnormalization->outframe->buffer, &outinfo, GST_MAP_WRITE);
					gst_buffer_map(momentnormalization->prevBuffer, &ininfo, GST_MAP_READ);
					memcpy(outinfo.data, ininfo.data, ininfo.size);
					gst_buffer_unmap(momentnormalization->outframe->buffer, &outinfo);
					gst_buffer_unmap(momentnormalization->prevBuffer, &ininfo);
				}
				ierror = normalization_success;
			}

			if(ierror != normalization_success){
				GST_ERROR_OBJECT(momentnormalization, "Problem performing normalization of the blobs ...\n");
				return GST_FLOW_ERROR;
			}

			/*
			 * Setting up the metadata to be used in the Normalization for processing purposes
			 * its important to mention that the whole structure is coming and is to be used
			 * when processing the frame in the normalization element
			 * */
//			gst_buffer_map(momentnormalization->outframe->buffer, &outinfo, GST_MAP_WRITE);
//			if(momentnormalization->outframe->buffer != NULL && GST_IS_BUFFER (momentnormalization->outframe->buffer)){
//				GST_DEBUG_OBJECT (momentnormalization, "Adding metadata to buffer in moment normalization");
//				gst_buffer_add_momentnormalization_meta(momentnormalization->outframe->buffer, momentnormalization);
//				GST_DEBUG_OBJECT (momentnormalization, "Finished adding metadata to buffer in moment normalization");
//			}
//			gst_buffer_unmap(momentnormalization->outframe->buffer, &outinfo);
		}else{
			gst_buffer_unmap(momentnormalization->inframe->buffer, &ininfo);
			//no object to normalize nothing done for now
		}
	}else if(srcformat==GST_VIDEO_FORMAT_GRAY8){

		momentnormalization->channels = 1;//always this number since we support only binary
		momentnormalization->height = height;
		momentnormalization->width = width;
		momentnormalization->channels_format = planar;
		momentnormalization->image_type = srcformat;

		/*
		 * Extracts the metadata information making sure we are not
		 * accessing the wrong chunk of memory, this is done by using
		 * the "gst_meta_get_info" function from Gstreamer
		 * */
		gst_buffer_map(momentnormalization->inframe->buffer, &ininfo, GST_MAP_READ);
		BlobMeta = (GstMetaBLOB*)gst_buffer_get_meta(momentnormalization->inframe->buffer, info->api);
//		while ((meta = gst_buffer_iterate_meta(inframe->buffer, &state))) {
//			if (meta->info->api == info->api) {
//				GST_DEBUG_OBJECT(momentnormalization, "Metadata found with ID: %g \n", meta->info->api);
//				BlobMeta = (GstMetaBLOB*)meta;
//				break;//since we found the metainfo needed
//			}
//		}
//		gst_buffer_map(inframe->buffer, &tmpinfo, GST_MAP_READ);
//		GST_DEBUG("inframe->buffer: %p", inframe->buffer);
//		ROImeta = gst_buffer_get_video_region_of_interest_meta_id(inframe->buffer, 0);
//		GST_DEBUG("ROImeta: %p", ROImeta);
//		gst_buffer_unmap(inframe->buffer, &tmpinfo);

		//avoid copy the data, perform just assignation
		if(BlobMeta->n_blobs > 0){//ROImeta!=NULL
			GST_DEBUG_OBJECT(momentnormalization, "Metadata found..!! ID=%i", info->api);//ROImeta->id);
			//we have to process Bounding boxes
			momentnormalization->n_blobs = BlobMeta->n_blobs;//1;//
			momentnormalization->blobs = (BLOBMOMENT*)g_malloc(sizeof(BLOBMOMENT)*momentnormalization->n_blobs);
			GST_DEBUG_OBJECT(momentnormalization, "%i PATCHES FOUND: \n", momentnormalization->n_blobs);
			for(i=0;i<momentnormalization->n_blobs;i++){
				PATCH_PADDING_X[i] = momentnormalization->padding_x;
				PATCH_PADDING_Y[i] = momentnormalization->padding_y;
				GST_DEBUG_OBJECT(momentnormalization, "Padding of blob %i is ==> <%i, %i>\n", i,
						PATCH_PADDING_X[i], PATCH_PADDING_Y[i]);
				momentnormalization->blobs[i].x = BlobMeta->x[i];//ROImeta->x;//
				momentnormalization->blobs[i].y = BlobMeta->y[i];//ROImeta->y;//
				momentnormalization->blobs[i].width = BlobMeta->width[i];//ROImeta->w;//
				momentnormalization->blobs[i].height = BlobMeta->height[i];//ROImeta->h;//

				if((momentnormalization->blobs[i].x + momentnormalization->blobs[i].width) &&
						(momentnormalization->blobs[i].y + momentnormalization->blobs[i].height))
					process_it=TRUE;//at least one blob has to be processed

				GST_DEBUG_OBJECT(momentnormalization, "PATCH %i has dimensions ==> <%i, %i, %i, %i>", i,
						momentnormalization->blobs[i].x, momentnormalization->blobs[i].y,
						momentnormalization->blobs[i].width, momentnormalization->blobs[i].height);
			}
			gst_buffer_unmap(momentnormalization->inframe->buffer, &ininfo);

			//CALLING NORMALIZATION
			if(process_it){
				ierror = Normalization_error_handler(perform_normalization(momentnormalization));
				GST_DEBUG_OBJECT (momentnormalization, "Motion detected (skin) = %4.3f%%; "
						"Motion detected (non-skin) = %4.3f%%", momentnormalization->diff_acumm_skin*100.0f
						, momentnormalization->diff_acumm_nonskin*100.0f);

				//20% and 15% motion in skin and non-skin pixels respectively
				if(momentnormalization->diff_acumm_skin > 0.20 || momentnormalization->diff_acumm_nonskin > 0.15){
					GST_WARNING_OBJECT(momentnormalization, "CONDITION: %lld, AWAKE ..!!", (g_get_real_time()/1000));
					if(momentnormalization->output_logfile){
//						fprintf(momentnormalization->img_proc_Log, "%f %f\n", momentnormalization->diff_acumm_skin,
//								momentnormalization->diff_acumm_nonskin);

						fprintf(momentnormalization->img_proc_Log, "%lld : AWAKE\n", (g_get_real_time()/1000));
					}
				}else{
					GST_WARNING_OBJECT(momentnormalization, "CONDITION: %lld, SKIN_DETECTED ..!!", (g_get_real_time()/1000));
					if(momentnormalization->output_logfile){
//						fprintf(momentnormalization->img_proc_Log, "%f %f\n", momentnormalization->diff_acumm_skin,
//								momentnormalization->diff_acumm_nonskin);

						fprintf(momentnormalization->img_proc_Log, "%lld : SKIN_DETECTED\n", (g_get_real_time()/1000));
					}
				}
			}else{
				/*
				 * In case the blob detector fails in finding a blob we will not process this frame
				 * in the moment normalization filter, and we will just output the difference previously
				 * calculated
				 * */
				GST_WARNING_OBJECT(momentnormalization, "CONDITION: UNKNOWN ..!!");
				if(momentnormalization->any_buffer){
					gst_buffer_map(momentnormalization->outframe->buffer, &outinfo, GST_MAP_WRITE);
					gst_buffer_map(momentnormalization->prevBuffer, &ininfo, GST_MAP_READ);
					memcpy(outinfo.data, ininfo.data, ininfo.size);
					gst_buffer_unmap(momentnormalization->outframe->buffer, &outinfo);
					gst_buffer_unmap(momentnormalization->prevBuffer, &ininfo);
				}
				ierror = normalization_success;
			}

			if(ierror != normalization_success){
				GST_ERROR_OBJECT(momentnormalization, "Problem performing normalization of the blobs ...");
				return GST_FLOW_ERROR;
			}

			/*
			 * Setting up the metadata to be used in the Normalization for processing purposes
			 * its important to mention that the whole structure is coming and is to be used
			 * when processing the frame in the normalization element
			 * */
//			gst_buffer_map(momentnormalization->outframe->buffer, &outinfo, GST_MAP_WRITE);
//			if(momentnormalization->outframe->buffer != NULL && GST_IS_BUFFER (momentnormalization->outframe->buffer)){
//				GST_DEBUG_OBJECT (momentnormalization, "Adding metadata to buffer in moment normalization");
//				gst_buffer_add_momentnormalization_meta(momentnormalization->outframe->buffer, momentnormalization);
//				GST_DEBUG_OBJECT (momentnormalization, "Finished adding metadata to buffer in moment normalization");
//			}
//			gst_buffer_unmap(momentnormalization->outframe->buffer, &outinfo);
		}else{
			gst_buffer_unmap(momentnormalization->inframe->buffer, &ininfo);
			//no object to normalize nothing done for now
		}
	}else{
		GST_ERROR_OBJECT (momentnormalization, "The input format of the frame is not supported by this element MOMENTNORMALIZATION: %s",
				gst_video_format_to_string(momentnormalization->base_momentnormalization.in_info.finfo->format));
		return GST_FLOW_ERROR;
	}
	end = clock();
	GST_DEBUG_OBJECT(momentnormalization, "++++++++++++++++++++++++++++++++++++++++++++++\n"
			"++++++++++++++++++++++++++++++++++++++++++++++\n"
			" Total time in element Moment Normalization: %f ms.\n"
			"++++++++++++++++++++++++++++++++++++++++++++++\n"
			"++++++++++++++++++++++++++++++++++++++++++++++\n",1000 * (((float) end - start) / CLOCKS_PER_SEC));
	momentnormalization->internal_pt = 1000*(((float)(end-start))/CLOCKS_PER_SEC);

	/*
	 * Unlocking the data for releasing the changes and output
	 * the modified frame
	 * */
	GST_OBJECT_UNLOCK(momentnormalization);

	return GST_FLOW_OK;
}

#if 0
static gboolean plugin_init (GstPlugin * plugin){

  return gst_element_register(plugin, "momentnormalization", GST_RANK_NONE, GST_TYPE_MOMENTNORMALIZATION);
}


#ifndef VERSION
#define VERSION "0.0.1"
#endif
#ifndef PACKAGE
#define PACKAGE "PeiLin_Normalization_API"
#endif
#ifndef PACKAGE_NAME
#define PACKAGE_NAME "Image_proc_API"
#endif
#ifndef GST_PACKAGE_ORIGIN
#define GST_PACKAGE_ORIGIN "None"
#endif

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR, GST_VERSION_MINOR, momentnormalization,
    "Performs object normalization of binary patches or bouding boxes using PeiLin moments",
    plugin_init, VERSION, "LGPL", PACKAGE_NAME, GST_PACKAGE_ORIGIN)

#endif

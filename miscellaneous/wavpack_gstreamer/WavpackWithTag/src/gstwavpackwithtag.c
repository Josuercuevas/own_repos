/* GStreamer Wavpack encoder and tagger plugin
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


#include <string.h>
#include <gst/gst.h>
#include <gst/gstbufferlist.h>
#include <glib/gprintf.h>
#include <gst/audio/gstaudioringbuffer.h>
#include <gst/audio/gstaudioencoder.h>
#include <wavpack/wavpack.h>
#include "gstwavpackwithtag.h"
#include "gstwavpackcommon.h"

#define DEFAULT_CHUNKS (5)
#define DEFAULT_LABEL (199)//unknown cry label

static gboolean gst_wavpack_enc_start (GstAudioEncoder * enc);
static gboolean gst_wavpack_enc_stop (GstAudioEncoder * enc);
static gboolean gst_wavpack_enc_set_format (GstAudioEncoder * enc, GstAudioInfo * info);
static GstFlowReturn gst_wavpack_enc_handle_frame (GstAudioEncoder * enc, GstBuffer * in_buf);
static gboolean gst_wavpack_enc_sink_event (GstAudioEncoder * enc, GstEvent * event);

static int gst_wavpack_enc_push_block (void *id, void *data, int32_t count);
static GstFlowReturn gst_wavpack_enc_drain (GstWavpackEnc * enc);

static void gst_wavpack_enc_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_wavpack_enc_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);


static gboolean buffer_list_copy_data (GstBuffer ** buf, guint idx, gpointer data);

enum{
  ARG_0,
  ARG_MODE,
  ARG_BITRATE,
  ARG_BITSPERSAMPLE,
  ARG_CORRECTION_MODE,
  ARG_MD5,
  ARG_EXTRA_PROCESSING,
  ARG_JOINT_STEREO_MODE,
  PROP_CHUNKS,
  PROP_LABEL
};

GST_DEBUG_CATEGORY_STATIC (gst_wavpack_enc_debug);
#define GST_CAT_DEFAULT gst_wavpack_enc_debug

static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE ("sink",
		GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS ("audio/x-raw, "
        "format = (string) " "S32LE, " "layout = (string) interleaved, "
        "channels = (int) [ 1, 8 ], " "rate = (int) [ 6000, 192000 ]")
    );

static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
		GST_PAD_SRC, GST_PAD_ALWAYS,  GST_STATIC_CAPS ("audio/x-wavpack, "
        "depth = (int) [ 1, 32 ], " "channels = (int) [ 1, 8 ], "
        "rate = (int) [ 6000, 192000 ], " "framed = (boolean) TRUE")
    );

static GstStaticPadTemplate wvcsrc_factory = GST_STATIC_PAD_TEMPLATE ("wvcsrc",
		GST_PAD_SRC, GST_PAD_SOMETIMES, GST_STATIC_CAPS("audio/x-wavpack-correction, "
				"framed = (boolean) TRUE")
    );

enum{
	GST_WAVPACK_ENC_MODE_VERY_FAST = 0,
	GST_WAVPACK_ENC_MODE_FAST,
	GST_WAVPACK_ENC_MODE_DEFAULT,
	GST_WAVPACK_ENC_MODE_HIGH,
	GST_WAVPACK_ENC_MODE_VERY_HIGH
};

#define GST_TYPE_WAVPACK_ENC_MODE (gst_wavpack_enc_mode_get_type ())
static GType gst_wavpack_enc_mode_get_type (void){
  static GType qtype = 0;

  if (qtype == 0) {
	  static const GEnumValue values[] = {
#if 0
      /* Very Fast Compression is not supported yet, but will be supported
       * in future wavpack versions */
      {GST_WAVPACK_ENC_MODE_VERY_FAST, "Very Fast Compression", "veryfast"},
#endif
      {GST_WAVPACK_ENC_MODE_FAST, "Fast Compression", "fast"},
      {GST_WAVPACK_ENC_MODE_DEFAULT, "Normal Compression", "normal"},
      {GST_WAVPACK_ENC_MODE_HIGH, "High Compression", "high"},
      {GST_WAVPACK_ENC_MODE_VERY_HIGH, "Very High Compression", "veryhigh"},
      {0, NULL, NULL}
    };

    qtype = g_enum_register_static ("GstWavpackEncMode", values);
  }
  return qtype;
}

enum{
  GST_WAVPACK_CORRECTION_MODE_OFF = 0,
  GST_WAVPACK_CORRECTION_MODE_ON,
  GST_WAVPACK_CORRECTION_MODE_OPTIMIZED
};


#define GST_TYPE_WAVPACK_ENC_CORRECTION_MODE (gst_wavpack_enc_correction_mode_get_type ())
static GType gst_wavpack_enc_correction_mode_get_type (void){
  static GType qtype = 0;

  if (qtype == 0) {
    static const GEnumValue values[] = {
      {GST_WAVPACK_CORRECTION_MODE_OFF, "Create no correction file", "off"},
      {GST_WAVPACK_CORRECTION_MODE_ON, "Create correction file", "on"},
      {GST_WAVPACK_CORRECTION_MODE_OPTIMIZED,
          "Create optimized correction file", "optimized"},
      {0, NULL, NULL}
    };

    qtype = g_enum_register_static ("GstWavpackEncCorrectionMode", values);
  }
  return qtype;
}

enum{
  GST_WAVPACK_JS_MODE_AUTO = 0,
  GST_WAVPACK_JS_MODE_LEFT_RIGHT,
  GST_WAVPACK_JS_MODE_MID_SIDE
};

#define GST_TYPE_WAVPACK_ENC_JOINT_STEREO_MODE (gst_wavpack_enc_joint_stereo_mode_get_type ())
static GType gst_wavpack_enc_joint_stereo_mode_get_type (void){
  static GType qtype = 0;

  if (qtype == 0) {
    static const GEnumValue values[] = {
      {GST_WAVPACK_JS_MODE_AUTO, "auto", "auto"},
      {GST_WAVPACK_JS_MODE_LEFT_RIGHT, "left/right", "leftright"},
      {GST_WAVPACK_JS_MODE_MID_SIDE, "mid/side", "midside"},
      {0, NULL, NULL}
    };

    qtype = g_enum_register_static ("GstWavpackEncJSMode", values);
  }
  return qtype;
}

#define gst_wavpack_enc_parent_class parent_class
G_DEFINE_TYPE (GstWavpackEnc, gst_wavpack_enc, GST_TYPE_AUDIO_ENCODER);

static void gst_wavpack_enc_class_init (GstWavpackEncClass * wavpackclass){
	GObjectClass *gobject_class = (GObjectClass *) wavpackclass;
	GstElementClass *element_class = (GstElementClass *) (wavpackclass);
	GstAudioEncoderClass *base_class = (GstAudioEncoderClass *) (wavpackclass);

	/* add pad templates */
	gst_element_class_add_pad_template (element_class,
	  gst_static_pad_template_get (&sink_factory));
	gst_element_class_add_pad_template (element_class,
	  gst_static_pad_template_get (&src_factory));
	gst_element_class_add_pad_template (element_class,
	  gst_static_pad_template_get (&wvcsrc_factory));

	/* set element details */
	gst_element_class_set_static_metadata (element_class, "Wavpack audio encoder with tagger rutine",
	  "Codec/Encoder/Audio",
	  "Encodes audio with the Wavpack lossless/lossy audio codec",
	  "Sebastian Dröge <slomo@circular-chaos.org>");

	/* set property handlers */
	gobject_class->set_property = gst_wavpack_enc_set_property;
	gobject_class->get_property = gst_wavpack_enc_get_property;

	base_class->start = GST_DEBUG_FUNCPTR (gst_wavpack_enc_start);
	base_class->stop = GST_DEBUG_FUNCPTR (gst_wavpack_enc_stop);
	base_class->set_format = GST_DEBUG_FUNCPTR (gst_wavpack_enc_set_format);
	base_class->handle_frame = GST_DEBUG_FUNCPTR (gst_wavpack_enc_handle_frame);
	base_class->sink_event = GST_DEBUG_FUNCPTR (gst_wavpack_enc_sink_event);

	/* install all properties */
	g_object_class_install_property (gobject_class, ARG_MODE, g_param_spec_enum ("mode", "Encoding mode",
		  "Speed versus compression tradeoff.", GST_TYPE_WAVPACK_ENC_MODE, GST_WAVPACK_ENC_MODE_DEFAULT,
		  G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property (gobject_class, ARG_BITRATE,
	  g_param_spec_uint ("bitrate", "Bitrate", "Try to encode with this average bitrate (bits/sec). "
		  "This enables lossy encoding, values smaller than 24000 disable it again.",
		  0, 9600000, 0, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property (gobject_class, ARG_BITSPERSAMPLE, g_param_spec_double ("bits-per-sample", "Bits per sample",
		  "Try to encode with this amount of bits per sample. "
		  "This enables lossy encoding, values smaller than 2.0 disable it again.",
		  0.0, 24.0, 0.0, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property (gobject_class, ARG_CORRECTION_MODE, g_param_spec_enum ("correction-mode", "Correction stream mode",
		  "Use this mode for the correction stream. Only works in lossy mode!",
		  GST_TYPE_WAVPACK_ENC_CORRECTION_MODE, GST_WAVPACK_CORRECTION_MODE_OFF,
		  G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property (gobject_class, ARG_MD5, g_param_spec_boolean ("md5", "MD5",
		  "Store MD5 hash of raw samples within the file.", FALSE,
		  G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property (gobject_class, ARG_EXTRA_PROCESSING, g_param_spec_uint ("extra-processing", "Extra processing",
		  "Use better but slower filters for better compression/quality.",
		  0, 6, 0, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property (gobject_class, ARG_JOINT_STEREO_MODE, g_param_spec_enum ("joint-stereo-mode", "Joint-Stereo mode",
		  "Use this joint-stereo mode.", GST_TYPE_WAVPACK_ENC_JOINT_STEREO_MODE,
		  GST_WAVPACK_JS_MODE_AUTO,
		  G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property (gobject_class, PROP_LABEL,
		g_param_spec_int ("cry-label", "Cry L",
			"Label of a given chunk of data packed", -100, 200, DEFAULT_LABEL,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property (gobject_class, PROP_CHUNKS,
			g_param_spec_uint ("chunks", "Chunks PF",
				"Number of chunks to be written in the file", 1, 20, DEFAULT_CHUNKS,
				G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
}

static void gst_wavpack_enc_reset (GstWavpackEnc * enc){
	/* close and free everything stream related if we already did something */
	if (enc->wp_context){
		WavpackCloseFile (enc->wp_context);
		enc->wp_context = NULL;
	}

	if (enc->wp_config){
		g_free (enc->wp_config);
		enc->wp_config = NULL;
	}

	if (enc->first_block){
		g_free (enc->first_block);
		enc->first_block = NULL;
	}

	enc->first_block_size = 0;

	if (enc->md5_context){
		g_checksum_free (enc->md5_context);
		enc->md5_context = NULL;
	}

	if (enc->pending_segment)
		gst_event_unref (enc->pending_segment);

	enc->pending_segment = NULL;

	if (enc->pending_buffer){
		gst_buffer_unref (enc->pending_buffer);
		enc->pending_buffer = NULL;
		enc->pending_offset = 0;
	}

	/* reset the last returns to GST_FLOW_OK. This is only set to something else
	* while WavpackPackSamples() or more specific gst_wavpack_enc_push_block()
	* so not valid anymore */
	enc->srcpad_last_return = enc->wvcsrcpad_last_return = GST_FLOW_OK;

	/* reset stream information */
	if(enc->reset_encoder){
		enc->samplerate = 0;
		enc->depth = 0;
		enc->channels = 0;
		enc->channel_mask = 0;
		enc->need_channel_remap = FALSE;
		enc->chunks_count = 0;
		enc->timestamp_offset = GST_CLOCK_TIME_NONE;
		enc->next_ts = GST_CLOCK_TIME_NONE;
	}

	/* resetting bufferlist */
	GST_DEBUG_OBJECT(enc, "Freeing list..!");
	if(enc->data_in_list){
		/*
		 * we have data in the list, this should not happen unless we
		 * suddenly stop the pipeline, however it is a precaution
		 * */
		if(GST_IS_BUFFER_LIST(enc->bufferlist)){
			if(gst_buffer_list_length(enc->bufferlist)>0){
				gst_buffer_list_remove(enc->bufferlist, 0, gst_buffer_list_length(enc->bufferlist));
			}
		}
	}
	GST_DEBUG_OBJECT(enc, "Freeing list done..!");

}

/* For cleaning the whole data */
static gboolean buffer_list_copy_data(GstBuffer ** buf, guint idx, gpointer data){
	GstBuffer *dest = data;
	guint num, i;

	if (idx == 0)
		gst_buffer_copy_into (dest, *buf, GST_BUFFER_COPY_METADATA, 0, -1);

	num = gst_buffer_n_memory (*buf);

	for (i = 0; i < num; ++i) {
		GstMemory *mem;
		mem = gst_buffer_get_memory (*buf, i);
		gst_buffer_append_memory (dest, mem);
	}

	return TRUE;
}

static void gst_wavpack_enc_init (GstWavpackEnc * enc){
	GstAudioEncoder *benc = GST_AUDIO_ENCODER (enc);
	gint i;

	/* initialize object attributes */
	enc->wp_config = NULL;
	enc->wp_context = NULL;
	enc->first_block = NULL;
	enc->md5_context = NULL;
	enc->reset_encoder = TRUE;

	GST_DEBUG_OBJECT(enc, "resetting encoder..!");
	gst_wavpack_enc_reset (enc);
	GST_DEBUG_OBJECT(enc, "resetting encoder done..!");

	enc->wv_id.correction = FALSE;
	enc->wv_id.wavpack_enc = enc;
	enc->wv_id.passthrough = FALSE;
	enc->wvc_id.correction = TRUE;
	enc->wvc_id.wavpack_enc = enc;
	enc->wvc_id.passthrough = FALSE;

	/* set default values of params */
	enc->n_chunks = DEFAULT_CHUNKS;
	enc->chunks_count = 0;
	enc->label = DEFAULT_LABEL;
	enc->mode = GST_WAVPACK_ENC_MODE_DEFAULT;
	enc->bitrate = 0;
	enc->bps = 0.0;
	enc->correction_mode = GST_WAVPACK_CORRECTION_MODE_OFF;
	enc->md5 = FALSE;
	enc->extra_processing = 0;
	enc->joint_stereo_mode = GST_WAVPACK_JS_MODE_AUTO;

	/* require perfect ts */
	gst_audio_encoder_set_perfect_timestamp (benc, TRUE);

	/* setting out buffer list */
	enc->data_in_list = FALSE;
	enc->bufferlist = NULL;
	for(i=0;i<80;i++){
		enc->cry_list[i] = DEFAULT_LABEL;
	}
}


static gboolean gst_wavpack_enc_start (GstAudioEncoder * enc){
  GST_DEBUG_OBJECT (enc, "start");

  /*
   * in case something else has to be done that is not handled in the
   * initialization part
   * */

  return TRUE;
}

static gboolean gst_wavpack_enc_stop (GstAudioEncoder * enc){
	GstWavpackEnc *wpenc = GST_WAVPACK_ENC (enc);

	GST_DEBUG_OBJECT (enc, "Stopping encoder and freeing buffer list");

	/*
	* In stopping the encoder we need to reset all the variables and flush memory as
	* well, or next time we use it we are going to have discontinuity in the encoded
	* file
	* */
	gst_wavpack_enc_reset (wpenc);

	/*
	* If there is any data in the list which does not have to be we proceed to free
	* it
	* */
	if(wpenc->bufferlist && wpenc->data_in_list){
		if(GST_IS_BUFFER_LIST(wpenc->bufferlist)){
			gst_buffer_list_unref(wpenc->bufferlist);
		}
	}

	return TRUE;
}

static gboolean gst_wavpack_enc_set_format (GstAudioEncoder * benc, GstAudioInfo * info){
	GstWavpackEnc *enc = GST_WAVPACK_ENC (benc);
	GstAudioChannelPosition *pos;
	GstAudioChannelPosition opos[64] = { GST_AUDIO_CHANNEL_POSITION_INVALID, };
	GstCaps *caps;
	guint64 mask = 0;

	/* we may be configured again, but that change should have cleanup context */
	g_assert (enc->wp_context == NULL);

	enc->channels = GST_AUDIO_INFO_CHANNELS (info);
	enc->depth = GST_AUDIO_INFO_DEPTH (info);
	enc->samplerate = GST_AUDIO_INFO_RATE (info);

	pos = info->position;
	g_assert (pos);

	/* If one channel is NONE they'll be all undefined */
	if (pos != NULL && pos[0] == GST_AUDIO_CHANNEL_POSITION_NONE) {
		goto invalid_channels;
	}

	enc->channel_mask = gst_wavpack_get_channel_mask_from_positions (pos, enc->channels);
	enc->need_channel_remap = gst_wavpack_set_channel_mapping (pos, enc->channels,
			enc->channel_mapping);

	/* wavpack caps hold gst mask, not wavpack mask */
	gst_audio_channel_positions_to_mask (opos, enc->channels, FALSE, &mask);

	/* set fixed src pad caps now that we know what we will get */
	caps = gst_caps_new_simple ("audio/x-wavpack", "channels", G_TYPE_INT, enc->channels,
	  "rate", G_TYPE_INT, enc->samplerate, "depth", G_TYPE_INT, enc->depth, "framed",
	  G_TYPE_BOOLEAN, TRUE, NULL);

	if (mask)
		gst_caps_set_simple (caps, "channel-mask", GST_TYPE_BITMASK, mask, NULL);

	if (!gst_audio_encoder_set_output_format (benc, caps))
		goto setting_src_caps_failed;

	gst_caps_unref (caps);

	/* no special feedback to base class; should provide all available samples */

	return TRUE;

	/* ERRORS */
	setting_src_caps_failed:
	{
		GST_DEBUG_OBJECT (enc,
			"Couldn't set caps on source pad: %" GST_PTR_FORMAT, caps);
		gst_caps_unref (caps);
		return FALSE;
	}
	invalid_channels:
	{
		GST_DEBUG_OBJECT (enc, "input has invalid channel layout");
		return FALSE;
	}

}

static void gst_wavpack_enc_set_wp_config (GstWavpackEnc * enc){
	enc->wp_config = g_new0(WavpackConfig, 1);
	/* set general stream informations in the WavpackConfig */
	enc->wp_config->bytes_per_sample = GST_ROUND_UP_8(enc->depth) / 8;
	enc->wp_config->bits_per_sample = enc->depth;
	enc->wp_config->num_channels = enc->channels;
	enc->wp_config->channel_mask = enc->channel_mask;
	enc->wp_config->sample_rate = enc->samplerate;

	/*
	* Set parameters in WavpackConfig
	*/

	/* Encoding mode */
	switch (enc->mode){
		#if 0
			case GST_WAVPACK_ENC_MODE_VERY_FAST:
				enc->wp_config->flags |= CONFIG_VERY_FAST_FLAG;
				enc->wp_config->flags |= CONFIG_FAST_FLAG;
				break;
		#endif
			case GST_WAVPACK_ENC_MODE_FAST:
				enc->wp_config->flags |= CONFIG_FAST_FLAG;
				break;
			case GST_WAVPACK_ENC_MODE_DEFAULT:
				break;
			case GST_WAVPACK_ENC_MODE_HIGH:
				enc->wp_config->flags |= CONFIG_HIGH_FLAG;
				break;
			case GST_WAVPACK_ENC_MODE_VERY_HIGH:
				enc->wp_config->flags |= CONFIG_HIGH_FLAG;
				enc->wp_config->flags |= CONFIG_VERY_HIGH_FLAG;
				break;
	}

	/* Bitrate, enables lossy mode */
	if (enc->bitrate){
		enc->wp_config->flags |= CONFIG_HYBRID_FLAG;
		enc->wp_config->flags |= CONFIG_BITRATE_KBPS;
		enc->wp_config->bitrate = enc->bitrate / 1000.0;
	}else if (enc->bps){
		enc->wp_config->flags |= CONFIG_HYBRID_FLAG;
		enc->wp_config->bitrate = enc->bps;
	}

	/* Correction Mode, only in lossy mode */
	if (enc->wp_config->flags & CONFIG_HYBRID_FLAG){
		if (enc->correction_mode > GST_WAVPACK_CORRECTION_MODE_OFF){
			GstCaps *caps = gst_caps_new_simple ("audio/x-wavpack-correction", "framed", G_TYPE_BOOLEAN, TRUE, NULL);

			enc->wvcsrcpad = gst_pad_new_from_static_template (&wvcsrc_factory, "wvcsrc");

			/* try to add correction src pad, don't set correction mode on failure */
			GST_DEBUG_OBJECT (enc, "Adding correction pad with caps %" GST_PTR_FORMAT, caps);
			if (!gst_pad_set_caps (enc->wvcsrcpad, caps)){
				enc->correction_mode = 0;
				GST_WARNING_OBJECT (enc, "setting correction caps failed");
			}else{
				gst_pad_use_fixed_caps (enc->wvcsrcpad);
				gst_pad_set_active (enc->wvcsrcpad, TRUE);
				gst_element_add_pad (GST_ELEMENT (enc), enc->wvcsrcpad);
				enc->wp_config->flags |= CONFIG_CREATE_WVC;
				if (enc->correction_mode == GST_WAVPACK_CORRECTION_MODE_OPTIMIZED) {
					enc->wp_config->flags |= CONFIG_OPTIMIZE_WVC;
				}
			}
			gst_caps_unref (caps);
		}
	}else{
		if(enc->correction_mode > GST_WAVPACK_CORRECTION_MODE_OFF){
			enc->correction_mode = 0;
			GST_WARNING_OBJECT (enc, "setting correction mode only has any effect if a bitrate is provided.");
		}
	}

	/* no more pads to be created for this element */
	gst_element_no_more_pads (GST_ELEMENT (enc));

	/* MD5, setup MD5 context */
	if ((enc->md5) && !(enc->md5_context)){
		enc->wp_config->flags |= CONFIG_MD5_CHECKSUM;
		enc->md5_context = g_checksum_new (G_CHECKSUM_MD5);
	}

	/* Extra encode processing */
	if (enc->extra_processing){
		enc->wp_config->flags |= CONFIG_EXTRA_MODE;
		enc->wp_config->xmode = enc->extra_processing;
	}

	/* Joint stereo mode */
	switch (enc->joint_stereo_mode){
		case GST_WAVPACK_JS_MODE_AUTO:
			break;
		case GST_WAVPACK_JS_MODE_LEFT_RIGHT:
			enc->wp_config->flags |= CONFIG_JOINT_OVERRIDE;
			enc->wp_config->flags &= ~CONFIG_JOINT_STEREO;
			break;
		case GST_WAVPACK_JS_MODE_MID_SIDE:
			enc->wp_config->flags |= (CONFIG_JOINT_OVERRIDE | CONFIG_JOINT_STEREO);
			break;
	}
}

static gint sinked=0;
static gboolean is_tags=FALSE;
/* Writing function which is basically just a pushing function to the next element */
static int gst_wavpack_enc_push_block (void *id, void *data, int32_t count){
	GstWavpackEncWriteID *wid = (GstWavpackEncWriteID *) id;
	GstWavpackEnc *enc = GST_WAVPACK_ENC (wid->wavpack_enc);
	GstFlowReturn *flow;
	GstBuffer *buffer;
	GstPad *pad;
	guchar *block = (guchar *) data;
	gint samples = 0, i;

	pad = (wid->correction) ? enc->wvcsrcpad : GST_AUDIO_ENCODER_SRC_PAD (enc);
	flow = (wid->correction) ? &enc->wvcsrcpad_last_return : &enc->srcpad_last_return;

	buffer = gst_buffer_new_and_alloc (count);
	gst_buffer_fill (buffer, 0, data, count);

	if (count > sizeof (WavpackHeader) && memcmp (block, "wvpk", 4) == 0){
		/* if it's a Wavpack block set buffer timestamp and duration, etc */
		WavpackHeader wph;

		GST_LOG_OBJECT (enc, "got %d bytes of encoded wavpack %sdata", count, (wid->correction) ? "correction " : "");

		gst_wavpack_read_header (&wph, block);

		/* Only set when pushing the first buffer again, in that case
		 * we don't want to delay the buffer or push newsegment events
		 */
		if (!wid->passthrough) {
			/* Only push complete blocks */
			if (enc->pending_buffer == NULL) {
				enc->pending_buffer = buffer;
				enc->pending_offset = wph.block_index;
			} else if (enc->pending_offset == wph.block_index) {
				enc->pending_buffer = gst_buffer_append (enc->pending_buffer, buffer);
			} else {
				GST_ERROR ("Processing Got incomplete block, dropping");
				gst_buffer_unref (enc->pending_buffer);
				enc->pending_buffer = buffer;
				enc->pending_offset = wph.block_index;
			}

			/* Is this the not-final block of multi-channel data? If so, just
			* accumulate and return here. */
			if (!(wph.flags & FINAL_BLOCK) && ((block[32] & ID_OPTIONAL_DATA) == 0))
				return TRUE;

			buffer = enc->pending_buffer;
			enc->pending_buffer = NULL;
			enc->pending_offset = 0;

			/* only send segment on correction pad,
			* regular pad is handled normally by baseclass */
			if (wid->correction && enc->pending_segment) {
				gst_pad_push_event (pad, enc->pending_segment);
				enc->pending_segment = NULL;
			}

			if (wph.block_index == 0) {
				/* save header for later reference, so we can re-send it later on
				 * EOS with fixed up values for total sample count etc. */
				if (enc->first_block == NULL && !wid->correction) {
					GstMapInfo map;

					gst_buffer_map (buffer, &map, GST_MAP_READ);
					enc->first_block = g_memdup (map.data, map.size);
					enc->first_block_size = map.size;
					gst_buffer_unmap (buffer, &map);
				}
			}
		}
		samples = wph.block_samples;

		GST_BUFFER_OFFSET (buffer) = wph.block_index;
		GST_BUFFER_OFFSET_END (buffer) = wph.block_index + wph.block_samples;
	}else{
		/* if it's something else set no timestamp and duration on the buffer */
		GST_DEBUG_OBJECT (enc, "got %d bytes of unknown data", count);
	}


	if(wid->correction || wid->passthrough){
		/*
		 * push the buffer and forward error, only for correction
		 * which is not considered here
		 * */
		GST_DEBUG_OBJECT (enc, "pushing buffer with %" G_GSIZE_FORMAT " bytes", gst_buffer_get_size (buffer));
		*flow = gst_pad_push (pad, buffer);
	}else if(sinked<(gint)(enc->n_chunks*CHUNKS_LENGTH) && !is_tags){//*REF*
		/*
		 * Only complete blocks are going to be sink in the part of the source code
		 * otherwise they are going to be considered in the next part of this if, example:
		 * (5 chunks * 2.048 ) = 10.240 then only 10 seconds are considered as complete
		 * blocks
		 * */
		sinked++;
		GST_DEBUG_OBJECT(enc, "Processing: handing frame %d of %" G_GSIZE_FORMAT " bytes", sinked, gst_buffer_get_size (buffer));
		*flow = gst_audio_encoder_finish_frame (GST_AUDIO_ENCODER (enc), buffer, samples);//samples
	}else if(!is_tags){
		/*
		 * The incomplete blocks which are basically the remaining time rounded towards zero, example:
		 * (5 chunks * 2.048 ) = 10.240 then only 0.240 seconds are considered here, however if is a TAG and
		 * not an audio sample is going to be encoded in the else part
		 * */
		GST_DEBUG_OBJECT (enc, "Processing: handing frame %d of %" G_GSIZE_FORMAT " bytes", sinked, gst_buffer_get_size (buffer));
		sinked = 0;
		gst_pad_push (pad, buffer);
		*flow = gst_audio_encoder_finish_frame (GST_AUDIO_ENCODER (enc), NULL, -1);//samples
	}else{
		/*
		 * for coding the tags of each chunk ( cry=1 or no-cry=199 )
		 * */
		GST_DEBUG_OBJECT(enc, "Processing: handing TAG %d of %" G_GSIZE_FORMAT " bytes", sinked, gst_buffer_get_size (buffer));
		*flow = gst_audio_encoder_finish_frame (GST_AUDIO_ENCODER (enc), buffer, samples);//samples
	}

	encoding_failed:
		if (*flow != GST_FLOW_OK){
			GST_WARNING_OBJECT (enc, "flow on %s:%s = %s", GST_DEBUG_PAD_NAME (pad), gst_flow_get_name (*flow));
			return FALSE;
		}

	return TRUE;
}

static void gst_wavpack_enc_fix_channel_order(GstWavpackEnc * enc, gint32 * data, gint nsamples){
	gint i, j;
	gint32 tmp[8];

	for(i = 0; i < nsamples / enc->channels; i++){
		for(j = 0; j < enc->channels; j++){
			tmp[enc->channel_mapping[j]] = data[j];
		}
		for(j = 0; j < enc->channels; j++){
			data[j] = tmp[j];
		}
		data += enc->channels;
	}
}


/*
 * Used just for debugging purposes
 * */
static gboolean buffer_list_calc_size (GstBuffer ** buf, guint idx, gpointer data){
  guint *p_size = data;
  gsize buf_size;

  buf_size = gst_buffer_get_size (*buf);
  GST_TRACE ("buffer %u has size %f", idx, (gfloat)buf_size/1024.0);
  *p_size += buf_size;

  return TRUE;
}


/* Main Function where we write the blocks to file */
static GstFlowReturn gst_wavpack_enc_handle_frame(GstAudioEncoder * benc, GstBuffer * buf){
	GstWavpackEnc *enc = GST_WAVPACK_ENC (benc);
	uint32_t sample_count;
	GstFlowReturn ret = GST_FLOW_OK;
	GstMapInfo map;
	guint i=0, extra_chunks=0;
	GstBuffer *Buffer_to_push=NULL;
	TagInfo *detection;
	guint crycount=0;
	time_t start,end;

	/* base class ensures configuration */
	g_return_val_if_fail (enc->depth != 0, GST_FLOW_NOT_NEGOTIATED);

	/* reset the last returns to GST_FLOW_OK. This is only set to something else
	* while WavpackPackSamples() or more specific gst_wavpack_enc_push_block()
	* so not valid anymore */
	enc->srcpad_last_return = enc->wvcsrcpad_last_return = GST_FLOW_OK;

	if (G_UNLIKELY (!buf)){
		return gst_wavpack_enc_drain (enc);
	}

	if(enc->n_chunks>1){
		extra_chunks = 1;
	}

	/* debugging info */
	GST_DEBUG_OBJECT (enc, "Incoming Buffer is of size %f bytes\n", (gfloat)gst_buffer_get_size(buf));

	if(enc->bufferlist==NULL){
		/*
		 * We have no list to store the incoming buffer so we create an empty one
		 * */
		enc->bufferlist = gst_buffer_list_new ();

		/*
		 * Mapping the incoming buffer to be stored in the list
		 * afterwards, we actually store a copy or we wont be able to
		 * unref later since this element does not have ownership of
		 * the incoming buffer.
		 * */
		gst_buffer_map (buf, &map, GST_MAP_READ);
		sample_count = gst_buffer_get_size (buf);
		GstBuffer *temp_buf = gst_buffer_new_and_alloc (sample_count);
		gst_buffer_copy_into(temp_buf, buf, GST_BUFFER_COPY_METADATA, 0, -1);
		gst_buffer_fill (temp_buf, 0, map.data, sample_count);
		gst_buffer_unmap (buf, &map);

		/*
		 * Since it is first buffer just insert it directly
		 * */
		gst_buffer_list_add(enc->bufferlist, temp_buf);

		/*
		 * Recording the label and chunk Id coming from
		 * the element upstream
		 * */
		enc->cry_list[enc->chunks_count] = enc->label;
		enc->chunks_count++;

		/*
		 * Generates a message upstream for the pipeline to know that NO key-unit
		 * have to be generated since we havent sink anything yet
		 * */
		{
			GstStructure *s;
			s = gst_structure_new ("GstWavPackWithTag", "generate_keyunit",
					G_TYPE_BOOLEAN, FALSE, NULL);

			gst_element_post_message (GST_ELEMENT_CAST (enc),
					gst_message_new_element (GST_OBJECT_CAST (enc), s));
		}
		goto exit;//we return upstream
	}else if(gst_buffer_list_length(enc->bufferlist) < (enc->n_chunks+extra_chunks)){

		/*
		 * Keeping track of the list size for debugging purposes
		 * */
		GST_DEBUG_OBJECT (enc, "List length: %d", gst_buffer_list_length(enc->bufferlist));

		/*
		 * Mapping the incoming buffer and making a copy to be stored in the
		 * list, to be later processed with the encoder.
		 * */
		gst_buffer_map (buf, &map, GST_MAP_READ);
		sample_count = gst_buffer_get_size (buf);
		GstBuffer *temp_buf = gst_buffer_new_and_alloc (sample_count);
		gst_buffer_copy_into(temp_buf, buf, GST_BUFFER_COPY_METADATA, 0, -1);
		gst_buffer_fill (temp_buf, 0, map.data, sample_count);
		gst_buffer_unmap (buf, &map);


		/*
		 * Since we haven't full up the list, we just insert
		 * it at the end
		 * */
		gst_buffer_list_add(enc->bufferlist, temp_buf);

		/*
		 * Recording the label and chunk Id coming from
		 * the element upstream
		 * */
		enc->cry_list[enc->chunks_count] = enc->label;
		enc->chunks_count++;

		/*
		 * Generates a message upstream for the pipeline to know that NO key-unit
		 * have to be generated since we haven't sink anything yet
		 * */
		{
			GstStructure *s;
			s = gst_structure_new ("GstWavPackWithTag", "generate_keyunit",
					G_TYPE_BOOLEAN, FALSE, NULL);

			gst_element_post_message (GST_ELEMENT_CAST (enc),
					gst_message_new_element (GST_OBJECT_CAST (enc), s));
		}
		goto exit;//we return upstream
	}else{
		/*
		 * NOTE:
		 * This is the heart of this element, where we determine what to do with the chunks collected so far,
		 * that is to say, we sink them if a cry burst at any of them is detected, otherwise we just skip it
		 * and flush the internal adapter used by the encoder
		 * */
		start = clock();//to have some performance reference

		/*
		 * check if we need to encode the chunks collected so far
		 * */
		for(i=0;i<enc->n_chunks;i++){
			if(enc->cry_list[i] != DEFAULT_LABEL){
				/*
				 * This means that this chunk is reported to be
				 * a cry burst
				 * */
				GST_DEBUG_OBJECT (enc, "Index: %i",i);
				crycount++;
			}
		}

		if(crycount==0){
			/*
			 * This simply means, no cry burst detected, therefore just skip this 10 seconds and flush
			 * the internal adapter
			 * */
			GST_DEBUG_OBJECT (enc, "List_size: %d, Cry_count: %d, Sync: %s", gst_buffer_list_length(enc->bufferlist),
					crycount, "FALSE");

			/*
			 * Leave only one chunk to allow overlapping
			 * */
			guint size = 0;
			gst_buffer_list_foreach (enc->bufferlist, buffer_list_calc_size, &size);
			GST_LOG_OBJECT (enc, "Dropping %f Kbytes from list %p: ", (gfloat)size/1024.f, enc->bufferlist);
			gst_buffer_list_remove(enc->bufferlist, 0, enc->n_chunks);


			/*
			 * Generates a message upstream for the pipeline to know that NO key-unit
			 * have to be generated since we haven't sink anything yet
			 * */
			{
				GstStructure *s;
				s = gst_structure_new ("GstWavPackWithTag", "generate_keyunit",
						G_TYPE_BOOLEAN, FALSE, NULL);

				gst_element_post_message (GST_ELEMENT_CAST (enc),
						gst_message_new_element (GST_OBJECT_CAST (enc), s));
			}

			/*
			 * IMPORTANT:
			 * Remember that we still have the incoming buffer which we did not considered above
			 * therefore we have to map it and make a copy to be stored in the list
			 * */
			gst_buffer_map (buf, &map, GST_MAP_READ);
			sample_count = gst_buffer_get_size (buf);
			GstBuffer *temp_buf = gst_buffer_new_and_alloc (sample_count);
			gst_buffer_copy_into(temp_buf, buf, GST_BUFFER_COPY_METADATA, 0, -1);
			gst_buffer_fill (temp_buf, 0, map.data, sample_count);
			gst_buffer_unmap (buf, &map);


			/*
			 * Since we removed buffers from the list a simple insertion is all
			 * we need
			 * */
			gst_buffer_list_add(enc->bufferlist, temp_buf);


			/*
			 * The label of the last previously stored buffer becomes the label
			 * if the current FIRST buffer (shift)
			 * */
			enc->cry_list[0] = enc->cry_list[enc->n_chunks];//first element all the time
			GST_DEBUG_OBJECT (enc, "Label_000: %i",enc->cry_list[0]);

			/*
			 * The label of the new buffer is assigned accordingly*/
			enc->cry_list[1] = enc->label;
			GST_DEBUG_OBJECT (enc, "Label_111: %i",enc->cry_list[1]);


			/*
			 * Keeping track of our list status
			 * */
			enc->chunks_count = gst_buffer_list_length(enc->bufferlist);


			/*
			 * For debugging purposes to know if we have the correct number of
			 * buffers and amount of memory
			 * */
			size = 0;
			gst_buffer_list_foreach (enc->bufferlist, buffer_list_calc_size, &size);
			GST_LOG_OBJECT (enc, "Total size of buffer_list %p: %f Kbytes", enc->bufferlist, (gfloat)size/1024.f);

			/*
			 * FIXME: This is really the memory leak..!!!
			 * If we don't finish the frame the internal adapter for the audio encoder will
			 * not be flushed keeping garbage data that has been already removed, this should not be
			 * required in reality but the adapter is managed by Gstreamer ... how to remove it?
			 * */
			GST_DEBUG_OBJECT (enc, "NO need for Processing ... Finishing Segment..!!!!");
			ret = gst_audio_encoder_finish_frame(GST_AUDIO_ENCODER(enc), NULL, -1);// -1 to clear the adapter

			goto exit;//we return upstream
		}else{
			/*
			 * In this case we have to encode the first 10 seconds stored in the list, therefore we
			 * extract the necessary buffers and send it to our Wavpack-Encoder
			 * */

			GST_DEBUG_OBJECT (enc, "List_size: %d, Cry_count: %d, Sync: %s", gst_buffer_list_length(enc->bufferlist),
								crycount, "TRUE");

			/*
			 * Extracting the necessary buffers
			 * */
			Buffer_to_push = gst_buffer_new();
			gst_buffer_list_foreach(enc->bufferlist, buffer_list_copy_data, Buffer_to_push);

			/*
			 * Leave only one chunk to allow overlapping
			 * */
			guint size = 0;
			gst_buffer_list_foreach (enc->bufferlist, buffer_list_calc_size, &size);
			GST_LOG_OBJECT (enc, "Processing: Removing %f Kbytes from list %p: ", (gfloat)size/1024.f, enc->bufferlist);
			gst_buffer_list_remove(enc->bufferlist, 0, enc->n_chunks);
		}
	}


	/*
	 * Using the extracted buffers from the list to encode using wavpack library
	 * using 4-bytes per sample
	 * */
	sample_count = gst_buffer_get_size(Buffer_to_push)/4;
	GST_DEBUG_OBJECT(enc, "Processing: %f Kbytes or %d raw samples\n",
			(gfloat)gst_buffer_get_size(Buffer_to_push)/1024.f, sample_count);


	/*
	 * Check if we already have a valid WavpackContext, otherwise make one
	 * */
	if (!enc->wp_context){
		/* create raw context */
		enc->wp_context = WavpackOpenFileOutput (gst_wavpack_enc_push_block, &enc->wv_id,
				(enc->correction_mode > 0) ? &enc->wvc_id : NULL);
		if (!enc->wp_context)
		  goto context_failed;

		/* set the WavpackConfig according to our parameters */
		gst_wavpack_enc_set_wp_config (enc);

		/* set the configuration to the context now that we know everything
		 * and initialize the encoder */
		if (!WavpackSetConfiguration (enc->wp_context, enc->wp_config, (uint32_t) (-1))
			|| !WavpackPackInit (enc->wp_context)){
			WavpackCloseFile (enc->wp_context);
			goto config_failed;
		}

		GST_DEBUG_OBJECT (enc, "Processing: setup of encoding context successful...!!");
	}

	/*
	 * To make sure we can modify this buffer, otherwise the mapping will not be possible,
	 * but this is for safety only, in reality should not be required since this element owns
	 * it, and therefore writability is by default allowed
	 * */
	Buffer_to_push = gst_buffer_make_writable (Buffer_to_push);

	if(enc->need_channel_remap){
		GST_DEBUG_OBJECT (enc, "Processing: Re-mapping the channels..");
		gst_buffer_map (Buffer_to_push, &map, GST_MAP_WRITE);
		gst_wavpack_enc_fix_channel_order (enc, (gint32*) map.data, sample_count);
		gst_buffer_unmap (Buffer_to_push, &map);
		GST_DEBUG_OBJECT (enc, "Processing: Re-mapping the channels finished..");
	}

	if(!gst_buffer_map(Buffer_to_push, &map, GST_MAP_READ)){//Buffer_to_push
		GST_ERROR_OBJECT (enc, "Error while mapping the buffer to be encoded..");
		goto encoding_failed;
	}

	/* if we want to append the MD5 sum to the stream update it here
	* with the current raw samples */
	if(enc->md5){
		g_checksum_update(enc->md5_context, map.data, map.size);
	}


	/*
	 * Encode and handle return values from encoding , the parameter"sample_count / enc->channels"
	 * is used since Wavpack processes individual channels internally which were set during
	 * the context creation above
	 * */
	GST_DEBUG_OBJECT(enc, "Processing: %d samples with WavpackPack..", sample_count);
	is_tags=FALSE;//simple flag
	if(WavpackPackSamples (enc->wp_context, (gint32*) map.data, sample_count / enc->channels)){
		GST_LOG_OBJECT (enc, "Processing: samples successful, now writing the tag for the corresponding blocks");
		is_tags = TRUE;//simple flag
		for(i=0;i<enc->chunks_count;i++){
			/*
			 * Labeling the sinked chunks
			 * */
			detection = g_malloc0(sizeof(TagInfo));
			sprintf(detection->item , "Block_%d_label",i);
			sprintf(detection->value , "%d", enc->cry_list[i]);
			detection->vsize = strlen(detection->item)+strlen(detection->value);
			gint res = WavpackAppendTagItem (enc->wp_context, detection->item, detection->value , detection->vsize);
			if (!res || !WavpackWriteTag (enc->wp_context)) {
				GST_ERROR_OBJECT(enc, "Error writing tag for the block");

				/*
				 * freeing memory used before going to error handling
				 * part
				 * */
				g_free(detection);
				goto encoding_failed;
			}

			/*
			 * freeing memory used before continuing
			 * */
			g_free(detection);
		}
		GST_LOG_OBJECT (enc, "Processing: done writing the tag for the blocks");
		gst_buffer_unmap (Buffer_to_push, &map);


		/*
		 * IMPORTANT:
		 * Remember that we still have the incoming buffer which we did not considered above
		 * therefore we have to map it and make a copy to be stored in the list
		 * */
		gst_buffer_map (buf, &map, GST_MAP_READ);
		sample_count = gst_buffer_get_size (buf);
		GstBuffer *temp_buf = gst_buffer_new_and_alloc (sample_count);
		gst_buffer_copy_into(temp_buf, buf, GST_BUFFER_COPY_METADATA, 0, -1);
		gst_buffer_fill (temp_buf, 0, map.data, sample_count);
		gst_buffer_unmap (buf, &map);

		/*
		 * Since we removed buffers from the list a simple insertion is all
		 * we need
		 * */
		gst_buffer_list_add(enc->bufferlist, temp_buf);


		/*
		 * The label of the last previously stored buffer becomes the label
		 * if the current FIRST buffer (shift)
		 * */
		enc->cry_list[0] = enc->cry_list[enc->n_chunks];//first element all the time
		GST_DEBUG_OBJECT (enc, "Label_000: %i",enc->cry_list[0]);

		/*
		 * The label of the new buffer is assigned accordingly*/
		enc->cry_list[1] = enc->label;
		GST_DEBUG_OBJECT (enc, "Label_111: %i",enc->cry_list[1]);


		/*
		 * Keeping track of our list status
		 * */
		enc->chunks_count = gst_buffer_list_length(enc->bufferlist);


		/*
		 * For debugging purposes to know if we have the correct number of
		 * buffers and amount of memory
		 * */
		guint size = 0;
		gst_buffer_list_foreach (enc->bufferlist, buffer_list_calc_size, &size);
		GST_LOG_OBJECT (enc, "Processing: Total size of buffer_list %p: %f Kbytes", enc->bufferlist, (gfloat)size/1024.f);

		/*
		 * "pending_buffer" for incomplete data blocks, but since we don't care about it
		 * we unref them since  we don't need them anymore, the only part that cares about it is the drain
		 * and that will only happens if "incomplete samples" is activated, in which case only
		 * complete data blocks are to be used, the incomplete ones are going to be kept in a pool
		 * to later be drained.
		 * */
		if (enc->pending_buffer){
			gst_buffer_unref (enc->pending_buffer);
			enc->pending_buffer = NULL;
			enc->pending_offset = 0;
		}else{
			enc->pending_offset = 0;
		}

		/*
		 * Unref the "Buffer_to_push" since we don't need it anymore and remember that what we encoded
		 * was really just the data inside not the buffer itself, therefore when pushing the
		 * encoded data in the pads we do not un-ref this "Buffer_to_push" */
		if(Buffer_to_push && GST_IS_BUFFER(Buffer_to_push)){
			GST_DEBUG_OBJECT (enc, "Processing: UN-REFING \"Buffer_to_push\"..");
			gst_buffer_unref(Buffer_to_push);
			Buffer_to_push=NULL;
		}
	}else{
		gst_buffer_unmap (Buffer_to_push, &map);

		if(Buffer_to_push){
			gst_buffer_unref (Buffer_to_push);//freeing the memory blocks
			Buffer_to_push=NULL;
		}

		/*
		 * Generates a message upstream for the pipeline to know that NO key-unit
		 * have to be generated since we haven't sink anything yet
		 * */
		{
			GstStructure *s;
			s = gst_structure_new ("GstWavPackWithTag", "generate_keyunit",
					G_TYPE_BOOLEAN, FALSE, NULL);

			gst_element_post_message (GST_ELEMENT_CAST (enc),
					gst_message_new_element (GST_OBJECT_CAST (enc), s));
		}

		/*
		 * IMPORTANT:
		 * EVEN WE FAILED ENCODING, Remember that we still have the incoming buffer which we
		 * did not considered above therefore we have to map it and make a copy to be
		 * stored in the list
		 * */
		gst_buffer_map (buf, &map, GST_MAP_READ);
		sample_count = gst_buffer_get_size (buf);
		GstBuffer *temp_buf = gst_buffer_new_and_alloc (sample_count);
		gst_buffer_copy_into(temp_buf, buf, GST_BUFFER_COPY_METADATA, 0, -1);
		gst_buffer_fill (temp_buf, 0, map.data, sample_count);
		gst_buffer_unmap (buf, &map);


		/*
		 * Since we removed buffers from the list a simple insertion is all
		 * we need
		 * */
		gst_buffer_list_add(enc->bufferlist, temp_buf);


		/*
		 * The label of the last previously stored buffer becomes the label
		 * if the current FIRST buffer (shift)
		 * */
		enc->cry_list[0] = enc->cry_list[enc->n_chunks];//first element all the time
		GST_DEBUG_OBJECT (enc, "Label_000: %i",enc->cry_list[0]);

		/*
		 * The label of the new buffer is assigned accordingly*/
		enc->cry_list[1] = enc->label;
		GST_DEBUG_OBJECT (enc, "Label_111: %i",enc->cry_list[1]);


		/*
		 * Keeping track of our list status
		 * */
		enc->chunks_count = gst_buffer_list_length(enc->bufferlist);


		/*
		 * For debugging purposes to know if we have the correct number of
		 * buffers and amount of memory
		 * */
		guint size = 0;
		gst_buffer_list_foreach (enc->bufferlist, buffer_list_calc_size, &size);
		GST_LOG_OBJECT (enc, "Processing: Total size of buffer_list %p: %f Kbytes", enc->bufferlist, (gfloat)size/1024.f);


		/*
		 * Since we did not encode the segment of 10 seconds, then we have to
		 * flush the internal adapter or we are going to have memory leak
		 * */
		GST_DEBUG_OBJECT (enc, "Processing: Finishing Segment after samples and tags encoding");
		ret = gst_audio_encoder_finish_frame(GST_AUDIO_ENCODER(enc), NULL, -1);// -1 to clear the adapter

		if ((enc->srcpad_last_return == GST_FLOW_OK) ||
					(enc->wvcsrcpad_last_return == GST_FLOW_OK)){
				ret = GST_FLOW_OK;
		}else if((enc->srcpad_last_return == GST_FLOW_NOT_LINKED) &&
				(enc->wvcsrcpad_last_return == GST_FLOW_NOT_LINKED)){
			ret = GST_FLOW_NOT_LINKED;
		}else if((enc->srcpad_last_return == GST_FLOW_FLUSHING) &&
				(enc->wvcsrcpad_last_return == GST_FLOW_FLUSHING)){
			ret = GST_FLOW_FLUSHING;
		}else{
			goto encoding_failed;
		}
	}


	/*
	 * NOTE:
	 * At this point is the only moment we are good to go and
	 * send a message upstream to generate a key-unit, so a new file
	 * should be created next time we sink another segment of 10 seconds
	 * otherwise we will keep sinking in the same file.
	 * */
	{
		GstStructure *s;
		s = gst_structure_new ("GstWavPackWithTag", "generate_keyunit",
				G_TYPE_BOOLEAN, TRUE, NULL);

		gst_element_post_message (GST_ELEMENT_CAST (enc),
				gst_message_new_element (GST_OBJECT_CAST (enc), s));
	}

	/*
	 * Resetting the encoder from Wavpack library so we don't screw
	 * the output file putting data that we don't have anymore
	 * */
	enc->data_in_list = FALSE;
	enc->reset_encoder = FALSE;
	gst_wavpack_enc_stop(enc);

	/*
	 * reset encoder context in Wavpack Library, but without
	 * restarting the whole encoder element, or the whole
	 * pipeline will crash after the first segment is compressed
	 * and send to multifile sink element
	 * */
	gst_wavpack_enc_set_wp_config(enc);

	end = clock();//to have some performance reference
	GST_LOG_OBJECT(enc, "The time spent during compression and tag encoding was: %f ms.", ((gfloat)(end-start)/CLOCKS_PER_SEC)*1000);

exit:
{
	return ret;
}

  /* ERRORS */
encoding_failed:
{
	GST_ELEMENT_ERROR (enc, LIBRARY, ENCODE, (NULL), ("encoding samples failed"));
	ret = GST_FLOW_ERROR;
	goto exit;
}
config_failed:
{
	GST_ELEMENT_ERROR (enc, LIBRARY, SETTINGS, (NULL), ("error setting up WAVPACK encoding context"));
	ret = GST_FLOW_ERROR;
	goto exit;
}
context_failed:
{
	GST_ELEMENT_ERROR (enc, LIBRARY, INIT, (NULL), ("error creating WAVPACK context"));
	ret = GST_FLOW_ERROR;
	goto exit;
}
}

/* Usually this problem can't be avoided in wavpack */
static void gst_wavpack_enc_rewrite_first_block (GstWavpackEnc * enc){
	GstSegment segment;
	gboolean ret;
	GstQuery *query;
	gboolean seekable = FALSE;

	g_return_if_fail (enc);
	g_return_if_fail (enc->first_block);

	/* update the sample count in the first block */
	WavpackUpdateNumSamples (enc->wp_context, enc->first_block);

	/* try to seek to the beginning of the output */
	query = gst_query_new_seeking (GST_FORMAT_BYTES);
	if (gst_pad_peer_query (GST_AUDIO_ENCODER_SRC_PAD (enc), query)){
		GstFormat format;

		gst_query_parse_seeking (query, &format, &seekable, NULL, NULL);
		if (format != GST_FORMAT_BYTES)
			seekable = FALSE;
	}else{
		GST_LOG_OBJECT (enc, "SEEKING query not handled");
	}

	gst_query_unref (query);

	if(!seekable){
		GST_DEBUG_OBJECT (enc, "downstream not seekable; not rewriting");
		return;
	}

	gst_segment_init (&segment, GST_FORMAT_BYTES);
	ret = gst_pad_push_event (GST_AUDIO_ENCODER_SRC_PAD (enc), gst_event_new_segment (&segment));
	if (ret){
		/* try to rewrite the first block */
		GST_DEBUG_OBJECT (enc, "rewriting first block ...");
		enc->wv_id.passthrough = TRUE;
		ret = gst_wavpack_enc_push_block (&enc->wv_id, enc->first_block, enc->first_block_size);
		enc->wv_id.passthrough = FALSE;
		g_free (enc->first_block);
		enc->first_block = NULL;
	} else {
	GST_WARNING_OBJECT (enc, "rewriting of first block failed. "
		"Seeking to first block failed!");
	}
}

/* Cleaning all data if needed specially when closing or receiving invalid data */
static GstFlowReturn gst_wavpack_enc_drain (GstWavpackEnc * enc){
	if (!enc->wp_context)
		return GST_FLOW_OK;

	GST_DEBUG_OBJECT (enc, "draining INSIDE WAVPACK...!!!!");

	/* Encode all remaining samples and flush them to the src pads */
	WavpackFlushSamples (enc->wp_context);

	/* Drop all remaining data, this is no complete block otherwise
	* it would've been pushed already */
	if (enc->pending_buffer) {
		gst_buffer_unref (enc->pending_buffer);
		enc->pending_buffer = NULL;
		enc->pending_offset = 0;
	}

	/* write the MD5 sum if we have to write one */
	if ((enc->md5) && (enc->md5_context)) {
		guint8 md5_digest[16];
		gsize digest_len = sizeof (md5_digest);

		g_checksum_get_digest (enc->md5_context, md5_digest, &digest_len);
		if (digest_len == sizeof (md5_digest)){
			WavpackStoreMD5Sum (enc->wp_context, md5_digest);
			WavpackFlushSamples (enc->wp_context);
		}else
			GST_WARNING_OBJECT (enc, "Calculating MD5 digest failed");
	}

	/* Try to rewrite the first frame with the correct sample number */
	if (enc->first_block)
		gst_wavpack_enc_rewrite_first_block (enc);

	/* close the context if not already happened */
	if (enc->wp_context) {
		WavpackCloseFile (enc->wp_context);
		enc->wp_context = NULL;
	}

	return GST_FLOW_OK;
}

static gboolean gst_wavpack_enc_sink_event (GstAudioEncoder * benc, GstEvent * event){
	GstWavpackEnc *enc = GST_WAVPACK_ENC (benc);

	GST_DEBUG_OBJECT (enc, "Received %s event on sinkpad",
	  GST_EVENT_TYPE_NAME (event));

	switch (GST_EVENT_TYPE (event)){
		case GST_EVENT_SEGMENT:
			if(enc->wp_context){
				GST_WARNING_OBJECT (enc, "got NEWSEGMENT after encoding already started");
			}
			/* peek and hold NEWSEGMENT events for sending on correction pad */
			if (enc->pending_segment){
				gst_event_unref (enc->pending_segment);
			}

			enc->pending_segment = gst_event_ref (event);
			break;
		case GST_EVENT_EOS:
			GST_DEBUG_OBJECT(enc, "EOS proceeding to free data");
			enc->data_in_list = TRUE;
			enc->reset_encoder = TRUE;
			break;
		default:
		  break;
	}

	/* baseclass handles rest */
	return GST_AUDIO_ENCODER_CLASS (parent_class)->sink_event (benc, event);
}

static void gst_wavpack_enc_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec){
	GstWavpackEnc *enc = GST_WAVPACK_ENC (object);

	switch (prop_id){
		case ARG_MODE:
			enc->mode = g_value_get_enum (value);
			break;
		case ARG_BITRATE:{
			guint val = g_value_get_uint (value);

			if ((val >= 24000) && (val <= 9600000)) {
			enc->bitrate = val;
			enc->bps = 0.0;
			} else {
			enc->bitrate = 0;
			enc->bps = 0.0;
			}
			break;
		}
		case ARG_BITSPERSAMPLE:{
			gdouble val = g_value_get_double (value);

			if ((val >= 2.0) && (val <= 24.0)) {
			enc->bps = val;
			enc->bitrate = 0;
			} else {
			enc->bps = 0.0;
			enc->bitrate = 0;
			}
			break;
		}
		case ARG_CORRECTION_MODE:
			enc->correction_mode = g_value_get_enum (value);
			break;
		case ARG_MD5:
			enc->md5 = g_value_get_boolean (value);
			break;
		case ARG_EXTRA_PROCESSING:
			enc->extra_processing = g_value_get_uint (value);
			break;
		case ARG_JOINT_STEREO_MODE:
			enc->joint_stereo_mode = g_value_get_enum (value);
			break;
		case PROP_CHUNKS:
			enc->n_chunks = g_value_get_uint (value);
			break;
		case PROP_LABEL:
			enc->label = g_value_get_int (value);
			if(enc->label<0)
				enc->label = DEFAULT_LABEL;
			break;
		default:
			G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
			break;
	}
}

static void gst_wavpack_enc_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec){
	GstWavpackEnc *enc = GST_WAVPACK_ENC (object);

	switch (prop_id){
		case ARG_MODE:
			g_value_set_enum (value, enc->mode);
			break;
		case ARG_BITRATE:
			if (enc->bps == 0.0) {
				g_value_set_uint (value, enc->bitrate);
			} else {
				g_value_set_uint (value, 0);
			}
			break;
		case ARG_BITSPERSAMPLE:
			if (enc->bitrate == 0) {
				g_value_set_double (value, enc->bps);
			} else {
				g_value_set_double (value, 0.0);
			}
			break;
		case ARG_CORRECTION_MODE:
			g_value_set_enum (value, enc->correction_mode);
			break;
			case ARG_MD5:
			g_value_set_boolean (value, enc->md5);
			break;
		case ARG_EXTRA_PROCESSING:
			g_value_set_uint (value, enc->extra_processing);
			break;
		case ARG_JOINT_STEREO_MODE:
			g_value_set_enum (value, enc->joint_stereo_mode);
			break;
		case PROP_CHUNKS:
			g_value_set_uint (value, enc->n_chunks);
			break;
		case PROP_LABEL:
			if(enc->label<0)
			enc->label = DEFAULT_LABEL;
			g_value_set_int (value, enc->label);
			break;
		default:
			G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
			break;
	}
}

static gboolean gst_wavpack_enc_plugin_init (GstPlugin * plugin){
  if (!gst_element_register (plugin, "wavpackwithtag", GST_RANK_NONE, GST_TYPE_WAVPACK_ENC))
    return FALSE;

  GST_DEBUG_CATEGORY_INIT (gst_wavpack_enc_debug, "wavpackwithtag", 0, "Wavpack encoder with tags");

  return TRUE;
}

#ifndef VERSION
#define VERSION "0.0.1"
#endif
#ifndef PACKAGE
#define PACKAGE "WAVPACKENC_WITHTAG_API"
#endif
#ifndef PACKAGE_NAME
#define PACKAGE_NAME "WAVPACKENC_WITHTAG_API"
#endif
#ifndef GST_PACKAGE_ORIGIN
#define GST_PACKAGE_ORIGIN "http://www.automodules.com/"
#endif


GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
	wavpackwithtag,
    "WavpackEncoderWithTag lossless/lossy audio format handling",
	gst_wavpack_enc_plugin_init, VERSION, "LGPL", PACKAGE_NAME, GST_PACKAGE_ORIGIN)

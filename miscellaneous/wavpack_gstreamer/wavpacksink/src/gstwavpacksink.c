/* GStreamer
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


#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "gstwavpacksink.h"
#include <gst/pbutils/pbutils.h>
#include <gst/audio/audio.h>
#include <gst/video/video.h>
#include <gio/gio.h>
#include <glib/gstdio.h>
#include <memory.h>
#include "metainfo.h"


GST_DEBUG_CATEGORY_STATIC (gst_wavpack_sink_debug);
#define GST_CAT_DEFAULT gst_wavpack_sink_debug


/*****************CHANGE THIS IF WE WANT TO SAVE THE FILE IN ANOTHER PLACE ****************/
//josue computer
#if 1
	#define DEFAULT_LOCATION ("segment%05d.wv")
#else
	#define DEFAULT_LOCATION ("PUT HERE THE PATH")
#endif
/*****************CHANGE THIS IF WE WANT TO SAVE THE FILE IN ANOTHER PLACE ****************/

#define DEFAULT_CHUNKS (5)
#define DEFAULT_LENGTH (2.048)
#define DEFAULT_SYNC (TRUE)
#define DEFAULT_ASYNC (TRUE)

enum{
  PROP_0,
  PROP_LOCATION,
  PROP_CHUNKS,
  PROP_COMPRESSION_MODE,
  PROP_SYNC,
  PROP_ASYNC
};


/* Compression mode for Wavpack Library */
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

    qtype = g_enum_register_static ("WavpackEncMode", values);
  }
  return qtype;
}




/*
 * To link wavpack sink with external elements which need any type of updating
 * in their function as the samples pass through here
 * */
struct _GstWavpackSinkPrivate
{
	/*function to be in charge of communicating wavpack sink with external elements*/
	GstWavpackSinkCallbacks callbacks;

	/*data passed from external elements, which are to be updated*/
	gpointer user_data;

	/*
	* Ring buffer like system in case external elements want
	* this information
	* */
	GMutex data_access_locker;
	GstBufferList *bufferlist;
};







static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK, GST_PAD_ALWAYS, GST_STATIC_CAPS_ANY);

#define gst_wavpack_sink_parent_class parent_class
G_DEFINE_TYPE (GstWavpackSink, gst_wavpack_sink, GST_TYPE_BIN);

static void gst_wavpack_sink_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * spec);
static void gst_wavpack_sink_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * spec);
static void gst_wavpack_sink_handle_message (GstBin * bin, GstMessage * message);
static GstPadProbeReturn gst_wavpack_sink_ghost_event_probe (GstPad * pad,
    GstPadProbeInfo * info, gpointer data);
static GstPadProbeReturn gst_wavpack_sink_ghost_buffer_probe (GstPad * pad,
    GstPadProbeInfo * info, gpointer data);
static void gst_wavpack_sink_reset (GstWavpackSink * sink);
static GstStateChangeReturn gst_wavpack_sink_change_state (GstElement * element, GstStateChange trans);
static gboolean schedule_next_key_unit (GstWavpackSink * sink);

static void gst_wavpack_sink_dispose (GObject * object){
	GstWavpackSink *sink = GST_WAVPACK_SINK_CAST(object);
	GstWavpackSinkPrivate *priv = sink->private_info;

	/*FREE LIST*/
	GST_DEBUG_OBJECT(sink, "Disposing");
	g_mutex_lock(&priv->data_access_locker);

	if(GST_IS_BUFFER_LIST(priv->bufferlist)){
		gst_buffer_list_remove(priv->bufferlist, 0, gst_buffer_list_length(priv->bufferlist));//delete all samples
	}


	g_mutex_unlock(&priv->data_access_locker);

	G_OBJECT_CLASS (parent_class)->dispose((GObject *) sink);
}

static void gst_wavpack_sink_finalize (GObject * object){
	GstWavpackSink *sink = GST_WAVPACK_SINK_CAST (object);
	GstWavpackSinkPrivate *priv = sink->private_info;
	g_free (sink->location);

	/*remove lock which is not needed anymore*/
	g_mutex_clear(&priv->data_access_locker);

	G_OBJECT_CLASS (parent_class)->finalize ((GObject *) sink);
}

static void gst_wavpack_sink_class_init (GstWavpackSinkClass * klass){
	GObjectClass *gobject_class;
	GstElementClass *element_class;
	GstBinClass *bin_class;

	gobject_class = (GObjectClass *) klass;
	element_class = GST_ELEMENT_CLASS (klass);
	bin_class = GST_BIN_CLASS (klass);

	gst_element_class_add_pad_template (element_class, gst_static_pad_template_get (&sink_template));

	gst_element_class_set_static_metadata (element_class,
			"Wavpack multifile sink", "Sink", "Wavpack multifile sink",
			"Josue Cuevas <josuercuevas@gmail.com>");

	element_class->change_state = GST_DEBUG_FUNCPTR (gst_wavpack_sink_change_state);

	bin_class->handle_message = gst_wavpack_sink_handle_message;
	gobject_class->dispose = gst_wavpack_sink_dispose;
	gobject_class->finalize = gst_wavpack_sink_finalize;
	gobject_class->set_property = gst_wavpack_sink_set_property;
	gobject_class->get_property = gst_wavpack_sink_get_property;

	/* install all properties */
		g_object_class_install_property (gobject_class, PROP_COMPRESSION_MODE, g_param_spec_enum ("enc-mode", "Encoding mode",
			  "Speed versus compression tradeoff.", GST_TYPE_WAVPACK_ENC_MODE, GST_WAVPACK_ENC_MODE_DEFAULT,
			  G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property (gobject_class, PROP_LOCATION, g_param_spec_string("location",
			"File Location", "Location of the file to write", DEFAULT_LOCATION,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property (gobject_class, PROP_CHUNKS, g_param_spec_uint ("chunks",
			"Chunks to be stored per file", "Number of chunks to be written in the file", 1, 20,
			DEFAULT_CHUNKS, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property (gobject_class, PROP_SYNC, g_param_spec_boolean("sync",
			"Buffer sync wrt GstClock", "Sinks the buffer only if their GstClock time is not too late", DEFAULT_SYNC,
			G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
	g_object_class_install_property (gobject_class, PROP_ASYNC, g_param_spec_boolean("async",
				"Buffer async wrt GstClock", "Sinks the buffer only if their GstClock time is not too late", DEFAULT_ASYNC,
				G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));


	/*adding private data*/
	g_type_class_add_private(klass, sizeof(GstWavpackSinkPrivate));
}

static void gst_wavpack_sink_init (GstWavpackSink * sink){
	GstPadTemplate *templ = gst_static_pad_template_get (&sink_template);
	gint i;
	GstWavpackSinkPrivate *priv;

	/*
	 * starting the private data with default values in case we dont need interaction with
	 * external elements in the pipe
	 * */
	priv = sink->private_info = G_TYPE_INSTANCE_GET_PRIVATE(sink, GST_TYPE_WAVPACK_SINK, GstWavpackSinkPrivate);
	priv->callbacks.eos = NULL;
	priv->callbacks.new_preroll = NULL;
	priv->callbacks.new_sample = NULL;
	priv->user_data = NULL;
	priv->bufferlist = NULL;
	g_mutex_init(&priv->data_access_locker);

	/* pad to the encoder with tag process */
	sink->ghostpad = gst_ghost_pad_new_no_target_from_template ("sink", templ);

	/* pad to the fakesink part for rejected segments */
	//sink->fakepad = gst_ghost_pad_new_no_target_from_template ("sink", templ);

	/* creating the real processing pad */
	gst_object_unref (templ);
	gst_element_add_pad (GST_ELEMENT_CAST(sink), sink->ghostpad);
	gst_pad_add_probe (sink->ghostpad, GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM, gst_wavpack_sink_ghost_event_probe, sink, NULL);
	gst_pad_add_probe (sink->ghostpad, GST_PAD_PROBE_TYPE_BUFFER, gst_wavpack_sink_ghost_buffer_probe, sink, NULL);

	/* creating the fakepad pad */
	//gst_element_add_pad (GST_ELEMENT_CAST(sink), sink->fakepad);
	sink->location = g_strdup (DEFAULT_LOCATION);
	sink->n_chunks = DEFAULT_CHUNKS;
	sink->sync = DEFAULT_SYNC;//synchronous buffer sinking
	sink->async = DEFAULT_ASYNC;//asynchronous buffer sinking
	sink->multifilesink_setup=FALSE;//to check if the element is set
	sink->generate_keyunit = FALSE;
	sink->mode = GST_WAVPACK_ENC_MODE_DEFAULT;
	/* haven't added a sink yet, make it is detected as a sink meanwhile */
	GST_OBJECT_FLAG_SET (sink, GST_ELEMENT_FLAG_SINK);

	gst_wavpack_sink_reset (sink);
}

static void gst_wavpack_sink_reset(GstWavpackSink * sink){
	sink->index = 0;
	sink->count = 0;
	sink->timeout_id = 0;
	sink->timestamp = 0;
	sink->stream_time = 0;
	sink->running_time = 0;
	sink->waiting_fku = FALSE;
	sink->generate_keyunit = FALSE;
	gst_event_replace (&sink->force_key_unit_event, NULL);
	gst_segment_init (&sink->segment, GST_FORMAT_UNDEFINED);
}

static gboolean gst_wavpack_sink_create_elements(GstWavpackSink * sink){
	GstPad *pad = NULL;
	GstCaps *caps = NULL;

	GST_DEBUG_OBJECT (sink, "Creating internal elements (audioconvert ! wavpackwithtag ! multifilesink) ");

	if (sink->elements_created)
	return TRUE;

	sink->audioconvertS16 = gst_element_factory_make ("audioconvert", NULL);
	sink->capsfilter = gst_element_factory_make ("capsfilter", NULL);
	sink->audioconvertS32 = gst_element_factory_make ("audioconvert", NULL);
	sink->wavpackwithtag = gst_element_factory_make ("wavpackwithtag", NULL);
	sink->multifilesink = gst_element_factory_make ("multifilesink", NULL);


	if (sink->wavpackwithtag==NULL){
		gst_element_post_message(GST_ELEMENT_CAST (sink), gst_missing_element_message_new (GST_ELEMENT_CAST (sink),
				  "WavpackWithTag"));
		GST_ELEMENT_ERROR(sink, CORE, MISSING_PLUGIN, (("Missing element '%s' - check your GStreamer installation."),
			  "WavpackWithTag"), (NULL));
		return FALSE;
	}

	if (sink->audioconvertS16==NULL){
		gst_element_post_message(GST_ELEMENT_CAST (sink), gst_missing_element_message_new (GST_ELEMENT_CAST (sink),
				  "audioconvertS16"));
		GST_ELEMENT_ERROR(sink, CORE, MISSING_PLUGIN, (("Missing element '%s' - check your GStreamer installation."),
			  "audioconvertS16"), (NULL));
		return FALSE;
	}

	if (sink->capsfilter==NULL){
		gst_element_post_message(GST_ELEMENT_CAST (sink), gst_missing_element_message_new (GST_ELEMENT_CAST (sink),
				  "capsfilter"));
		GST_ELEMENT_ERROR(sink, CORE, MISSING_PLUGIN, (("Missing element '%s' - check your GStreamer installation."),
			  "capsfilter"), (NULL));
		return FALSE;
	}

	if (sink->audioconvertS32==NULL){
		gst_element_post_message(GST_ELEMENT_CAST (sink), gst_missing_element_message_new (GST_ELEMENT_CAST (sink),
				  "audioconvertS32"));
		GST_ELEMENT_ERROR(sink, CORE, MISSING_PLUGIN, (("Missing element '%s' - check your GStreamer installation."),
			  "audioconvertS32"), (NULL));
		return FALSE;
	}

	if (sink->multifilesink==NULL){
		gst_element_post_message(GST_ELEMENT_CAST (sink), gst_missing_element_message_new (GST_ELEMENT_CAST (sink),
			  "multifilesink"));
		GST_ELEMENT_ERROR(sink, CORE, MISSING_PLUGIN, (("Missing element '%s' - check your GStreamer installation."),
			  "multifilesink"), (NULL));
		return FALSE;
	}

	/* Setting audioconvertS16 */
	//no need to do dowsampling from F32

	/* Setting the capsfilter for the connection between the audioconverts */
	caps = gst_caps_from_string ("audio/x-raw,format=(string)S16LE");
	g_object_set (sink->capsfilter, "caps", caps, NULL);
	gst_caps_unref(caps);

	/* Setting audioconvertS16 */
	//no need to do upsampling S16 to S32

	/* Setting wavpackwithtag */
	//this is dynamically controlled using the data coming from crying detector


	/* Add all the elements and terminating with null to avoid bugs */
	gst_bin_add_many(GST_BIN_CAST (sink), sink->audioconvertS16, sink->capsfilter, sink->audioconvertS32, sink->wavpackwithtag, sink->multifilesink, NULL);//
	gst_element_link_many(sink->audioconvertS16, sink->capsfilter, sink->audioconvertS32, sink->wavpackwithtag, sink->multifilesink, NULL);//



	/* The element to receive all the info is the audioconverter */
	pad = gst_element_get_static_pad (sink->audioconvertS16, "sink");
	gst_ghost_pad_set_target (GST_GHOST_PAD(sink->ghostpad), pad);
	gst_object_unref(pad);

	sink->elements_created = TRUE;
	return TRUE;
}



static void gst_wavpack_sink_handle_message (GstBin * bin, GstMessage * message){
	GstWavpackSink *sink = GST_WAVPACK_SINK_CAST (bin);

	switch (message->type) {
		case GST_MESSAGE_ELEMENT:
		{
			GFile *file;
			const char *filename, *title;
			GstClockTime running_time;
			gboolean discont = FALSE;
			GError *error = NULL;
			const GstStructure *structure;

			structure = gst_message_get_structure (message);
			if (!strcmp (gst_structure_get_name (structure), "GstMultiFileSink")){
				/* The key unit has been generated without any problem */
				filename = gst_structure_get_string(structure, "filename");
				gst_structure_get_clock_time(structure, "running-time", &running_time);

				sink->last_running_time = running_time;

				file = g_file_new_for_path(filename);
				title = "ciao";
				GST_INFO_OBJECT(sink, "COUNT %d", sink->index);

				/* multifilesink is starting a new file. It means that upstream sent a key
				* unit and we can schedule the next key unit now.
				*/
				sink->generate_keyunit = FALSE;
				sink->waiting_fku = FALSE;
				gst_segment_init(&sink->segment, GST_FORMAT_TIME);

				/*
				 * since we have a buffer that is already gone, we don't
				 * actually generate key unit, we just reset the flags and
				 * dispose the sinkpad which was not unref when the real
				 * key unit was generated
				 * */
				schedule_next_key_unit(sink);

				/* multifilesink is an internal implementation detail. If applications
				* need a notification, we should probably do our own message */
				GST_DEBUG_OBJECT(bin, "dropping message %" GST_PTR_FORMAT, message);
				gst_message_unref(message);
				g_object_unref(file);
				message = NULL;
			}else if(!strcmp (gst_structure_get_name(structure), "GstWavPackWithTag")){
				/* Message coming from the tagger element */
				gst_structure_get_boolean(structure, "generate_keyunit", &sink->generate_keyunit);

				/*
				 * signaling that we have processed the chunk successfully , by dropping
				 * it or sink it to a file
				 * */
				GstWavpackSinkPrivate *priv = sink->private_info;
				if(priv->callbacks.new_sample){
					/*we have a function set for callback*/
					GST_INFO_OBJECT(sink, "Updating the callback function");
					priv->callbacks.new_sample(sink, priv->user_data);
				}
			}

			break;
		}
		default:
			break;
	}

  if (message)
    GST_BIN_CLASS (parent_class)->handle_message (bin, message);
}

static GstStateChangeReturn gst_wavpack_sink_change_state(GstElement * element, GstStateChange trans){
	GstStateChangeReturn ret = GST_STATE_CHANGE_SUCCESS;
	GstWavpackSink *sink = GST_WAVPACK_SINK_CAST (element);
	gint fd;

	switch (trans) {
		case GST_STATE_CHANGE_NULL_TO_READY:
			/*
			 * test if we can create a file in the corresponding
			 * location, or just exiting from from here
			 * */
			fd = g_creat(sink->location, S_IWRITE);
			if(fd<0){
				GST_ERROR_OBJECT(sink, "We cannot create the file \"%s\", please"
						"make sure the path exists or you have the right permissions...!!", sink->location);
				return GST_STATE_CHANGE_FAILURE;
			}else{
				g_close(fd, NULL);
				g_remove(sink->location);
			}


			/*creating elements*/
			if (!gst_wavpack_sink_create_elements(sink)) {
				return GST_STATE_CHANGE_FAILURE;
			}
			break;
		case GST_STATE_CHANGE_PAUSED_TO_PLAYING:
			break;
		default:
			break;
	}

	ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, trans);

	switch (trans) {
		case GST_STATE_CHANGE_PLAYING_TO_PAUSED:
			break;
		case GST_STATE_CHANGE_PAUSED_TO_READY:
			gst_wavpack_sink_reset (sink);
			break;
		case GST_STATE_CHANGE_READY_TO_NULL:
			gst_wavpack_sink_reset (sink);
			break;
		default:
			break;
	}

	return ret;
}

static void gst_wavpack_sink_set_property(GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec){
	GstWavpackSink *sink = GST_WAVPACK_SINK_CAST (object);

	switch (prop_id) {
		case PROP_COMPRESSION_MODE:
			sink->mode = g_value_get_enum (value);
			break;
		case PROP_LOCATION:
			g_free (sink->location);
			sink->location = g_value_dup_string (value);
			if (sink->multifilesink)
				g_object_set(sink->multifilesink, "location", sink->location, NULL);
			break;
		case PROP_CHUNKS:
			sink->n_chunks = g_value_get_uint (value);
			break;
		case PROP_SYNC:
			sink->sync = g_value_get_boolean (value);
			break;
		case PROP_ASYNC:
			sink->async = g_value_get_boolean (value);
			break;
		default:
			G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
			break;
	}
}

static void gst_wavpack_sink_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec){
	GstWavpackSink *sink = GST_WAVPACK_SINK_CAST (object);

	switch (prop_id) {
		case PROP_COMPRESSION_MODE:
			g_value_set_enum(value, sink->mode);
			break;
		case PROP_LOCATION:
			g_value_set_string (value, sink->location);
			break;
		case PROP_CHUNKS:
			g_value_set_uint (value, sink->n_chunks);
			break;
		case PROP_SYNC:
			g_value_set_boolean (value, sink->sync);
			break;
		case PROP_ASYNC:
			g_value_set_boolean (value, sink->async);
			break;
		default:
			G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
			break;
	}
}

static GstPadProbeReturn gst_wavpack_sink_ghost_event_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer data){
	GstWavpackSink *sink = GST_WAVPACK_SINK_CAST (data);
	GstEvent *event = gst_pad_probe_info_get_event (info);

	switch (GST_EVENT_TYPE (event)) {
		case GST_EVENT_SEGMENT:
		{
			GST_INFO_OBJECT (sink, "New segment...");
			gst_event_copy_segment (event, &sink->segment);
			break;
		}
		case GST_EVENT_FLUSH_STOP:
			GST_INFO_OBJECT (sink, "Reseting segment...");
			gst_segment_init (&sink->segment, GST_FORMAT_UNDEFINED);
			break;
		case GST_EVENT_CUSTOM_DOWNSTREAM:
		{
			GstClockTime timestamp;
			GstClockTime running_time, stream_time;
			gboolean all_headers;
			guint count;

			GST_INFO_OBJECT (sink, "New GST_EVENT_CUSTOM_DOWNSTREAM to generate key-unit...");

			if (!gst_video_event_is_force_key_unit (event))
				break;

			gst_event_replace (&sink->force_key_unit_event, event);
			gst_video_event_parse_downstream_force_key_unit (event, &timestamp,
					&stream_time, &running_time, &all_headers, &count);
			GST_INFO_OBJECT (sink, "setting index %d", count);
			sink->index = count;

			break;
		}
		default:
			break;
	}

	return GST_PAD_PROBE_OK;
}

static gboolean schedule_next_key_unit (GstWavpackSink *sink){
	gboolean res = TRUE;
	GstClockTime running_time;
	GstPad *sinkpad = NULL;
	GstEvent *event = NULL;
	sinkpad = gst_element_get_static_pad (GST_ELEMENT (sink), "sink");

	if (!sink->generate_keyunit){
		/*
		 * if is false mean we dont need to generate new file
		 * this is very important since if we dont have this,
		 * we will make a mess if the message from multifilesink
		 * takes longer than expected to arrive, since we will keep
		 * generating key-units like crazy (garbage files)
		 * */
		goto out;
	}

	running_time = sink->last_running_time + DEFAULT_LENGTH*GST_SECOND;


	GST_INFO_OBJECT (sink, "sending downstream force-key-unit, file-index %d \n"
	  "time now %" GST_TIME_FORMAT "\ntarget time %" GST_TIME_FORMAT "\nstream %" GST_TIME_FORMAT,
	  sink->index + 1, GST_TIME_ARGS (sink->last_running_time),
	  GST_TIME_ARGS (running_time), GST_TIME_ARGS (sink->stream_time));

	/* estimating the values makes the segment smaller */
	event = gst_video_event_new_downstream_force_key_unit (sink->timestamp,sink->stream_time,running_time, TRUE, sink->index+1);

	//sending event
	if (!gst_video_event_is_force_key_unit (event))
		GST_ERROR_OBJECT (sink, "Key unit was not generated");

	if (!(res = gst_pad_send_event (sinkpad, event) )) {
		GST_ERROR_OBJECT (sink, "Failed to push upstream force key unit event");
	}

out:
	/*
	 * mark as waiting for a fku event if the app schedules them or if we just
	* successfully scheduled one
	*/
	sink->waiting_fku = res;
	gst_object_unref (sinkpad);
	return res;
}


static GstPadProbeReturn gst_wavpack_sink_ghost_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer data){
	GstWavpackSink *sink = GST_WAVPACK_SINK_CAST (data);
	GstBuffer *buffer = gst_pad_probe_info_get_buffer (info);
	GstClockTime timestamp;
	gint cry_label=-1;

	/*
	 * check the incoming result, we can avoid this part
	 * if we include the gstcrydetector in this bin
	 * however, we need to check the feasibility of doing so
	 * */
	const GstMetaInfo *infomet = gst_buffer_get_cry_meta(METAIMPL);
	GstCryMeta *met=NULL;
	GstMeta *temp;
	gpointer state = NULL;
	while ((temp = gst_buffer_iterate_meta (buffer, &state))) {
		if(infomet->api == temp->info->api){
			met = (GstCryMeta*)temp;
			break;
		}
	}


	/*
	 * Checking the result coming from the crying detector
	 * which was pushed to the ghostpad
	 * */
	if(!met){
		GST_DEBUG_OBJECT (sink, "NO METADATA FOUND: %d, %d", infomet->api, cry_label);
	}else{
		cry_label = met->label;
		GST_DEBUG_OBJECT (sink, "Label_FOUND: %d, %d, %d\n\n", met->label, infomet->api, cry_label);
	}


	/*
	 * Setting downstream element since these three
	 * parameters could be changed by the user
	 * or any other application
	 * */
	g_object_set(sink->wavpackwithtag, "chunks", sink->n_chunks, "cry-label", cry_label, "mode", sink->mode, NULL);

	if(!sink->multifilesink_setup){
		/* Setting multifilesink */
		//uses keyunit
		GST_INFO_OBJECT(sink, "Setting multi-file-sink element with-> { sync: %d, async: %d, location: %s, key-unit-type: %d, Post-message: %d }",
				sink->sync, sink->async, sink->location, 3, TRUE);
		g_object_set(sink->multifilesink, "sync", sink->sync, "async", sink->async, "location", sink->location,
				"next-file", 3, "post-messages", TRUE, NULL);
		sink->multifilesink_setup = TRUE;
	}


	/*
	 * making a copy to keep it in the ring
	 * only if we have a callback function
	 * */
	if(sink->private_info->callbacks.new_sample){
		/*we need to keep a ring buffer*/
		GstWavpackSinkPrivate *priv = sink->private_info;
		GstMapInfo map;
		gst_buffer_map (buffer, &map, GST_MAP_READ);
		gsize sample_count = gst_buffer_get_size (buffer);
		GstBuffer *temp_buf = gst_buffer_new_and_alloc (sample_count);
		gst_buffer_copy_into(temp_buf, buffer, GST_BUFFER_COPY_METADATA, 0, -1);
		gst_buffer_fill (temp_buf, 0, map.data, sample_count);
		gst_buffer_unmap (buffer, &map);

		/*locating the buffer in the right location*/
		if(priv->bufferlist == NULL){//
			/*empty list so we proceed to create a new list*/
			priv->bufferlist = gst_buffer_list_new ();
		}

		/*insert incoming buffer*/
		g_mutex_lock(&priv->data_access_locker);
			gst_buffer_list_add(priv->bufferlist, temp_buf);
		g_mutex_unlock(&priv->data_access_locker);
	}


	timestamp = GST_BUFFER_TIMESTAMP (buffer);
	if (!GST_CLOCK_TIME_IS_VALID (timestamp) || sink->waiting_fku){
	  if(sink->generate_keyunit)//force key-unit
		  goto generate_key;
	  return GST_PAD_PROBE_OK;
	}

generate_key:
	/*
	 * Great we got the message from the wavpackwithtag element,
	 * now we need to check how to generate the key unit correctly
	 * so the generated file are indexed correctly and the data makes
	 * sense
	 *  */
	sink->timestamp = timestamp-DEFAULT_LENGTH*GST_SECOND;
	sink->last_running_time = gst_segment_to_running_time (&sink->segment,
		  GST_FORMAT_TIME, timestamp);
	sink->stream_time = gst_segment_to_stream_time (&sink->segment,
		  GST_FORMAT_TIME, timestamp);
	schedule_next_key_unit (sink);
	return GST_PAD_PROBE_OK;
}


GstSample * gst_wavpack_sink_pull_sample(GstWavpackSink *wavpacksink){
	  GstSample *sample = NULL;
	  GstBuffer *buffer;
	  GstWavpackSinkPrivate *priv;

	  g_return_val_if_fail(GST_IS_WAVPACK_SINK(wavpacksink), NULL);

	  priv = wavpacksink->private_info;

	  g_mutex_lock(&priv->data_access_locker);
	  {
		  //get always first element
		  buffer = gst_buffer_list_get(priv->bufferlist, 0);

		  //getting the samples
		  sample = gst_sample_new(buffer, NULL, NULL, NULL);

		  //remove first element
		  gst_buffer_list_remove(priv->bufferlist, 0, 1);
	  }
	  g_mutex_unlock(&priv->data_access_locker);
	  return sample;
}




/*
 * sets the function callback to communicate sinking events in this bin
 * */
void gst_wavpack_sink_set_callbacks(GstWavpackSink * wavpacksink, GstWavpackSinkCallbacks *callbacks, gpointer user_data){
	GstWavpackSinkPrivate *priv;

	/*check if what we received is the right type*/
	g_return_if_fail(GST_IS_WAVPACK_SINK(wavpacksink));

	/*
	 * check if we do have a function to call to or we're gonna
	 * be screwed when calling this*/
	g_return_if_fail(callbacks != NULL);

	priv = wavpacksink->private_info;

	/*This has to be done with care*/
	GST_OBJECT_LOCK(wavpacksink);

		priv->callbacks = *callbacks;
		priv->user_data = user_data;

	GST_OBJECT_UNLOCK(wavpacksink);
}




static gboolean gst_wavpack_sink_plugin_init (GstPlugin * plugin){
  GST_DEBUG_CATEGORY_INIT (gst_wavpack_sink_debug, "wavpacksink", 0, "WavpackSink");
  return gst_element_register (plugin, "wavpacksink", GST_RANK_NONE, gst_wavpack_sink_get_type ());
}


#ifndef VERSION
#define VERSION "0.0.1"
#endif
#ifndef PACKAGE
#define PACKAGE "WAVPACKSINKER_API"
#endif
#ifndef PACKAGE_NAME
#define PACKAGE_NAME "WAVPACKSINKER_API"
#endif
#ifndef GST_PACKAGE_ORIGIN
#define GST_PACKAGE_ORIGIN "http://www.automodules.com/"
#endif


GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR, wavpacksink, "WavpackSink element to multifile",
	gst_wavpack_sink_plugin_init, VERSION, "LGPL", PACKAGE_NAME, GST_PACKAGE_ORIGIN)

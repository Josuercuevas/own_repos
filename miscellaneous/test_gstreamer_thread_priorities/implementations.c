/*
 * implementations.c
 *
 *  Created on: Sep 7, 2015
 *      Author: josue
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

#include <gst/gst.h>
#include <gst/gstcaps.h>
#include <gst/gstpad.h>
#include <gst/gsttask.h>
#include <gst/gsttaskpool.h>
#include <gst/gstutils.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include "defintions.h"


G_DEFINE_TYPE (TestRTPool, test_rt_pool, GST_TYPE_TASK_POOL);

/*
 * Own buffer pool created for handling multi threads and its corresponding priorities
 * */
static void default_prepare (GstTaskPool * pool, GError ** error);
static void default_cleanup (GstTaskPool * pool);
static gpointer default_push (GstTaskPool * pool, GstTaskPoolFunction func, gpointer data,
    GError ** error);
static void default_join(GstTaskPool * pool, gpointer id);
static void test_rt_pool_class_init (TestRTPoolClass * klass);
static void test_rt_pool_init (TestRTPool * pool);
static void test_rt_pool_finalize (GObject * object);
static GstTaskPool *test_rt_pool_new (void);
static void test_rt_pool_finalize (GObject * object);
static void pad_added_handler(GstElement *src, GstPad *new_pad, CustomData *data);





static void on_stream_status(GstBus *bus, GstMessage *message, gpointer user_data){
	GstStreamStatusType type;
	GstElement *owner;
	const GValue *val;
	GstTask *task = NULL;

	gst_message_parse_stream_status(message, &type, &owner);

	val = gst_message_get_stream_status_object(message);

	/* see if we know how to deal with this object */
	if (G_VALUE_TYPE(val) == GST_TYPE_TASK) {
		task = g_value_get_object(val);
	}

	switch(type){
	case GST_STREAM_STATUS_TYPE_CREATE:
		if(task){
			/*********************** PUT MEANINFUL NAME HERE  WITH A CALL BACK FUNCTION ***********************/
			const gchar *name = gst_object_get_name(GST_OBJECT_CAST(owner));
			gst_object_set_name(GST_OBJECT_CAST(task), name);
			g_message("The name of the thread in this task is: %s", name);
			GstTaskPool *pool;
			pool = test_rt_pool_new();
			gst_task_set_pool(task, pool);
		}
		break;
	default:
		break;
	}
}

static void on_error(GstBus *bus, GstMessage *message, gpointer user_data){
	GMainLoop *loop = (GMainLoop*)user_data;
	GError *errorType;
	gchar *name;
	gst_message_parse_error(message, &errorType, &name);
	g_message("received ERROR: %s", name);
	gst_object_unref(GST_OBJECT_CAST(errorType));
	g_free(name);
	g_main_loop_quit(loop);
}

static void on_eos(GstBus *bus, GstMessage *message, gpointer user_data){
	GMainLoop *loop = (GMainLoop*)user_data;
	g_message("End of stream...");
	g_main_loop_quit (loop);
}



int main (int argc, char *argv[]){
	GMainLoop *loop;
	CustomData data;
	GstBus *bus;
	GstStateChangeReturn ret;

	gst_init (&argc, &argv);

	/* create a new bin to hold the elements */
	data.pipeline = gst_pipeline_new("pipeline");
	g_assert (data.pipeline);

	/* create a source */
	data.filesrc = gst_element_factory_make("filesrc", "filesrc");
	g_assert(data.filesrc);
	g_object_set(data.filesrc, "location", "PATH_TO_AN_MP4_FILE");//

	/*demuxer*/
	data.qtdemux = gst_element_factory_make("qtdemux", "qtdemux");
	g_assert(data.qtdemux);
	g_object_set(data.qtdemux, "name", "demuxer");

	/*queues*/
	data.videoqueue = gst_element_factory_make("queue", "videoqueue");
	g_assert(data.videoqueue);
	data.audioqueue = gst_element_factory_make("queue", "audioqueue");
	g_assert(data.audioqueue);

	/*video handling*/
	data.video_parser = gst_element_factory_make("h264parse", "h264parse");
	g_assert(data.video_parser);
	data.video_decoder = gst_element_factory_make("avdec_h264", "avdec_h264");
	g_assert(data.video_decoder);

	/*audio handling*/
	data.audio_parser = gst_element_factory_make("aacparse", "aacparse");
	g_assert(data.audio_parser);
	data.audio_decoder = gst_element_factory_make("avdec_aac", "avdec_aac");
	g_assert(data.audio_decoder);

	/*converters*/
	data.video_convert = gst_element_factory_make("videoconvert", "videoconvert");
	g_assert(data.video_convert);
	data.audio_convert = gst_element_factory_make("audioconvert", "audioconvert");
	g_assert(data.audio_convert);

	/* sinkers */
	data.video_sink = gst_element_factory_make("ximagesink", "ximagesink");
	g_assert(data.video_sink);
	data.audio_sink = gst_element_factory_make("alsasink", "alsasink");
	g_assert(data.audio_sink);


	/* add objects to the main pipeline */
	gst_bin_add_many(GST_BIN(data.pipeline), data.filesrc, data.qtdemux, data.videoqueue, data.video_parser, data.video_decoder, data.video_convert,
			data.video_sink, data.audioqueue, data.audio_parser, data.audio_decoder, data.audio_convert, data.audio_sink, NULL);

	/* link the elements */
	gst_element_link(data.filesrc, data.qtdemux);
	gst_element_link_many(data.videoqueue, data.video_parser, data.video_decoder, data.video_convert,
			data.video_sink, NULL);
	gst_element_link_many(data.audioqueue, data.audio_parser, data.audio_decoder, data.audio_convert, data.audio_sink, NULL);


	g_signal_connect (data.qtdemux, "pad-added", (GCallback)pad_added_handler, &data);


	loop = g_main_loop_new(NULL, FALSE);

	/* get the bus, we need to install a sync handler */
	bus = gst_pipeline_get_bus(GST_PIPELINE (data.pipeline));
	gst_bus_enable_sync_message_emission(bus);
	gst_bus_add_signal_watch(bus);

	g_signal_connect(bus, "sync-message::stream-status", (GCallback)on_stream_status, NULL);
	g_signal_connect (bus, "message::error", (GCallback)on_error, loop);
	g_signal_connect (bus, "message::eos", (GCallback)on_eos, loop);

	/* start playing */
	ret = gst_element_set_state(data.pipeline, GST_STATE_PLAYING);
	if(ret == GST_STATE_CHANGE_FAILURE || ret == GST_STATE_CHANGE_NO_PREROLL){
		g_message("failed to change state, %i", ret);
		return -1;
	}

	/* Run event loop listening for bus messages until EOS or ERROR */
	g_main_loop_run(loop);

	/* stop the bin */
	gst_element_set_state(data.pipeline, GST_STATE_NULL);
	gst_object_unref(bus);
	g_main_loop_unref(loop);




	return 0;
}





























/**************************************** GST TASK POOL CONTROLLED BY US *******************************/
static void default_prepare(GstTaskPool * pool, GError ** error){
	/*
	 * we don't do anything here. We could construct a pool of threads here that
	 * we could reuse later but we don't
	 * */
	g_message ("prepare Realtime pool %p", pool);
}

static void default_cleanup(GstTaskPool * pool){
	g_message("cleanup Realtime pool %p", pool);
}

static gpointer default_push(GstTaskPool * pool, GstTaskPoolFunction func, gpointer data,
    GError ** error){
	TestRTId *tid;
	gint res;
	pthread_attr_t attr;
	struct sched_param param;

	g_message("The name of the element calling this Task is: %s", GST_OBJECT_NAME(data));

	g_message("pushing Realtime pool %p, %p", pool, func);
	tid = g_slice_new0(TestRTId);

	g_message("set policy");
	pthread_attr_init(&attr);
	if ((res = pthread_attr_setschedpolicy(&attr, SCHED_RR)) != 0)//SCHED_RR
		g_warning("setschedpolicy: failure: %p", g_strerror(res));

	g_message("set priority");
	if(g_strcmp0(GST_OBJECT_NAME(data), "videoqueue")==0 || g_strcmp0(GST_OBJECT_NAME(data), "audioqueue")==0){
		/*setting lower priority*/
		param.sched_priority = 1;
	}else{
		param.sched_priority = 99;
	}

	if ((res = pthread_attr_setschedparam(&attr, &param)) != 0)
		g_warning("setschedparam: failure: %p", g_strerror(res));

	g_message ("set inherit");
	if ((res = pthread_attr_setinheritsched (&attr, PTHREAD_EXPLICIT_SCHED)) != 0)
		g_warning ("setinheritsched: failure: %p", g_strerror(res));

	g_message("create thread");
	res = pthread_create(&tid->thread, &attr, (void *(*)(void *)) func, data);
	pthread_setname_np(tid->thread, GST_OBJECT_NAME(data));
	if (res != 0){
		g_set_error(error, G_THREAD_ERROR, G_THREAD_ERROR_AGAIN,
			"Error creating thread: %s", g_strerror(res));
		g_slice_free(TestRTId, tid);
		tid = NULL;
	}

	return tid;
}

static void default_join(GstTaskPool * pool, gpointer id){
	TestRTId *tid = (TestRTId *) id;
	g_message ("joining Realtime pool %p", pool);
	pthread_join (tid->thread, NULL);
	g_slice_free (TestRTId, tid);
}

static void test_rt_pool_class_init(TestRTPoolClass * klass){
	GObjectClass *gobject_class;
	GstTaskPoolClass *gsttaskpool_class;
	gobject_class = (GObjectClass *) klass;
	gsttaskpool_class = (GstTaskPoolClass *) klass;
	gobject_class->finalize = GST_DEBUG_FUNCPTR(test_rt_pool_finalize);

	gsttaskpool_class->prepare = default_prepare;
	gsttaskpool_class->cleanup = default_cleanup;
	gsttaskpool_class->push = default_push;
	gsttaskpool_class->join = default_join;
}

static void test_rt_pool_init(TestRTPool * pool){
	/*nothing here*/
}

static void test_rt_pool_finalize(GObject * object){
	G_OBJECT_CLASS (test_rt_pool_parent_class)->finalize (object);
}

static GstTaskPool *test_rt_pool_new(void){
	GstTaskPool *pool;
	pool = g_object_new(TEST_TYPE_RT_POOL, NULL);
	return pool;
}







/* This function will be called by the pad-added signal */
static void pad_added_handler(GstElement *src, GstPad *new_pad, CustomData *data){
	g_message("Inside the pad_added_handler method from: %s", GST_OBJECT_NAME(src));
	GstPad *audiosinkpad, *videosinkpad;
	GstElement *video_queue = (GstElement*)data->videoqueue;
	GstElement *audio_queue = (GstElement*)data->audioqueue;

	/* We can now link this pad with the h264dec sink pad */
	g_print("Dynamic pad created, linking demuxer/decoder\n");

	videosinkpad = gst_element_get_static_pad(video_queue, "sink");
	audiosinkpad = gst_element_get_static_pad(audio_queue, "sink");

	gst_pad_link(new_pad, videosinkpad);
	gst_pad_link(new_pad, audiosinkpad);

	gst_object_unref(videosinkpad);
	gst_object_unref(audiosinkpad);
}

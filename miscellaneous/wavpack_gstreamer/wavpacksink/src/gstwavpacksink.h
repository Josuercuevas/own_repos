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
#ifndef _GST_WAVPACK_SINK_H_
#define _GST_WAVPACK_SINK_H_

#include <gst/gst.h>
#include "metainfo.h"

G_BEGIN_DECLS

#define GST_TYPE_WAVPACK_SINK   (gst_wavpack_sink_get_type())
#define GST_WAVPACK_SINK(obj)   (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_WAVPACK_SINK,GstWavpackSink))
#define GST_WAVPACK_SINK_CAST(obj)   ((GstWavpackSink *) obj)
#define GST_WAVPACK_SINK_CLASS(klass)   (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_WAVPACK_SINK,GstWavpackSinkClass))
#define GST_IS_WAVPACK_SINK(obj)   (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_WAVPACK_SINK))
#define GST_IS_WAVPACK_SINK_CLASS(obj)   (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_WAVPACK_SINK))

typedef struct _GstWavpackSink GstWavpackSink;
typedef struct _GstWavpackSinkClass GstWavpackSinkClass;
typedef struct _GstWavpackSinkPrivate GstWavpackSinkPrivate;

struct _GstWavpackSink
{
  GstBin bin;

  GstPad *ghostpad, *fakepad;
  GstElement *multifilesink, *audioconvertS16, *audioconvertS32;
  GstElement *wavpackwithtag, *capsfilter;
  gboolean elements_created;
  GstEvent *force_key_unit_event;

  gchar *location;
  guint n_chunks, mode;
  gboolean generate_keyunit, sync, async;
  guint index;
  gint count;
  guint timeout_id;
  GstSegment segment;
  gboolean waiting_fku;
  GstClockTime timestamp;
  GstClockTime stream_time;
  GstClockTime running_time;
  GstClockTime last_running_time;

  gboolean multifilesink_setup;


  /*
   * < Private data that cannot be accessed by external elements >
   * */
  GstWavpackSinkPrivate *private_info;
};


typedef struct {
  void          (*eos)              (GstWavpackSink *wavpacksink, gpointer user_data);//not in use
  GstFlowReturn (*new_preroll)      (GstWavpackSink *wavpacksink, gpointer user_data);//not in use
  GstFlowReturn (*new_sample)       (GstWavpackSink *wavpacksink, gpointer user_data);//signal that new data has came
} GstWavpackSinkCallbacks;


struct _GstWavpackSinkClass
{
  GstBinClass bin_class;
};

GType gst_wavpack_sink_get_type (void);

/*
 * Signals the external elements that data has been processed by the WavpackSink, therefore
 * we are good to go and continue processing data, no data is to be created otherwise
 * we need to find a way to free it ... WHERE?
 * */
void gst_wavpack_sink_set_callbacks(GstWavpackSink * wavpacksink, GstWavpackSinkCallbacks *callbacks,
		gpointer user_data);

/*
 * To pull samples in external implementations
 * */
GstSample * gst_wavpack_sink_pull_sample(GstWavpackSink *wavpacksink);

G_END_DECLS

#endif

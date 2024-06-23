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
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA 02110-1301, USA.
 */

#ifndef _GST_COLORRETINEX_H_
#define _GST_COLORRETINEX_H_

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/video/gstvideofilter.h>
#include "includes/RetinexLib.h"
#include "includes/norm.h"

G_BEGIN_DECLS

#define GST_TYPE_COLORRETINEX   (gst_colorretinex_get_type())
#define GST_COLORRETINEX(obj)   (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_COLORRETINEX,GstColorRetinex))
#define GST_COLORRETINEX_CLASS(klass)   (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_COLORRETINEX,GstColorRetinexClass))
#define GST_IS_COLORRETINEX(obj)   (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_COLORRETINEX))
#define GST_IS_COLORRETINEX_CLASS(obj)   (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_COLORRETINEX))

typedef struct _GstColorRetinex GstColorRetinex;
typedef struct _GstColorRetinexClass GstColorRetinexClass;

struct _GstColorRetinex
{
  GstVideoFilter base_colorretinex;
  guint RetinexThreshold;
  gfloat *indata, *outdata;
};

struct _GstColorRetinexClass
{
  GstVideoFilterClass base_colorretinex_class;
};

guint8 min_val(guint8 a, guint8 b);
guint8 max_val(gfloat a, gfloat b);

GType gst_colorretinex_get_type (void);

G_END_DECLS

#endif

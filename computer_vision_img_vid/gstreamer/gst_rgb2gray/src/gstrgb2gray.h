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

#ifndef _GST_RGB2GRAY_H_
#define _GST_RGB2GRAY_H_

#include <gst/video/video.h>
#include <gst/video/gstvideofilter.h>

G_BEGIN_DECLS

#define GST_TYPE_RGB2GRAY   (gst_rgb2gray_get_type())
#define GST_RGB2GRAY(obj)   (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_RGB2GRAY,GstRgb2gray))
#define GST_RGB2GRAY_CLASS(klass)   (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_RGB2GRAY,GstRgb2grayClass))
#define GST_IS_RGB2GRAY(obj)   (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_RGB2GRAY))
#define GST_IS_RGB2GRAY_CLASS(obj)   (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_RGB2GRAY))

typedef struct _GstRgb2gray GstRgb2gray;
typedef struct _GstRgb2grayClass GstRgb2grayClass;

struct _GstRgb2gray
{
  GstVideoFilter base_rgb2gray;

};

struct _GstRgb2grayClass
{
  GstVideoFilterClass base_rgb2gray_class;
};

GType gst_rgb2gray_get_type (void);

G_END_DECLS

#endif

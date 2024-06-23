/* eglvivsink GStreamer 1.0 plugin definition
 * Copyright (C) 2013  Carlos Rafael Giani
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
 * License along with this library; if not, write to the Free
 * Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */


#include "config.h"
#include <gst/gst.h>
#include "gst-BlobDetector/src/gstblobdetector.h"
#include "gst-Normalizer/src/gstmomentnormalization.h"
#include "gst-Retinex/src/gstcolorretinex.h"
#include "gst-SkinDetector/src/gstskindetector.h"
#include "gst-DistanceTransform/src/gstdistancetransform.h"

static gboolean plugin_init(GstPlugin *plugin)
{
	gboolean ret = TRUE;
	ret = gst_element_register (plugin, "blobdetector", GST_RANK_NONE, GST_TYPE_BLOBDETECTOR);
	ret = gst_element_register(plugin, "momentnormalization", GST_RANK_NONE, GST_TYPE_MOMENTNORMALIZATION);
	ret = gst_element_register (plugin, "colorretinex", GST_RANK_NONE, GST_TYPE_COLORRETINEX);
	ret = gst_element_register (plugin, "skindetector", GST_RANK_NONE, GST_TYPE_SKINDETECTOR);
	ret = gst_element_register (plugin, "distancetransform", GST_RANK_NONE, GST_TYPE_DISTANCETRANSFORM);
	return ret;
}

#ifndef PACKAGE_NAME
#define PACKAGE_NAME "Image_proc_API"
#endif
#ifndef GST_PACKAGE_ORIGIN
#define GST_PACKAGE_ORIGIN "none"
#endif

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR,
	img_procdetector, "img_proc detector", plugin_init,
	VERSION, "LGPL", PACKAGE_NAME, GST_PACKAGE_ORIGIN)

/*
 * Metainfo.h
 *
 *  Created on: Jan 9, 2015
 *      Author: josue
 *
 *		General structures needed for each element in the
 * 		open-close eyes API, as well as for handling and declaring
 * 		metadata information to be transferred between elements when
 * 		final pipeline is constructed:
 * 		videosrc ! decode_frame(if needed) ! retinex ! skin_detector ! blob_detector ! normalization ! sink (if needed)
 */

#ifndef METAINFO_H_
#define METAINFO_H_

#if __cplusplus
extern "C"{
#endif

#include <gst/gst.h>
#include <gst/video/video.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>


//basic structures needded in all the elements
typedef struct _GstMetaBLOB GstMetaBLOB;

#define MAXBLOBS (20)
#define MAXLABEL (100000)
#define METAIMPL ("BLOBmeta")
#define METAAPI ("BLOBmetaAPI")

/*
 * Error codes in case there is anyone present in the
 * elements created for the API
 * */
enum retvals{
	//skin detector part
	skin_success=0,
	skin_wrong_img_type,

	//blob extraction part
	blob_success,
	blob_nobinary,
	blob_mallocerror,

	//peilin normalization part
	normalization_success,
	normalization_nobinary,
	normalization_mallocerror,

	//unknown errors
	normalization_unknown,
	blob_unknown,
	skin_unknown
};


struct _GstMetaBLOB{
	GstMeta Meta;//to contain all the info from the metadata registration

	/*
	 * additional info for the meta-registration
	 * we will support up to a maximum of "MAXBLOBS" BBs
	 * */
	guint n_blobs;//number of BBs found
	gint x[MAXBLOBS], y[MAXBLOBS];//starting location of the BB
	gint width[MAXBLOBS], height[MAXBLOBS];//width and height of the BB
};

GType BLOB_meta_api_get_type (void);
const GstMetaInfo *BLOB_meta_get_info(void);
#define BLOB_META_API_TYPE (BLOB_meta_api_get_type())
#define BLOB_META_INFO (BLOB_meta_get_info())
#define gst_buffer_get_blob_meta(b) ((GstMetaInfo*)gst_meta_get_info(b))

/* Functions needed for buffer metadata handling */
static gboolean GstMetaBLOB_InitFun(GstMeta *meta, gpointer params, GstBuffer *buffer);
static gboolean GstMetaBLOB_TransFun(GstBuffer * transbuf, GstMeta * meta,
    GstBuffer * buffer, GQuark type, gpointer data);

#ifdef __cplusplus
}
#endif

#endif /* METAINFO_H_ */

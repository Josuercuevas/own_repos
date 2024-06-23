/*
 * Metainfo.h
 *
 *  Created on: Jan 9, 2015
 *      Author: josue
 *
 *		
 */

#ifndef METAINFO_H_
#define METAINFO_H_

#if __cplusplus
extern "C"{
#endif

#include <gst/gst.h>
#include <gst/audio/audio.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>

const GstMetaInfo *Cry_meta_get_info (void);
GType Cry_meta_api_get_type(void);


//basic structures needded in all the elements
typedef struct _GstCryMeta GstCryMeta;



#define METAIMPL ("CryMeta")
#define METAAPI ("CryMetaAPI")
#define CRY_META_API_TYPE (Cry_meta_api_get_type())
#define CRY_META_INFO (Cry_meta_get_info())
#define gst_buffer_get_cry_meta(b) ((GstMetaInfo*)gst_meta_get_info(b))


struct _GstCryMeta{
	GstMeta meta;
	gint label;
};

#ifdef __cplusplus
}
#endif

#endif /* METAINFO_H_ */

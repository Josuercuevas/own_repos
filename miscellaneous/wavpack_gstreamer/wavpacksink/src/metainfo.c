/*
 * Metainfo.c
 *
 *  Created on: Jan 9, 2015
 *      Author: josue
 *
 *		
 */

#include "metainfo.h"

/* Functions needed for buffer metadata handling */
static gboolean GstCryMeta_InitFun(GstMeta *meta, gpointer params, GstBuffer *buffer){
	GstCryMeta *crymeta = (GstCryMeta*)meta;
	crymeta->label=0;
	return TRUE;
}

static gboolean GstCryMeta_TransFun(GstBuffer * transbuf, GstMeta * meta,
    GstBuffer * buffer, GQuark type, gpointer data)
{
	GstCryMeta *crymeta = (GstCryMeta*)meta;

	/* we always copy no matter what transform */
	g_return_val_if_fail (GST_IS_BUFFER (transbuf), NULL);

	//add the meta to transformed buffer
	crymeta = (GstCryMeta*)gst_buffer_add_meta(transbuf,CRY_META_INFO, NULL);
	crymeta->label=0;
	return TRUE;
}


/* our metadata */
const GstMetaInfo *Cry_meta_get_info (void){
  static const GstMetaInfo *meta_info = NULL;

  if (g_once_init_enter (&meta_info)) {
    const GstMetaInfo *mi = gst_meta_register(CRY_META_API_TYPE, METAIMPL, sizeof (GstCryMeta),
    		(GstMetaInitFunction)GstCryMeta_InitFun,
        (GstMetaFreeFunction)NULL, //we dont have to free anything inside the meta-data
        (GstMetaTransformFunction)GstCryMeta_TransFun);
    g_once_init_leave (&meta_info, mi);
  }
  return meta_info;
}


/* API registration of our metadata */
GType Cry_meta_api_get_type(void)
{
	static volatile GType type;
	static const gchar *tags[] = {GST_META_TAG_AUDIO_STR, GST_META_TAG_MEMORY_STR, NULL };

	if (g_once_init_enter (&type)) {
		GType _type = gst_meta_api_type_register (METAAPI, tags);
		g_once_init_leave (&type, _type);
	}
	return type;
}

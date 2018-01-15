/*
 * Definitions of the meta-info file function
 * for later processing in the normalization element
 */

#include "metainfo.h"

static gboolean GstMetaBLOB_InitFun(GstMeta *meta, gpointer params, GstBuffer *buffer){
	GstMetaBLOB *BLOBMETA = (GstMetaBLOB*)meta;
	memset(BLOBMETA->x, 0, sizeof(gint)*MAXBLOBS);
	memset(BLOBMETA->y, 0, sizeof(gint)*MAXBLOBS);
	memset(BLOBMETA->width, 0, sizeof(gint)*MAXBLOBS);
	memset(BLOBMETA->height, 0, sizeof(gint)*MAXBLOBS);
	return TRUE;
}

static gboolean GstMetaBLOB_TransFun(GstBuffer * transbuf, GstMeta * meta,
    GstBuffer * buffer, GQuark type, gpointer data)
{
	GstMetaBLOB *BLOBMETA = (GstMetaBLOB*) meta;

	/* we always copy no matter what transform */
	g_return_val_if_fail (GST_IS_BUFFER (transbuf), NULL);

	//add the meta to transformed buffer
	meta = (GstMetaBLOB*)gst_buffer_add_meta(transbuf,BLOB_META_INFO, NULL);
	memset(BLOBMETA->x, 0, sizeof(gint)*MAXBLOBS);
	memset(BLOBMETA->y, 0, sizeof(gint)*MAXBLOBS);
	memset(BLOBMETA->width, 0, sizeof(gint)*MAXBLOBS);
	memset(BLOBMETA->height, 0, sizeof(gint)*MAXBLOBS);
	return TRUE;
}


/* our metadata */
const GstMetaInfo *BLOB_meta_get_info (void){
  static const GstMetaInfo *meta_info = NULL;

  if (g_once_init_enter (&meta_info)) {
    const GstMetaInfo *mi = gst_meta_register(BLOB_META_API_TYPE, METAIMPL, sizeof (GstMetaBLOB),
    		(GstMetaInitFunction)GstMetaBLOB_InitFun,
        (GstMetaFreeFunction)NULL, //we dont have to free anything inside the meta-data
        (GstMetaTransformFunction)GstMetaBLOB_TransFun);
    g_once_init_leave (&meta_info, mi);
  }
  return meta_info;
}



/* API registration of our metadata */
GType BLOB_meta_api_get_type(void)
{
	static volatile GType type;
	static const gchar *tags[] = {GST_META_TAG_VIDEO_STR, GST_META_TAG_MEMORY_STR, NULL };

	if (g_once_init_enter (&type)) {
		GType _type = gst_meta_api_type_register (METAAPI, tags);
		g_once_init_leave (&type, _type);
	}
	return type;
}

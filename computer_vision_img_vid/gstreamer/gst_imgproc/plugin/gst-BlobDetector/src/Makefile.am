plugin_LTLIBRARIES = libgstblobdetector.la

# sources used to compile this plug-in
libgstblobdetector_la_SOURCES = \
	gstblobdetector.c \
	gstblobdetector.h \
	../../common/metainfo.h \
	../../common/metainfo.c \
	BlobDetection.c \
	Extractor_core.c \
	includes/BlobDetection.h \
	includes/Extractor.h

# compiler and linker flags used to compile this plugin, set in configure.ac
libgstblobdetector_la_CFLAGS = \
	$(GST_PLUGINS_BASE_CFLAGS) \
	$(GST_BASE_CFLAGS) \
	$(GST_CFLAGS)

#including needed libraries to run the plugin successfully
libgstblobdetector_la_LIBADD = \
	$(GST_BASE_LIBS) \
	$(GST_PLUGINS_BASE_LIBS) -lgstvideo-1.0 \
	$(GST_LIBS)

#LD flags in case we need any
libgstblobdetector_la_LDFLAGS = $(GST_PLUGIN_LDFLAGS)


libgstblobdetector_la_LIBTOOLFLAGS = --tag=disable-static

# headers we need but don't want installed
noinst_HEADERS = \
	gstblobdetector.h \
	../../common/metainfo.h \
	includes/BlobDetection.h \
	includes/Extractor.h

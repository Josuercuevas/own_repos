plugin_LTLIBRARIES = libgstmomentnormalization.la

# sources used to compile this plug-in
libgstmomentnormalization_la_SOURCES = \
	gstmomentnormalization.c \
	gstmomentnormalization.h \
	../../common/metainfo.h \
	../../common/metainfo.c \
	MomentNormalizer.c \
	includes/MomentNormalizer.h \
	includes/Normalizer.h
	

# compiler and linker flags used to compile this plugin, set in configure.ac
libgstmomentnormalization_la_CFLAGS = \
	$(GST_PLUGINS_BASE_CFLAGS) \
	$(GST_BASE_CFLAGS) \
	$(GST_CFLAGS)

#including needed libraries to run the plugin successfully
libgstmomentnormalization_la_LIBADD = \
	$(GST_BASE_LIBS) \
	$(GST_PLUGINS_BASE_LIBS) -lgstvideo-1.0 \
	$(GST_LIBS)

#LD flags in case we need any
libgstmomentnormalization_la_LDFLAGS = $(GST_PLUGIN_LDFLAGS)


libgstmomentnormalization_la_LIBTOOLFLAGS = --tag=disable-static

# headers we need but don't want installed
noinst_HEADERS = \
	gstmomentnormalization.h \
	../../common/metainfo.h \
	includes/MomentNormalizer.h \
	includes/Normalizer.h

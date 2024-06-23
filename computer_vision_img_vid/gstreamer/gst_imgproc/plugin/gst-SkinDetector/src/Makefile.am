plugin_LTLIBRARIES = libgstskindetector.la

# sources used to compile this plug-in
libgstskindetector_la_SOURCES = \
	gstskindetector.c \
	gstskindetector.h \
	../../common/metainfo.h \

# compiler and linker flags used to compile this plugin, set in configure.ac
libgstskindetector_la_CFLAGS = \
	$(GST_PLUGINS_BASE_CFLAGS) \
	$(GST_BASE_CFLAGS) \
	$(GST_CFLAGS)

#including needed libraries to run the plugin successfully
libgstskindetector_la_LIBADD = \
	$(GST_BASE_LIBS) \
	$(GST_PLUGINS_BASE_LIBS) -lgstvideo-1.0 \
	$(GST_LIBS)

#LD flags in case we need any
libgstskindetector_la_LDFLAGS = $(GST_PLUGIN_LDFLAGS)


libgstskindetector_la_LIBTOOLFLAGS = --tag=disable-static

# headers we need but don't want installed
noinst_HEADERS = \
	gstskindetector.h \
	../../common/metainfo.h \







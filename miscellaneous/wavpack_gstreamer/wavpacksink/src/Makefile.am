plugin_LTLIBRARIES = libgstwavpacksink.la

# sources used to compile this plug-in
libgstwavpacksink_la_SOURCES = \
	gstwavpacksink.h \
	gstwavpacksink.c \
	metainfo.c \
	metainfo.h 

# compiler and linker flags used to compile this plugin, set in configure.ac
libgstwavpacksink_la_CFLAGS = \
	$(GST_PLUGINS_BASE_CFLAGS) \
	$(GST_BASE_CFLAGS) \
	$(GST_CFLAGS) \
	$(GIO_CFLAGS)

#including needed libraries to run the plugin successfully
libgstwavpacksink_la_LIBADD = \
	$(GST_BASE_LIBS) \
	$(GST_PLUGINS_BASE_LIBS) \
	$(GST_LIBS) \
	$(GIO_LIBS) \
	-lgstaudio-1.0 \
	-lgstvideo-1.0 \
	-lgstpbutils-1.0 \
	-lgio-2.0 \
	-lrt

#LD flags in case we need any
libgstwavpacksink_la_LDFLAGS = $(GST_PLUGIN_LDFLAGS)


libgstwavpacksink_la_LIBTOOLFLAGS = --tag=disable-static

# headers we need but don't want installed
noinst_HEADERS = \
	gstwavpacksink.h \
	metainfo.h 

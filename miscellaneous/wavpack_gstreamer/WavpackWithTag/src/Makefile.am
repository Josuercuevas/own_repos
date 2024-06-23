plugin_LTLIBRARIES = libgstwavpackwithtag.la

# sources used to compile this plug-in
libgstwavpackwithtag_la_SOURCES = \
	gstwavpackwithtag.h \
	gstwavpackwithtag.c \
	gstwavpackcommon.h \
	gstwavpackcommon.c 

# compiler and linker flags used to compile this plugin, set in configure.ac
libgstwavpackwithtag_la_CFLAGS = \
	$(GST_PLUGINS_BASE_CFLAGS) \
	$(GST_BASE_CFLAGS) \
	$(GST_CFLAGS)

#including needed libraries to run the plugin successfully
libgstwavpackwithtag_la_LIBADD = \
	$(GST_BASE_LIBS) \
	$(GST_PLUGINS_BASE_LIBS) \
	$(GST_LIBS) \
	-lgstaudio-1.0 \
	-lwavpack \
	-lrt

#LD flags in case we need any
libgstwavpackwithtag_la_LDFLAGS = $(GST_PLUGIN_LDFLAGS)


libgstwavpackwithtag_la_LIBTOOLFLAGS = --tag=disable-static

# headers we need but don't want installed
noinst_HEADERS = \
	gstwavpackwithtag.h \
	gstwavpackcommon.h

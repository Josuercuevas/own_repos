plugin_LTLIBRARIES = libgstrgbmapping.la

# sources used to compile this plug-in
libgstrgbmapping_la_SOURCES = \
	gstrgbmapping.c \
	gstrgbmapping.h

# compiler and linker flags used to compile this plugin, set in configure.ac
libgstrgbmapping_la_CFLAGS = \
	$(GST_PLUGINS_BASE_CFLAGS) \
	$(GST_BASE_CFLAGS) \
	$(GST_CFLAGS)

#including needed libraries to run the plugin successfully
libgstrgbmapping_la_LIBADD = \
	$(GST_BASE_LIBS) \
	$(GST_PLUGINS_BASE_LIBS) -lgstvideo-1.0 \
	$(GST_LIBS)

#LD flags in case we need any
libgstrgbmapping_la_LDFLAGS = $(GST_PLUGIN_LDFLAGS)


libgstrgbmapping_la_LIBTOOLFLAGS = --tag=disable-static

# headers we need but don't want installed
noinst_HEADERS = \
	gstrgbmapping.h






plugin_LTLIBRARIES = libgstdistancetransform.la

# sources used to compile this plug-in
libgstdistancetransform_la_SOURCES = \
	gstdistancetransform.c \
	gstdistancetransform.h

# compiler and linker flags used to compile this plugin, set in configure.ac
libgstdistancetransform_la_CFLAGS = \
	$(GST_PLUGINS_BASE_CFLAGS) \
	$(GST_BASE_CFLAGS) \
	$(GST_CFLAGS)

#including needed libraries to run the plugin successfully
libgstdistancetransform_la_LIBADD = \
	$(GST_BASE_LIBS) \
	$(GST_PLUGINS_BASE_LIBS) -lgstvideo-1.0 \
	$(GST_LIBS)

#LD flags in case we need any
libgstdistancetransform_la_LDFLAGS = $(GST_PLUGIN_LDFLAGS)


libgstdistancetransform_la_LIBTOOLFLAGS = --tag=disable-static

# headers we need but don't want installed
noinst_HEADERS = \
	gstdistancetransform.h

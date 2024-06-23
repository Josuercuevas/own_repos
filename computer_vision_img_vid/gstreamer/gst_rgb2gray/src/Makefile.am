plugin_LTLIBRARIES = libgstrgb2gray.la

# sources used to compile this plug-in
libgstrgb2gray_la_SOURCES = \
	gstrgb2gray.c \
	gstrgb2gray.h

# compiler and linker flags used to compile this plugin, set in configure.ac
libgstrgb2gray_la_CFLAGS = \
	$(GST_PLUGINS_BASE_CFLAGS) \
	$(GST_BASE_CFLAGS) \
	$(GST_CFLAGS)

#including needed libraries to run the plugin successfully
libgstrgb2gray_la_LIBADD = \
	$(GST_BASE_LIBS) \
	$(GST_PLUGINS_BASE_LIBS) -lgstvideo-1.0 \
	$(GST_LIBS)

#LD flags in case we need any
libgstrgb2gray_la_LDFLAGS = $(GST_PLUGIN_LDFLAGS)


libgstrgb2gray_la_LIBTOOLFLAGS = --tag=disable-static

# headers we need but don't want installed
noinst_HEADERS = \
	gstrgb2gray.h






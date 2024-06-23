plugin_LTLIBRARIES = libgstcolorretinex.la

# sources used to compile this plug-in
libgstcolorretinex_la_SOURCES = \
	gstcolorretinex.c \
	gstcolorretinex.h \
	RetinexLib.c \
	includes/RetinexLib.h \
	norm.c \
	includes/norm.h

# compiler and linker flags used to compile this plugin, set in configure.ac
libgstcolorretinex_la_CFLAGS = \
	$(GST_PLUGINS_BASE_CFLAGS) \
	$(GST_BASE_CFLAGS) \
	$(GST_CFLAGS)

# libraries to add for compilation
libgstcolorretinex_la_LIBADD = \
	$(GST_BASE_LIBS) \
	$(GST_PLUGINS_BASE_LIBS) -lgstvideo-1.0 \
	$(GST_LIBS) \
	-lfftw3f -lfftw3f_threads

# extra libraries flags
libgstcolorretinex_la_LDFLAGS = $(GST_PLUGIN_LDFLAGS)

# lib tools for compilation
libgstcolorretinex_la_LIBTOOLFLAGS = --tag=disable-static

# headers we need but don't want installed
noinst_HEADERS = \
	gstcolorretinex.h \
	includes/RetinexLib.h \
	includes/norm.h

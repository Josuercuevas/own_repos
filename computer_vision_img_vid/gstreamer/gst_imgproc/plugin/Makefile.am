plugin_LTLIBRARIES = libgstimg_procdetector.la

# sources used to compile this plug-in

libgstimg_procdetector_la_SOURCES = \
    plugin.c \
    common/metainfo.h \
    common/metainfo.c

# Add for BlobDetector
libgstimg_procdetector_la_SOURCES += \
    gst-BlobDetector/src/gstblobdetector.c \
    gst-BlobDetector/src/gstblobdetector.h \
    gst-BlobDetector/src/BlobDetection.c \
    gst-BlobDetector/src/Extractor_core.c \
    gst-BlobDetector/src/includes/BlobDetection.h \
    gst-BlobDetector/src/includes/Extractor.h

# Added for Normalizer
libgstimg_procdetector_la_SOURCES += \
    gst-Normalizer/src/gstmomentnormalization.h \
    gst-Normalizer/src/gstmomentnormalization.c \
    gst-Normalizer/src/MomentNormalizer.c \
    gst-Normalizer/src/includes/MomentNormalizer.h \
    gst-Normalizer/src/includes/Normalizer.h

# Added for Retinex
libgstimg_procdetector_la_SOURCES += \
    gst-Retinex/src/gstcolorretinex.c \
    gst-Retinex/src/gstcolorretinex.h \
    gst-Retinex/src/RetinexLib.c \
    gst-Retinex/src/includes/RetinexLib.h \
    gst-Retinex/src/norm.c \
    gst-Retinex/src/includes/norm.h

# Added for SkinDetector
libgstimg_procdetector_la_SOURCES += \
    gst-SkinDetector/src/gstskindetector.c \
    gst-SkinDetector/src/gstskindetector.h

# Added for DistanceTransform
libgstimg_procdetector_la_SOURCES += \
    gst-DistanceTransform/src/gstdistancetransform.c \
    gst-DistanceTransform/src/gstdistancetransform.h 

# compiler and linker flags used to compile this plugin, set in configure.ac
libgstimg_procdetector_la_CFLAGS = \
    $(GST_PLUGINS_BASE_CFLAGS) \
    $(GST_BASE_CFLAGS) \
    $(GST_CFLAGS)

#including needed libraries to run the plugin successfully
libgstimg_procdetector_la_LIBADD = \
    $(GST_BASE_LIBS) \
    $(GST_PLUGINS_BASE_LIBS) -lgstvideo-1.0\
    $(GST_LIBS)

# Added for Retinex
libgstimg_procdetector_la_LIBADD += \
    -lfftw3f -lfftw3f_threads

#LD flags in case we need any
libgstimg_procdetector_la_LDFLAGS = $(GST_PLUGIN_LDFLAGS)

libgstimg_procdetector_la_LIBTOOLFLAGS = --tag=disable-static

# headers we need but don't want installed
noinst_HEADERS = \
    common/metainfo.h \
    gst-BlobDetector/src/includes/BlobDetection.h \
    gst-BlobDetector/src/includes/Extractor.h \
    gst-BlobDetector/src/gstblobdetector.h \
    gst-Normalizer/src/includes/MomentNormalizer.h \
    gst-Normalizer/src/includes/Normalizer.h \
    gst-Normalizer/src/gstmomentnormalization.h \
    gst-Retinex/src/gstcolorretinex.h \
    gst-Retinex/src/includes/RetinexLib.h \
    gst-Retinex/src/includes/norm.h \
    gst-SkinDetector/src/gstskindetector.h \
    gst-DistanceTransform/src/gstdistancetransform.h

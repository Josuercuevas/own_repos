# Make file for LBP extractor main functions
# author Josue Cuevas

##Application name
APP_NAME := extract_lbp

## General folders
BINDIR := ../bin
SRCDIR := src/*
CXX = g++
LIBS :=   -lm -ldl -lpthread -lwavpack -lfftw3 -lX11
CFLAGS := -Wl,--no-as-needed
FLAGS :=  $(CFLAGS) -Wall -lstdc++

LBP_FEATURE_EXTRACTOR := $(BINDIR)/$(APP_NAME)

.PHONY:  app all clean

all: app

app: $(LBP_FEATURE_EXTRACTOR)

clean:
	@echo "******************************** Cleaning object files..! ********************************"
	rm -f $(LBP_FEATURE_EXTRACTOR) $(BINDIR)/*.o $(BINDIR)/*.d $(BINDIR)/debug_images/* $(BINDIR)/debug_histograms/*
	@echo "****************************** Done cleaning object files..! ****************************"
	@echo ""
	@echo ""

$(LBP_FEATURE_EXTRACTOR): $(SRCDIR)
		@echo "******************************* Creating write audio tester binary ..! *******************************"
		$(CXX) -o $@ $(FLAGS) $(CFLAGS) $(LIBS) -ldl $^
		@echo "******************************** Done creating write audio tester binary ..! ****************************"
		@echo ""

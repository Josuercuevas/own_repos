################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../plugin/gst-BlobDetector/src/BlobDetection.c \
../plugin/gst-BlobDetector/src/Extractor_core.c \
../plugin/gst-BlobDetector/src/gstblobdetector.c 

OBJS += \
./plugin/gst-BlobDetector/src/BlobDetection.o \
./plugin/gst-BlobDetector/src/Extractor_core.o \
./plugin/gst-BlobDetector/src/gstblobdetector.o 

C_DEPS += \
./plugin/gst-BlobDetector/src/BlobDetection.d \
./plugin/gst-BlobDetector/src/Extractor_core.d \
./plugin/gst-BlobDetector/src/gstblobdetector.d 


# Each subdirectory must supply rules for building sources it contributes
plugin/gst-BlobDetector/src/%.o: ../plugin/gst-BlobDetector/src/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	gcc -I/usr/include/glib-2.0 -I/usr/local/include/gstreamer-1.0 -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



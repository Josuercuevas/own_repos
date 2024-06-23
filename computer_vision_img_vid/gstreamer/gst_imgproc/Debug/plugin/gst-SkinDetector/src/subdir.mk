################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../plugin/gst-SkinDetector/src/Detector_core.c \
../plugin/gst-SkinDetector/src/SkinDetector.c \
../plugin/gst-SkinDetector/src/gstskindetector.c 

OBJS += \
./plugin/gst-SkinDetector/src/Detector_core.o \
./plugin/gst-SkinDetector/src/SkinDetector.o \
./plugin/gst-SkinDetector/src/gstskindetector.o 

C_DEPS += \
./plugin/gst-SkinDetector/src/Detector_core.d \
./plugin/gst-SkinDetector/src/SkinDetector.d \
./plugin/gst-SkinDetector/src/gstskindetector.d 


# Each subdirectory must supply rules for building sources it contributes
plugin/gst-SkinDetector/src/%.o: ../plugin/gst-SkinDetector/src/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	gcc -I/usr/include/glib-2.0 -I/usr/local/include/gstreamer-1.0 -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



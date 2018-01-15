################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../plugin/gst-DistanceTransform/src/gstdistancetransform.c 

OBJS += \
./plugin/gst-DistanceTransform/src/gstdistancetransform.o 

C_DEPS += \
./plugin/gst-DistanceTransform/src/gstdistancetransform.d 


# Each subdirectory must supply rules for building sources it contributes
plugin/gst-DistanceTransform/src/%.o: ../plugin/gst-DistanceTransform/src/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	gcc -I/usr/include/glib-2.0 -I/usr/local/include/gstreamer-1.0 -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



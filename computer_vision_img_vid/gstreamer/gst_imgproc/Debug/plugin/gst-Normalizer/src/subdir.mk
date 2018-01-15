################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../plugin/gst-Normalizer/src/MomentNormalizer.c \
../plugin/gst-Normalizer/src/gstmomentnormalization.c 

OBJS += \
./plugin/gst-Normalizer/src/MomentNormalizer.o \
./plugin/gst-Normalizer/src/gstmomentnormalization.o 

C_DEPS += \
./plugin/gst-Normalizer/src/MomentNormalizer.d \
./plugin/gst-Normalizer/src/gstmomentnormalization.d 


# Each subdirectory must supply rules for building sources it contributes
plugin/gst-Normalizer/src/%.o: ../plugin/gst-Normalizer/src/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	gcc -I/usr/include/glib-2.0 -I/usr/local/include/gstreamer-1.0 -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



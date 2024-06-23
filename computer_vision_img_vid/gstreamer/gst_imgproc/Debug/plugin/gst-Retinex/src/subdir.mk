################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../plugin/gst-Retinex/src/RetinexLib.c \
../plugin/gst-Retinex/src/gstcolorretinex.c \
../plugin/gst-Retinex/src/norm.c 

OBJS += \
./plugin/gst-Retinex/src/RetinexLib.o \
./plugin/gst-Retinex/src/gstcolorretinex.o \
./plugin/gst-Retinex/src/norm.o 

C_DEPS += \
./plugin/gst-Retinex/src/RetinexLib.d \
./plugin/gst-Retinex/src/gstcolorretinex.d \
./plugin/gst-Retinex/src/norm.d 


# Each subdirectory must supply rules for building sources it contributes
plugin/gst-Retinex/src/%.o: ../plugin/gst-Retinex/src/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	gcc -I/usr/include/glib-2.0 -I/usr/local/include/gstreamer-1.0 -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../plugin/common/metainfo.c 

OBJS += \
./plugin/common/metainfo.o 

C_DEPS += \
./plugin/common/metainfo.d 


# Each subdirectory must supply rules for building sources it contributes
plugin/common/%.o: ../plugin/common/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	gcc -I/usr/include/glib-2.0 -I/usr/local/include/gstreamer-1.0 -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



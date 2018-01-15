################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include plugin/gst-SkinDetector/src/subdir.mk
-include plugin/gst-Retinex/src/subdir.mk
-include plugin/gst-Normalizer/src/subdir.mk
-include plugin/gst-DistanceTransform/src/subdir.mk
-include plugin/gst-BlobDetector/src/subdir.mk
-include plugin/common/subdir.mk
-include plugin/.libs/subdir.mk
-include plugin/subdir.mk
-include subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
endif

-include ../makefile.defs

# Add inputs and outputs from these tool invocations to the build variables 

# All Target
all: gst_img_procdetector

# Tool invocations
gst_img_procdetector: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: Cross GCC Linker'
	gcc  -o "gst_img_procdetector" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(OBJS)$(C_DEPS)$(EXECUTABLES) gst_img_procdetector
	-@echo ' '

.PHONY: all clean dependents
.SECONDARY:

-include ../makefile.targets

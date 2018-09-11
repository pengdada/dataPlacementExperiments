# Add the make.common for common paths
include ../../make.common

# build the versions without using any OpenCL code
srcFiles= heat.cu heat_globalmem.cu \
	heat_2d.cu heat_2d_globalmem.cu GPUTuner-main-MM.cu
exeFiles =$(srcFiles:.cu=.out)

NVLM_POWER_HOME=../../nvml-power

# For CUPTI profiler
# ifeq ($(CUPTI_PATH),)
# CUPTI_PATH := /usr/local/cuda/extras/CUPTI/
# # CUPTI_PATH := /home/sayket111/abdullah/CUPTI/
# # /home/sayket111/abdullah/GPUTuning/nvml-power/
# endif
# -lnvidia-ml

all:$(exeFiles)  

# For cupti_profiler
# CUPTI_PATH=/usr/local/cuda/extras/CUPTI/
INCLUDES = -I ../../cupti_profiler/include/ -I $(CUPTI_PATH)/include/
CXXARGS = -std=c++11 -g
CXXARGS += -Xcompiler -DDEBUG
LIBS = -lcuda -Xlinker -rpath,$(CUPTI_PATH)/lib64 -L$(CUPTI_PATH)/lib64 -lcupti

#@GPUTuning Test for nvml-power
HEADERNVMLAPI = -I$(CUDA_HOME)/include -I$(NVLM_POWER_HOME)/include -I/usr/include -Xlinker -rpath,$(CUDA_HOME)/lib64/ -L$(CUDA_HOME)/lib64/ -L/usr/lib64 -lnvidia-ml -lcuda -lpthread 
INCLUDEPROG = $(NVLM_POWER_HOME)/src/nvmlpower.cpp


%.out:%.cu $(INCLUDEPROG)
	# nvcc $(GPUTuner_FLAG) $(INCLUDECUDA) -G -DNO_OPENCL -o $@ $+ $(HEADERNVMLAPI)
	nvcc $(GPUTuner_FLAG) $(CXXARGS) $(INCLUDES) $(LIBS) -G -DNO_OPENCL  $< $(INCLUDEPROG) -lm -o $@ $(HEADERNVMLAPI)
	
# -DDIM=1024 -DBLOCKSIZE=32

#
clean:
	rm -rf *.out


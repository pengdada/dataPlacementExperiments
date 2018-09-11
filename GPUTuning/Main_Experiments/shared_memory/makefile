# Add the make.common for common paths
include ../../make.common

srcFiles= mm-naive.cu mm-shmem.cu GPUTuner-main-MM.cu
exeFiles =$(srcFiles:.cu=.out)

NVLM_POWER_HOME=../../nvml-power

# -lnvidia-ml

all:$(exeFiles)  

#@GPUTuning Test for nvml-power
HEADERNVMLAPI = -I$(CUDA_HOME)/include -I$(NVLM_POWER_HOME)/include -I/usr/include -Xlinker -rpath,$(CUDA_HOME)/lib64/ -L$(CUDA_HOME)/lib64/ -L/usr/lib64 -lnvidia-ml -lcuda -lpthread 
INCLUDEPROG = $(NVLM_POWER_HOME)/src/nvmlpower.cpp

%.out:%.cu $(INCLUDEPROG)
	nvcc $(GPUTuner_FLAG) $(INCLUDECUDA) -o $@ $+ $(HEADERNVMLAPI)

# -DMSIZE=1024 -DBLOCK_SIZE=32

#
clean:
	rm -rf *.out


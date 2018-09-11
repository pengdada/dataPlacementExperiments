# I removed the OpenGL display stuff so no glut is needed anymore
# ray_globalMem.cu ray_constMem.cu
# Add the make.common for common paths
include ../../make.common

srcFiles= const_global_mem.cu
exeFiles =$(srcFiles:.cu=.out)

NVLM_POWER_HOME=../../nvml-power

all:$(exeFiles)  

#@GPUTuning Test for nvml-power
HEADERNVMLAPI = -I$(CUDA_HOME)/include -I$(NVLM_POWER_HOME)/include -I/usr/include -Xlinker -rpath,$(CUDA_HOME)/lib64/ -L$(CUDA_HOME)/lib64/ -L/usr/lib64 -lnvidia-ml -lcuda -lpthread 
INCLUDEPROG = $(NVLM_POWER_HOME)/src/nvmlpower.cpp

%.out:%.cu $(INCLUDEPROG)
	nvcc $(GPUTuner_FLAG) $(INCLUDECUDA) -o $@ $+ $(HEADERNVMLAPI)

#GRIDSIZE: 1 for 1D grid 2 for 2D grid
#BLOCKSIZE: 16, 32
#MEMTYPE: shared (1), constant(2), global(3)
#SPHERES: 1, 20, 256, 2340
#DIM: 1024, 2048, 4096, 20480, 60000
#	nvcc $< -lglut -lGL -lGLU -o $@
#
clean:
	rm -rf *.out


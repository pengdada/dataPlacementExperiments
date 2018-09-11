include ../../make.common
# CUDAHOME=/usr/local/cuda/
NVML_PATH=/usr/local/cuda-9.2/targets/x86_64-linux/
#         /usr/local/cuda-9.2/targets/x86_64-linux/include/

# --path-to-nvml=/usr/local/cuda-9.2/targets/x86_64-linux/lib/stubs/libnvidia-ml.so 

# * Add the following to the Makefile:

# ```sh
# INCLUDECUDA = -I/usr/local/cuda-7.0/samples/common/inc/
# HEADERNVMLAPI = -L/usr/lib64/nvidia -lnvidia-ml -L/usr/lib64 -lcuda -I/usr/include -lpthread -I/(**Path_to_the_nvmlPower_files**)
# INCLUDEPROG = (**Path_to_the_nvmlPower_files**)/nvmlPower.cpp

# radixSortThrust: radixSortThrust.cu $(INCLUDEPROG)
# 	nvcc $(INCLUDECUDA) -o $@ $+ $(HEADERNVMLAPI)

# -I$(NVML_PATH)/include -Wl,-rpath=/usr/lib/nvidia-396/ -L/usr/lib/nvidia-396/
# --compiler-options '-fPIC'
all:
	g++ -shared -fPIC -I../include -I$(CUDA_HOME)/include -L$(CUDA_HOME)/lib -L$(CUDA_HOME)/lib64 -lcuda -lcudart  -lnvidia-ml -lpthread -c nvmlpower.cpp -o ../lib/libnvmlpower.so -ldl

clean:
	rm -rf ../lib/*.so

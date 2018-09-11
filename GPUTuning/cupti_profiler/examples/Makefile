all: demo

CUPTI_PATH=/usr/local/cuda/extras/CUPTI
INCLUDES = -I ../include -I /usr/local/cuda/extras/CUPTI/include
CXXARGS = -std=c++11 -g
CXXARGS += -Xcompiler -DDEBUG
LIBS = -lcuda -L$(CUPTI_PATH)/lib64 -lcupti
# ../include/cupti_profiler.cpp
demo: demo.cu
	nvcc $(CXXARGS) $(INCLUDES) $(LIBS)  demo.cu  -o demo

clean:
	rm -f *.o demo

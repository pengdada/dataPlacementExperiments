nvcc -O3 -use_fast_math -arch $2     -maxrregcount 85 --ptxas-options=-v -I. -I/usr/tce/packages/cuda/cuda-9.0.176/samples/common/inc -o $1 $1.cu -L /usr/tce/packages/cuda/cuda-9.0.176/lib -lm

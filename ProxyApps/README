This folder contains different data placement versions of 5 lulesh kernels.
* All data in global memory
  globalall.cu: same as original programs
  globalall_1.cu: time recorder for each kernel
  globalall_total.cu: time recorder for whole execution time. 

* All readonly data in texture memory
  texall.cu: time recorder for overheads + kernel execution time for each kernel invocation
  texall_1.cu: time recorder for pure kernel execution time for each kernel invocation
  texall_total.cu: time recorder for whole execution time. 

* All readonly data in constant memory ( Works when edgeElem is small (e.g, =4)
  constantall.cu: time recorder for overheads + kernel execution time for each kernel invocation
  constantall_1.cu: time recorder for pure kernel execution time for each kernel invocation
  constantall_total.cu: time recorder for whole execution time. 

* All readonly data will be copied to shared memory ( works when edgeElem is small (e.g, =4))
  sharedall.cu: time recorder for overheads + kernel execution time for each kernel invocation
  sharedall_1.cu: time recorder for pure kernel execution time for each kernel invocation
  sharedall_total.cu: time recorder for whole execution time. 

To compile:
./compile.sh globalall  //excluding .cu

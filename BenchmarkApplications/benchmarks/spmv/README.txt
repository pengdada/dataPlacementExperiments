To collect execution time data, type
* make check

12 versions of working cuda code will be compiled and run.
The kernel timing (10x repetition) will be saved to *.screen

spmv.cu: original version

type make check to compile and run

------------------------------------------
sparse matrix vector multiplication

For the sparse matrix: read-only
* val[]
* cols[]
* rowDelimiters[]

The input vector: read-only
* vec[]

The output vector: writen
* out[]

sparse factor: non-zero elements total elements/50  
------------------------------------------

Working versions
-------------------------------
1.cu      rowDeli[] in shared memory, naive way to copy
2.cu      rowDeli[] in constant memory
4.cu      vec[] in texture, rowDeli[] in shared, better arrangement
5.cu      val[] in texture
6.cu      vec[] in constant, rowDeli[] in texture
7.cu      vec[] in texture 
8.cu      vec[] in texture, rowDeli[] in shared,  
9.cu      val[], col[] in texture, rowDeli[] in constant
10.cu     vec[], cols[], val[] texture
11.cu     cols[] texture
spmv.cu         : original version
spmv_rule.cu    : rule-based version, val[], col[], vec[] and rowDeli[] in texture

Versions with the __ldg undefined error: 
------------------------
combine.cu      : cols[], val[], vec[] read-only cache
spmv_read.cu    : read-only cache cols[] and vec[] // not working
spmv_adative.cu :  adaptive version ** // not working
                rowDeli[] in constant, val[]-readonly cache , vec[] in texture
spmv_surf.cu  : surface object

spmv_cpu.cu     : original version, comparing with CPU OpenMP version

Debugging versions:
------------------------
spmv_index.cu   : debugging version, record and write out some index values
spmv_kernel.cu  : original + total time
spmv_prof.cu  : debugging version with printf


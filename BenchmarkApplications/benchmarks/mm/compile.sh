nvcc mm-shmem.cu -o mm_shmem 
nvcc mm_rule.cu -o mm_rule
nvcc mm-naive.cu -o mm_naive
nvcc mm.cu -o mm
nvcc 1.cu -o 1
nvcc 3.cu -o 3
nvcc 6.cu -o 6
nvcc 8.cu -o 8 # not sure if this should be used??
#nvcc surf_1D.cu -o surf_1D
nvcc surface.cu -o surface

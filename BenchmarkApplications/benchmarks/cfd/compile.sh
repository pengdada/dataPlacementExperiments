nvcc cfd.cu -o cfd -arch=sm_35
nvcc cfd_new.cu -o cfd_new -arch=sm_35
nvcc cfd_rule.cu -o cfd_rule -arch=sm_35
#nvcc 6_overhead.cu -o 6_overhead -arch=sm_35
nvcc 1.cu -o 1 -arch=sm_35
nvcc 2.cu -o 2 -arch=sm_35
nvcc 3.cu -o 3 -arch=sm_35
nvcc 4.cu -o 4 -arch=sm_35
#nvcc 5.cu -o 5 -arch=sm_35
nvcc 6.cu -o 6 -arch=sm_35
nvcc 7.cu -o 7 -arch=sm_35

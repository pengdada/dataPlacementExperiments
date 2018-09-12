all: mm-naive.out mm-shmem.out
mm-naive.out: mm-naive.cu
	nvcc $< -lm -o $@
mm-shmem.out: mm-shmem.cu
	nvcc $< -lm -o $@
mm.out: mm.cu
	nvcc $< -o $@
# the version with OpenMP CPU, blocked version 	
mm_cpu.out: mm_cpu.cu
	nvcc -Xcompiler -openmp $< -o $@
check: mm-naive.out mm-shmem.out
	./mm-naive.out && ./mm-shmem.out
clean:
	rm -rf *.o *.out

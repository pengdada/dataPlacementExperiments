SrcFiles = \
  1.cu \
  2.cu \
  4.cu \
  5.cu \
  6.cu \
  7.cu \
  8.cu \
  9.cu \
  10.cu \
  11.cu \
  spmv.cu \
  spmv_rule.cu

ExeFiles = $(SrcFiles:.cu=.out)
  
all: $(ExeFiles) 

%.out:%.cu
	nvcc $< spmv_support.cu -o $@
clean:
	rm -rf $(ExeFiles) $(OutFiles) *.core

# run to generate screen output saved to files
OutFiles = $(ExeFiles:.out=.screen)

%.screen:%.out
	./$< | grep "kernel exe time" | cut -f 4 -d' ' > $@

check: $(OutFiles)	


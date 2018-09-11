#!/bin/bash

DIM=(2048)
GPUBLKSIZE=(32)

NO_OF_DIM=${#DIM[*]}
NO_OF_GPUBLKSIZE=${#GPUBLKSIZE[*]}

# mkdir tmp
# mv output.csv tmp/

for (( i=0; i<=$(( $NO_OF_DIM -1 )); i++ ))
do
	# sed -i "s/^#define MSIZE .*/#define MSIZE ${MSIZE[$i]} /" parameters.h
	sed -i "s/^#define DIM .*/#define DIM ${DIM[$i]} /" parameters.h 
	for (( k=0; k<=$(( $NO_OF_GPUBLKSIZE -1 )); k++ ))
	do		
		make clean
		GPUTuner_FLAG="-DBLOCKSIZE=${GPUBLKSIZE[$k]}"
		GPUTuner_FLAG=$GPUTuner_FLAG make
		echo "value = $GPUTuner_FLAG"
		# echo $dim $blksz
		python3 script-texture-memory-CUPTI.py		
	done
done

# exit
# exit
# exit
# sudo shutdown


#!/bin/bash

MSIZE=(1024)
GPUBLKSIZE=(32)

NO_OF_MSIZE=${#MSIZE[*]}
NO_OF_GPUBLKSIZE=${#GPUBLKSIZE[*]}

# mkdir tmp
# mv output.csv tmp/

for (( i=0; i<=$(( $NO_OF_MSIZE -1 )); i++ ))
do
	sed -i "s/^#define MSIZE .*/#define MSIZE ${MSIZE[$i]} /" parameters.h 
	for (( k=0; k<=$(( $NO_OF_GPUBLKSIZE -1 )); k++ ))
	do		
		make clean
		GPUTuner_FLAG="-DBLOCK_SIZE=${GPUBLKSIZE[$k]}"
		GPUTuner_FLAG=$GPUTuner_FLAG make
		echo "value = $GPUTuner_FLAG"
		# echo $dim $blksz
		python3 script-shared-memory-CUPTI.py		
	done
done

# exit
# exit
# exit
# sudo shutdown


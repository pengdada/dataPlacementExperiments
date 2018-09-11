#!/bin/bash

DIM=(20480)
BLKSIZE=(1000)
GPUBLKSIZE=(32)
MEMTYPE=(2 3)

NO_OF_DIM=${#DIM[*]}
NO_OF_BLKSIZE=${#BLKSIZE[*]}
NO_OF_GPUBLKSIZE=${#GPUBLKSIZE[*]}
NO_OF_MEMTYPE=${#MEMTYPE[*]}

# mkdir tmp
# mv output.csv tmp/

for (( i=0; i<=$(( $NO_OF_DIM -1 )); i++ ))
do
	sed -i "s/^#define DIM .*/#define DIM ${DIM[$i]} /" parameters.h 
	for (( j=0; j<=$(( $NO_OF_BLKSIZE -1 )); j++ ))
	do
		sed -i "s/^#define SPHERES.*/#define SPHERES ${BLKSIZE[$j]}/" parameters.h
		for (( k=0; k<=$(( $NO_OF_GPUBLKSIZE -1 )); k++ ))
		do
			for (( l=0; l<=$(( $NO_OF_MEMTYPE -1 )); l++ ))
			do
				# ${DATA_SIZE[$j]}
				make clean
                GPUTuner_FLAG="-DBLOCKSIZE=${GPUBLKSIZE[$k]} -DMEMTYPE=${MEMTYPE[$l]}"
				GPUTuner_FLAG=$GPUTuner_FLAG make
                echo "value = $GPUTuner_FLAG"
				# echo $dim $blksz
				python3 script-constant-memory-CUPTI.py
			done
		done
	done
done

# exit
# exit
# exit
# sudo shutdown


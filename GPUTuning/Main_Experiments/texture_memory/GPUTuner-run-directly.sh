#!/bin/bash

if [[ -z "${GPUTuning_PATH}" ]]; then
  GPUTuning_PATH=~/
else
  GPUTuning_PATH="${GPUTuning_PATH}"
fi

echo ${GPUTuning_PATH}

# Export necessary parameters to set up the environment for GPUTuner
export HARMONY_HOME=${GPUTuning_PATH}/GPUTuning/activeharmony-GPUTuner
export STRATEGY=exhaustive.so
export LAYERS=log.so:agg.so
export LOG_FILE=search.log
export AGG_TIMES=5
export AGG_FUNC=median

make clean
make
# run experiments

# small to max size for matrices
for dim in 1024 2048 4096
do 
	sed -i "s/^#define DIM .*/#define DIM $dim /" parameters.h 
	echo $dim
	../../activeharmony-GPUTuner/bin/GPUTuner -m=output -e=-DMEMTYPE,0,1,2,3 -e=-DBLOCKSIZE,4,8,16,24,32 -n=500 --compile ~-DBLOCKSIZE ./GPUTuner-main-MM.out %-DMEMTYPE %-DBLOCKSIZE $dim
done

# exit
# exit
# exit
# sudo shutdown


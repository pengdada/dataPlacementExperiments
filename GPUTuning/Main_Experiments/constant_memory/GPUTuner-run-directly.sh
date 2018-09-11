#!/bin/bash

if [[ -z "${GPUTuning_PATH}" ]]; then
  GPUTuning_PATH=~/
else
  GPUTuning_PATH="${GPUTuning_PATH}"
fi
# Export necessary parameters to set up the environment for GPUTuner
export HARMONY_HOME=${GPUTuning_PATH}/GPUTuning/activeharmony-GPUTuner
export STRATEGY=exhaustive.so
export LAYERS=log.so:agg.so
export LOG_FILE=search.log
export AGG_TIMES=5
export AGG_FUNC=median

# run experiments
# small to big size for bitMap in global mem
for dim in 1024 4096 10240 20480
do
# small to max size for spheres in constant mem
for blksz in 20 200 500 1000 2340
  do 
    sed -i "s/^#define DIM .*/#define DIM $dim /" parameters.h 
    sed -i "s/^#define SPHERES.*/#define SPHERES $blksz/" parameters.h
    echo $dim $blksz
    ../../activeharmony-GPUTuner/bin/GPUTuner -m=output -e=-DBLOCKSIZE,4,8,16,24,32 -e=-DMEMTYPE,2,3 -e=-DDIM,20480,40960 -n=500 --compile ~-DBLOCKSIZE ~-DMEMTYPE ./const_global_mem.out %-DBLOCKSIZE %-DMEMTYPE $blksz $dim
  done
done

# exit
# exit
# exit
# sudo shutdown


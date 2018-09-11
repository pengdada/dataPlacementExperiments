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

make clean
make
# run experiments

# small to max size for matrices
for msize in 512 # 1024 2048
do 
	sed -i "s/^#define MSIZE .*/#define MSIZE $msize /" parameters.h 
	echo $msize
	../../activeharmony-GPUTuner/bin/GPUTuner -m=output -e=-DMEMTYPE,0,1 -e=-DBLOCK_SIZE,4,8,16,24,32 -n=500 --compile ~-DBLOCK_SIZE ./GPUTuner-main-MM.out %-DMEMTYPE %-DBLOCK_SIZE $msize
done

# exit
# exit
# exit
# sudo shutdown


#!/bin/bash

if [[ -z "${GPUTuning_PATH}" ]]; then
  GPUTuning_PATH=~/
else
  GPUTuning_PATH="${GPUTuning_PATH}"
fi

# Install GPUTuner
cd ${GPUTuning_PATH}/GPUTuning/activeharmony-GPUTuner/
make clean
make
make install

# run experiments
# constant memroy
# cd ${GPUTuning_PATH}/GPUTuning/Main_Experiments/constant_memory/
# source GPUTuner-run-directly.sh

# texture memory
cd ${GPUTuning_PATH}/GPUTuning/Main_Experiments/texture_memory/
source GPUTuner-run-directly.sh

# shared memory
# cd ${GPUTuning_PATH}/GPUTuning/Main_Experiments/shared_memory/
# source GPUTuner-run-directly.sh


# exit
# exit
# exit
# sudo shutdown


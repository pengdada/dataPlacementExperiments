#!/bin/bash

if [[ -z "${GPUTuning_PATH}" ]]; then
  GPUTuning_PATH=~/
else
  GPUTuning_PATH="${GPUTuning_PATH}"
fi


# run experiments
# constant memroy
# cd ${GPUTuning_PATH}/GPUTuning/CUPTI_Experiments/constant_memory/
# source CUPTI-run-directly.sh

# texture memory
cd ${GPUTuning_PATH}/GPUTuning/CUPTI_Experiments/texture_memory/
source CUPTI-run-directly.sh

# # shared memory
# cd ${GPUTuning_PATH}/GPUTuning/CUPTI_Experiments/shared_memory
# source CUPTI-run-directly.sh

# exit
# exit
# exit
# sudo shutdown


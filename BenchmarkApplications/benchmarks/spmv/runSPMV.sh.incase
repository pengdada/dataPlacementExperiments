#!/bin/sh

HOST=`hostname`

if [[ $HOST = "ray"* ]]; then
   module load cuda/9.2.88
else
   module load cudatoolkit/9.1
fi

make clean && make
./spmv

#!/bin/bash

if [ -z "$1" ]
  then
    echo "Directory for data must be supplied!"
    exit 1
else
	DATADIR=$1
fi

if [ ! -d "$DATADIR" ]; then
	mkdir $DATADIR
fi
PERFDIR=$DATADIR"/performance/"

if [ ! -d "$LOGDIR" ]; then
	mkdir $LOGDIR
fi

if [ ! -d "$PERFDIR" ]; then
	mkdir $PERFDIR
fi

FILE=compile.sh
ITERATIONS=10
MINEXPMETRICS=141

HOST=`hostname | awk -F'-|[0-9]' '{ print $1 }'`

CONFIG="config.h"
MSIZES_FILE="msizes.txt"
BLOCKSIZES_FILE="blocksizes.txt"


find ../ -name $FILE | awk -F'/' '{ print $2 }' | while read DIR;do
	echo $DIR
	cd ../$DIR

	IFS=$'\n' read -d '' -r -a MSIZES < $MSIZES_FILE
	IFS=$'\n' read -d '' -r -a BLOCKSIZES < $BLOCKSIZES_FILE

	echo "${BLOCKSIZES[@]}"
	echo "${MSIZES[@]}"

	chmod +x $FILE
	COMPILE_CMD=./$FILE
	TEST=$( $COMPILE_CMD )

	cat $FILE | grep -v -e '^$' | grep -v "^ *#" | awk -F'-o' '{ print $2 }' | awk '{ print $1}' | while read BIN;do
		

		for BLOCK_SZ in "${BLOCKSIZES[@]}"; do
			for SIZE in "${MSIZES[@]}"; do

				for i in $( seq 1 $ITERATIONS )
				do	

					sed -i "/MSIZE/s/.*/#define MSIZE $SIZE/" $CONFIG
					sed -i "/BLOCK_SIZE/s/.*/#define BLOCK_SIZE $BLOCK_SZ/" $CONFIG

					TIMESTAMP=`date +%s`
					VAL=$(($TIMESTAMP + $i))

					FNAME="nvprof-"$DIR"-"$BIN"-"$HOST"-"$SIZE"-"$BLOCK_SZ"-"$VAL

					PERF_FILE=$PERFDIR$FNAME".time"
					
					chmod +x $BIN
					if [ "$DIR" == "acoustic" ] || [ "$DIR" == "dafx2015stencil" ] 
					then 
						continue
					else
						RUN_CMD=./$BIN
					fi
					OUTPUT=`eval "$RUN_CMD"`
					TIME=$(echo $OUTPUT | egrep "time:" | awk -F'PROGRAM TOOK|time:|Time =' '{ print $NF }' | awk '{ print $1 }')
					echo $TIME > $PERF_FILE

				done
			done
		done 
	done 
done

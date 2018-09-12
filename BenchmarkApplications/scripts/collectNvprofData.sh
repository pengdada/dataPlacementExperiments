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

if [ ! -d "$PERFDIR" ]; then
	mkdir $PERFDIR
fi

FILE=compile.sh
ITERATIONS=10
MINEXPMETRICS=141

HOST=`hostname | awk -F'-|[0-9]' '{ print $1 }'`

find ../benchmarks/ -name $FILE | awk -F'/' '{ print $3 }' | while read DIR;do
	echo $DIR
	cd ../benchmarks/$DIR
	chmod +x $FILE
	COMPILE_CMD=./$FILE

	cat $FILE | grep -v -e '^$' | grep -v "^ *#" | awk -F'-o' '{ print $2 }' | awk '{ print $1}' | while read BIN;do

		TEST=$( $COMPILE_CMD )
		for i in $( seq 1 $ITERATIONS )
		do	

			TIMESTAMP=`date +%s`
			VAL=$(($TIMESTAMP + $i))

			FNAME="nvprof-"$DIR"-"$BIN"-"$HOST"-"$VAL

			PERF_FILE=$PERFDIR$FNAME".time"
			
			chmod +x $BIN
			RUN_CMD=./$BIN

			OUTPUT=`eval "$RUN_CMD"`
			TIME=$(echo $OUTPUT | egrep "time:" | awk -F'PROGRAM TOOK|time:|Time =' '{ print $NF }' | awk '{ print $1 }')
			echo $TIME > $PERF_FILE

		done 
	done 
done

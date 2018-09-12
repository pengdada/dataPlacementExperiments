#!/bin/bash

# print platform and benchmark names with performance times

if [ $# -lt 1 ]
  then
    echo "Directory must be supplied!"
    exit 1
else
	DIR=$1
fi


# loop over all performance files averaging performance values by same Platform/Benchmark/Name
find $DIR -path "*.time" | xargs tail -n +1 | egrep -v "^$" | paste -d " " - - |  awk -v dir="$DIR"   'BEGIN { FS=dir } { print $2 }'  | awk 'BEGIN { ORS=" " } { print $1" ";printf("%.8f\n",$NF) }' | awk -F'-| ' 'BEGIN { ORS=" "} {  print $(NF-5)":"$(NF-4)":"$(NF-3)" "; printf("%.8f", $NF); print "\n"} ' | awk 'BEGIN{ ORS=" "} { avg[$1]+=$2; rec[$1]++ } END { for(v in avg){ print v" "; printf("%.8f\n",avg[v]/rec[v])     }}'| awk -F':' '{print $1" "$2" "$3" "$4}' | while read LINE;do 
	
		PLATFORM=$(echo $LINE | awk '{ print $3}')
		BENCHMARK=$(echo $LINE | awk '{ print $1 }')
		NAME=$(echo $LINE | awk '{ print $2 }')
		TIME=$(echo $LINE | awk '{ print $4 }')
		CLASS=$( python getMemoryClassificationAll.py $BENCHMARK $NAME )
			
		echo $PLATFORM" "$BENCHMARK" "$NAME" "$CLASS" "$TIME	
		
done | sort -k1,1 -k2,2 -k5,5


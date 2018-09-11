#################################################################

# Assumption, This script is run on the following directory structure
# .
# ..
# /Volta
# /Pascal
# /Kepler
# /Graph
# Data_Analyzer_to_graph_script.sh
# We also assume all the GPU folders has a input file called exec-time.txt

#################################################################

# Get the command line parameters
# Get the current directory [Needed for the grpah creating R scripts]
# @Help = https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -d|--dir)
    CurDir="$2"
    shift # past argument
    shift # past value
    ;;
    -i|--inputfile)
    InputFile="$2"
    shift # past argument
    shift # past value
    ;;
    -o|--outputfile)
    OutPutFile="$2"
    shift # past argument
    shift # past value
    ;;
    --default)
    DEFAULT=YES
    shift # past argument
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

#


# Get the input and output files

# Add appropiate header to the exec-time.txt
# e.g(Executable,Memtype,GPUBlockSize,NumSpheres,Dim,ExecTime)
# Probably do it manually for different application

GPU=(Volta Pascal Maxwell Kepler)
NO_OF_GPU=${#GPU[*]}

for (( i=0; i<=$(( $NO_OF_GPU -1 )); i++ ))
do
	# Go to a specific GPU folder
    echo ${CurDir}${GPU[$i]}
	# cd /${CurDir}${GPU[$i]}
	# Run the exec_time_data_summarizer,
	# This pyhton script summarizes the execution time file
	# It mostly averages the values across different runs
	# How to run "'exec_time_data_summarizer.py -i <inputfile> -o <outputfile>'"
	summaryInFile="${GPU[$i]}/exec-time.txt"
    echo $summaryInFile
	summaryOutFile="${GPU[$i]}/summary.csv"
	python3 exec_time_data_summarizer.py -i $summaryInFile -o $summaryOutFile
	# Run the exec_time_data_normalizer
	# This pyhton script normalizes the execution time data based on default configuration
	# How to run "'exec_time_data_normalizer.py -i <inputfile> -o <outputfile>'"
	normOutFile="${GPU[$i]}/Normalized_Exec_Data-${GPU[$i]}.csv"
	python3 exec_time_data_normalizer.py -i $summaryOutFile -o $normOutFile

done

# Now start creating Graphs
# Pass the CurDir to the graph routine
Rscript --vanilla Graph_script_Mem_Impact_Across_GPUBlock_Across_GPU.R -d ${CurDir}
Rscript --vanilla Graph_script_Execution_Time_Across_GPU.R -d ${CurDir}
# ${pwd}
# ${CurDir}

# We may want to create other graphs if necessary

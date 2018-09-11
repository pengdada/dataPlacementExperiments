import csv
import pandas as pd
import numpy as np

import sys, getopt

def parseArgs(argv):
	# InputFileName = ''
	# OutputFileName = ''
	try:
		opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
	except getopt.GetoptError:
		print('exec_time_data_summarizer.py -i <inputfile> -o <outputfile>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('exec_time_data_summarizer.py -i <inputfile> -o <outputfile>')
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg.strip()
		elif opt in ("-o", "--ofile"):
			outputfile = arg.strip()
	return (inputfile, outputfile)
	print ('Input file is =', inputfile)
	print ('Output file is =', outputfile)



if __name__ == '__main__':
	#Input and Output File Names
	#Input File Name
	# inputFileName=''
	# #Output File Name
	# outputFileName=''
	(inputFileName, outputFileName) = parseArgs(sys.argv[1:])
	print ('Input file is=', inputFileName)
	print ('Output file is=', outputFileName)

	##############   @AppSpecific 	###############
	header=['Executable', 'GPUBlockSize', 'DataSize', 'ExecTime']
	configOptions=['Executable', 'GPUBlockSize', 'DataSize']
	###############################################

	execData=pd.DataFrame()
	# for i in range(0, len(app)):
	# Generate current file name that we want to summarize
	data=pd.read_csv(inputFileName, names = header)
	#First do the average of the same configuration runs
	# This the we will have only one number associated with one configurtion
	data=data.groupby(configOptions, as_index=False).agg('mean')
	# data=data.groupby('Option 1', as_index=False).agg({'Execution Time (s)':'mean', 'Power Consumption (W)':'mean'})
	execData=execData.append(data)
	print(execData)
		
	execData.to_csv(outputFileName, sep=',', encoding='utf-8')	




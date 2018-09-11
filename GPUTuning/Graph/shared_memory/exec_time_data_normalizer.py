from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
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
	inputFileName=''
	#Output File Name
	outputFileName=''
	(inputFileName, outputFileName) = parseArgs(sys.argv[1:])
	print ('Input file is=', inputFileName)
	print ('Output file is=', outputFileName)
	data=pd.read_csv(inputFileName)	
##############   @AppSpecific 	###############
	# Create dataset size based on input parameters
###############################################
	dataSize = data.DataSize.unique()
	gpuBlockSize = data['GPUBlockSize'].unique()#.astype(str)
	# memTypes = data.Memtype.unique()
	# print(DataSize)
	# print(GPUBlockSize)
	# print(memTypes)
	finalData=pd.DataFrame()
	# Normalize the execution time for each type of memory based on GPU block size

	for ds in dataSize:
		tmpDF = data[(data['DataSize'] == ds)]
		##############   @AppSpecific 	###############
		# Chose the default configuration, may have to use different filtering criteria for different apps 
		# Normalize based on the execution time of the default config (Global Memory, Memtype = 3) and ((GPU Block size = 32))
		defExec = tmpDF[(tmpDF['Executable'] == './mm-naive.out') & (tmpDF['GPUBlockSize'] == 32)]['ExecTime']
		###############################################		
		print(defExec)
		tmpDF['NormExec'] = tmpDF['ExecTime']/float(defExec)
		tmpDF['Improvement'] = ((float(defExec) - tmpDF['ExecTime'])*100)/float(defExec)
		tmpDF['SpeedUp'] = float(defExec)/tmpDF['ExecTime']

		# # Is improvement, checks if there was improvement compared to the global memory
		# # Both comfiguration uses same GPUBlockSize
		# constMemExec = tmpDF[(tmpDF['Memtype'] == 3)]['ExecTime']
		# tmpDF['isImprovement'] = ((float(constMemExec) - tmpDF['ExecTime'])*100)/float(defExec)
		# print(tmpDF)
		finalData = finalData.append(tmpDF)
	print(finalData)
	# Saving the data to csv for future use
	finalData.to_csv(outputFileName, sep=',', encoding='utf-8')

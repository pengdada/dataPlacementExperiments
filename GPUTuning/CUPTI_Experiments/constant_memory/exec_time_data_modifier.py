import csv
import pandas as pd
import numpy as np

if __name__ == '__main__':
	###################### Things to modify based on apps #################
	#Input and Output File Names
	#Input File Name
	InputFileName="exec-time-Kepler.txt"
	#Output File Name
	OutputFileName="exec_summary-Kepler.csv"

	ConfigOptions=['Executable', 'Memtype', 'GPU-BlockSize', 'NumSpheres', 'Dim']

	execData=pd.DataFrame()
	# for i in range(0, len(app)):
	# Generate current file name that we want to summarize
	curInFile=InputFileName
	print(curInFile)
	data=pd.read_csv(curInFile)
	#First do the average of the same configuration runs
	# This the we will have only one number associated with one configurtion
	data=data.groupby(ConfigOptions, as_index=False).agg('mean')
	# data=data.groupby('Option 1', as_index=False).agg({'Execution Time (s)':'mean', 'Power Consumption (W)':'mean'})
	execData=execData.append(data)
	print(execData)
		
	execData.to_csv(OutputFileName, sep=',', encoding='utf-8')	

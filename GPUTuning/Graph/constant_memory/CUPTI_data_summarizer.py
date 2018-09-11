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

###################### Things to modify based on apps #################
#Input and Output File Names
(inputFileName, outputFileName) = parseArgs(sys.argv[1:])
print ('Input file is=', inputFileName)
print ('Output file is=', outputFileName)
# #Input File Name
# InputFileName="/home/sayket111/abdullah/PORPLE/OrgAndOptBenchmarks/MM/OUTPUT/7_26_2018/output.csv"
# #Output File Name
# OutputFileName="/home/sayket111/abdullah/PORPLE/OrgAndOptBenchmarks/MM/OUTPUT/7_26_2018/summary_data.csv"

# Set up this parameter for other apps
# It always starts with 'Kernel'
# Next params will be the actual configuration (e.g. dim, numspheres, memtype)
# These params may change based on different experiment types and applications
ConfigOptions=['Kernel']

######################################################################


data=pd.read_csv(inputFileName)

#First do the average of the same configuration runs
# This the we will have only one number associated with one configurtion
data=data.groupby('Configuration', as_index=False).agg({'Kernel':'first', 'Metric':'first', 'Value':'mean'})

# Split the configuration name to extract different configuration options
data['conf']=data['Configuration'].str.split("_")

# Add the configuration options to the main dataframe
# Excluding the 'Kenrel' in ConfigOptions
for i in range(1,len(ConfigOptions)):
	data[ConfigOptions[i]]=data['conf'].str[-i]

# Now group all the results associalted with same configuration in one data frame
# This grouping would give us all the metrics value for a single run
# eachConfigResult=data.groupby(['memtype', 'numshperes', 'dim'])
eachConfigResult=data.groupby(ConfigOptions)

#####################################################

summaryData=pd.DataFrame()
flag=0
for name,group in eachConfigResult:
	# print (group.columns)
	print(type(name))
	print(type(group))
	# Convert it to a pivot table, where index is the params defined at ConfigOptions
	# This phase converts all the metrics values of a single Configuration(ConfigOptions) to one row
	# Format is ike this: <ConfigOptions, metric>
	# groupT=pd.pivot_table(group, index=['Kernel', 'memtype', 'numshperes', 'dim'], columns='Metric', values='Value')
	groupT=pd.pivot_table(group, index=ConfigOptions, columns='Metric', values='Value')
	# Now append each group(configuration) to the summaryData dataframe
	summaryData=summaryData.append(groupT)
# Write the summarized data to the output file
summaryData.to_csv(outputFileName, sep=',', encoding='utf-8')
summaryData=pd.read_csv(outputFileName)

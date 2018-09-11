#include "nvmlpower.hpp"

/*
These may be encompassed in a class if desired. Trivial CUDA programs written for the purpose of benchmarking might prefer this approach.
*/
volatile bool pollThreadStatus = false;
unsigned int deviceCount = 0;
char deviceNameStr[64];

/* Varibale to track average power consumption and execution time */
struct timeval start, end;
 

nvmlReturn_t nvmlResult;
nvmlDevice_t nvmlDeviceID;
nvmlPciInfo_t nvmPCIInfo;
nvmlEnableState_t pmmode;
nvmlComputeMode_t computeMode;

pthread_t powerPollThread;

char finalOutPutFileP[MAX_NAME_LEN];
char timeStampedOutPutFileP[MAX_NAME_LEN];

TuningParams params;
 
/* Variable to save the profile data collected by NVML power*/
NVMLData profileData;


/*
Poll the GPU using nvml APIs.
*/
void *powerPollingFunc(void *ptr)
{

	unsigned int powerLevel = 0;
	/*FILE *myFile;
	_openTimeStampedFile(timeStampedOutPutFileP, &myFile);
	_writeTimeStampedHeader(myFile);
*/
	while (pollThreadStatus)
	{
		pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, 0);

		// Get the power management mode of the GPU.
		nvmlResult = nvmlDeviceGetPowerManagementMode(nvmlDeviceID, &pmmode);

		// The following function may be utilized to handle errors as needed.
		_getNVMLError(nvmlResult);

		// Check if power management mode is enabled.
		if (pmmode == NVML_FEATURE_ENABLED)
		{
			// Get the power usage in milliWatts.
			nvmlResult = nvmlDeviceGetPowerUsage(nvmlDeviceID, &powerLevel);
		}

		/* Add up power data for average power calculation */
		profileData.sumPower += (powerLevel)/1000.0;
		profileData.count++;

		/* Get the timestamp */
		struct timeval tmpTime;
		int ret;
		double timeStamp;
		if((ret = gettimeofday(&tmpTime, NULL)) != 0)
		{
			fprintf(stderr,"Error - gettimeofday() failed, errno = %d\n",errno);
		}

		timeStamp = (tmpTime.tv_sec - start.tv_sec);
		timeStamp += (tmpTime.tv_usec - start.tv_usec) / 1000000.0; /* us to sec */

		/*The output file stores power in Watts.*/
		// _writeTimeStampedDataToFile(myFile, timeStamp, (powerLevel)/1000.0);

		/* @Test*/
		// fprintf(stderr, "---------- Sum power = %f, Count = %d \n", profileData.sumPower, profileData.count);
		pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, 0);
	}

	// _closeTimeStampedFile(myFile);
	pthread_exit(0);
}

/*
Start power measurement by spawning a pthread that polls the GPU.
Function needs to be modified as per usage to handle errors as seen fit.
*/
void nvmlAPIRun()
{
	int i;

	// Initialize nvml.
	nvmlResult = nvmlInit();
	if (NVML_SUCCESS != nvmlResult)
	{
		fprintf(stderr,"NVML Init fail: %s\n", nvmlErrorString(nvmlResult));
		exit(0);
	}
	// Count the number of GPUs available.
	nvmlResult = nvmlDeviceGetCount(&deviceCount);
	if (NVML_SUCCESS != nvmlResult)
	{
		fprintf(stderr,"Failed to query device count: %s\n", nvmlErrorString(nvmlResult));
		exit(0);
	}
	for (i = 0; i < deviceCount; i++)
	{
		// Get the device ID.
		nvmlResult = nvmlDeviceGetHandleByIndex(i, &nvmlDeviceID);
		if (NVML_SUCCESS != nvmlResult)
		{
			printf("Failed to get handle for device %d: %s\n", i, nvmlErrorString(nvmlResult));
			exit(0);
		}

		// Get the name of the device.
		nvmlResult = nvmlDeviceGetName(nvmlDeviceID, deviceNameStr, sizeof(deviceNameStr)/sizeof(deviceNameStr[0]));
		if (NVML_SUCCESS != nvmlResult)
		{
			printf("Failed to get name of device %d: %s\n", i, nvmlErrorString(nvmlResult));
			exit(0);
		}

		// Get PCI information of the device.
		nvmlResult = nvmlDeviceGetPciInfo(nvmlDeviceID, &nvmPCIInfo);
		if (NVML_SUCCESS != nvmlResult)
		{
			printf("Failed to get PCI info of device %d: %s\n", i, nvmlErrorString(nvmlResult));
			exit(0);
		}

		// Get the compute mode of the device which indicates CUDA capabilities.
		nvmlResult = nvmlDeviceGetComputeMode(nvmlDeviceID, &computeMode);
		if (NVML_ERROR_NOT_SUPPORTED == nvmlResult)
		{
			printf("This is not a CUDA-capable device.\n");
		}
		else if (NVML_SUCCESS != nvmlResult)
		{
			printf("Failed to get compute mode for device %i: %s\n", i, nvmlErrorString(nvmlResult));
			exit(0);
		}
	}

	// This statement assumes that the first indexed GPU will be used.
	// If there are multiple GPUs that can be used by the system, this needs to be done with care.
	// Test thoroughly and ensure the correct device ID is being used.
	nvmlResult = nvmlDeviceGetHandleByIndex(0, &nvmlDeviceID);

	pollThreadStatus = true;
	const char *message = "Test";
	int ret = 0;
	/* Initialize and reset the profileData */
	profileData.sumPower = 0;
	profileData.avgPower = 0;
	profileData.execTime = 0;
	profileData.startTime = 0; 
	profileData.endTime = 0;
	profileData.count = 0;

	/* Create and open the file to save profileData and timestamped power value */

	/* ************** */

	if((ret = gettimeofday(&start, NULL)) != 0)
	{
		fprintf(stderr,"Error - gettimeofday() failed, errno = %d\n",errno);
	}

	int iret = pthread_create(&powerPollThread, NULL, powerPollingFunc, (void*) message);
	if (iret)
	{
		fprintf(stderr,"Error - pthread_create() return code: %d\n",iret);
		exit(0);
	}
}

/*
End power measurement. This ends the polling thread.
*/
void nvmlAPIEnd()
{
	time_t elapsedTime;
	int ret;
	pollThreadStatus = false;
	pthread_join(powerPollThread, NULL);

	if((ret = gettimeofday(&end, NULL)) != 0)
	{
		fprintf(stderr,"Error - gettimeofday() failed, errno = %d\n",errno);
	}

	profileData.execTime = (end.tv_sec - start.tv_sec);
	profileData.execTime += (end.tv_usec - start.tv_usec) / 1000000.0; /* us to sec */
	profileData.avgPower = profileData.sumPower / profileData.count;

	/* @Test*/
	// fprintf(stderr, "---------- Sum power = %f, Count = %d \n", profileData.sumPower, profileData.count);

	nvmlResult = nvmlShutdown();
	if (NVML_SUCCESS != nvmlResult)
	{
		printf("Failed to shut down NVML: %s\n", nvmlErrorString(nvmlResult));
		exit(0);
	}

	/* Write the final result in here*/
	_writeFinalOutPut(finalOutPutFileP);
	/*@Test: Just print the results*/
	// fprintf(stderr, "Exectime = %f, Avg power = %f\n", profileData.execTime, profileData.avgPower);

}

/*
 * @Summary, Setup tuning parameter values, used for formatted outputs
 */
void setUpTuningParams(int argc, char* argv[])
{
	int i;
	params.numParams = argc - 1;
	strcpy(params.executable, argv[0]);
	params.paramVals = (char **)malloc(sizeof(char*) * params.numParams);
	for (int i = 0; i < params.numParams; ++i)
	{
		params.paramVals[i] = (char*)malloc(sizeof(char) * MAX_PARAM_VAL_LEN);
		strcpy(params.paramVals[i], argv[i+1]);
	}
	_setUpOutPutFilePath();
}



/*
 * @Summary, Setup output file paths
 */
void _setUpOutPutFilePath()
{
	int i;
	char buf[10];
	/* Set up the path for the final_aggregate file*/
	strcpy(finalOutPutFileP, params.executable);
	strcat(finalOutPutFileP, "_Final_data.csv");

	/* Set up path for timestamped file */
	strcpy(timeStampedOutPutFileP, params.executable);
	for(i=0; i<params.numParams; i++)
	{		
		strcat(timeStampedOutPutFileP, "_");
		strcat(timeStampedOutPutFileP, params.paramVals[i]);
	}
	strcat(timeStampedOutPutFileP, ".csv");
}
/*
Return a number with a specific meaning. This number needs to be interpreted and handled appropriately.
*/
int _getNVMLError(nvmlReturn_t resultToCheck)
{
	if (resultToCheck == NVML_ERROR_UNINITIALIZED)
		return 1;
	if (resultToCheck == NVML_ERROR_INVALID_ARGUMENT)
		return 2;
	if (resultToCheck == NVML_ERROR_NOT_SUPPORTED)
		return 3;
	if (resultToCheck == NVML_ERROR_NO_PERMISSION)
		return 4;
	if (resultToCheck == NVML_ERROR_ALREADY_INITIALIZED)
		return 5;
	if (resultToCheck == NVML_ERROR_NOT_FOUND)
		return 6;
	if (resultToCheck == NVML_ERROR_INSUFFICIENT_SIZE)
		return 7;
	if (resultToCheck == NVML_ERROR_INSUFFICIENT_POWER)
		return 8;
	if (resultToCheck == NVML_ERROR_DRIVER_NOT_LOADED)
		return 9;
	if (resultToCheck == NVML_ERROR_TIMEOUT)
		return 10;
	if (resultToCheck == NVML_ERROR_IRQ_ISSUE)
		return 11;
	if (resultToCheck == NVML_ERROR_LIBRARY_NOT_FOUND)
		return 12;
	if (resultToCheck == NVML_ERROR_FUNCTION_NOT_FOUND)
		return 13;
	if (resultToCheck == NVML_ERROR_CORRUPTED_INFOROM)
		return 14;
	if (resultToCheck == NVML_ERROR_GPU_IS_LOST)
		return 15;
	if (resultToCheck == NVML_ERROR_UNKNOWN)
		return 16;

	return 0;
}

/*
 * @Summary, sets up the header for the output file
 */
void __writeOutputHeader(FILE* myfile)
{
	int i;
	for (int i = 0; i < params.numParams; ++i)
	{
		fprintf(myfile,"Option %d,", i);
	}
	fprintf(myfile,"Execution Time (s),");
	fprintf(myfile, "Power Consumption (W)\n");
}

/*
 * @Summary, internal function that writes output data to the output file
 */
void __writeFinalOutputDataToFile(FILE* myfile)
{
	int i;
	for (int i = 0; i < params.numParams; ++i)
	{
		fprintf(myfile,"%s,", params.paramVals[i]);
	}
	fprintf(myfile,"%f,", profileData.execTime);
	fprintf(myfile, "%f\n", profileData.avgPower);
}

/*
 * @Summary, wrapper function that writes data to output file
 */
void _writeFinalOutPut(char *fileName)
{
	FILE* myfile;
	int i;
	
	if( access( fileName, F_OK ) != -1 ) 
	{
		/* file exists */
		fprintf(stderr, "File already created, no need to write the header again\n");
		/*Just Open the file write the data */
		myfile = fopen(fileName, "a");
		if(myfile == NULL)
		{	
			fprintf(stderr, "*************File Error, Can not open the file***********\n");
			return;
		}
		__writeFinalOutputDataToFile(myfile);
	} 
	else 
	{
		myfile = fopen(fileName, "w");
		if(myfile == NULL)
		{	
			printf("*************File Error, Can not create the output file***********\n");
			return;
		}
		__writeOutputHeader(myfile);
		__writeFinalOutputDataToFile(myfile);
		
	}
	fclose(myfile);
}


void inline _openTimeStampedFile(char *fileName, FILE** myfile)
{
	// FILE* myfile = ;
	*myfile = fopen(fileName, "w+");
	if(myfile == NULL)
	{	
		fprintf(stderr, "*************File Error, Can not open the file***********\n");
		return;
	}
}

void inline _writeTimeStampedHeader(FILE *myfile)
{
	fprintf(myfile,"Time Stamp (s),");
	fprintf(myfile,"Power (W),");
	fprintf(myfile,"\n");
}

void inline _writeTimeStampedDataToFile(FILE *myfile, double timeStamp, double power)
{
	fprintf(myfile,"%.4lf,", timeStamp);
	fprintf(myfile,"%.4lf,", power);
	fprintf(myfile,"\n");
}

void inline _closeTimeStampedFile(FILE *myfile)
{
	fclose(myfile);
}
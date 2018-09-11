/*
Header file including necessary nvml headers.
*/

#ifndef __INCLNVML__
#define __INCLNVML__

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <nvml.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

/* Maximum length of a TUNING param value*/
#define MAX_PARAM_VAL_LEN 512
#define MAX_NAME_LEN 512

/*
 * Struct to save the necessary data from NVML 
 */
typedef struct _NVML_DATA_
{
	double execTime;
	double startTime;
	double endTime;
	/* 
	 * This variable holds the summation of the power consumption 
	 * at different timestamps 
	 */
	double sumPower;
	/* 
	 * Counts the number of times power data was poled, 
	 * used for avgPower calculation 
	 * avgPower = sumPower/count
	 */
	long int count; 
	/* Average power consumption of the kernel*/
	double avgPower;

	/* @TODO: Add other fields/features that we may want to collect*/

}NVMLData;


/* 
 * Struct to keep the input parameters (tuning parameters)
 * This values are used to differentiate between different profile data
 * We use these values to set up appropiate rows and file names
 */

typedef struct _TUNING_PARAMS_
{
	int numParams;	/*No of tuning params*/
	char **paramVals; /* string values of those params */
	char executable[MAX_NAME_LEN]; /* Name of the executable */
}TuningParams;

void setUpTuningParams(int argc, char* argv[]);
void nvmlAPIRun();
void nvmlAPIEnd();

/* Internal functions */
void __writeFinalOutputHeader(FILE* myfile);
void __writeFinalOutputDataToFile(FILE* myfile);
void _writeFinalOutPut(char *fileName);


void inline _openTimeStampedFile(char *fileName, FILE** myfile);
void inline _writeTimeStampedHeader(FILE *myfile);
void inline _writeTimeStampedDataToFile(FILE *myfile, double timeStamp, double power);
void inline _closeTimeStampedFile(FILE *myfile);

void _setUpOutPutFilePath();

void * _powerPollingFunc(void *ptr);
int _getNVMLError(nvmlReturn_t resultToCheck);

#endif

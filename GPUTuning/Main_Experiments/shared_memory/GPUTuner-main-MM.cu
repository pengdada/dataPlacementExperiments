/* 
 * @Abdullah, this code is more like a driver program to execute different 
 * version of program using different memories 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_STR_LEN 256

char apps[2][MAX_STR_LEN] = {"mm-naive.out", "mm-shmem.out"};

int main(int argc, char const *argv[])
{
	/* 
	 * The argv will be in the following sequence
	 * argv[0] app
	 * argv[1] = specific version of the program [e.g. 1 for mm-1.out]
	 * argv[2] = GPU Block size
	 * // App specific params
	 * argv[3]
	 * argv[4]
	 * ......
	 * argv[n]
	 */
	/*int app_ind_params, app_dep_params;
	app_ind_params = 3;
	app_dep_params = argc - app_ind_params;*/
	int app_type = atoi(argv[1]);
	
	char *app_command;

	/* Allocate space for application command line params */
	app_command = (char *)malloc(sizeof(*app_command) * (argc - 1) * MAX_STR_LEN);
	/*for (int i = 0; i < argc - 1; ++i)
	{
		app_command[i] = (char *) malloc(sizeof(app_command[i]) * MAX_STR_LEN);
	}*/

	/* Now make the command line string */
	sprintf(app_command, "./%s ", apps[app_type]);
	printf("%s\n", app_command);

	for (int i = 2; i < argc; ++i)
	{
		sprintf(app_command, "%s %s", app_command, argv[i]);
	}
	printf("%s\n", app_command);
	system(app_command);


	/*switch(app_type):
	case 0:
	case 1:*/

	return 0;
}


/*../../../activeharmony-GPUTuner/bin/GPUTuner -m=output -e=-DMEMTYPE,0,1 -e=-DMSIZE,1024,2048,4096,8192 -e=-DBLOCK_SIZE,4,8,16,32 -n=500 --compile ~-DMSIZE ~-DBLOCKSIZE ./a.out %-DMEMTYPE %-DMSIZE %-DBLOCKSIZE*/
#include <cuda_runtime_api.h>
#include <cuda.h>            
#include <iostream>          
#include <stdio.h>
#include <list>
#include <map>
#include <math.h>
#include <stdlib.h>         
#include <vector>
#include <set>
#include <algorithm>
#include <iterator>
#include <fstream>
#include <assert.h>
#define K 1
using namespace std;

// command parameters for SpMV

#define ITERATIONS 10
// Original size
#define MSIZE 12*8*21 //22
// Half size
//#define MSIZE 6*8*21 //22
// Double size
//#define MSIZE 24*8*21 //22
// Four time size
//#define MSIZE 48*8*21 //22
// 8x time size
//#define MSIZE 96*8*21 //22

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define SFactor 50 // sparse factor

static const double MAX_RELATIVE_ERROR = .02;
static const int PAD_FACTOR = 16;


//----------supportive functions
void HandleError( cudaError_t err,
                             const char *file,
                             int line ); 

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

// fill array elements with a random number 
void fill(float *A, const int n, const float maxi);
// init a sparse matrix with random nunbers
void initRandomMatrix(int *cols, int *rowDelimiters, const int n, const int dim);

void convertToPadded(float *A, int *cols, int dim, int *rowDelimiters,
        float **newA_ptr, int **newcols_ptr, int *newIndices,
	    int *newSize);

// CPU reference version
void spmvCpu(const float *val, const int *cols, const int *rowDelimiters,
        const float *vec, int dim, float *out);

// cross check the results
void spmv_verifyResults(const float *cpuResults, const float *gpuResults,
        const int size); 

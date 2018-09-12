#include "config.h"

void HandleError( cudaError_t err,
                             const char *file,
			     int line ) {
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
	file, line );
    exit( EXIT_FAILURE );
  }
}
// common supportive functions
void fill(float *A, const int n, const float maxi)
{
  for (int j = 0; j < n; j++) 
  {
    A[j] = ((float) maxi * (rand() / (RAND_MAX + 1.0f)));
  }
}

void initRandomMatrix(int *cols, int *rowDelimiters, const int n, const int dim)
{
  long long nnzAssigned = 0;

  // Figure out the probability that a nonzero should be assigned to a given
  // spot in the matrix
  double prob = (double)n / ((double)dim * (double)dim);

  // Seed random number generator
  srand48(2013);

  // Randomly decide whether entry i,j gets a value, but ensure n values
  // are assigned
  bool fillRemaining = false;
  for (int i = 0; i < dim; i++)
  {
    rowDelimiters[i] = nnzAssigned;
    for (int j = 0; j < dim; j++)
    {
      long long numEntriesLeft = (dim * dim) - ((i * dim) + j);
      long long needToAssign   = n - nnzAssigned;
      if (numEntriesLeft <= needToAssign) {
        fillRemaining = true;
      }
      //if ((nnzAssigned < n && drand48() <= prob) || fillRemaining)
      // must check boundary all the time to avoid out-of-bound access
      if ((nnzAssigned < n) && (drand48() <= prob || fillRemaining))
      {
        // Assign (i,j) a value
        cols[nnzAssigned] = j;
        nnzAssigned++;
      }
    }
  }
  // Observe the convention to put the number of non zeroes at the end of the
  // row delimiters array
  rowDelimiters[dim] = n;
  assert(nnzAssigned == n);
}

void convertToPadded(float *A, int *cols, int dim, int *rowDelimiters, 
    float **newA_ptr, int **newcols_ptr, int *newIndices, 
    int *newSize) 
{
  // determine total padded size and new row indices
  int paddedSize = 0;  
  int rowSize; 

  for (int i=0; i<dim; i++) 
  {    
    newIndices[i] = paddedSize; 
    rowSize = rowDelimiters[i+1] - rowDelimiters[i]; 
    if (rowSize % PAD_FACTOR != 0) 
    {
      rowSize += PAD_FACTOR - rowSize % PAD_FACTOR; 
    } 
    paddedSize += rowSize; 
  }
  *newSize = paddedSize; 
  newIndices[dim] = paddedSize; 

  HANDLE_ERROR(cudaMallocHost(newA_ptr, paddedSize * sizeof(float))); 
  HANDLE_ERROR(cudaMallocHost(newcols_ptr, paddedSize * sizeof(int))); 

  float *newA = *newA_ptr; 
  int *newcols = *newcols_ptr; 

  memset(newA, 0, paddedSize * sizeof(float)); 

  // fill newA and newcols
  for (int i=0; i<dim; i++) 
  {
    for (int j=rowDelimiters[i], k=newIndices[i]; j<rowDelimiters[i+1]; 
        j++, k++) 
    {
      newA[k] = A[j]; 
      newcols[k] = cols[j]; 
    }
  }
}

void spmvCpu(const float *val, const int *cols, const int *rowDelimiters, 
    const float *vec, int dim, float *out) 
{
  for (int i=0; i<dim; i++) 
  {
    float t = 0; 
    for (int j = rowDelimiters[i]; j < rowDelimiters[i + 1]; j++)
    {
      int col = cols[j]; 
      t += val[j] * vec[col];
    }    
    out[i] = t; 
  }
}

void spmv_verifyResults(const float *cpuResults, const float *gpuResults,
    const int size) 
{
  for (int i = 0; i < size; i++)
  {
    if (fabs(cpuResults[i] - gpuResults[i]) / cpuResults[i] 
        > MAX_RELATIVE_ERROR) 
    {
      cout << "Failed! Mismatch at i: "<< i << " ref: " << cpuResults[i] <<
        " dev: " << gpuResults[i] << endl;
      abort();
      return;
    }
  }
}



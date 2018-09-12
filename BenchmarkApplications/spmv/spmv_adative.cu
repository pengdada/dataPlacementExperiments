
#include <cassert>
#include <cfloat>
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
#include "../include/common.h"
#define K 1
using namespace std;

__constant__ int c_row[64516/4];
__constant__ int gotoGlobal[4];

#define MSIZE 12*8*21 //22
#define BLOCK_SIZE 256
#define WARP_SIZE 32

texture<float,1,cudaReadModeElementType> tex_vec;
texture<int,1,cudaReadModeElementType> tex_cols;
texture<float,1,cudaReadModeElementType> tex_row;
texture<float,1,cudaReadModeElementType> tex_val;

static const double MAX_RELATIVE_ERROR = .02;
static const int PAD_FACTOR = 16;

void fill(float *A, const int n, const float maxi)
{
  for (int j = 0; j < n; j++) 
  {
    A[j] = ((float) maxi * (rand() / (RAND_MAX + 1.0f)));
  }
}

void initRandomMatrix(int *cols, int *rowDelimiters, const int n, const int dim)
{
  int nnzAssigned = 0;

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
      int numEntriesLeft = (dim * dim) - ((i * dim) + j);
      int needToAssign   = n - nnzAssigned;
      if (numEntriesLeft <= needToAssign) {
        fillRemaining = true;
      }
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

  cudaMallocHost(newA_ptr, paddedSize * sizeof(float)); 
  cudaMallocHost(newcols_ptr, paddedSize * sizeof(int)); 

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
      t += val[j] * vec[col];//tex1Dfetch(tex_vec,col);
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
      abort ();
      return;
    }
  }

  cout << "spmv passed" << endl;
}
  __global__ void 
spmv_kernel( float*  val,
    int  * cols,
    int*   rowDelimiters,
    float*  vec,
    const int dim, float * out,int p0,int p1,int p2,int p3)
{
  // Thread ID in block
  int t = threadIdx.x; 
  // Thread ID within warp
  int id = t & (WARP_SIZE-1);
  int warpsPerBlock = blockDim.x / WARP_SIZE;
  // One row per warp
  int myRow = (blockIdx.x * warpsPerBlock) + (t / WARP_SIZE);
  //__shared__ int rowDeli[BLOCK_SIZE/WARP_SIZE+1];
  __shared__ volatile float partialSums[BLOCK_SIZE];
  //if (threadIdx.x<BLOCK_SIZE/WARP_SIZE+1)
  //rowDeli[threadIdx.x]=rowDelimiters[myRow+threadIdx.x];

  //__syncthreads();
  int _temp1,_temp2;
  if (myRow < dim) 
  {

    if(p0==0)
    {_temp1= c_row[myRow];_temp2=c_row[myRow+1];}
    else if(p0==1)
    {_temp1=tex1Dfetch(tex_row,myRow);_temp2=tex1Dfetch(tex_row,myRow+1);}
    else if(p0==3)
    {_temp1= __ldg(&rowDelimiters[myRow]);_temp2= __ldg(&rowDelimiters[myRow+1]);}
    int warpStart = _temp1;//c_row[myRow];
    int warpEnd = _temp2;//c_row[myRow+1];
    float mySum = 0;
    for (int j = warpStart + id; j < warpEnd; j += WARP_SIZE)
    {
      /*if(p1==0)
        {_temp1= cols[j];}
        else if(p1==1)
        {_temp1=tex1Dfetch(tex_cols,j);}
        else if(p1==3)
        {_temp1= __ldg(&cols[j]);}*/
      int col = cols[j]; 
      /*if(p2==0)
        {_temp1= vec[col];}
        else if(p2==1)
        {_temp1=tex1Dfetch(tex_vec,col);}
        else if(p2==3)
        {_temp1= __ldg(&vec[col]);}*/
      if(p3==0)
      {_temp2= val[j];}
      else if(p3==1)
      {_temp2=tex1Dfetch(tex_val,j);}
      else if(p3==3)
      {_temp2= __ldg(&val[j]);}
      mySum += tex1Dfetch(tex_vec,col)*_temp2;
    }
    partialSums[t] = mySum;

    // Reduce partial sums
    if (id < 16) partialSums[t] += partialSums[t+16];
    if (id <  8) partialSums[t] += partialSums[t+ 8];
    if (id <  4) partialSums[t] += partialSums[t+ 4];
    if (id <  2) partialSums[t] += partialSums[t+ 2];
    if (id <  1) partialSums[t] += partialSums[t+ 1];

    // Write result 
    if (id == 0)
    {
      out[myRow] = partialSums[t];
    }
  }
}

int main(int argc, char **argv) {
  cudaSetDevice(1);
  srand(2013);
  float *h_spmv_val, *h_spmv_valPad;
  int *h_spmv_cols, *h_spmv_colsPad;
  int *h_rowDelimiters, *h_rowDelimitersPad;
  float *h_spmv_vec, *h_spmv_out, *spmv_refOut;
  int spmv_nItems, nItemsPadded, spmv_numRows;

  spmv_numRows = MSIZE * (BLOCK_SIZE/WARP_SIZE);
  spmv_nItems = spmv_numRows * (spmv_numRows/10) ; // 1% of entries will be non-zero
  float maxval = 200.0;
  cudaMallocHost(&h_spmv_val, spmv_nItems * sizeof(float)); 
  cudaMallocHost(&h_spmv_cols, spmv_nItems * sizeof(int)); 
  cudaMallocHost(&h_rowDelimiters, (spmv_numRows + 1) * sizeof(int)); 
  fill(h_spmv_val, spmv_nItems, maxval); 
  initRandomMatrix(h_spmv_cols, h_rowDelimiters, spmv_nItems, spmv_numRows);

  // Set up remaining host data
  int paddedSize = spmv_numRows + (PAD_FACTOR - spmv_numRows % PAD_FACTOR);
  cudaMallocHost(&h_spmv_vec, spmv_numRows * sizeof(float)) ;
  spmv_refOut = new float[spmv_numRows];
  cudaMallocHost(&h_rowDelimitersPad, (spmv_numRows + 1) * sizeof(int)); 
  fill(h_spmv_vec, spmv_numRows, maxval);

  cudaMallocHost(&h_spmv_out, paddedSize * sizeof(float)); 
  convertToPadded(h_spmv_val, h_spmv_cols, spmv_numRows, h_rowDelimiters, &h_spmv_valPad,
      &h_spmv_colsPad, h_rowDelimitersPad, &nItemsPadded);

  // Compute reference solution
  spmvCpu(h_spmv_val, h_spmv_cols, h_rowDelimiters, h_spmv_vec, spmv_numRows, spmv_refOut);

  float *d_spmv_val, *d_spmv_vec, *d_spmv_out;
  int *d_spmv_cols, *d_rowDelimiters;

  // Allocate device memory
  cudaMalloc(&d_spmv_val,  spmv_nItems * sizeof(float));
  cudaMalloc(&d_spmv_cols, spmv_nItems * sizeof(int));
  cudaMalloc(&d_spmv_vec,  spmv_numRows * sizeof(float));
  cudaMalloc(&d_spmv_out,  spmv_numRows * sizeof(float));
  cudaMalloc(&d_rowDelimiters, (spmv_numRows+1) * sizeof(int));

  // Transfer data to device
  cudaMemcpy(d_spmv_val, h_spmv_val,   spmv_nItems * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_spmv_cols, h_spmv_cols, spmv_nItems * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_spmv_vec, h_spmv_vec, spmv_numRows * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rowDelimiters, h_rowDelimiters, (spmv_numRows+1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(c_row,h_rowDelimiters,(spmv_numRows+1)*sizeof(int));
  cudaBindTexture(0,tex_vec,d_spmv_vec,spmv_numRows * sizeof(float));
  //  cudaBindTexture(0,tex_val,d_spmv_val,spmv_nItems * sizeof(float));
  // cudaBindTexture(0,tex_col,d_spmv_cols,spmv_nItems * sizeof(int)); 
  //int *Global=(int *)malloc(4*sizeof(int)) ;

  //p0=0;p1=3;p2=1;p3=3;

  //cudaMemcpyToSymbol(gotoGlobal,Global,4*sizeof(int));

  cudaEvent_t kernel_start, kernel_stop;
  cudaEventCreate(&kernel_start);
  cudaEventCreate(&kernel_stop);
  float kernel_time = 0.0f;

  cudaEventRecord(kernel_start, 0);

  // Setup thread configuration
  int spmv_grid = (int) ceil(spmv_numRows / (float)(BLOCK_SIZE / WARP_SIZE));

  for(int i=0;i<10;i++)
    spmv_kernel <<<spmv_grid, BLOCK_SIZE>>>
      (d_spmv_val, d_spmv_cols, d_rowDelimiters, d_spmv_vec, spmv_numRows, d_spmv_out,0,3,1,3);

  cudaDeviceSynchronize();

  cudaEventRecord(kernel_stop, 0);
  cudaEventSynchronize(kernel_stop);

  // get elapsed time
  kernel_time = 0.0f;
  cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop);
  kernel_time *= 1.e-3; // Convert to seconds

  cout << "kernel exe time: " << kernel_time << endl;
  cudaMemcpy(h_spmv_out, d_spmv_out, spmv_numRows * sizeof(float), cudaMemcpyDeviceToHost);
  spmv_verifyResults(spmv_refOut, h_spmv_out, spmv_numRows);

  return 0;
}


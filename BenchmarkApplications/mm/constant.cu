
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
#include "config.h"
#define K 1
using namespace std;

#define mm_BLOCK_SIZE BLOCK_SIZE 
#define WA MSIZE // Matrix A width
#define HA MSIZE // Matrix A height
#define WB MSIZE // Matrix B width
#define HB WA  // Matrix B height
#define WC WB  // Matrix C width 
#define HC HA  // Matrix C height

#define mm_GRID_X (MSIZE/mm_BLOCK_SIZE)
#define mm_GRID_Y (MSIZE/mm_BLOCK_SIZE)
#define mm_NBLOCKS (mm_GRID_X*mm_GRID_Y)

#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]

void randomInit(float* data, int size)
{
  for (int i = 0; i < size; ++i)
    data[i] = rand() / (float)RAND_MAX;
}

/* 
   compute the gold standard: the reference results using the naive implementation 
   C[hA][wB] = A[hA][wA]xB[hB][wB]
   All matrices are linearized. 
 */
void
computeGold(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
  // outer i, j: for C[i][j]
  for (unsigned int i = 0; i < hA; ++i)
    for (unsigned int j = 0; j < wB; ++j) {
      double sum = 0.0;
      // inner k: iterate wA and hB
      for (unsigned int k = 0; k < wA; ++k) {
        double a = A[i * wA + k];
        double b = B[k * wB + j];
        sum += a * b;
      }
      C[i * wB + j] = (float)sum;
    }
}
__global__ void
mm_kernel( float* C, float* A, float* B, int wA, int wB)
{
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  // Index of the first sub-matrix of A processed by the block
  int aBegin = wA * mm_BLOCK_SIZE * by;

  // Index of the last sub-matrix of A processed by the block
  int aEnd   = aBegin + wA - 1;

  // Step size used to iterate through the sub-matrices of A
  int aStep  = mm_BLOCK_SIZE;

  // Index of the first sub-matrix of B processed by the block
  int bBegin = mm_BLOCK_SIZE * bx;

  // Step size used to iterate through the sub-matrices of B
  int bStep  = mm_BLOCK_SIZE * wB;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub = 0;

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int a = aBegin, b = bBegin;
      a <= aEnd;
      a += aStep, b += bStep) 
  {

    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[mm_BLOCK_SIZE][mm_BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[mm_BLOCK_SIZE][mm_BLOCK_SIZE];

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    AS(ty, tx) = A[a + wA * ty + tx];
    BS(ty, tx) = B[b + wB * ty + tx];

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll
    for (int k = 0; k < mm_BLOCK_SIZE; ++k)
      Csub += AS(ty, k) * BS(k, tx);

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  int c = wB * mm_BLOCK_SIZE * by + mm_BLOCK_SIZE * bx;

  C[c + wB * ty + tx] = Csub;
//if (threadIdx.x==0&&threadIdx.y==0) atomicAdd(d_flag,1);

}

int main(int argc, char **argv) {
  //  cudaSetDevice(1);
  srand(2013);
  unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;

  uiWA = WA;
  uiHA = HA;
  uiWB = WB;
  uiHB = HB;
  uiWC = WC;
  uiHC = HC;

  // allocate host memory for matrices A and B
  unsigned int size_A = uiWA * uiHA;
  unsigned int mem_size_A = sizeof(float) * size_A;
  float* h_A = (float*)malloc(mem_size_A);
  unsigned int size_B = uiWB * uiHB;
  unsigned int mem_size_B = sizeof(float) * size_B;
  float* h_B = (float*)malloc(mem_size_B);
  //printf("size A = %d bytes,size B=%d bytes\n",mem_size_A,mem_size_B);
  // initialize host memory
  randomInit(h_A, size_A);
  randomInit(h_B, size_B);

  // allocate device memory
  float* d_A, *d_B, *d_C;
  unsigned int size_C = uiWC * uiHC;
  unsigned int mem_size_C = sizeof(float) * size_C;
  printf("size A = %d bytes,size B=%d bytes,size C=%d bytes\n",mem_size_A,mem_size_B,mem_size_C);

  // allocate host memory for the result
  float* h_C      = (float*) malloc(mem_size_C);
  float* h_CUBLAS = (float*) malloc(mem_size_C);

  checkCudaErrors(cudaMalloc((void**) &d_A, mem_size_A));
  checkCudaErrors(cudaMalloc((void**) &d_B, mem_size_B));

  // copy host memory to device
  checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice) );
  checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice) );

  checkCudaErrors(cudaMalloc((void**) &d_C, mem_size_C));

  // setup execution parameters
  dim3 mm_grid(mm_GRID_X, mm_GRID_Y);
  dim3 mm_block(mm_BLOCK_SIZE, mm_BLOCK_SIZE);
  
  // warm up the GPU 

  printf("dimGrid: %dx%d dimBlock: %dx%d mat size: %dx%d\n",mm_GRID_X,mm_GRID_Y,mm_BLOCK_SIZE,mm_BLOCK_SIZE,uiWA,uiHA);
  for (int rpt=0; rpt<5; rpt++)
  {
  	mm_kernel<<< mm_grid, mm_block>>>(d_C, d_A, d_B, uiWA, uiWB);
  }

  cudaEvent_t kernel_start, kernel_stop;
  cudaEventCreate(&kernel_start);
  cudaEventCreate(&kernel_stop);
  float kernel_time = 0.0f;

  cudaEventRecord(kernel_start, 0);
  for (int rpt=0; rpt<ITERATIONS; rpt++)
  {
  	mm_kernel<<< mm_grid, mm_block>>>(d_C, d_A, d_B, uiWA, uiWB);
  }

  cudaDeviceSynchronize();
  cudaEventRecord(kernel_stop, 0);
  cudaEventSynchronize(kernel_stop);

  // get elapsed time
  kernel_time = 0.0f;
  cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop);
  kernel_time *= 1.e-3; // Convert to seconds

  cout << "kernel exe time: " << kernel_time/ITERATIONS<< endl;
  // copy result from device to host
  checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost) );

/*
*/
  // compute reference solution
  float* reference = (float*)malloc(mem_size_C);
  computeGold(reference, h_A, h_B, uiHA, uiWA, uiWB);

  // check result (matrixMul)
  bool resCUDA = sdkCompareL2fe(reference, h_C, size_C, 1.0e-6f);
  printf("CUDA matrixMul compares %s\n\n", (true == resCUDA) ? "passed" : "FAIL");
  free(reference);

  // clean up memory
  free(h_A);
  free(h_B);
  free(h_C);
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));

  return 0;
}


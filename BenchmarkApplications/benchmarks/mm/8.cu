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
texture<float,1,cudaReadModeElementType> tex_A;
texture<float,1,cudaReadModeElementType> tex_B;

void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

void
computeGold(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
  for (unsigned int i = 0; i < hA; ++i)
    for (unsigned int j = 0; j < wB; ++j) {
      double sum = 0;
      for (unsigned int k = 0; k < wA; ++k) {
        double a = A[i * wA + k];
        double b = B[k * wB + j];
        sum += a * b;
      }
      C[i * wB + j] = (float)sum;
    }
}
__global__ void
mm_kernel( float *A,float *B,float* C, int wA, int wB)
{
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x%16;
  int ty = threadIdx.x/16;

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
      a += aStep, b += bStep) {

    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[mm_BLOCK_SIZE][mm_BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[mm_BLOCK_SIZE][mm_BLOCK_SIZE];

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    AS(ty, tx) = A[a+wA*ty+tx];//tex1Dfetch(tex_A,a+wA*ty+tx);
    BS(ty, tx) = tex1Dfetch(tex_B,b + wB * ty + tx);

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

  uiWA = WA ;
  uiHA = HA ;
  uiWB = WB ;
  uiHB = HB ;
  uiWC = WC ;
  uiHC = HC ;

  // allocate host memory for matrices A and B
  unsigned int size_A = uiWA * uiHA;
  unsigned int mem_size_A = sizeof(float) * size_A;
  float* h_A = (float*)malloc(mem_size_A);
  unsigned int size_B = uiWB * uiHB;
  unsigned int mem_size_B = sizeof(float) * size_B;
  float* h_B = (float*)malloc(mem_size_B);

  // initialize host memory
  randomInit(h_A, size_A);
  randomInit(h_B, size_B);

  // allocate device memory
  float* d_A, *d_B, *d_C;
  unsigned int size_C = uiWC * uiHC;
  unsigned int mem_size_C = sizeof(float) * size_C;

  // allocate host memory for the result
  float* h_C      = (float*) malloc(mem_size_C);
  float* h_CUBLAS = (float*) malloc(mem_size_C);

  checkCudaErrors(cudaMalloc((void**) &d_A, mem_size_A));
  checkCudaErrors(cudaMalloc((void**) &d_B, mem_size_B));
  
   cudaChannelFormatDesc channelDescA =  cudaCreateChannelDesc<float>();
   cudaChannelFormatDesc channelDescB =  cudaCreateChannelDesc<float>();
    cudaArray* A_Array, *B_Array;
    cudaMallocArray(&A_Array, &channelDescA, uiWA, uiHA);
    cudaMallocArray(&B_Array, &channelDescB, uiWB, uiHB);

    // Copy to device memory some data located at address h_data
    // in host memory 
    cudaMemcpyToArray(A_Array, 0, 0, h_A, uiWA * uiHA * sizeof(float),
                      cudaMemcpyHostToDevice);
    cudaMemcpyToArray(B_Array, 0, 0, h_B, uiWB * uiHB * sizeof(float),
                      cudaMemcpyHostToDevice);

    // Set texture reference parameters
    tex_A.addressMode[0] = cudaAddressModeWrap;
    tex_A.addressMode[1] = cudaAddressModeWrap;
    tex_A.filterMode     = cudaFilterModePoint;
    tex_B.addressMode[0] = cudaAddressModeWrap;
    tex_B.addressMode[1] = cudaAddressModeWrap;
    tex_B.filterMode     = cudaFilterModePoint;
cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
    // Bind the array to the texture reference
    cudaBindTexture(0,tex_A, d_A,mem_size_A);
    cudaBindTexture(0,tex_B,d_B,mem_size_B);
   // cudaBindTextureToArray(tex_B, B_Array, channelDescB);
  // copy host memory to device
  //checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice) );
  //checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice) );

  checkCudaErrors(cudaMalloc((void**) &d_C, mem_size_C));

  dim3 mm_grid(mm_GRID_X, mm_GRID_Y);
  dim3 mm_block(mm_BLOCK_SIZE, mm_BLOCK_SIZE);
  
  // warm up the GPU 

  for (int rpt=0; rpt<5; rpt++)
  {
  	mm_kernel<<< mm_grid, mm_block >>>(d_A,d_B,d_C, uiWA, uiWB);
  }

  cudaEvent_t kernel_start, kernel_stop;
  cudaEventCreate(&kernel_start);
  cudaEventCreate(&kernel_stop);
  float kernel_time = 0.0f;

  printf("dimGrid: %dx%d dimBlock: %dx%d mat size: %dx%d\n",mm_GRID_X,mm_GRID_Y,mm_BLOCK_SIZE,mm_BLOCK_SIZE,uiWA,uiHA);
  cudaEventRecord(kernel_start, 0);
  // setup execution parameters
 // int mm_grid=mm_GRID_X*mm_GRID_Y;
  for (int rpt=0; rpt<ITERATIONS; rpt++)
  {
  mm_kernel<<< mm_grid, 16*16>>>(d_A,d_B,d_C, uiWA, uiWB);
  }

  cudaDeviceSynchronize();

  cudaEventRecord(kernel_stop, 0);
  cudaEventSynchronize(kernel_stop);

  // get elapsed time
  kernel_time = 0.0f;
  cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop);
  kernel_time *= 1.e-3; // Convert to seconds
  
  cout << "kernel exe time: " << kernel_time/ITERATIONS << endl;
  // copy result from device to host
  checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost) );

/* 
*/
  // compute reference solution
  float* reference = (float*)malloc(mem_size_C);
  //computeGold(reference, h_A, h_B, uiHA, uiWA, uiWB);

  // check result (matrixMul)
 // bool resCUDA = sdkCompareL2fe(reference, h_C, size_C, 1.0e-6f);
 //. printf("CUDA matrixMul compares %s\n\n", (true == resCUDA) ? "passed" : "FAIL");
  free(reference);

//   ofstream f1("mm_correct.txt");
//   for(int i=0; i<size_C; ++i)
//     f1 << reference[i] << endl;
//   f1.close();
// 
//   ofstream f2("mm_gpu.txt");
//   for(int i=0; i<size_C; ++i)
//     f2 << h_C[i] << endl;
//   f2.close();


  // clean up memory
  free(h_A);
  free(h_B);
  free(h_C);
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));


  return 0;
}


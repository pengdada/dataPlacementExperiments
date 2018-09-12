#include <omp.h>
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

//#define mm_BLOCK_SIZE_x 8
#define mm_BLOCK_SIZE 16
//#define mm_SUPER_BLOCKS_PER_SM 4
//int mm_SUPER_BLOCKS_PER_SM = 4;

#define iSizeMultiple 4 //must be multipes of 15

// A: 64x64
#define HA (4 * mm_BLOCK_SIZE) // Matrix A height/ rows
#define WA (4 * mm_BLOCK_SIZE) // Matrix A width / cols

// B: 64 x 960
// very strange width of B: 15 times of A's size ????
#define HB WA  // Matrix B height / rows
#define WB (60 * mm_BLOCK_SIZE) // Matrix B width / cols

// C: 64 x 960
#define HC HA  // Matrix C height/ rows
#define WC WB  // Matrix C width / cols 


#define mm_GRID_X (WC*iSizeMultiple/mm_BLOCK_SIZE)
#define mm_GRID_Y (HC*iSizeMultiple/mm_BLOCK_SIZE)
#define mm_NBLOCKS (mm_GRID_X*mm_GRID_Y)

#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]

int line[100000][6];
int yy = 0;

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

void
mm_kernel_cpu( float* C, float* A, float* B, int wA, int wB)
{
  // Block index
  FILE *f = fopen("hha.txt","w");
  int bx = 0;
  int by = 0;
  float As[mm_BLOCK_SIZE][mm_BLOCK_SIZE];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  float Bs[mm_BLOCK_SIZE][mm_BLOCK_SIZE];
  omp_set_num_threads(16); 
#pragma omp parallel for  // Thread index
  for(int tx =0;tx<16;tx++)
  {
    for(int ty = 0;ty<16;ty++)
    {
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


        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        AS(ty, tx) = A[a + wA * ty + tx];
        BS(ty, tx) = B[b + wB * ty + tx];
        //fprintf(f,"0 0 0 %d %d %d\n",(a-aBegin)/aStep,ty*16+tx,a+ wA * ty + tx);
        //fprintf(f,"1 0 0 %d %d %d\n",(a-aBegin)/aStep,ty*16+tx,b +wB * ty + tx);
        if((ty*16+tx)%32<4)
        {
          line[yy][0]=0;
          line[yy][1]=0;
          line[yy][2]=0;
          line[yy][3]=(a-aBegin)/aStep;
          line[yy][4]=ty*16+tx;
          line[yy][5]=a+ wA * ty + tx;
          yy++;
          line[yy][0]=0;
          line[yy][1]=0;
          line[yy][2]=0;
          line[yy][3]=(a-aBegin)/aStep;
          line[yy][4]=ty*16+tx;
          line[yy][5]=a+ wA * ty + tx;
          yy++;
        }
      }
    }
    omp_set_num_threads(16);
#pragma omp parallel for    // Synchronize to make sure the matrices are loaded
    for(int tx =0;tx<16;tx++)
    {
      for(int ty = 0;ty<16;ty++)
      {
        //int nthreads = omp_get_num_threads();
        //printf("%d thread\n",omp_get_num_threads());
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

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll
        for (int k = 0; k < mm_BLOCK_SIZE; ++k)
          Csub += AS(ty, k) * BS(k, tx);

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
      }
    }
  }

  omp_set_num_threads(16); 
#pragma omp parallel for 
  for(int tx =0;tx<16;tx++){
    for(int ty = 0;ty<16;ty++){

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
      int c = wB * mm_BLOCK_SIZE * by + mm_BLOCK_SIZE * bx;

      C[c + wB * ty + tx] = Csub;
      //fprintf(f,"2 1 0 0 %d %d\n",ty*16+tx,c+ wB * ty + tx);
      if((ty*16+tx)%32<4){
        line[yy][0]=2;
        line[yy][1]=1;
        line[yy][2]=0;
        line[yy][3]=0;
        line[yy][4]=ty*16+tx;
        line[yy][5]=c+ wB * ty + tx;
        yy++;
      }
    }
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

int main(int argc, char **argv) 
{
  //  cudaSetDevice(1);
  struct timespec t1,t2,t3,t4;
  clock_gettime(CLOCK_MONOTONIC,&t1);
  srand(2013);
  unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;

  // why 4x size for all arrays??
  uiHA = HA * iSizeMultiple;
  uiWA = WA * iSizeMultiple;

  uiHB = HB * iSizeMultiple;
  uiWB = WB * iSizeMultiple;

  uiHC = HC * iSizeMultiple;
  uiWC = WC * iSizeMultiple;

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

  // allocate host memory for the result C
  float* h_C      = (float*) malloc(mem_size_C);
  float* h_CUBLAS = (float*) malloc(mem_size_C);

  checkCudaErrors(cudaMalloc((void**) &d_A, mem_size_A));
  checkCudaErrors(cudaMalloc((void**) &d_B, mem_size_B));
  checkCudaErrors(cudaMalloc((void**) &d_C, mem_size_C));

  // copy host memory to device
  checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice) );
  checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice) );

  // cpu computation: 16 threads
  clock_gettime(CLOCK_MONOTONIC,&t3);
  mm_kernel_cpu(h_C, h_A, h_B, uiWA, uiWB);
  clock_gettime(CLOCK_MONOTONIC,&t4);
  printf("profiling time: %f\n",t4.tv_sec-t3.tv_sec+(t4.tv_nsec-t3.tv_nsec)/1.e9);

  // GPU computation
  cudaEvent_t kernel_start, kernel_stop;
  cudaEventCreate(&kernel_start);
  cudaEventCreate(&kernel_stop);
  float kernel_time = 0.0f;

  cudaEventRecord(kernel_start, 0);
  // setup execution parameters
  dim3 mm_grid(mm_GRID_X, mm_GRID_Y);
  dim3 mm_block(mm_BLOCK_SIZE, mm_BLOCK_SIZE);
  // int mm_grid=mm_GRID_X*mm_GRID_Y;
  mm_kernel<<< mm_grid, mm_block>>>(d_C, d_A, d_B, uiWA, uiWB);
  cudaDeviceSynchronize();
  cudaEventRecord(kernel_stop, 0);
  cudaEventSynchronize(kernel_stop);

  // get elapsed time
  kernel_time = 0.0f;
  cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop);
  kernel_time *= 1.e-3; // Convert to seconds

  cout << "kernel exe time: " << kernel_time << endl;
  // copy result from device to host
  checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost) );

  // compute reference solution
  float* reference = (float*)malloc(mem_size_C);
  computeGold(reference, h_A, h_B, uiHA, uiWA, uiWB);

  // check result (matrixMul)
  bool resCUDA = sdkCompareL2fe(reference, h_C, size_C, 1.0e-6f);
  printf("CUDA matrixMul compares %s\n\n", (true == resCUDA) ? "passed" : "FAIL");

  // clean up memory
  free(h_A);
  free(h_B);
  free(h_C);
  free(reference);
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));
  clock_gettime(CLOCK_MONOTONIC,&t2);
  //printf("profiling time: %f\n",t2.tv_sec-t1.tv_sec+(t2.tv_nsec-t1.tv_nsec)/1.e9);
  return 0;
}


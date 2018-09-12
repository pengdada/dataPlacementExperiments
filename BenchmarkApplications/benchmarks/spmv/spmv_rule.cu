#include "config.h"

texture<float,1,cudaReadModeElementType> tex_val;
texture<int,1,cudaReadModeElementType> tex_col;
texture<float,1,cudaReadModeElementType> tex_vec;
texture<int,1,cudaReadModeElementType> tex_row;

__global__ void 
spmv_kernel(const float* val,
                       const int    * cols,
                       const int    * rowDelimiters,
                       const float  * vec,
                       const int dim, float * out)
{
  // Thread ID in block
  int t = threadIdx.x; 
  // Thread ID within warp
  int id = t & (WARP_SIZE-1);
  int warpsPerBlock = blockDim.x / WARP_SIZE;
  // One row per warp
  int myRow = (blockIdx.x * warpsPerBlock) + (t / WARP_SIZE);
  __shared__ volatile float partialSums[BLOCK_SIZE];

  if (myRow < dim) 
  {
    int warpStart =tex1Dfetch(tex_row,myRow);
    int warpEnd = tex1Dfetch(tex_row,myRow+1);
    float mySum = 0;
    for (int j = warpStart + id; j < warpEnd; j += WARP_SIZE)
    {
      int col = tex1Dfetch(tex_col,j); 
      mySum += tex1Dfetch(tex_val,j) *tex1Dfetch(tex_vec,col);
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
  spmv_nItems = spmv_numRows * (spmv_numRows/ SFactor); 
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

  cudaBindTexture(0,tex_vec,d_spmv_vec,spmv_numRows * sizeof(float));
  cudaBindTexture(0,tex_val,d_spmv_val,spmv_nItems * sizeof(float));
  cudaBindTexture(0,tex_row,d_rowDelimiters, (spmv_numRows+1) * sizeof(int));
  cudaBindTexture(0,tex_col,d_spmv_cols,spmv_nItems * sizeof(int));  

  // Setup thread configuration
  int spmv_grid = (int) ceil(spmv_numRows / (float)(BLOCK_SIZE / WARP_SIZE));
// warm up the GPU
for(int i=0;i<5;i++)
{
  spmv_kernel <<<spmv_grid, BLOCK_SIZE>>>
  (d_spmv_val, d_spmv_cols, d_rowDelimiters, d_spmv_vec, spmv_numRows, d_spmv_out);
}

  cudaEvent_t kernel_start, kernel_stop;
  cudaEventCreate(&kernel_start);
  cudaEventCreate(&kernel_stop);
  float kernel_time = 0.0f;

  cudaEventRecord(kernel_start, 0);

  // Setup thread configuration
for(int i=0;i<ITERATIONS;i++)
{
  spmv_kernel <<<spmv_grid, BLOCK_SIZE>>>
  (d_spmv_val, d_spmv_cols, d_rowDelimiters, d_spmv_vec, spmv_numRows, d_spmv_out);
}

  cudaDeviceSynchronize();

  cudaEventRecord(kernel_stop, 0);
  cudaEventSynchronize(kernel_stop);

  // get elapsed time
  kernel_time = 0.0f;
  cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop);
  kernel_time *= 1.e-3; // Convert to seconds
  
  cout << "kernel exe time: " << kernel_time/ITERATIONS << endl;
  cudaMemcpy(h_spmv_out, d_spmv_out, spmv_numRows * sizeof(float), cudaMemcpyDeviceToHost);
//  spmv_verifyResults(spmv_refOut, h_spmv_out, spmv_numRows);

  // Don't forget to unbind texture memory
  cudaUnbindTexture(tex_vec);
  cudaUnbindTexture(tex_val);
  cudaUnbindTexture(tex_row);
  cudaUnbindTexture(tex_col);  

  return 0;
}


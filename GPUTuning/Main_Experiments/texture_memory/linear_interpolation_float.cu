/*   Linear Intepolation Demo 
 *    
 *    Copyright (C) 2012-2013 Orange Owl Solutions.  
 *
 *    This file is part of  Linear Intepolation Demo
 *    Linear Intepolation Demo is free software: you can redistribute it and/or modify
 *    it under the terms of the Lesser GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    Bluebird Library is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    Lesser GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with Linear Intepolation Demo.  If not, see <http://www.gnu.org/licenses/>.
 *
 *
 *    For any request, question or bug reporting please visit http://www.orangeowlsolutions.com/
 *    or send an e-mail to: info@orangeowlsolutions.com
 *
 *
 */

// includes, system
#include <cstdlib> 
#include <conio.h>
#include <math.h>
#include <fstream>
#include <iostream> 
#include <iomanip>

// includes, cuda 
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

texture<float, 1, cudaReadModeElementType> data_d_texture_filtering;
texture<float, 1> data_d_texture;

#define BLOCK_SIZE 256

/******************/
/* ERROR CHECKING */
/******************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) { getch(); exit(code); }
  }
}
/************/
/* LINSPACE */
/************/
// --- Generates N equally spaced, increasing points between a and b and stores them in x 
void linspace(float* x, float a, float b, int N) {
  float delta_x=(b-a)/(float)N;
  x[0]=a;
  for(int k=1;k<N;k++) x[k]=x[k-1]+delta_x;
}

/*************/
/* RANDSPACE */
/*************/
// --- Generates N randomly spaced, increasing points between a and b and stores them in x 
void randspace(float* x, float a, float b, int N) {
  float delta_x=(b-a)/(float)N;
  x[0]=a;
  for(int k=1;k<N;k++) x[k]=x[k-1]+delta_x+(((float)rand()/(float)RAND_MAX-0.5)*(1./(float)N));
}

/******************/
/* DATA GENERATOR */
/******************/
// --- Generates N complex random data points, with real and imaginary parts ranging in (0.f,1.f)
void Data_Generator(float* data, int N) {
  for(int k=0;k<N;k++) {
    data[k]=(float)rand()/(float)RAND_MAX;
  }
}

/*************************************/
/* LINEAR INTERPOLATION KERNEL - CPU */
/*************************************/
float linear_kernel_CPU(float in)
{
  float d_y;
  return 1.-abs(in);
}

/***************************************/
/* LINEAR INTERPOLATION FUNCTION - CPU */
/***************************************/
void linear_interpolation_function_CPU(float* result_GPU, float* data, float* x_in, float* x_out, int M, int N){

  float a;
  for(int j=0; j<N; j++){
    int k = floor(x_out[j]+M/2);
    a = x_out[j]+M/2-floor(x_out[j]+M/2);
    result_GPU[j] = a * data[k+1] + (-data[k] * a + data[k]);
  }	

}

/*************************************/
/* LINEAR INTERPOLATION KERNEL - GPU */
/*************************************/
__device__ float linear_kernel_GPU(float in)
{
  float d_y;
  return 1.-abs(in);
}

/**************************************************************/
/* LINEAR INTERPOLATION KERNEL FUNCTION - GPU - GLOBAL MEMORY */
/**************************************************************/
__global__ void linear_interpolation_kernel_function_GPU(float* __restrict__ result_d, const float* __restrict__ data_d, const float* __restrict__ x_out_d, const int M, const int N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;

  if(j<N)
  {
    float reg_x_out = x_out_d[j]+M/2;
    int k = __float2int_rz(reg_x_out); 
    float a = reg_x_out - truncf(reg_x_out);
    float dk = data_d[k];
    float dkp1 = data_d[k+1];
    result_d[j] = a * dkp1 + (-dk * a + dk);
  } 
}

/***************************************************************/
/* LINEAR INTERPOLATION KERNEL FUNCTION - GPU - TEXTURE MEMORY */
/***************************************************************/
__global__ void linear_interpolation_kernel_function_GPU_texture(float* __restrict__ result_d, const float* __restrict__ x_out_d, const int M, const int N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;

  if(j<N)
  {
    float reg_x_out = x_out_d[j]+M/2;
    int k = __float2int_rz(reg_x_out); 
    float a = reg_x_out - truncf(reg_x_out);
    float dk = tex1Dfetch(data_d_texture,k);
    float dkp1 = tex1Dfetch(data_d_texture,k+1);
    result_d[j] = a * dkp1 + (-dk * a + dk);
  } 
}

__global__ void linear_interpolation_kernel_function_GPU_texture_filtering(float* __restrict__ result_d, const float* __restrict__ x_out_d, const int M, const int N)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x; 
  if(j<N) result_d[j] = tex1D(data_d_texture_filtering,float(x_out_d[j]+M/2+0.5));
}

/***************************************/
/* LINEAR INTERPOLATION FUNCTION - GPU */
/***************************************/
void linear_interpolation_function_GPU(float* result_d, float* data_d, float* x_in_d, float* x_out_d, int M, int N){

  dim3 dimBlock(BLOCK_SIZE,1); dim3 dimGrid(N/BLOCK_SIZE + (N%BLOCK_SIZE == 0 ? 0:1),1);
  linear_interpolation_kernel_function_GPU<<<dimGrid,dimBlock>>>(result_d, data_d, x_out_d, M, N);
}

/*************************************************/
/* LINEAR INTERPOLATION FUNCTION - GPU - TEXTURE */
/*************************************************/
void linear_interpolation_function_GPU_texture(float* result_d, float* data_d, float* x_in_d, float* x_out_d, int M, int N){

  cudaBindTexture(NULL, data_d_texture, data_d, M*sizeof(float));

  dim3 dimBlock(BLOCK_SIZE,1); dim3 dimGrid(N/BLOCK_SIZE + (N%BLOCK_SIZE == 0 ? 0:1),1);
  linear_interpolation_kernel_function_GPU_texture<<<dimGrid,dimBlock>>>(result_d, x_out_d, M, N);
}

void linear_interpolation_function_GPU_texture_filtering(float* result_d, cudaArray* data_d_cudaArray, float* x_in_d, float* x_out_d, int M, int N){

  gpuErrchk(cudaBindTextureToArray(data_d_texture_filtering, data_d_cudaArray));
  data_d_texture_filtering.normalized = false;
  data_d_texture_filtering.filterMode = cudaFilterModeLinear;

  dim3 dimBlock(BLOCK_SIZE,1); dim3 dimGrid(N/BLOCK_SIZE + (N%BLOCK_SIZE == 0 ? 0:1),1);
  linear_interpolation_kernel_function_GPU_texture_filtering<<<dimGrid,dimBlock>>>(result_d, x_out_d, M, N);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
}

/********/
/* MAIN */
/********/
int main()
{
  // --- Number of input points
  // int M=1024*256;
  int M=1024*256;

  // --- Number of output points
  // int N=1024*256;
  int N=1024*256;

  // --- Number of computations for time measurement
  int Nit = 100;

  // --- Input sampling
  float* x_in=(float*)malloc(M*sizeof(float));
  float* x_in_d;		gpuErrchk(cudaMalloc((void**)&x_in_d,sizeof(float)*M));
  linspace(x_in,-M/2.,M/2.,M);
  gpuErrchk(cudaMemcpy(x_in_d,x_in,sizeof(float)*M,cudaMemcpyHostToDevice));

  // --- Input data
  float *data;		data=(float*)malloc((M+1)*sizeof(float));
  float* data_d;		gpuErrchk(cudaMalloc((void**)&data_d,sizeof(float)*(M+1)));
  cudaMemset(data_d,0,sizeof(float)*(M+1));
  Data_Generator(data,M);
  data[M]=0.;
  gpuErrchk(cudaMemcpy(data_d,data,sizeof(float)*M,cudaMemcpyHostToDevice));

  // --- Output sampling
  float* x_out;		x_out=(float*)malloc(N*sizeof(float));
  float* x_out_d;		gpuErrchk(cudaMalloc((void**)&x_out_d,sizeof(float)*N));
  randspace(x_out,-M/2.,M/2.,N);
  gpuErrchk(cudaMemcpy(x_out_d,x_out,sizeof(float)*N,cudaMemcpyHostToDevice));

  //cudaArray* data_d_cudaArray = NULL; gpuErrchk(cudaMallocArray (&data_d_cudaArray, &data_d_texture_filtering.channelDesc, M, 1));
  //gpuErrchk(cudaMemcpyToArray(data_d_cudaArray, 0, 0, data, sizeof(float)*M, cudaMemcpyHostToDevice));

  // --- Result allocation
  float *result_GPU;							result_GPU=(float*)malloc(N*sizeof(float));
  float *result_texture;						result_texture=(float*)malloc(N*sizeof(float));
  float *result_texture_filtering;			result_texture_filtering=(float*)malloc(N*sizeof(float));
  float *result_CPU;							result_CPU=(float*)malloc(N*sizeof(float));
  float *result_d;							gpuErrchk(cudaMalloc((void**)&result_d,sizeof(float)*N));
  float *result_d_texture;					gpuErrchk(cudaMalloc((void**)&result_d_texture,sizeof(float)*N));
  float *result_d_texture_filtering;			gpuErrchk(cudaMalloc((void**)&result_d_texture_filtering,sizeof(float)*N));

  // --- Reference interpolation result as evaluated on the CPU
  linear_interpolation_function_CPU(result_CPU, data, x_in, x_out, M, N);

  float time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  for (int k=0; k<Nit; k++) linear_interpolation_function_GPU(result_d, data_d, x_in_d, x_out_d, M, N);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cout << "GPU Global memory [ms]: " << setprecision (10) << time/Nit << endl;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  //for (int k=0; k<Nit; k++) linear_interpolation_function_GPU_texture_filtering(result_d_texture_filtering, data_d_cudaArray, x_in_d, x_out_d, M, N);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cout << "GPU Texture filtering [ms]: " << setprecision (10) << time/Nit << endl;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  for (int k=0; k<Nit; k++) linear_interpolation_function_GPU_texture(result_d_texture, data_d, x_in_d, x_out_d, M, N);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cout << "GPU Texture [ms]: " << setprecision (10) << time/Nit << endl;

  gpuErrchk(cudaMemcpy(result_GPU,result_d,sizeof(float)*N,cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(result_texture_filtering,result_d_texture_filtering,sizeof(float)*N,cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(result_texture,result_d_texture,sizeof(float)*N,cudaMemcpyDeviceToHost));

  float diff_norm=0.f, norm=0.f;
  for(int j=0; j<N; j++) {
    diff_norm = diff_norm + (result_CPU[j]-result_GPU[j])*(result_CPU[j]-result_GPU[j]);
    norm      = norm      + result_CPU[j]*result_CPU[j];
    //printf("%f %f\n",result_CPU[j],result_GPU[j]);
  }
  printf("Error GPU [percentage] = %f\n",100.*sqrt(diff_norm/norm));

  float diff_norm_texture_filtering=0.f;
  for(int j=0; j<N; j++) {
    diff_norm_texture_filtering = diff_norm_texture_filtering + (result_CPU[j]-result_texture_filtering[j])*(result_CPU[j]-result_texture_filtering[j]);
  }
  printf("Error texture filtering [percentage] = %f\n",100.*sqrt(diff_norm_texture_filtering/norm));

  float diff_norm_texture=0.f;
  for(int j=0; j<N; j++) {
    diff_norm_texture = diff_norm_texture + (result_CPU[j]-result_texture[j])*(result_CPU[j]-result_texture[j]);
  }
  printf("Error texture [percentage] = %f\n",100.*sqrt(diff_norm_texture/norm));

  cudaDeviceReset();

  ofstream outfile;
  outfile.open("x_in.dat", ios::out | ios::binary);							for(int i=0; i<M; i++){ outfile.write( (char*)&x_in[i], sizeof(float)); } outfile.close();
  outfile.open("data.dat", ios::out | ios::binary);							for(int i=0; i<M; i++){ outfile.write( (char*)&data[i], sizeof(float)); } outfile.close();
  outfile.open("x_out.dat", ios::out | ios::binary);							for(int i=0; i<N; i++){ outfile.write( (char*)&x_out[i], sizeof(float)); } outfile.close();
  outfile.open("result_texture_filtering.dat", ios::out | ios::binary);		for(int i=0; i<N; i++){ outfile.write( (char*)&result_texture_filtering[i], sizeof(float)); } outfile.close();
  outfile.open("result_texture.dat", ios::out | ios::binary);					for(int i=0; i<N; i++){ outfile.write( (char*)&result_texture[i], sizeof(float)); } outfile.close();
  outfile.open("result_GPU.dat", ios::out | ios::binary);						for(int i=0; i<N; i++){ outfile.write( (char*)&result_GPU[i], sizeof(float)); } outfile.close();
  outfile.open("result_CPU.dat", ios::out | ios::binary);						for(int i=0; i<N; i++){ outfile.write( (char*)&result_CPU[i], sizeof(float)); } outfile.close();

  getch();

  return 0;
}



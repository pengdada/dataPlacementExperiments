/* Another version of MM using shared memory: more straightforward 
*
* multShare.h
*
* Robert Hochberg
* January 24, 2012
*
* Based nearly entirely on the code from the CUDA C Programming Guide
* Updated Liao, 2018/4/18
*   error checking with gold standard (serial version)
*/
#include <fstream>

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>

#include "nvmlpower.hpp"
#include "cupti_profiler.h"

#include "parameters.h"




typedef double REAL;
void HandleError( cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
        file, line );
    exit( EXIT_FAILURE );
  }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
  int width;
  int height;
  REAL* elements;
  int stride; // the same value as width: each +1 for row pointer, how many elements it swipes 
              // for sub matrix: use the original whole matrix's stride
} Matrix;


/* 
 * @Abdullah, Naming the configuration used
 * This is just for testing purpose, so coding a little crude
 */
char* SetUpConfigurationName(int argc, char* argv[])
{
    int i;
    char *conf;
    conf = (char *)malloc(sizeof(*conf) * 512);
    /* Set up the path for the final_aggregate file*/
    strcpy(conf, argv[0]);

    for(i=1; i<argc; i++)
    {       
        strcat(conf, "_");
        strcat(conf, argv[i]);
    }
    return conf;
}
/***************************/

REAL sumOfSquareOfElements (const REAL *data, int rows, int cols )
{
  double ref=0.0; 
  for (unsigned int i = 0; i < rows*cols; ++i)
    ref += data[i] * data[i];
  return (REAL)ref; 
}

/* 
   compute the gold standard: the reference results using the naive implementation 
   C[hA][wB] = A[hA][wA]xB[hB][wB]
   All matrices are linearized. 
 */
void
computeGold(REAL* C, const REAL* A, const REAL* B, int hA, int wA, int wB)
{
  // outer i, j: for C[i][j]
  for (int i = 0; i < hA; ++i)
    for (int j = 0; j < wB; ++j) {
      double sum = 0.0;
      // inner k: iterate wA and hB
      for (int k = 0; k < wA; ++k) {
        double a = A[i * wA + k];
        double b = B[k * wB + j];
        sum += a * b;
      }
      C[i * wB + j] = (REAL)sum;
    }
}

/* compare data to reference based on L^2 norm ratio of diff/ref */
bool sdkCompareL2fe(const REAL *reference, const REAL *data,
               const unsigned int len, const REAL epsilon, REAL* diffRatio)
{
  assert(epsilon >= 0);
  REAL error = 0;
  REAL ref = 0;

  // step 1: L^2 norm of the difference vector
  // error = sum of (ref[i]-data[i])^2 for all element i
  // normError = sqrt (error)

  // step 2: L^2 norm of the reference vector
  // ref = sum of (ref[i])^2 for all element i
  // normRef = sqrt(ref)

  // step 3: ratio of two norms
  // error = normError/normRef

  for (unsigned int i = 0; i < len; ++i)
  {
    REAL diff = reference[i] - data[i];
    error += diff * diff;
    ref += reference[i] * reference[i];
  }

//  printf ("sdkCompareL2fe() square sum for error =%g, ref =%g\n", error, ref);

  // power 2 sum of references are too small ? 
  if (fabs(ref) < 1e-7)
  {
    printf ("sdkCompareL2fe() Warning: ref square sum is too small, return false\n");
    return false;
  }

  REAL normRef = sqrtf(ref);
  REAL normError = sqrtf(error);

  error = normError / normRef;
  if (diffRatio!=NULL) 
    *diffRatio = error; 

  bool result = error < epsilon;
  return result;
}

// Get a matrix element: universal for both original and sub-matrix
__device__ REAL GetElement(const Matrix A, int row, int col) {
  return A.elements[row * A.stride + col];
}

// Set a matrix element: universal for both original and sub-matrix
__device__ void SetElement(Matrix A, int row, int col, REAL value) {
  A.elements[row * A.stride + col] = value;
}

// use row, col to retrieve a submatrix: row = 0 to origin_row/blocksize
// Get the (BLOCK_SIZE x BLOCK_SIZE) sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col) {
  Matrix Asub;
  Asub.width = BLOCK_SIZE;
  Asub.height = BLOCK_SIZE;
  Asub.stride = A.stride;
  Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
  return Asub;
}

// Matrix multiplication kernel on gpu: 2-D thread block mapping
// retrieve submatrices for current thread block for calculation
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) 
{
  // Block row and column ids for the current thread
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;

  // Each thread block computes one sub-matrix Csub of C
  Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
  // Each thread computes one element of Csub
  // by accumulating results into Cvalue
  REAL Cvalue = 0.0;

  // Thread row and column within Csub
  int row = threadIdx.y;
  int col = threadIdx.x;

  // Loop over all the sub-matrices of A and B that are
  // required to compute Csub
  // Multiply each pair of sub-matrices together
  // and accumulate the results
  for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) 
  {
    // Get sub-matrix Asub of A
    Matrix Asub = GetSubMatrix(A, blockRow, m);
    // Get sub-matrix Bsub of B
    Matrix Bsub = GetSubMatrix(B, m, blockCol);

    // Shared memory used to store Asub and Bsub respectively
    __shared__ REAL As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ REAL Bs[BLOCK_SIZE][BLOCK_SIZE];
    // Load Asub and Bsub from device memory to shared memory
    // Each thread loads one element of each sub-matrix
    As[row][col] = GetElement(Asub, row, col);
    Bs[row][col] = GetElement(Bsub, row, col);
    // Synchronize to make sure the sub-matrices are loaded
    // before starting the computation
    __syncthreads();

    // Multiply Asub and Bsub together
    for (int e = 0; e < BLOCK_SIZE; ++e)
      Cvalue += As[row][e] * Bs[e][col];
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }
  // Write Csub to device memory
  // Each thread writes one element
  SetElement(Csub, row, col, Cvalue);
}

int main(int argc, char* argv[])
{

  
  // cuda events
  cudaEvent_t kernel_start, kernel_stop;
  cudaEventCreate(&kernel_start);
  cudaEventCreate(&kernel_stop);

  Matrix A, B, C;
  int a1, a2, b1, b2;
  a1 = MSIZE; // atoi(argv[1]); /* Height of A */
  a2 = MSIZE; // atoi(argv[2]); /* Width of A */
  b1 = a2; /* Height of B */
  b2 = MSIZE; //atoi(argv[3]); /* Width of B */

  srand48(time(NULL));


  /* @Abdullah */
    std::vector<std::string> event_names;
    std::vector<std::string> metric_names;
    metric_names.push_back(argv[1]);
    cupti_profiler::profiler profiler(event_names, metric_names);
    // Get #passes required to compute all metrics and events
    const int passes = profiler.get_passes();
    /* @NVML-power */
    setUpTuningParams(argc, argv);
    /*************/

  // allocate and initialize the matrices
  printf("Allocating A, B, C ... size=%d thread blocksize=%d\n",MSIZE, BLOCK_SIZE);
  A.height = a1;
  A.width = a2;
  A.elements = (REAL*)malloc(A.width * A.height * sizeof(REAL));
  if (A.elements ==NULL)
    printf ("malloc() fails for A\n");

  B.height = b1;
  B.width = b2;
  B.elements = (REAL*)malloc(B.width * B.height * sizeof(REAL));
  if (B.elements ==NULL)
    printf ("malloc() fails for B\n");

  C.height = A.height;
  C.width = B.width;
  C.elements = (REAL*)malloc(C.width * C.height * sizeof(REAL));
  if (C.elements ==NULL)
    printf ("malloc() fails for C\n");

  // initialize A and B
  printf("Initializing A, B ...\n");
  for(int i = 0; i < A.height; i++)
    for(int j = 0; j < A.width; j++)
      A.elements[i*A.width + j] = 5*i * drand48();   // (arc4random() % 3);

  for(int i = 0; i < B.height; i++)
    for(int j = 0; j < B.width; j++)
      B.elements[i*B.width + j] = 7*j* drand48(); // (arc4random() % 2);

  //  printf ("After initialization: \n \tdebug A squre = %g, B square sum=%g\n", 
  //       sumOfSquareOfElements (A.elements, A.height, A.width),  sumOfSquareOfElements (B.elements, B.height, B.width));
  // call the kernel

  // Load A and B to device memory
  printf(" Allocate and copy device versions of A, B ...\n");
  Matrix d_A;
  d_A.width = d_A.stride = A.width;
  d_A.height = A.height;
  size_t size = A.width * A.height * sizeof(REAL);
  HANDLE_ERROR( cudaMalloc(&d_A.elements, size));
  HANDLE_ERROR (cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice));

  Matrix d_B;
  d_B.width = d_B.stride = B.width;
  d_B.height = B.height;
  size = B.width * B.height * sizeof(REAL);
  HANDLE_ERROR(cudaMalloc(&d_B.elements, size));
  HANDLE_ERROR(cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice));

  // Allocate C in device memory
  Matrix d_C;
  d_C.width = d_C.stride = C.width;
  d_C.height = C.height;
  size = C.width * C.height * sizeof(REAL);
  HANDLE_ERROR(cudaMalloc(&d_C.elements, size));

  // Invoke kernel
  //-----------------------------
  printf("Invoking kernel 10 times ...block_size=%d\n",BLOCK_SIZE);
  // start the event  
  cudaEventRecord(kernel_start, 0);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  // row, col  -> y, x // a bit reversed, ok if consistent
  dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);

      /* 
     * @Abdullah, used this to remove the side effect of 
     * overlapping computation and communication in data placement efficency
     */
    profiler.start();
    HANDLE_ERROR(  cudaDeviceSynchronize()  );

    for(int i=0; i<passes; ++i) 
    {
        // kernel launch
        nvmlAPIRun();
        MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

        HANDLE_ERROR(  cudaDeviceSynchronize()  );
        nvmlAPIEnd();
    }
    profiler.stop();

    profiler.print_metric_values_to_file("output.csv", SetUpConfigurationName(argc, argv));

    /*******************************/
  //  HANDLE_ERROR(cudaThreadSynchronize());
  HANDLE_ERROR(cudaDeviceSynchronize());

  // stop the event  
  cudaEventRecord(kernel_stop, 0);
  cudaEventSynchronize(kernel_stop);
  nvmlAPIEnd();

  // get elapsed time from the start/end of events
  float kernel_time = 0.0;
  cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop);
  kernel_time *= 1.e-3; // Convert to seconds
  // printf( "kernel exe time x10 :%f \n", kernel_time);
  printf( "%f\n", kernel_time);

  std::ofstream ofs ("exec-time.txt", std::ofstream::out | std::ofstream::app);
  ofs << argv[0] << "," << argv[1] << "," << argv[2] << "," << kernel_time << std::endl;
  ofs.close();

  // Read C from device memory
  // printf("Copy C off of device ...\n");
  HANDLE_ERROR(cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost));

  //verify the results  

  // compute reference solution from A and B
  // printf("Compute reference result ...\n");
  REAL* reference = (REAL*)malloc(C.height* C.width*sizeof(REAL));
  if (reference==NULL)
    printf ("malloc() fails for reference\n");
  //  printf ("Before compuateGold: \n \tdebug A squre = %g, B square sum=%g\n", 
  //       sumOfSquareOfElements (A.elements, A.height, A.width),  sumOfSquareOfElements (B.elements, B.height, B.width));
  computeGold(reference, A.elements, B.elements, A.height, A.width, B.width);

  // check result (matrixMul): reference vs. C
  // printf("Check correctness ...\n");
  REAL diffRatio=0.0; 
  bool resCUDA = sdkCompareL2fe(reference, C.elements, C.height* C.width, 1.0e-6f, &diffRatio);
  // printf("CUDA matrixMul diff ratio=%g %s\n\n", diffRatio, (true == resCUDA) ? "passed" : "FAIL");

  // Free device memory
  HANDLE_ERROR(cudaFree(d_A.elements));
  HANDLE_ERROR(cudaFree(d_B.elements));
  HANDLE_ERROR(cudaFree(d_C.elements));

  // Free host memory
  free(A.elements);
  free(B.elements);
  free(C.elements);
  free(reference);
  return 0;
}


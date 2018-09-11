/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

#include <fstream>
#include "cuda.h"
#include "../common/book.h"
#include "../common/cpu_anim.h"

#include "nvmlpower.hpp"
#include "cupti_profiler.h"
#include "parameters.h"

#if 0
// these exist on the GPU side
texture<float>  texConstSrc;
texture<float>  texIn;
texture<float>  texOut;
#endif

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


// this kernel takes in a 2-d array of floats
// it updates the value-of-interest by a scaled value based
// on itself and its nearest neighbors
__global__ void blend_kernel( float *dst,  bool dstOut, float *dev_in, float* dev_out ) {
  // map from threadIdx/BlockIdx to pixel position
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  int left = offset - 1;
  int right = offset + 1;
  if (x == 0)   left++;
  if (x == DIM-1) right--; 

  int top = offset - DIM;
  int bottom = offset + DIM;
  if (y == 0)   top += DIM;
  if (y == DIM-1) bottom -= DIM;

  float   t, l, c, r, b;

  if (dstOut) {
    t = dev_in[top]; //tex1Dfetch(texIn,top);
    l = dev_in[left]; //tex1Dfetch(texIn,left);
    c = dev_in[offset]; //tex1Dfetch(texIn,offset);
    r = dev_in[right]; //tex1Dfetch(texIn,right);
    b = dev_in[bottom]; //tex1Dfetch(texIn,bottom);
  } else {
    t = dev_out[top]; //tex1Dfetch(texOut,top);
    l = dev_out[left]; //tex1Dfetch(texOut,left);
    c = dev_out[offset]; //tex1Dfetch(texOut,offset);
    r = dev_out[right]; //tex1Dfetch(texOut,right);
    b = dev_out[bottom]; //tex1Dfetch(texOut,bottom);
  }

  dst[offset] = c + SPEED * (t + b + r + l - 4 * c);
}

// NOTE - texOffsetConstSrc could either be passed as a
// parameter to this function, or passed in __constant__ memory
// if we declared it as a global above, it would be
// a parameter here: 
// __global__ void copy_const_kernel( float *iptr,
//                                    size_t texOffset )
__global__ void copy_const_kernel( float *iptr , float *dev_const) {
  // map from threadIdx/BlockIdx to pixel position
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  float c = dev_const[offset]; //tex1Dfetch(texConstSrc,offset);
  if (c != 0)
    iptr[offset] = c;
}

// globals needed by the update routine
struct DataBlock {
  unsigned char   *output_bitmap;
  float           *dev_inSrc;
  float           *dev_outSrc;
  float           *dev_constSrc;
  CPUAnimBitmap  *bitmap;

  cudaEvent_t     start, stop;
  float           totalTime;
  float           frames;
};

void anim_gpu( DataBlock *d, int ticks ) {
  HANDLE_ERROR( cudaEventRecord( d->start, 0 ) );
  dim3    blocks(DIM/BLOCKSIZE,DIM/BLOCKSIZE);
  dim3    threads(BLOCKSIZE,BLOCKSIZE);
  CPUAnimBitmap  *bitmap = d->bitmap;

  /* 
  * @Abdullah, used this to remove the side effect of 
  * overlapping computation and communication in data placement efficency
  */
  HANDLE_ERROR(  cudaDeviceSynchronize()  );
  nvmlAPIRun();
  // since tex is global and bound, we have to use a flag to
  // select which is in/out per iteration
  volatile bool dstOut = true;
  for (int i=0; i<REPEAT; i++) {
    float   *in, *out;
    if (dstOut) {
      in  = d->dev_inSrc;
      out = d->dev_outSrc;
    } else {
      out = d->dev_inSrc;
      in  = d->dev_outSrc;
    }
    copy_const_kernel<<<blocks,threads>>>( in , d->dev_constSrc);
    blend_kernel<<<blocks,threads>>>( out, dstOut, d->dev_inSrc, d->dev_outSrc);
    dstOut = !dstOut;
  }

  float_to_color<<<blocks,threads>>>( d->output_bitmap,
      d->dev_inSrc );

  /* @Abdullah */
  HANDLE_ERROR(  cudaDeviceSynchronize()  );
  /* @Abdullah, Moved this part before the cudaMemCPY to make sure we only consider GPU time*/
  HANDLE_ERROR( cudaEventRecord( d->stop, 0 ) );
  HANDLE_ERROR( cudaEventSynchronize( d->stop ) );
  /* @Abdullah */
  nvmlAPIEnd();

  HANDLE_ERROR( cudaMemcpy( bitmap->get_ptr(),
	d->output_bitmap,
	bitmap->image_size(),
	cudaMemcpyDeviceToHost ) );

  float   elapsedTime;
  HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
	d->start, d->stop ) );
  d->totalTime += elapsedTime;
  ++d->frames;
  /*printf( "Average Time per frame:  %3.1f ms\n",
      d->totalTime/d->frames  );*/
  printf( "%3.1f\n", d->totalTime/d->frames  );
}

// clean up memory allocated on the GPU
void anim_exit( DataBlock *d ) {

#if 0
  cudaUnbindTexture( texIn );
  cudaUnbindTexture( texOut );
  cudaUnbindTexture( texConstSrc );
#endif

  HANDLE_ERROR( cudaFree( d->dev_inSrc ) );
  HANDLE_ERROR( cudaFree( d->dev_outSrc ) );
  HANDLE_ERROR( cudaFree( d->dev_constSrc ) );

  HANDLE_ERROR( cudaEventDestroy( d->start ) );
  HANDLE_ERROR( cudaEventDestroy( d->stop ) );
}


int main( int argc, char* argv[] ) {
  DataBlock   data;
  CPUAnimBitmap bitmap( DIM, DIM, &data );
  data.bitmap = &bitmap;
  data.totalTime = 0;
  data.frames = 0;

  printf ("Execution parameters:\n");
  printf ("\tDIM:%d\n", DIM);
  printf ("\tBLOCKSIZE:%d\n", BLOCKSIZE);
  printf ("\tREPEAT:%d\n", REPEAT);

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

  HANDLE_ERROR( cudaEventCreate( &data.start ) );
  HANDLE_ERROR( cudaEventCreate( &data.stop ) );

  int imageSize = bitmap.image_size();

  HANDLE_ERROR( cudaMalloc( (void**)&data.output_bitmap,
	imageSize ) );

  // assume float == 4 chars in size (ie rgba)
  HANDLE_ERROR( cudaMalloc( (void**)&data.dev_inSrc,
	imageSize ) );
  HANDLE_ERROR( cudaMalloc( (void**)&data.dev_outSrc,
	imageSize ) );
  HANDLE_ERROR( cudaMalloc( (void**)&data.dev_constSrc,
	imageSize ) );

#if 0  
  HANDLE_ERROR( cudaBindTexture( NULL, texConstSrc,
	data.dev_constSrc,
	imageSize ) );

  HANDLE_ERROR( cudaBindTexture( NULL, texIn,
	data.dev_inSrc,
	imageSize ) );

  HANDLE_ERROR( cudaBindTexture( NULL, texOut,
	data.dev_outSrc,
	imageSize ) );
#endif

  // intialize the constant data
  float *temp = (float*)malloc( imageSize );
  for (int i=0; i<DIM*DIM; i++) {
    temp[i] = 0;
    int x = i % DIM;
    int y = i / DIM;
    if ((x>300) && (x<600) && (y>310) && (y<601))
      temp[i] = MAX_TEMP;
  }
  temp[DIM*100+100] = (MAX_TEMP + MIN_TEMP)/2;
  temp[DIM*700+100] = MIN_TEMP;
  temp[DIM*300+300] = MIN_TEMP;
  temp[DIM*200+700] = MIN_TEMP;
  for (int y=800; y<900; y++) {
    for (int x=400; x<500; x++) {
      temp[x+y*DIM] = MIN_TEMP;
    }
  }
  HANDLE_ERROR( cudaMemcpy( data.dev_constSrc, temp,
	imageSize,
	cudaMemcpyHostToDevice ) );    

  // initialize the input data
  for (int y=800; y<DIM; y++) {
    for (int x=0; x<200; x++) {
      temp[x+y*DIM] = MAX_TEMP;
    }
  }
  HANDLE_ERROR( cudaMemcpy( data.dev_inSrc, temp,
	imageSize,
	cudaMemcpyHostToDevice ) );
  free( temp );

#ifdef NO_OPENCL  
  static int ticks = 1;
  /* 
     * @Abdullah, used this to remove the side effect of 
     * overlapping computation and communication in data placement efficency
     */
    profiler.start();
    HANDLE_ERROR(  cudaDeviceSynchronize()  );

    for(int i=0; i<passes; ++i) 
    {
        nvmlAPIRun();
        anim_gpu( &data, ticks++ );
        nvmlAPIEnd();
    }
    profiler.stop();

    profiler.print_metric_values_to_file("output.csv", SetUpConfigurationName(argc, argv));

/*******************************/
  anim_exit(&data );
#else
  bitmap.anim_and_exit( (void (*)(void*,int))anim_gpu,
      (void (*)(void*))anim_exit );
#endif
  /* @Abdullah */
  std::ofstream ofs ("exec-time.txt", std::ofstream::out | std::ofstream::app);
  ofs << argv[0] << "," << argv[1] << "," << argv[2] << "," << data.totalTime/data.frames << std::endl;
  ofs.close();
  return 0;
}

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
#include "parameters.h"


#if 0
// these exist on the GPU side
// 2 dimension float type texture array
texture<float,2>  texConstSrc;
texture<float,2>  texIn;
texture<float,2>  texOut;
#endif

// dstOut: a flag to swap Input and Output arrays
__global__ void blend_kernel( float *dst, bool dstOut , float* dev_inSrc, float* dev_outSrc) {
  // map from threadIdx/BlockIdx to pixel position
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  // retrieve 5 point stencil
  float   t, l, c, r, b;
#if 0
  if (dstOut) {
    t = dev_inSrc[x*DIM + y-1]; //tex2D(texIn,x,y-1);
    l = dev_inSrc[(x-1)*DIM + y]; //tex2D(texIn,x-1,y);
    c = dev_inSrc[x*DIM + y]; //tex2D(texIn,x,y);
    r = dev_inSrc[(x+1)*DIM + y]; //tex2D(texIn,x+1,y);
    b = dev_inSrc[x*DIM +y+1]; //tex2D(texIn,x,y+1);
  } else {
    t = dev_outSrc[x*DIM + y-1]; //tex2D(texOut,x,y-1);
    l = dev_outSrc[(x-1)*DIM + y]; //tex2D(texOut,x-1,y);
    c = dev_outSrc[x*DIM + y]; //tex2D(texOut,x,y);
    r = dev_outSrc[(x+1)*DIM + y]; //tex2D(texOut,x+1,y);
    b = dev_outSrc[x*DIM +y+1]; //tex2D(texOut,x,y+1);
  }
#endif  
  // inversed x,y !!
  if (dstOut) {
    t = dev_inSrc[x +DIM * (y-1)]; //tex2D(texIn,x,y-1);
    l = dev_inSrc[(x-1) +DIM * y]; //tex2D(texIn,x-1,y);
    c = dev_inSrc[x+DIM * y]; //tex2D(texIn,x,y);
    r = dev_inSrc[(x+1)+ DIM * y]; //tex2D(texIn,x+1,y);
    b = dev_inSrc[x+DIM *(y+1)]; //tex2D(texIn,x,y+1);
  } else {
    t = dev_outSrc[x + DIM * (y-1)]; //tex2D(texOut,x,y-1);
    l = dev_outSrc[(x-1)+ DIM * y]; //tex2D(texOut,x-1,y);
    c = dev_outSrc[x+DIM * y]; //tex2D(texOut,x,y);
    r = dev_outSrc[(x+1)+DIM * y]; //tex2D(texOut,x+1,y);
    b = dev_outSrc[x+DIM *(y+1)]; //tex2D(texOut,x,y+1);
  }

  // stencil calculation
  dst[offset] = c + SPEED * (t + b + r + l - 4 * c);
}

// copy nonzero const dev_constSrc  to another array iptr
__global__ void copy_const_kernel( float *iptr, float* dev_constSrc) {
  // map from threadIdx/BlockIdx to pixel position
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  float c = dev_constSrc[x+DIM*y] ; // tex2D(texConstSrc,x,y);
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

void anim_gpu( DataBlock *d, int ticks) {
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
 // iterate  steps 
  for (int i=0; i<REPEAT; i++) {
    float   *in, *out;
    if (dstOut) {
      in  = d->dev_inSrc;
      out = d->dev_outSrc;

    } else {
      out = d->dev_inSrc;
      in  = d->dev_outSrc;
    }

    copy_const_kernel<<<blocks,threads>>>( in, d->dev_constSrc );
    blend_kernel<<<blocks,threads>>>( out, dstOut, d->dev_inSrc, d->dev_outSrc );
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

  printf ("DEBUG: bitmap->get_ptr() =%p\n", bitmap->get_ptr()); 
  printf ("DEBUG: d->output_bitmap =%p\n", d->output_bitmap); 
  printf ("DEBUG: bitmap->image_size() =%d\n", bitmap->image_size()); 
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
  /* @Abdullah */
  printf( "%3.1fn", d->totalTime/d->frames  );
  
}

// clean up memory allocated on the GPU
void anim_exit( DataBlock *d ) {
#if 0
  cudaUnbindTexture( texIn );
  cudaUnbindTexture( texOut );
  cudaUnbindTexture( texConstSrc );
#endif  
  HANDLE_ERROR( cudaFree( d->output_bitmap) );
  HANDLE_ERROR( cudaFree( d->dev_inSrc ) );
  HANDLE_ERROR( cudaFree( d->dev_outSrc ) );
  HANDLE_ERROR( cudaFree( d->dev_constSrc ) );

  HANDLE_ERROR( cudaEventDestroy( d->start ) );
  HANDLE_ERROR( cudaEventDestroy( d->stop ) );
}


int main( int argc, char* argv[]) {
  printf ("%s Execution parameters:\n", argv[0]);
  printf ("\tDIM:%d\n", DIM);
  printf ("\tBLOCKSIZE:%d\n", BLOCKSIZE);
  printf ("\tREPEAT:%d\n", REPEAT);

  DataBlock   data;
  CPUAnimBitmap bitmap( DIM, DIM, &data );
  data.bitmap = &bitmap;
  data.totalTime = 0;
  data.frames = 0;

  /* @NVML-power */
  setUpTuningParams(argc, argv);

  HANDLE_ERROR( cudaEventCreate( &data.start ) );
  HANDLE_ERROR( cudaEventCreate( &data.stop ) );

  int imageSize = bitmap.image_size();

  HANDLE_ERROR( cudaMalloc( (void**)&data.output_bitmap,
	imageSize ) );

  // assume float == 4 chars in size (ie rgba)
  // Allocate device memory for in, out, and const arrays
  HANDLE_ERROR( cudaMalloc( (void**)&data.dev_inSrc,
	imageSize ) );
  HANDLE_ERROR( cudaMalloc( (void**)&data.dev_outSrc,
	imageSize ) );
  HANDLE_ERROR( cudaMalloc( (void**)&data.dev_constSrc,
	imageSize ) );

#if 0  // no more binding texture
  cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
  HANDLE_ERROR( cudaBindTexture2D( NULL, texConstSrc,
	data.dev_constSrc,
	desc, DIM, DIM,
	sizeof(float) * DIM ) );

  HANDLE_ERROR( cudaBindTexture2D( NULL, texIn,
	data.dev_inSrc,
	desc, DIM, DIM,
	sizeof(float) * DIM ) );

  HANDLE_ERROR( cudaBindTexture2D( NULL, texOut,
	data.dev_outSrc,
	desc, DIM, DIM,
	sizeof(float) * DIM ) );
#endif 
  // initialize the constant data by copying temp[], i, j are reversed!!
  float *temp = (float*)malloc( imageSize );
  if(temp== NULL)
  {
    fprintf(stderr, "out of memory\n");
    abort();
  }

  for (int i=0; i<DIM*DIM; i++) {
    temp[i] = 0;
    int x = i % DIM; // x is col number
    int y = i / DIM;  // y is the row number,  reversed !!
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
// a version without depending on OpenCL library  
#ifdef NO_OPENCL  
  static int ticks = 1;
  anim_gpu( &data, ticks++ ) ;
  anim_exit(&data );
#else
// not interested in graphics display  
  bitmap.anim_and_exit( (void (*)(void*,int))anim_gpu, (void (*)(void*))anim_exit );
#endif
  /* @Abdullah */
  std::ofstream ofs ("exec-time.txt", std::ofstream::out | std::ofstream::app);
  ofs << argv[0] << "," << argv[1] << "," << argv[2] << "," << data.totalTime/data.frames << std::endl;
  ofs.close();
  return 0;
}


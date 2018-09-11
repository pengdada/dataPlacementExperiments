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
#include "../common/cpu_bitmap.h"

#include "parameters.h"
#include "nvmlpower.hpp"
#include "cupti_profiler.h"

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF     2e10f

// memory footprint: 7 float variables x 4 bytes= 28 bytes
struct Sphere {
    float   r,b,g;
    float   radius;
    float   x,y,z;
    __device__ float hit( float ox, float oy, float *n ) {
        float dx = ox - x;
        float dy = oy - y;
        if (dx*dx + dy*dy < radius*radius) {
            float dz = sqrtf( radius*radius - dx*dx - dy*dy );
            *n = dz / sqrtf( radius * radius );
            return dz + z;
        }
        return -INF;
    }
};

//max SPHERES = 64k/28bytes = 2340

/*
 * @Abdullah, defining the Sphere array for different types of memory
 * The options will be passed via compilation commands
 * 1 = shared mem
 * 2 = constant mem
 * 3 = device/global mem
 */

#if MEMTYPE == 2
    __constant__ Sphere s[SPHERES];
#elif MEMTYPE == 3
    __device__ Sphere s[SPHERES];
#else
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

__global__ void kernel( unsigned char *ptr ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    float   ox = (x - DIM/2);
    float   oy = (y - DIM/2);

    float   r=0, g=0, b=0;
    float   maxz = -INF;

    for(int i=0; i<SPHERES; i++) {
        float   n;
        float   t = s[i].hit( ox, oy, &n );
        if (t > maxz) {
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
            maxz = t;
        }
    } 

    ptr[offset*4 + 0] = (int)(r * 255);
    ptr[offset*4 + 1] = (int)(g * 255);
    ptr[offset*4 + 2] = (int)(b * 255);
    ptr[offset*4 + 3] = 255;
}

// globals needed by the update routine
struct DataBlock {
    unsigned char   *dev_bitmap;
};

int main( int argc, char* argv[] ) {
    DataBlock   data;
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
    // capture the start time
    cudaEvent_t     start, stop;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    

    CPUBitmap bitmap( DIM, DIM, &data );
    unsigned char   *dev_bitmap;
    printf ("DIM=%d, SPHERES=%d\n",DIM, SPHERES);
    printf ("BitMap size=%ld, SPHERES size=%lu\n", bitmap.image_size() , sizeof(Sphere) * SPHERES );


    // allocate memory on the GPU for the output bitmap
    HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap,
                              bitmap.image_size() ) );

    // allocate temp memory, initialize it, copy to constant
    // memory on the GPU, then free our temp memory
    Sphere *temp_s = (Sphere*)malloc( sizeof(Sphere) * SPHERES );
    for (int i=0; i<SPHERES; i++) {
        temp_s[i].r = rnd( 1.0f );
        temp_s[i].g = rnd( 1.0f );
        temp_s[i].b = rnd( 1.0f );
        temp_s[i].x = rnd( 1000.0f ) - 500;
        temp_s[i].y = rnd( 1000.0f ) - 500;
        temp_s[i].z = rnd( 1000.0f ) - 500;
        temp_s[i].radius = rnd( 100.0f ) + 20;
    }
    HANDLE_ERROR( cudaMemcpyToSymbol( s, temp_s, 
                                sizeof(Sphere) * SPHERES) );
    free( temp_s );

    // generate a bitmap from our sphere data
    dim3    grids(DIM/BLOCKSIZE,DIM/BLOCKSIZE);
    dim3    threads(BLOCKSIZE, BLOCKSIZE);

    /* 
     * @Abdullah, used this to remove the side effect of 
     * overlapping computation and communication in data placement efficency
     */
    HANDLE_ERROR(  cudaDeviceSynchronize()  );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );

    

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
        kernel<<<grids,threads>>>( dev_bitmap );
        HANDLE_ERROR(  cudaDeviceSynchronize()  );
        nvmlAPIEnd();
    }
    profiler.stop();

    profiler.print_metric_values_to_file("output.csv", SetUpConfigurationName(argc, argv));

    /*******************************/

    HANDLE_ERROR(  cudaDeviceSynchronize()  );
    HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( stop ) );
    nvmlAPIEnd();
    // copy our bitmap back from the GPU for display
    HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), dev_bitmap,
                              bitmap.image_size(),
                              cudaMemcpyDeviceToHost ) );

    // get stop time, and display the timing results
    
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
                                        start, stop ) );
    // printf( "Time to generate:  %3.1f ms\n", elapsedTime );
    printf( "%3.1f\n", elapsedTime );

    std::ofstream cupti_ofs ("exec-time.txt", std::ofstream::out | std::ofstream::app);
    cupti_ofs << argv[0] << "," << argv[1] << "," << argv[2] << "," << argv[3] << "," << argv[4] << "," << elapsedTime << std::endl;
    cupti_ofs.close();

    HANDLE_ERROR( cudaEventDestroy( start ) );
    HANDLE_ERROR( cudaEventDestroy( stop ) );

    HANDLE_ERROR( cudaFree( dev_bitmap ) );

    // display, has errors,comment out , Liao
    //bitmap.display_and_exit();
}


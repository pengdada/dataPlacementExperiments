// Liao 11/30/2011
// based on liao6@tux322:/usr/local/cuda/include/cuda_runtime_api.h
#include <stdio.h>

int main()
{
  cudaDeviceProp prop;
  int count;

  cudaGetDeviceCount (&count);

  for (int i =0; i< count; i++)
  {
    cudaGetDeviceProperties  (&prop, i);
    printf ("Name: %s\n", prop.name);
    printf ("Global Mem: %u\n", prop.totalGlobalMem);
    printf ("Shared Mem per Block: %d\n", prop.sharedMemPerBlock);
    printf ("regs per block: %d\n", prop.regsPerBlock);

    printf ("warpSize: %d\n", prop.warpSize);
    printf ("memPitch: %d\n", prop.memPitch);
    printf ("maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
    printf ("maxThreadsDim[0]: %d\n", prop.maxThreadsDim[0]);
    printf ("maxThreadsDim[1]: %d\n", prop.maxThreadsDim[1]);
    printf ("maxThreadsDim[2]: %d\n", prop.maxThreadsDim[2]);

    printf ("maxGridSize[0]: %d\n", prop.maxGridSize[0]);
    printf ("maxGridSize[1]: %d\n", prop.maxGridSize[1]);
    printf ("maxGridSize[2]: %d\n", prop.maxGridSize[2]);

    printf ("clockRate: %d\n", prop.clockRate);
    printf ("totalConstMem: %d\n", prop.totalConstMem);
    printf ("major: %d\n", prop.major);
    printf ("minor: %d\n", prop.minor);
    printf ("textureAlignment: %d\n", prop.textureAlignment);
    printf ("deviceOverlap: %d\n", prop.deviceOverlap);
    printf ("multiProcessorCount: %d\n", prop.multiProcessorCount);
    printf ("kernelExecTimeoutEnabled: %d\n", prop.kernelExecTimeoutEnabled);
    printf ("integrated: %d\n", prop.integrated);
    printf ("canMapHostMemory: %d\n", prop.canMapHostMemory);

    printf ("computeMode: %d\n", prop.computeMode);
    printf ("maxTexture1D: %d\n", prop.maxTexture1D);

    printf ("maxTexture2D[0]: %d\n", prop.maxTexture2D[0]);
    printf ("maxTexture2D[1]: %d\n", prop.maxTexture2D[1]);

    printf ("maxTexture3D[0]: %d\n", prop.maxTexture3D[0]);
    printf ("maxTexture3D[1]: %d\n", prop.maxTexture3D[1]);
    printf ("maxTexture3D[2]: %d\n", prop.maxTexture3D[2]);

    printf ("maxTexture1DLayered[0]: %d\n", prop.maxTexture1DLayered[0]);
    printf ("maxTexture1DLayered[1]: %d\n", prop.maxTexture1DLayered[1]);

    printf ("maxTexture2DLayered[0]: %d\n", prop.maxTexture2DLayered[0]);
    printf ("maxTexture2DLayered[1]: %d\n", prop.maxTexture2DLayered[1]);
    printf ("maxTexture2DLayered[2]: %d\n", prop.maxTexture2DLayered[2]);

    printf ("surfaceAlignment: %d\n", prop.surfaceAlignment);
    printf ("concurrentKernels: %d\n", prop.concurrentKernels);
    printf ("ECCEnabled: %d\n", prop.ECCEnabled);
    printf ("pciBusID: %d\n", prop.pciBusID);

    printf ("pciDeviceID: %d\n", prop.pciDeviceID);
    printf ("pciDomainID: %d\n", prop.pciDomainID);
    printf ("tccDriver: %d\n", prop.tccDriver);
    printf ("asyncEngineCount: %d\n", prop.asyncEngineCount);
    printf ("unifiedAddressing: %d\n", prop.unifiedAddressing);
    printf ("memoryClockRate: %d\n", prop.memoryClockRate);
    printf ("memoryBusWidth: %d\n", prop.memoryBusWidth);
    printf ("l2CacheSize: %d\n", prop.l2CacheSize);
    printf ("maxThreadsPerMultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);
  }

  return 0;
}

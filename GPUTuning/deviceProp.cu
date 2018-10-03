#include <stdio.h>

// code from mixbench
#define CUDA_SAFE_CALL( call) {                                    \
  cudaError err = call;                                                    \
  if( cudaSuccess != err) {                                                \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
        __FILE__, __LINE__, cudaGetErrorString( err) );              \
    exit(EXIT_FAILURE);                                                  \
  } }

// SM version to 32 FP cores
static inline int _ConvertSMVer2Cores(int major, int minor)
{
  switch(major){
    case 1:  return 8;
    case 2:  switch(minor){
               case 1:  return 48;
               default: return 32;
             }
    case 3:  return 192;
    case 6: switch(minor){
              case 0:  return 64;
              default: return 128;
            }
    case 7: switch(minor){
              case 0:  return 64;
              default: return 128;
            }
    default: return 128;
  }
}

static inline void GetDevicePeakInfo(double *aGIPS, double *aGBPS, cudaDeviceProp *aDeviceProp = NULL){
  cudaDeviceProp deviceProp;
  int current_device;
  if( aDeviceProp )
    deviceProp = *aDeviceProp;
  else{
    CUDA_SAFE_CALL( cudaGetDevice(&current_device) );
    CUDA_SAFE_CALL( cudaGetDeviceProperties(&deviceProp, current_device) );
  }
  const int TotalSPs = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor)*deviceProp.multiProcessorCount;
  *aGIPS = 1000.0 * deviceProp.clockRate * TotalSPs / (1000.0 * 1000.0 * 1000.0);  // Giga instructions/sec
  //        *aGIPS64 = 1000.0 * deviceProp.clockRate * TotalSPs / (1000.0 * 1000.0 * 1000.0);  // Giga instructions/sec
  *aGBPS = 2.0 * (double)deviceProp.memoryClockRate * 1000.0 * (double)deviceProp.memoryBusWidth / 8.0;
}

#if 0
static inline cudaDeviceProp GetDeviceProperties(void){
        cudaDeviceProp deviceProp;
        int current_device;
        CUDA_SAFE_CALL( cudaGetDevice(&current_device) );
        CUDA_SAFE_CALL( cudaGetDeviceProperties(&deviceProp, current_device) );
        return deviceProp;
}
#endif

// Print basic device information
static void StoreDeviceInfo(FILE *fout){
  cudaDeviceProp deviceProp;
  int current_device, driver_version;
  CUDA_SAFE_CALL( cudaGetDevice(&current_device) );
  CUDA_SAFE_CALL( cudaGetDeviceProperties(&deviceProp, current_device) );
  CUDA_SAFE_CALL( cudaDriverGetVersion(&driver_version) );
  fprintf(fout, "------------------------ Device specifications ------------------------\n");
  fprintf(fout, "Device:              %s\n", deviceProp.name);
  fprintf(fout, "CUDA driver version: %d.%d\n", driver_version/1000, driver_version%1000);
  fprintf(fout, "GPU clock rate:      %d MHz\n", deviceProp.clockRate/1000);
  fprintf(fout, "Memory clock rate:   %d MHz\n", deviceProp.memoryClockRate/1000/2); // TODO: why divide by 2 here??
  fprintf(fout, "Memory bus width:    %d bits\n", deviceProp.memoryBusWidth);
  fprintf(fout, "WarpSize:            %d\n", deviceProp.warpSize);
  fprintf(fout, "L2 cache size:       %d KB\n", deviceProp.l2CacheSize/1024);
  fprintf(fout, "Total global mem:    %d MB\n", (int)(deviceProp.totalGlobalMem/1024/1024));
  fprintf(fout, "ECC enabled:         %s\n", deviceProp.ECCEnabled?"Yes":"No");
  fprintf(fout, "Compute Capability:  %d.%d\n", deviceProp.major, deviceProp.minor);
  const int TotalSPs = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor)*deviceProp.multiProcessorCount;
  fprintf(fout, "Total SPs:           %d (%d MPs x %d SPs/MP)\n", TotalSPs, deviceProp.multiProcessorCount, _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor));
  double InstrThroughput, MemBandwidth;
  GetDevicePeakInfo(&InstrThroughput, &MemBandwidth, &deviceProp);
  fprintf(fout, "Compute throughput:  %.2f GFlops (theoretical single precision FMAs)\n", 2.0*InstrThroughput);
  fprintf(fout, "Memory bandwidth:    %.2f GB/sec\n", MemBandwidth/(1000.0*1000.0*1000.0));
  fprintf(fout, "-----------------------------------------------------------------------\n");
}

int main()
{
  int count;

  cudaGetDeviceCount (&count);
  printf ("Total GPU device count =%d\n", count);

#if 1
  for (int i =0; i< 1; i++)
  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties  (&prop, i);
    printf ("Name: %s\n", prop.name);
    printf ("Global Mem (GB): %zu\n", prop.totalGlobalMem/1024/1024/1024);
    printf ("Shared Mem per Block: %zd\n", prop.sharedMemPerBlock);
    printf ("regs per block: %d\n", prop.regsPerBlock);
    printf ("warpSize: %d\n", prop.warpSize); //5 

    printf ("memPitch: %zd\n", prop.memPitch);
    printf ("maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
    printf ("maxThreadsDim[0]: %d\n", prop.maxThreadsDim[0]);
    printf ("maxThreadsDim[1]: %d\n", prop.maxThreadsDim[1]);
    printf ("maxThreadsDim[2]: %d\n", prop.maxThreadsDim[2]);

    printf ("maxGridSize[0]: %d\n", prop.maxGridSize[0]);
    printf ("maxGridSize[1]: %d\n", prop.maxGridSize[1]);
    printf ("maxGridSize[2]: %d\n", prop.maxGridSize[2]);
    printf ("clockRate: %d\n", prop.clockRate); //10

    printf ("totalConstMem: %zd\n", prop.totalConstMem);
    printf ("major: %d\n", prop.major);
    printf ("minor: %d\n", prop.minor);
    printf ("textureAlignment: %zd\n", prop.textureAlignment);
    printf ("texturePitchAlignment: %zd\n", prop.texturePitchAlignment); //15

    printf ("deviceOverlap: %d\n", prop.deviceOverlap);
    printf ("multiProcessorCount: %d\n", prop.multiProcessorCount);
    printf ("kernelExecTimeoutEnabled: %d\n", prop.kernelExecTimeoutEnabled);
    printf ("integrated: %d\n", prop.integrated);
    printf ("canMapHostMemory: %d\n", prop.canMapHostMemory); // 20

    printf ("computeMode: %d\n", prop.computeMode);
    printf ("maxTexture1D: %d\n", prop.maxTexture1D);

    printf ("maxTexture1DMipmap: %d\n", prop.maxTexture1DMipmap);
    printf ("maxTexture1DLinear: %d\n", prop.maxTexture1DLinear);

    printf ("maxTexture2D[0]: %d\n", prop.maxTexture2D[0]);
    printf ("maxTexture2D[1]: %d\n", prop.maxTexture2D[1]); //25

    printf ("maxTexture3D[0]: %d\n", prop.maxTexture3D[0]);
    printf ("maxTexture3D[1]: %d\n", prop.maxTexture3D[1]);
    printf ("maxTexture3D[2]: %d\n", prop.maxTexture3D[2]);

    printf ("maxTexture1DLayered[0]: %d\n", prop.maxTexture1DLayered[0]);
    printf ("maxTexture1DLayered[1]: %d\n", prop.maxTexture1DLayered[1]);

    printf ("maxTexture2DLayered[0]: %d\n", prop.maxTexture2DLayered[0]);
    printf ("maxTexture2DLayered[1]: %d\n", prop.maxTexture2DLayered[1]);
    printf ("maxTexture2DLayered[2]: %d\n", prop.maxTexture2DLayered[2]);

    printf ("maxSurfaceCubemap: %d\n", prop.maxSurfaceCubemap); //40

    printf ("maxSurfaceCubemapLayered[0]: %d\n", prop.maxSurfaceCubemapLayered[0]);
    printf ("maxSurfaceCubemapLayered[1]: %d\n", prop.maxSurfaceCubemapLayered[1]);
    printf ("surfaceAlignment: %zd\n", prop.surfaceAlignment);
    printf ("concurrentKernels: %d\n", prop.concurrentKernels);
    printf ("ECCEnabled: %d\n", prop.ECCEnabled);
    printf ("pciBusID: %d\n", prop.pciBusID); //45

    printf ("pciDeviceID: %d\n", prop.pciDeviceID);
    printf ("pciDomainID: %d\n", prop.pciDomainID);
    printf ("tccDriver: %d\n", prop.tccDriver);
    printf ("asyncEngineCount: %d\n", prop.asyncEngineCount);
    printf ("unifiedAddressing: %d\n", prop.unifiedAddressing); // 50
    printf ("memoryClockRate: %d\n", prop.memoryClockRate);
    printf ("memoryBusWidth: %d\n", prop.memoryBusWidth);
    printf ("l2CacheSize: %d\n", prop.l2CacheSize);
    printf ("maxThreadsPerMultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf ("streamPrioritiesSupported: %d\n", prop.streamPrioritiesSupported); // 55


    printf ("globalL1CacheSupported: %d\n", prop.globalL1CacheSupported);
    printf ("localL1CacheSupported: %d\n", prop.localL1CacheSupported);
    printf ("sharedMemPerMultiprocessor: %zd\n", prop.sharedMemPerMultiprocessor);
    printf ("regsPerMultiprocessor: %d\n", prop.regsPerMultiprocessor);
    //    printf ("managedMemSupported: %d\n", prop.managedMemSupported); //60

    printf ("isMultiGpuBoard: %d\n", prop.isMultiGpuBoard);
    printf ("multiGpuBoardGroupID: %d\n", prop.multiGpuBoardGroupID);
    printf ("singleToDoublePrecisionPerfRatio: %d\n", prop.singleToDoublePrecisionPerfRatio);
    printf ("pageableMemoryAccess: %d\n", prop.pageableMemoryAccess);
    printf ("concurrentManagedAccess: %d\n", prop.concurrentManagedAccess); //65

    printf ("computePreemptionSupported: %d\n", prop.computePreemptionSupported);
    printf ("canUseHostPointerForRegisteredMem: %d\n", prop.canUseHostPointerForRegisteredMem);
    printf ("cooperativeLaunch: %d\n", prop.cooperativeLaunch);
    printf ("cooperativeMultiDeviceLaunch: %d\n", prop.cooperativeMultiDeviceLaunch);
  }
#endif
  cudaSetDevice(0);

  StoreDeviceInfo(stdout);

  return 0;
}

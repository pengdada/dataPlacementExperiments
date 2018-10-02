// 2-D image dimension size
// this number should be divisible by BLOCKSIZE
// work with -D compilation option
#ifndef DIM
//#define DIM 1024 // fit into L2 cache
//#define DIM 2048
// #define DIM 4096  // larger than L2 cache
#define DIM 20480 
//#define DIM 60000  // MAX to fit into global mem with spheres. 65327 if solely occupies the global memory
#endif

// number of spheres in space
#ifndef SPHERES
//#define SPHERES 20
//#define SPHERES 256
#define SPHERES 2340
#endif

// 2-D thread block size
#ifndef BLOCKSIZE
#define BLOCKSIZE 32
//#define BLOCKSIZE 16
#endif


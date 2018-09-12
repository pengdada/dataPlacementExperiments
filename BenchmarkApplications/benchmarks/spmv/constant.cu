#include <stdio.h>

#define MSIZE	12*8*21
#define BLOCK_SIZE	256
#define WARP_SIZE	32

//__constant__ int c_row[64*1024/4];
//__constant__ int c_row[MSIZE*BLOCK_SIZE/WARP_SIZE*4];
int main()
{

  printf ("%d\n", MSIZE*BLOCK_SIZE/WARP_SIZE*4);
   return 0;
}

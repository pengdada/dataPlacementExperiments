/* square matrix only: size */
#ifndef MSIZE
#define MSIZE 2048 
//#define MSIZE 2048 
//#define MSIZE 4096
#endif 
// sub matrix size:  thread block size
// original matrix size should be divisible by block size for simplicity
// 128/16 = 8:  8x8 =64 submatrices
#ifndef BLOCK_SIZE
//#define BLOCK_SIZE 16
#define BLOCK_SIZE 32
#endif
/* 
 * @Abdullah
 * Using BLOCK_SIZE
 * We are using BLOCK_SIZE of (8x8), (16x16), (32x32)
 * Because the shared mem verison uses keeps two matrices of double (8 bytes) 
 * of BLOCK_SIZExBLOCK_SIZE in shared memory
 * Our Volta GPU has maximum configurable shared memory of 48KB
 * So 32x32x8x2=16KB
 * We can't use the full 48KB 
 * because the original matrix has to be divisible by BLOCK_SIZE
 *
 * One more reason, here we are using a 2-D thread block, 
 * and the maximum thread block size of Volta, Pascal and Kepler is 1024. 
 * So  if we use a symmetric 2-D block size we can only go up to 32x32(=1024) blocks.

 */

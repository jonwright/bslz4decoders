
#include <stdint.h>

__global__ void shuf_8192_32(const uint32_t * __restrict__ in,
                                   uint32_t * __restrict__ out ){
    /*
    grid 32x32 threads
    each thread loads 4 bytes (aligned) = 128 bytes per row of 32
    total bytes loaded = 32x32x4 = 4096 bytes
                  x                y  z
    blocks = ( total_bytes / 8192, 2, 1 )
    */
    __shared__ uint32_t smem[32][33];
    /* This thread is going to load 4 bytes. Next thread in x will load
    the next 4 to be aligned. In total we pick up 32*4 = 128 bytes in this
    row of 32 (warp) for bit0.
    The next row (warp) is going to pick up bit1, etc
    The first grid starts at byte 0 + blockIdx.x * 2048
    The second grid starts at byte 8192/32/2
     */
    smem[ threadIdx.y ][ threadIdx.x ] =    in[ threadIdx.x     +   // Aligned loads. 32*4 = 128 bytes
                                                threadIdx.y*64  +   // Offset to next bit = 8192/32/4.
                                                blockIdx.x*2048 +   // Start of the block
                                                blockIdx.y*32 ];    // Next 32 reads
     __syncthreads();   /* Now we loaded 4 kB to smem.   Do the first level of transpose */
     uint32_t myval = smem[ threadIdx.x ][ threadIdx.y ];
     __syncthreads();
    for( int i = 0; i < 32; i++ )
         smem[i][threadIdx.y] = __ballot_sync(0xFFFFFFFFU,  myval & (1U<<i) );
    out[ threadIdx.x + threadIdx.y*32 + blockIdx.y*1024 + blockIdx.x*2048 ] = smem[threadIdx.x][threadIdx.y];
}

__global__ void shuf_8192_8(const uint32_t * __restrict__ in,
                                  uint32_t * __restrict__ out ){
    /*
    grid 32x8x4 threads
    each thread loads 4 bytes (aligned) = 128 bytes per row of 32
    total bytes loaded = 32x32x4 = 4096 bytes
                  x                y  z
    blocks = ( total_bytes / 8192, 2, 1 )
    */

    __shared__ uint32_t smem[8][4][32];
    /*                       z  y  x
    This thread is going to load 4 bytes.
    In total we pick up 32*4 = 128 bytes in this row of 32 (warp) for bit0.
    The next row (warp) is going to pick up bit1, etc
    The first grid starts at byte 0 + blockIdx.x * 2048
    The second grid starts at byte 8192/32/2

    smem[ threadIdx.z ][ threadIdx.y ][ threadIdx.x ] =    in[ threadIdx.x        +   // Aligned loads. 32*4 = 128 bytes
                                                               threadIdx.y*32     +   // End of x. Get 4 blocks.
                                                               threadIdx.z*256    +   // the next bit is in z index
                                                                blockIdx.x*2048   +   // Start of the block
                                                                blockIdx.y*32 ];      // Next 32 reads
     __syncthreads();
     uint32_t myval = smem[ threadIdx.x ][ threadIdx.y ][ threadIdx.z ];
     __syncthreads();
    for( int i = 0; i < 32; i++ )
         smem[i][threadIdx.y] = __ballot_sync(0xFFFFFFFFU,  myval & (1U<<i) );
    out[ threadIdx.x + threadIdx.y*32 + blockIdx.y*1024 + blockIdx.x*2048 ] = smem[threadIdx.z][threadIdx.x][threadIdx.y];
    */
}
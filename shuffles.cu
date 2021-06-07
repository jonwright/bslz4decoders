
/* Bitshuffle cuda kernels aimed at hdf5 bslz4 format from the Dectris Eiger detectors.
   Written by Jon Wright, ESRF, June 2021.

   */


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
    uint32_t v;
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
     v = smem[ threadIdx.x ][ threadIdx.y ];
     #pragma unroll 32
     for( int i = 0; i < 32; i++ )
         smem[i][threadIdx.y] = __ballot_sync(0xFFFFFFFFU,  v & (1U<<i) );
     __syncthreads();   /* Now we loaded 4 kB to smem.   Do the first level of transpose */
     out[ threadIdx.x + threadIdx.y*32 + blockIdx.y*1024 + blockIdx.x*2048 ] = smem[threadIdx.x][threadIdx.y];
}

__global__ void shuf_8192_16(const uint32_t * __restrict__ in,
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
    int bitsy = threadIdx.y % 16;
    int itemy = threadIdx.y / 16;
    int itemx = threadIdx.x / 16;
    int shft  = 16*itemx;
    uint32_t v, mask = 0xFFFFU << shft ;

    smem[ threadIdx.y ][ threadIdx.x ] =    in[ threadIdx.x     +   // Aligned loads. 32*4 = 128 bytes = 1024 bits
                                                itemy * 32      +   // end of the threadIdx.x reads
                                                bitsy * 128     +   // position of the next bit
                                                blockIdx.x*2048 +   // Start of the block
                                                blockIdx.y*64 ] ;   // Next 32*2 byte reads
     __syncthreads();   /* Now we loaded 4 kB to smem.   Do the first level of transpose */
                        /*  (0-32)x4x8=1024                                   */
    v = smem[ threadIdx.x ][ threadIdx.y ];
    #pragma unroll 32
    for( int i = 0; i < 32; i++ )
         smem[i][threadIdx.y] = __ballot_sync(0xFFFFFFFFU,  v & (1U<<i) );
    __syncthreads();
    /* output is [0, 1024] [1, 1025] [2,1026], ...
       smem[i = 0 -> 1024 outputs][2 each]

       output[x, y] = [2][1024]
    */
    v = ( ( smem[ 2 * (threadIdx.x%16)   ][ threadIdx.y ] & mask ) >> shft ) |
        ( ( smem[ 2 * (threadIdx.x%16)+1 ][ threadIdx.y ] & mask ) << 16-shft );
    out[ threadIdx.x + shft * 31 + threadIdx.y*16 + blockIdx.y*1024 + blockIdx.x*2048 ] = v;
}
__global__ void shuf_8192_8(const uint32_t * __restrict__ in,
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
    int bytx = threadIdx.x / 8;
    uint32_t t, v, mask=0xFFU << bytx*8;

    smem[ threadIdx.y ][ threadIdx.x ] =    in[ threadIdx.x              +   // Aligned loads. 32*4 = 128 bytes = 1024 bits
                                                (threadIdx.y / 8) * 32   +   // end of the threadIdx.x reads
                                                (threadIdx.y % 8) * 256  +   // position of the next bit
                                                blockIdx.x*2048          +   // Start of the block
                                                blockIdx.y*128 ] ;   // Next 32*4 byte reads
     __syncthreads();   /* Now we loaded 4 kB to smem.   Do the first level of transpose */
                        /*  (0-32)x4x8=1024                                   */
    v = smem[ threadIdx.x ][ threadIdx.y ];
    #pragma unroll 32
    for( int i = 0; i < 32; i++ ){
         smem[i][threadIdx.y] = __ballot_sync(0xFFFFFFFFU,  v & (1U<<i) );
    }
    v = 0U;
    __syncthreads();
    /* output is [0, 1024, 2048, 4096]
       smem[i = 0 -> 1024 outputs][4 each]

       output[x, y] = [4][1024]
    */
    #pragma unroll 4
    for( int i = 0 ; i < 4; i++ ){
         t = smem[ 4 * ( threadIdx.x % 8 ) + ( i + bytx ) % 4 ][ threadIdx.y ] & mask;
         v |= ((i - bytx) >= 0 ) ? t << (( i - bytx ) * 8) : t >> (( bytx - i )*8) ;
    }
    out[ threadIdx.x + bytx * 248  + threadIdx.y*8 + blockIdx.y*1024 + blockIdx.x*2048 ] = v ;
}

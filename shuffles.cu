
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
    uint32_t bitsy = threadIdx.y % 16;
    uint32_t itemy = threadIdx.y / 16; 
    smem[ threadIdx.y ][ threadIdx.x ] =    in[ threadIdx.x     +   // Aligned loads. 32*4 = 128 bytes = 1024 bits
                                                itemy * 32      +   // end of the threadIdx.x reads
                                                bitsy * 128     +   // position of the next bit
                                                blockIdx.x*2048 +   // Start of the block
                                                blockIdx.y*64 ] ;   // Next 32 reads
     __syncthreads();   /* Now we loaded 4 kB to smem.   Do the first level of transpose */
                        /*  (0-32)x4x8=1024                                   */
     uint32_t myval = smem[ threadIdx.x ][ threadIdx.y ];
     __syncthreads();
    for( int i = 0; i < 32; i++ )
         smem[i][threadIdx.y] = __ballot_sync(0xFFFFFFFFU,  myval & (1U<<i) );
    __syncthreads();
    /* output is [0, 1024] [1, 1025] [2,1026], ... 
       smem[i = 0 -> 1024 outputs][2 each]

       output[x, y] = [2][1024]
    */
    if( threadIdx.x < 16){
        uint32_t lo = smem[ 2 * (threadIdx.x%16)   ][ threadIdx.y ] & 0xFFFFU;
        uint32_t hi = smem[ 2 * (threadIdx.x%16)+1 ][ threadIdx.y ] & 0xFFFFU;
        out[ threadIdx.x + threadIdx.y*16 + blockIdx.y*1024 + blockIdx.x*2048 ] = lo | (hi << 16);
    } else {
        uint32_t lo = smem[ 2 * (threadIdx.x%16)   ][ threadIdx.y ] & 0xFFFF0000U;
        uint32_t hi = smem[ 2 * (threadIdx.x%16)+1 ][ threadIdx.y ] & 0xFFFF0000U;
        out[ threadIdx.x + 512 - 16 + threadIdx.y*16 + blockIdx.y*1024 + blockIdx.x*2048 ] = (lo >> 16) | hi;
    }
    // out[ threadIdx.x + threadIdx.y*32 + blockIdx.y*1024 + blockIdx.x*2048 ] = smem[threadIdx.x][threadIdx.y];
    
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
    __shared__ uint32_t smem[32][33];
    /*   
    This thread is going to load 4 bytes.
    In total we pick up 32*4 = 128 bytes in this row of 32 (warp) for bit0.
    The next row (warp) is going to pick up bit1, etc
    The first grid starts at byte 0 + blockIdx.x * 2048
    The second grid starts at byte 8192/32/2
    */
    uint32_t bity = threadIdx.y % 8;
    uint32_t byty = threadIdx.y / 8;
    uint32_t bitx = threadIdx.x % 8;
    uint32_t bytx = threadIdx.x / 8;
                      // 128bytes
    smem[threadIdx.y][ threadIdx.x ] =    in[ threadIdx.x        +   // Aligned loads. 32*4 = 128 bytes
                                              byty*32            +   // Next 32x4 bytes
                                              bity*256           +   // the next bit is 8192/8/4 = 256
                                              blockIdx.x*2048    +   // Start of the block
                                              blockIdx.y*128 ];      // One group reads 32*4
     __syncthreads();
     /*  tY = which bit is this 

      */
     uint32_t myval = smem[ threadIdx.x  ][ threadIdx.y ];
     __syncthreads();
    for( int i = 0; i < 32; i++ ){
        // 0 1 2 3      0 1 2 3 4 5 6 7
        // 4 5 6 7      8 9 ... 
        // 8 9 ..
         smem[ i ][threadIdx.y] = __ballot_sync(0xFFFFFFFFU,  myval & (1U<<i) );
    }
     __syncthreads();  
//    out[ threadIdx.x + threadIdx.y*32 + blockIdx.y*1024 + blockIdx.x*2048 ] = smem[threadIdx.x][threadIdx.y];
     /*                            y         y         x
         Reading :  smem =  [ 4 bytes ] [ 8 bits ] [ 32 threads ] [ 32bits ]
         Writing :  smem =  [ i 0..31 ] [ 32bits ] [ 32 threads ]
         4*8 = 32 = 8 * 4
     */
      out[ threadIdx.x + threadIdx.y*32 + blockIdx.y*1024 + blockIdx.x*2048 ] = smem[threadIdx.x][threadIdx.y];
  
}
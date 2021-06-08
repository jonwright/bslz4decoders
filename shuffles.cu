
/* Bitshuffle cuda kernels aimed at hdf5 bslz4 format from the Dectris Eiger detectors.
   Written by Jon Wright, ESRF, June 2021.

   */


#include <stdint.h>


__global__ void copybytes(uint8_t * src, uint32_t srcstart, uint8_t * dst, uint32_t dststart )
{
  dst[threadIdx.x + dststart] = src[threadIdx.x+srcstart];
}

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
    int itemx = threadIdx.x / 16;
    int shft  = 16*itemx;
    uint32_t v, mask = 0xFFFFU << shft ;
    smem[ threadIdx.y ][ threadIdx.x ] =    in[ threadIdx.x     +   // Aligned loads. 32*4 = 128 bytes = 1024 bits
                                                ( threadIdx.y / 16 ) * 32      +   // end of the threadIdx.x reads
                                                ( threadIdx.y % 16 ) * 128     +   // position of the next bit
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
    out[ threadIdx.x + shft * 31 + threadIdx.y*16 + blockIdx.y*1024 + blockIdx.x*2048 ] =
        ( ( smem[ 2 * (threadIdx.x%16)   ][ threadIdx.y ] & mask ) >> shft ) |
        ( ( smem[ 2 * (threadIdx.x%16)+1 ][ threadIdx.y ] & mask ) << 16-shft );
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
    uint32_t bitx = 4 *( threadIdx.x % 8 );
    uint32_t bytx = threadIdx.x / 8;
    uint32_t mask = 0xFFU << ( 8*bytx );
    smem[ threadIdx.y ][ threadIdx.x ] =    in[ threadIdx.x     +   // Aligned loads. 32*4 = 128 bytes = 1024 bits
                                                (threadIdx.y/8) * 32      +   // end of the threadIdx.x reads
                                                (threadIdx.y%8) * 256     +   // position of the next bit
                                                blockIdx.x*2048 +   // Start of the block
                                                blockIdx.y*128 ] ;   // Next 32*4 byte reads
    __syncthreads();   /* Now we loaded 4 kB to smem.   Do the first level of transpose */
                        /*  (0-32)x4x8=1024                                   */
    uint32_t v = smem[ threadIdx.x ][ threadIdx.y ];
    #pragma unroll 32
    for( int i = 0; i < 32; i++ )
         smem[i][threadIdx.y] = __ballot_sync(0xFFFFFFFFU,  v & (1U<<i) );
    v = 0U;
    __syncthreads();
    /* output is [0, 1024, 2048, 4096]
       smem[i = 0 -> 1024 outputs][4 each]
       output[x, y] = [4][1024]
    */
    switch (bytx){
        case 0:
            v |= ((smem[ bitx     ][ threadIdx.y ] & mask)       ) ;
            v |= ((smem[ bitx + 1 ][ threadIdx.y ] & mask) <<  8 ) ;
            v |= ((smem[ bitx + 2 ][ threadIdx.y ] & mask) << 16 ) ;
            v |= ((smem[ bitx + 3 ][ threadIdx.y ] & mask) << 24 ) ;
            break;
        case 1:
            v |= ((smem[ bitx + 1 ][ threadIdx.y ] & mask)       ) ;
            v |= ((smem[ bitx + 2 ][ threadIdx.y ] & mask) <<  8 ) ;
            v |= ((smem[ bitx + 3 ][ threadIdx.y ] & mask) << 16 ) ;
            v |= ((smem[ bitx     ][ threadIdx.y ] & mask) >>  8 ) ;
            break;
        case 2:
            v |= ((smem[ bitx + 2 ][ threadIdx.y ] & mask)       ) ;
            v |= ((smem[ bitx + 3 ][ threadIdx.y ] & mask) <<  8 ) ;
            v |= ((smem[ bitx     ][ threadIdx.y ] & mask) >> 16 ) ;
            v |= ((smem[ bitx + 1 ][ threadIdx.y ] & mask) >>  8 ) ;
            break;
        case 3:
            v |= ((smem[ bitx + 3 ][ threadIdx.y ] & mask)       ) ;
            v |= ((smem[ bitx     ][ threadIdx.y ] & mask) >> 24 ) ;
            v |= ((smem[ bitx + 1 ][ threadIdx.y ] & mask) >> 16 ) ;
            v |= ((smem[ bitx + 2 ][ threadIdx.y ] & mask) >>  8 ) ;
         break;
    }

    out[ threadIdx.x + (threadIdx.x/8) * 248  + threadIdx.y*8 + blockIdx.y*1024 + blockIdx.x*2048 ] = v;
}

__global__ void simple_shuffle(const uint8_t * __restrict__ in, uint8_t * __restrict__ out,
                          const uint32_t blocksize, const uint32_t total_bytes, const uint32_t elemsize ) {
  // slow : do not use except for debugging
  //                 0-32                        32
  uint32_t dest = threadIdx.x + blockIdx.x * blockDim.x;      // where to write output
  // first input byte :  bytes_per_elem==4
  uint32_t block_id = (dest * elemsize) / blocksize;             // which block is this block ?
  uint32_t block_start = block_id * blocksize;                   // where did the block start ?
  uint32_t nblocks = total_bytes / blocksize;                    // rounds down
  uint32_t bsize = blocksize;
  uint32_t tocopy = 0;
  uint32_t elements_in_block = bsize / elemsize;
  uint32_t position_in_block = dest % elements_in_block;         // 0 -> 2048
  int loop = 1;
  if( block_id == nblocks ) {                                    // this nmight not be a full length block.
     bsize = total_bytes % blocksize;
     tocopy = bsize % ( 8 * elemsize);
     bsize -= tocopy;
     elements_in_block = bsize / elemsize;
     if( position_in_block >= elements_in_block ){
         // this is a copy
         for( int i = 0 ; i < elemsize ; i++ ){
             out[ dest * elemsize + i ] = in[ dest * elemsize + i ];
         }
         loop = 0;
     } else  {
         position_in_block = position_in_block % elements_in_block;
     }
  }
  if (loop && block_id <= nblocks) {
     const uint8_t * mybyte = in + block_start + ( position_in_block / 8 );
     uint8_t mymask = 1U << (position_in_block % 8);
     uint32_t bytestride = bsize / ( 8 * elemsize );

     uint32_t myval = 0;
     for( int i = 0 ; i < elemsize*8 ; i ++ ) {       // grab my bits
        if( (*mybyte & mymask) > 0 ) {
            myval = myval | (1U << i);
        }
        mybyte = mybyte + bytestride;
     }
     for( int i = 0; i<elemsize ; i++){
         out[dest * elemsize + i] = (uint8_t) ((myval)>>(8*i));
         }
  }
}


__global__ void simple_shuffle_end(const uint8_t * __restrict__ in, uint8_t * __restrict__ out,
                          const uint32_t blocksize, const uint32_t total_bytes, const uint32_t elemsize,
                          const uint32_t startpos
                           ) {
  // slow : do not use except for debugging
  //                 0-32                        32
  uint32_t dest = threadIdx.x + blockIdx.x * blockDim.x;      // where to write output
  // first input byte :  bytes_per_elem==4
  uint32_t block_id = (dest * elemsize) / blocksize;             // which block is this block ?
  uint32_t block_start = block_id * blocksize;                   // where did the block start ?
  uint32_t nblocks = total_bytes / blocksize;                    // rounds down
  uint32_t bsize = blocksize;
  uint32_t tocopy = 0;
  uint32_t elements_in_block = bsize / elemsize;
  uint32_t position_in_block = dest % elements_in_block;         // 0 -> 2048
  int loop = 1;
  if( block_id == nblocks ) {                                    // this nmight not be a full length block.
     bsize = total_bytes % blocksize;
     tocopy = bsize % ( 8 * elemsize);
     bsize -= tocopy;
     elements_in_block = bsize / elemsize;
     if( position_in_block >= elements_in_block ){
         loop = 0;
     } else  {
         position_in_block = position_in_block % elements_in_block;
     }
  }
  if (loop && block_id <= nblocks) {
     const uint8_t * mybyte = in + startpos + block_start + ( position_in_block / 8 );
     uint8_t mymask = 1U << (position_in_block % 8);
     uint32_t bytestride = bsize / ( 8 * elemsize );
     uint32_t myval = 0;
     for( int i = 0 ; i < elemsize*8 ; i ++ ) {       // grab my bits
        if( (*mybyte & mymask) > 0 ) {
            myval = myval | (1U << i);
        }
        mybyte = mybyte + bytestride;
     }
     for( int i = 0; i<elemsize ; i++){
         out[startpos + dest * elemsize + i] = (uint8_t) ((myval)>>(8*i));
         }
  }
}


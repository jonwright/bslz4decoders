

/* Cuda kernel wrapper to call nvcomp on bslz4 data
   Jon Wright, ESRF, 2021.

   Use python to cat the nvcomp_extract.cu on the top of this file.
*/


inline __device__ uint32_t read32be( const uint8_t* address )
{
    return ( (uint32_t)(255 & (address)[0]) << 24 | (uint32_t)(255 & (address)[1]) << 16 |
             (uint32_t)(255 & (address)[2]) <<  8 |(uint32_t)(255 & (address)[3])       ) ;
}

__global__ void h5lz4dc (
    const uint8_t*  const compressed,     /* compressed data pointer */
    const uint32_t* const block_starts,   /* block start positions in compressed (bytes) */
    const uint32_t num_blocks,            /* number of blocks */
    const uint32_t blocksize,             /* blocksize in bytes */
    uint8_t* const decompressed           /* destination start pointer */
)
{

  const int blockid = blockIdx.x * blockDim.y + threadIdx.y;
  //                         Defined in ncvomp_extract
  __shared__ uint8_t buffer[DECOMP_INPUT_BUFFER_SIZE * DECOMP_CHUNKS_PER_BLOCK];

  if (blockid < num_blocks) {
    decompressStream( buffer + threadIdx.y * DECOMP_INPUT_BUFFER_SIZE,
                      decompressed + blockid * blocksize,         // output start
                      compressed + block_starts[blockid] + 4,     // input starts
                      read32be( compressed + block_starts[blockid] ) // numbers of compressed bytes
        );
  }
}


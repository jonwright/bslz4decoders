

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

/* Batch version that processes multiple images */
__global__ void h5lz4dc_batch (
    const uint8_t** const compressed_ptrs,      /* array of compressed data pointers */
    const uint32_t** const block_starts_ptrs,   /* array of block start arrays */
    const uint32_t* const num_blocks_array,     /* number of blocks per image */
    const uint32_t blocksize,                   /* blocksize in bytes */
    uint8_t** const decompressed_ptrs           /* array of destination pointers */
)
{
  const int image_id = blockIdx.z;
  const int blockid = blockIdx.x * blockDim.y + threadIdx.y;
  
  __shared__ uint8_t buffer[DECOMP_INPUT_BUFFER_SIZE * DECOMP_CHUNKS_PER_BLOCK];

  const uint8_t* compressed = compressed_ptrs[image_id];
  const uint32_t* block_starts = block_starts_ptrs[image_id];
  const uint32_t num_blocks = num_blocks_array[image_id];
  uint8_t* decompressed = decompressed_ptrs[image_id];

  if (blockid < num_blocks) {
    decompressStream( buffer + threadIdx.y * DECOMP_INPUT_BUFFER_SIZE,
                      decompressed + blockid * blocksize,
                      compressed + block_starts[blockid] + 4,
                      read32be( compressed + block_starts[blockid] )
        );
  }
}

/* Forward declarations for shuffle functions from shuffles.cu */
__device__ void shuffle_8192_block_32(const uint32_t* in, uint32_t* out, int block_offset);
__device__ void shuffle_8192_block_16(const uint32_t* in, uint32_t* out, int block_offset);
__device__ void shuffle_8192_block_8(const uint32_t* in, uint32_t* out, int block_offset);

/* Fused kernel: decompress + shuffle in one pass, batch processing
   This eliminates intermediate buffers and keeps everything in cache */
__global__ void h5lz4dc_shuffle_batch (
    const uint8_t** const compressed_ptrs,      /* array of compressed data pointers */
    const uint32_t** const block_starts_ptrs,   /* array of block start arrays */
    const uint32_t* const num_blocks_array,     /* number of blocks per image */
    const uint32_t* const total_bytes_array,    /* total output bytes per image */
    const uint32_t blocksize,                   /* blocksize in bytes (8192) */
    const uint32_t bpp,                         /* bytes per pixel (1, 2, or 4) */
    uint8_t** const output_ptrs                 /* array of final output pointers */
)
{
  const int image_id = blockIdx.z;
  const int blockid = blockIdx.x * blockDim.y + threadIdx.y;
  
  /* 8KB scratch buffer per decompression stream */
  __shared__ uint8_t decomp_buffer[DECOMP_INPUT_BUFFER_SIZE * DECOMP_CHUNKS_PER_BLOCK];
  /* 8KB buffer for shuffled output before writing */
  __shared__ uint8_t shuffle_buffer[8192 * DECOMP_CHUNKS_PER_BLOCK];

  const uint8_t* compressed = compressed_ptrs[image_id];
  const uint32_t* block_starts = block_starts_ptrs[image_id];
  const uint32_t num_blocks = num_blocks_array[image_id];
  const uint32_t total_bytes = total_bytes_array[image_id];
  uint8_t* output = output_ptrs[image_id];

  const uint32_t nblocks_full = total_bytes / 8192;  /* full 8KB blocks */
  const uint32_t bytes_to_copy = total_bytes % (bpp * 8);

  if (blockid < num_blocks) {
    /* Decompress 8KB block to scratch buffer */
    uint8_t* my_decomp = decomp_buffer + threadIdx.y * DECOMP_INPUT_BUFFER_SIZE;
    decompressStream( my_decomp,
                      my_decomp,  /* decompress in place to scratch */
                      compressed + block_starts[blockid] + 4,
                      read32be( compressed + block_starts[blockid] )
        );
    
    __syncthreads();  /* ensure decompression complete */
    
    /* Now shuffle this 8KB block directly to output */
    if (blockid < nblocks_full) {
      /* Full 8KB block - use optimized shuffle */
      uint32_t* decomp_u32 = (uint32_t*)my_decomp;
      uint32_t* output_u32 = (uint32_t*)(output + blockid * 8192);
      
      /* Call appropriate shuffle based on bpp */
      if (bpp == 4) {
        shuffle_8192_block_32(decomp_u32, output_u32, 0);
      } else if (bpp == 2) {
        shuffle_8192_block_16(decomp_u32, output_u32, 0);
      } else if (bpp == 1) {
        shuffle_8192_block_8(decomp_u32, output_u32, 0);
      }
    } else if (blockid == nblocks_full) {
      /* Partial block at end - use simple_shuffle_end logic */
      const uint32_t pos = nblocks_full * blocksize;
      const uint32_t todo = total_bytes - pos;
      
      if (todo > bytes_to_copy) {
        /* Shuffle the non-copied portion */
        for (uint32_t dest = threadIdx.x; dest < (todo - bytes_to_copy) / bpp; dest += blockDim.x) {
          uint32_t block_id = (dest * bpp) / blocksize;
          uint32_t block_start = block_id * blocksize;
          uint32_t bsize = blocksize;
          uint32_t elements_in_block = bsize / bpp;
          uint32_t position_in_block = dest % elements_in_block;
          
          const uint8_t* mybyte = my_decomp + (position_in_block / 8);
          uint8_t mymask = 1U << (position_in_block % 8);
          uint32_t bytestride = bsize / (8 * bpp);
          
          uint32_t myval = 0;
          for (int i = 0; i < bpp * 8; i++) {
            if ((*mybyte & mymask) > 0) {
              myval = myval | (1U << i);
            }
            mybyte = mybyte + bytestride;
          }
          
          for (int i = 0; i < bpp; i++) {
            output[pos + dest * bpp + i] = (uint8_t)((myval) >> (8 * i));
          }
        }
      }
      
      /* Copy the last few unshuffled bytes directly from compressed data */
      if (bytes_to_copy > 0 && threadIdx.x < bytes_to_copy) {
        /* These bytes are at the end of the compressed chunk */
        output[total_bytes - bytes_to_copy + threadIdx.x] = 
            compressed[block_starts[num_blocks - 1] + 4 + 
                      read32be(compressed + block_starts[num_blocks - 1]) - bytes_to_copy + threadIdx.x];
      }
    }
  }
}


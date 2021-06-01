




import pycuda.autoinit
import pycuda.driver as drv
import pycuda
import numpy as np, bitshuffle, hdf5plugin, h5py

from pycuda.compiler import SourceModule

modsrc = """

/* Copy + paste from nvcomp to build/run using pycuda

   This is for decompression only, for the bslz4 data from hdf5/Dectris
   ... with apologies to anyone reading, I do not speak gpu languages.
   ... So sorry for the mess
       JPW 2021
   */

/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cstdint>
#include <cassert>


using offset_type = uint16_t;
using word_type = uint32_t;

// This restricts us to 4GB chunk sizes (total buffer can be up to
// max(size_t)). We actually artificially restrict it to much less, to
// limit what we have to test, as well as to encourage users to exploit some
// parallelism.
using position_type = uint32_t;
using double_word_type = uint64_t;
using item_type = uint32_t;

/**
 * @brief The number of threads to use per chunk in decompression.
 */
constexpr const int DECOMP_THREADS_PER_CHUNK = 32;

/**
 * @brief The number of chunks to decompression concurrently per threadblock.
 */
constexpr const int DECOMP_CHUNKS_PER_BLOCK = 2; // better as 2. Not sure why

/**
 * @brief The size of the shared memory buffer to use per decompression stream.
 */
constexpr const position_type DECOMP_INPUT_BUFFER_SIZE
    = DECOMP_THREADS_PER_CHUNK * sizeof(double_word_type);

/**
 * @brief The threshold of reading from the buffer during decompression, that
 * more data will be loaded inot the buffer and its contents shifted.
 */
constexpr const position_type DECOMP_BUFFER_PREFETCH_DIST
    = DECOMP_INPUT_BUFFER_SIZE / 2;

inline __device__ void syncCTA()
{
  if (DECOMP_THREADS_PER_CHUNK > 32) {
    __syncthreads();
  } else {
    __syncwarp();
  }
}

inline __device__ int warpBallot(int vote)
{
  return __ballot_sync(0xffffffff, vote);
}

inline __device__ offset_type readWord(const uint8_t* const address)
{
  offset_type word = 0;
  for (size_t i = 0; i < sizeof(offset_type); ++i) {
    word |= address[i] << (8 * i);
  }
  return word;
}

struct token_type
{
  position_type num_literals;
  position_type num_matches;

  __device__ bool hasNumLiteralsOverflow() const
  {
    return num_literals >= 15;
  }

  __device__ bool hasNumMatchesOverflow() const
  {
    return num_matches >= 19;
  }

  __device__ position_type numLiteralsOverflow() const
  {
    if (hasNumLiteralsOverflow()) {
      return num_literals - 15;
    } else {
      return 0;
    }
  }

  __device__ uint8_t numLiteralsForHeader() const
  {
    if (hasNumLiteralsOverflow()) {
      return 15;
    } else {
      return num_literals;
    }
  }

  __device__ position_type numMatchesOverflow() const
  {
    if (hasNumMatchesOverflow()) {
      assert(num_matches >= 19);
      return num_matches - 19;
    } else {
      assert(num_matches < 19);
      return 0;
    }
  }

  __device__ uint8_t numMatchesForHeader() const
  {
    if (hasNumMatchesOverflow()) {
      return 15;
    } else {
      return num_matches - 4;
    }
  }
  __device__ position_type lengthOfLiteralEncoding() const
  {
    if (hasNumLiteralsOverflow()) {
      const position_type num = numLiteralsOverflow();
      const position_type length = (num / 0xff) + 1;
      return length;
    }
    return 0;
  }

  __device__ position_type lengthOfMatchEncoding() const
  {
    if (hasNumMatchesOverflow()) {
      const position_type num = numMatchesOverflow();
      const position_type length = (num / 0xff) + 1;
      return length;
    }
    return 0;
  }
};

class BufferControl
{
public:
  __device__ BufferControl(
      uint8_t* const buffer,
      const uint8_t* const compData,
      const position_type length) :
      m_offset(0),
      m_length(length),
      m_buffer(buffer),
      m_compData(compData)
  {
    // do nothing
  }
  inline __device__ position_type readLSIC(position_type& idx) const
  {
    position_type num = 0;
    uint8_t next = 0xff;
    // read from the buffer
    while (next == 0xff && idx < end()) {
      next = rawAt(idx)[0];
      ++idx;
      num += next;
    }
    // read from global memory
    while (next == 0xff) {
      next = m_compData[idx];
      ++idx;
      num += next;
    }
    return num;
  }

  inline __device__ const uint8_t* raw() const
  {
    return m_buffer;
  }

  inline __device__ const uint8_t* rawAt(const position_type i) const
  {
    return raw() + (i - begin());
  }
  inline __device__ uint8_t operator[](const position_type i) const
  {
    if (i >= m_offset && i - m_offset < DECOMP_INPUT_BUFFER_SIZE) {
      return m_buffer[i - m_offset];
    } else {
      return m_compData[i];
    }
  }

  inline __device__ void setAndAlignOffset(const position_type offset)
  {
    static_assert(
        sizeof(size_t) == sizeof(const uint8_t*),
        "Size of pointer must be equal to size_t.");

    const uint8_t* const alignedPtr = reinterpret_cast<const uint8_t*>(
        (reinterpret_cast<size_t>(m_compData + offset)
         / sizeof(double_word_type))
        * sizeof(double_word_type));

    m_offset = alignedPtr - m_compData;
  }

  inline __device__ void loadAt(const position_type offset)
  {
    setAndAlignOffset(offset);

    if (m_offset + DECOMP_INPUT_BUFFER_SIZE <= m_length) {
      assert(
          reinterpret_cast<size_t>(m_compData + m_offset)
              % sizeof(double_word_type)
          == 0);
      assert(
          DECOMP_INPUT_BUFFER_SIZE
          == DECOMP_THREADS_PER_CHUNK * sizeof(double_word_type));
      const double_word_type* const word_data
          = reinterpret_cast<const double_word_type*>(m_compData + m_offset);
      double_word_type* const word_buffer
          = reinterpret_cast<double_word_type*>(m_buffer);
      word_buffer[threadIdx.x] = word_data[threadIdx.x];
    } else {
#pragma unroll
      for (int i = threadIdx.x; i < DECOMP_INPUT_BUFFER_SIZE;
           i += DECOMP_THREADS_PER_CHUNK) {
        if (m_offset + i < m_length) {
          m_buffer[i] = m_compData[m_offset + i];
        }
      }
    }

    syncCTA();
  }

  inline __device__ position_type begin() const
  {
    return m_offset;
  }

  inline __device__ position_type end() const
  {
    return m_offset + DECOMP_INPUT_BUFFER_SIZE;
  }

private:
  // may potentially be negative for mis-aligned m_compData.
  int64_t m_offset;
  const position_type m_length;
  uint8_t* const m_buffer;
  const uint8_t* const m_compData;
}; // End BufferControl Class



inline __device__ void coopCopyNoOverlap(
    uint8_t* const dest,
    const uint8_t* const source,
    const position_type length)
{
  for (position_type i = threadIdx.x; i < length; i += blockDim.x) {
    dest[i] = source[i];
  }
}

inline __device__ void coopCopyRepeat(
    uint8_t* const dest,
    const uint8_t* const source,
    const position_type dist,
    const position_type length)
{
  // if there is overlap, it means we repeat, so we just
  // need to organize our copy around that
  for (position_type i = threadIdx.x; i < length; i += blockDim.x) {
    dest[i] = source[i % dist];
  }
}

inline __device__ void coopCopyOverlap(
    uint8_t* const dest,
    const uint8_t* const source,
    const position_type dist,
    const position_type length)
{
  if (dist < length) {
    coopCopyRepeat(dest, source, dist, length);
  } else {
    coopCopyNoOverlap(dest, source, length);
  }
}

inline __device__ token_type decodePair(const uint8_t num)
{
  return token_type{static_cast<uint8_t>((num & 0xf0) >> 4),
                    static_cast<uint8_t>(num & 0x0f)};
}


inline __device__ position_type lengthOfMatch(
    const uint8_t* const data,
    const position_type prev_location,
    const position_type next_location,
    const position_type length)
{
  assert(prev_location < next_location);

  position_type match_length = length - next_location - 5;
  for (position_type j = 0; j + next_location + 5 < length; j += blockDim.x) {
    const position_type i = threadIdx.x + j;
    int match = i + next_location + 5 < length
                    ? (data[prev_location + i] != data[next_location + i])
                    : 1;
    match = warpBallot(match);
    if (match) {
      match_length = j + __clz(__brev(match));
      break;
    }
  }

  return match_length;
}

inline __device__ void decompressStream(
    uint8_t* buffer,
    uint8_t* decompData,
    const uint8_t* compData,
    const position_type comp_end)
{
  BufferControl ctrl(buffer, compData, comp_end);
  ctrl.loadAt(0);

  position_type decomp_idx = 0;
  position_type comp_idx = 0;

  while (comp_idx < comp_end) {
    if (comp_idx + DECOMP_BUFFER_PREFETCH_DIST > ctrl.end()) {
      ctrl.loadAt(comp_idx);
    }

    // read header byte
    token_type tok = decodePair(*ctrl.rawAt(comp_idx));
    ++comp_idx;

    // read the length of the literals
    position_type num_literals = tok.num_literals;
    if (tok.num_literals == 15) {
      num_literals += ctrl.readLSIC(comp_idx);
    }
    const position_type literalStart = comp_idx;

    // copy the literals to the out stream
    if (num_literals + comp_idx > ctrl.end()) {
      coopCopyNoOverlap(
          decompData + decomp_idx, compData + comp_idx, num_literals);
    } else {
      // our buffer can copy
      coopCopyNoOverlap(
          decompData + decomp_idx, ctrl.rawAt(comp_idx), num_literals);
    }

    comp_idx += num_literals;
    decomp_idx += num_literals;

    // Note that the last sequence stops right after literals field.
    // There are specific parsing rules to respect to be compatible with the
    // reference decoder : 1) The last 5 bytes are always literals 2) The last
    // match cannot start within the last 12 bytes Consequently, a file with
    // less then 13 bytes can only be represented as literals These rules are in
    // place to benefit speed and ensure buffer limits are never crossed.
    if (comp_idx < comp_end) {

      // read the offset
      offset_type offset;
      if (comp_idx + sizeof(offset_type) > ctrl.end()) {
        offset = readWord(compData + comp_idx);
      } else {
        offset = readWord(ctrl.rawAt(comp_idx));
      }

      comp_idx += sizeof(offset_type);

      // read the match length
      position_type match = 4 + tok.num_matches;
      if (tok.num_matches == 15) {
        match += ctrl.readLSIC(comp_idx);
      }

      // copy match
      if (offset <= num_literals
          && (ctrl.begin() <= literalStart
              && ctrl.end() >= literalStart + num_literals)) {
        // we are using literals already present in our buffer

        coopCopyOverlap(
            decompData + decomp_idx,
            ctrl.rawAt(literalStart + (num_literals - offset)),
            offset,
            match);
        // we need to sync after we copy since we use the buffer
        syncCTA();
      } else {
        // we need to sync before we copy since we use decomp
        syncCTA();

        coopCopyOverlap(
            decompData + decomp_idx,
            decompData + decomp_idx - offset,
            offset,
            match);
      }
      decomp_idx += match;
    }
  }
  assert(comp_idx == comp_end);
}

inline __device__ uint32_t read32be( const uint8_t* address )
{
    return ( (uint32_t)(255 & (address)[0]) << 24 | (uint32_t)(255 & (address)[1]) << 16 |\\
             (uint32_t)(255 & (address)[2]) <<  8 |(uint32_t)(255 & (address)[3])       ) ;
}

__global__ void lz4dc_forBSLZ4 (
    const uint8_t* const device_in_ptr,    /* compressed start block pointer */
    const uint32_t* const device_in_pos,      /* data starts */
    const uint32_t batch_size,                   /* number of blocks */
    const uint32_t blocksize,                    /* blocksize */
    uint8_t* const device_out_ptr,        /* destination start pointer */
    const uint32_t copies,                 /* bytes to copy at the end */
    const uint32_t copystart
    )
{

  const int bid = blockIdx.x * blockDim.y + threadIdx.y;
  // threadIdx.x is 0->31 for parallel copies

  __shared__ uint8_t buffer[DECOMP_INPUT_BUFFER_SIZE * DECOMP_CHUNKS_PER_BLOCK];

  if (bid < batch_size) {

    size_t offset = device_in_pos[bid];
    const position_type chunk_length = read32be( device_in_ptr + offset );
    uint8_t* const decomp_ptr = device_out_ptr + bid * blocksize;
    const uint8_t* const comp_ptr = device_in_ptr + offset + 4;

    decompressStream(
        buffer + threadIdx.y * DECOMP_INPUT_BUFFER_SIZE,
        decomp_ptr,
        comp_ptr,
        chunk_length);

    /* remaining bytes at the end of the stream. One per thread.
       These should always be less than or equal to 32 == 4(bytes) * 8(bits)
    */
    if ( bid == (batch_size-1) && threadIdx.x < copies && threadIdx.y == 0 ){
        uint8_t * dest = device_out_ptr + copystart + threadIdx.x;
        const uint8_t * src = comp_ptr + chunk_length + threadIdx.x;
        *dest = *src;
    }
  }
}


__global__ void simple_shuffle(const uint8_t * __restrict__ in, uint8_t * __restrict__ out,
                          const uint32_t blocksize, const uint32_t total_bytes, const uint32_t elemsize ) {

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
"""




class BSLZ4CUDA:
    def __init__(self, total_output_bytes ):
        """ cache / reuse memory """
        self.total_output_bytes = total_output_bytes
        self.mod = SourceModule( modsrc )
        self.lz4dc_forBSLZ4 = self.mod.get_function("lz4dc_forBSLZ4")
        self.simple_shuffle = self.mod.get_function("simple_shuffle")
        self.output_d = drv.mem_alloc( total_output_bytes ) # holds the lz4 decompressed, still shuffled data
        self.shuf_d =   drv.mem_alloc( total_output_bytes ) # holds the final unshuffled data

    def __call__(self, chunk, shape, dtyp, blocksize, blocks, out=None ):
        bpp = dtyp.itemsize
        if out is not None:
            total_output_bytes = shape[0]*shape[1]*dtyp.itemsize
            output = np.empty( total_output_bytes, np.uint8 )
        else:
            output = out.ravel().view( np.uint8 )
        # LZ4 blocking :
        # a single block is 32 threads and this handles 2 chunks
        lz4block = (32,2,1)
        lz4grid = ((len(blocks)+2)//2,1,1)
        copies  = ( self.total_output_bytes % ( bpp * 8 ) )
        copystart = self.total_output_bytes - copies
        # shuffle blocking:
        shblock = (32,1,1)   # no good reason. Was told 32 in a warp.
        shgrid  = ((output.nbytes + 31) // 32, 1, 1) # Going to be wrong for 16 bits ?
        self.chunk_d = drv.mem_alloc( chunk.nbytes )        # the compressed data
        self.blocks_d = drv.mem_alloc( blocks.nbytes )      # 32 bit offsets into the compressed for blocks

        # drv.memcpy_htod( chunk_d, chunk )         # compressed data on the device
        # drv.memcpy_htod( blocks_d, blocks )
        self.lz4dc_forBSLZ4( drv.In( chunk  ),
                             drv.In( blocks ),
                             np.int32(len(blocks)),
                             np.int32(blocksize),
                             self.output_d,
                             np.int32(copies),               # todo : move the copy to directly go to output ?
                             np.int32(copystart),
                            block=lz4block, grid=lz4grid)

        self.simple_shuffle( self.output_d, self.shuf_d, np.uint32( blocksize ),
                            np.uint32( ref.nbytes ), np.uint32( ref.itemsize ),
                            block = shblock, grid = shgrid )
        # last block
        drv.memcpy_dtoh( output,  self.shuf_d )
        return output.view( dtyp ).reshape( shape )


import read_chunks, timeit
hname = "bslz4testcases.h5"
dset  = "data_uint32"

ref = h5py.File( hname, 'r' )[dset][0]

chunk, shape, dtyp  = read_chunks.get_chunk( hname, dset, 0 )
total_output_elem  = shape[1]*shape[2]
total_output_bytes = total_output_elem*dtyp.itemsize
assert total_output_bytes == ref.nbytes
assert total_output_elem == ref.size
blocksize, blocks = read_chunks.get_blocks( chunk, shape, dtyp )
output = np.empty( total_output_elem, dtyp )

start = timeit.default_timer()
decompressor = BSLZ4CUDA( total_output_bytes )
now   = timeit.default_timer()
print( "Create cuda thing",  now-start )

start = timeit.default_timer()
decomp = decompressor( chunk, (shape[1],shape[2]), dtyp, blocksize, blocks, out=output )
now = timeit.default_timer()
dt = now - start
print( "Call cuda thing %.3f ms, %.3f GB/s (metric)" % ( dt * 1000, decomp.nbytes / dt / 1e9 ) )
print( "total bytes = %d"%(decomp.nbytes))


if (ref==decomp).all():
    print("Test passes!!!")
else:
    print("FAILS!!!")
    print("decomp:")
    print(decomp.ravel()[:128])
    print("ref:")
    print(ref)
    err = abs(ref-decomp).ravel()
    ierr  = np.argmax( err)
    print(ierr, decomp.ravel()[ierr], ref.ravel()[ierr] )
    print(ref.ravel()[-10:])
    print(decomp.ravel()[-10:])
    import pylab as pl
    pl.imshow(ref,aspect='auto',interpolation='nearest')
    pl.figure()
    pl.imshow(output.view( ref.dtype ).reshape(ref.shape) - ref, aspect='auto', interpolation='nearest')
    pl.show()

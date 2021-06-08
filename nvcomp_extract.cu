
/* Extracted from https://github.com/NVIDIA/nvcomp
        src/lowlevel/LZ4CompressionKernels.cu

   Edited in order to :
      - remove compression and only keep decompression
      - suppress assert for pycuda caching

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
#define NDEBUG
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


#undef NDEBUG

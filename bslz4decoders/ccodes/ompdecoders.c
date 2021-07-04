
/* A curated collection of different BSLZ4 readers
   This is automatically generated code
   Edit this to change the original :
     codegen.py
   Created on :
     Sun Jul  4 16:17:30 2021
   Code generator written by Jon Wright.
*/

#include <stdint.h> /* uint32_t etc */
#include <stdio.h>  /* print error message before killing process(!?!?) */
#include <stdlib.h> /* malloc and friends */
#include <string.h> /* memcpy */

#ifdef USEIPP
#warning using ipp
#include <ippdc.h> /* for intel ... going to need a platform build system */
#else
#warning not using ipp
#include <lz4.h> /* assumes you have this already */
#endif

/* see https://justine.lol/endian.html */
#define READ32BE(p)                                                            \
  ((uint32_t)(255 & (p)[0]) << 24 | (uint32_t)(255 & (p)[1]) << 16 |           \
   (uint32_t)(255 & (p)[2]) << 8 | (uint32_t)(255 & (p)[3]))
#define READ64BE(p)                                                            \
  ((uint64_t)(255 & (p)[0]) << 56 | (uint64_t)(255 & (p)[1]) << 48 |           \
   (uint64_t)(255 & (p)[2]) << 40 | (uint64_t)(255 & (p)[3]) << 32 |           \
   (uint64_t)(255 & (p)[4]) << 24 | (uint64_t)(255 & (p)[5]) << 16 |           \
   (uint64_t)(255 & (p)[6]) << 8 | (uint64_t)(255 & (p)[7]))

#define ERRVAL -1

#define ERR(s)                                                                 \
  {                                                                            \
    fprintf(stderr, "ERROR %s\n", s);                                          \
    return ERRVAL;                                                             \
  }

#define CHECK_RETURN_VALS 1
/* Signature for omp_bslz4_make_starts_func */
int omp_bslz4(const uint8_t *, size_t, int, uint8_t *, size_t, uint8_t *,
              size_t, int);
/* Signature for omp_bslz4_with_starts_func */
int omp_bslz4_blocks(const uint8_t *, size_t, int, size_t, uint32_t *, int,
                     uint8_t *, size_t, uint8_t *, size_t, int);
/* Signature for omp_get_threads_func */
int omp_get_threads_used(int);
/* Definition for omp_bslz4_make_starts_func */
int omp_bslz4(const uint8_t *compressed, size_t compressed_length, int itemsize,
              uint8_t *output, size_t output_length, uint8_t *tmp,
              size_t tmp_length, int num_threads) {
  /* begin: total_length */

  size_t total_output_length;
  total_output_length = READ64BE(compressed);

  /* ends: total_length */
  /* begin: read_blocksize */

  int blocksize;
  blocksize = (int)READ32BE((compressed + 8));
  if (blocksize == 0) {
    blocksize = 8192;
  }

  /* ends: read_blocksize */
  /* begin: blocks_length */

  int blocks_length;
  blocks_length =
      (int)((total_output_length + (size_t)blocksize - 1) / (size_t)blocksize);

  /* ends: blocks_length */
  /* begin: create_starts */

  uint32_t *blocks;
  blocks = (uint32_t *)malloc(((size_t)blocks_length) * sizeof(uint32_t));
  if (blocks == NULL) {
    ERR("small malloc failed");
  }

  /* ends: create_starts */
  /* begin: read_starts */

  blocks[0] = 12;
  for (int i = 1; i < blocks_length; i++) {
    int nbytes = (int)READ32BE((compressed + blocks[i - 1]));
    blocks[i] = (uint32_t)(nbytes + 4) + blocks[i - 1];
    if (blocks[i] >= compressed_length) {
      ERR("Overflow reading starts");
    }
  }

  /* ends: read_starts */
  /* begin: omp_bslz4 */

  /* allow user to set the number of threads */
  if (num_threads == 0)
    num_threads = omp_get_max_threads();
  if ((blocksize * num_threads) > tmp_length)
    return -1;
  int error = 0;
  {
    int i; /* msvc does not let you put this inside the for */
#pragma omp parallel for num_threads(num_threads) shared(error)
    for (i = 0; i < blocks_length - 1; i++) {
      int t0 = omp_get_thread_num() * blocksize;
#ifdef USEIPP
      int bsize = blocksize;
      IppStatus ret = ippsDecodeLZ4_8u((Ipp8u *)&compressed[blocks[i] + 4u],
                                       (int)READ32BE(compressed + blocks[i]),
                                       &output[i * blocksize], &bsize);
      if (CHECK_RETURN_VALS && (ret != ippStsNoErr))
        error = 1;
#else
      int ret =
          LZ4_decompress_safe((char *)(compressed + blocks[i] + 4u),
                              (char *)(output + i * blocksize),
                              (int)READ32BE(compressed + blocks[i]), blocksize);
      if (CHECK_RETURN_VALS && (ret != (int)blocksize))
        error = 1;
#endif
      /* bitshuffle here */
      int64_t bref;
      bref = bshuf_trans_byte_bitrow_elem(&output[i * blocksize], &tmp[t0],
                                          blocksize / itemsize, itemsize);
      bref = bshuf_shuffle_bit_eightelem(&tmp[t0], &output[i * blocksize],
                                         blocksize / itemsize, itemsize);
    } /* end parallel loop */
    if (error)
      ERR("Error decoding LZ4");
    /* last block, might not be full blocksize */
    {
      int lastblock =
          (int)total_output_length - blocksize * (blocks_length - 1);
      /* last few bytes are copied flat */
      int copied = lastblock % (8 * itemsize);
      lastblock -= copied;
      memcpy(&output[total_output_length - (size_t)copied],
             &compressed[compressed_length - (size_t)copied], (size_t)copied);
      int nbytes = (int)READ32BE(compressed + blocks[blocks_length - 1]);
#ifdef USEIPP
      int bsize = lastblock;
      IppStatus ret = ippsDecodeLZ4_8u(
          (Ipp8u *)&compressed[blocks[blocks_length - 1] + 4u],
          (int)READ32BE(compressed + blocks[blocks_length - 1]),
          (Ipp8u *)&output[(blocks_length - 1) * blocksize], &bsize);
      if (CHECK_RETURN_VALS && (ret != ippStsNoErr))
        ERR("Error LZ4 block");
#else
      int ret = LZ4_decompress_safe(
          (char *)compressed + blocks[blocks_length - 1] + 4u,
          (char *)(output + (blocks_length - 1) * blocksize), nbytes,
          lastblock);
      if (CHECK_RETURN_VALS && (ret != lastblock))
        ERR("Error decoding last LZ4 block");
#endif
      /* bitshuffle here */
      int64_t bref;
      bref =
          bshuf_trans_byte_bitrow_elem(&output[(blocks_length - 1) * blocksize],
                                       &tmp[0], lastblock / itemsize, itemsize);
      bref = bshuf_shuffle_bit_eightelem(
          &tmp[0], &output[(blocks_length - 1) * blocksize],
          lastblock / itemsize, itemsize);
    }
  }

  /* ends: omp_bslz4 */
  /* begin: create_starts */
  free(blocks);
  /* end: create_starts */
  return 0;
}
/* Definition for omp_bslz4_with_starts_func */
int omp_bslz4_blocks(const uint8_t *compressed, size_t compressed_length,
                     int itemsize, size_t blocksize, uint32_t *blocks,
                     int blocks_length, uint8_t *output, size_t output_length,
                     uint8_t *tmp, size_t tmp_length, int num_threads) {
  /* begin: total_length */

  size_t total_output_length;
  total_output_length = READ64BE(compressed);

  /* ends: total_length */
  /* begin: omp_bslz4 */

  /* allow user to set the number of threads */
  if (num_threads == 0)
    num_threads = omp_get_max_threads();
  if ((blocksize * num_threads) > tmp_length)
    return -1;
  int error = 0;
  {
    int i; /* msvc does not let you put this inside the for */
#pragma omp parallel for num_threads(num_threads) shared(error)
    for (i = 0; i < blocks_length - 1; i++) {
      int t0 = omp_get_thread_num() * blocksize;
#ifdef USEIPP
      int bsize = blocksize;
      IppStatus ret = ippsDecodeLZ4_8u((Ipp8u *)&compressed[blocks[i] + 4u],
                                       (int)READ32BE(compressed + blocks[i]),
                                       &output[i * blocksize], &bsize);
      if (CHECK_RETURN_VALS && (ret != ippStsNoErr))
        error = 1;
#else
      int ret =
          LZ4_decompress_safe((char *)(compressed + blocks[i] + 4u),
                              (char *)(output + i * blocksize),
                              (int)READ32BE(compressed + blocks[i]), blocksize);
      if (CHECK_RETURN_VALS && (ret != (int)blocksize))
        error = 1;
#endif
      /* bitshuffle here */
      int64_t bref;
      bref = bshuf_trans_byte_bitrow_elem(&output[i * blocksize], &tmp[t0],
                                          blocksize / itemsize, itemsize);
      bref = bshuf_shuffle_bit_eightelem(&tmp[t0], &output[i * blocksize],
                                         blocksize / itemsize, itemsize);
    } /* end parallel loop */
    if (error)
      ERR("Error decoding LZ4");
    /* last block, might not be full blocksize */
    {
      int lastblock =
          (int)total_output_length - blocksize * (blocks_length - 1);
      /* last few bytes are copied flat */
      int copied = lastblock % (8 * itemsize);
      lastblock -= copied;
      memcpy(&output[total_output_length - (size_t)copied],
             &compressed[compressed_length - (size_t)copied], (size_t)copied);
      int nbytes = (int)READ32BE(compressed + blocks[blocks_length - 1]);
#ifdef USEIPP
      int bsize = lastblock;
      IppStatus ret = ippsDecodeLZ4_8u(
          (Ipp8u *)&compressed[blocks[blocks_length - 1] + 4u],
          (int)READ32BE(compressed + blocks[blocks_length - 1]),
          (Ipp8u *)&output[(blocks_length - 1) * blocksize], &bsize);
      if (CHECK_RETURN_VALS && (ret != ippStsNoErr))
        ERR("Error LZ4 block");
#else
      int ret = LZ4_decompress_safe(
          (char *)compressed + blocks[blocks_length - 1] + 4u,
          (char *)(output + (blocks_length - 1) * blocksize), nbytes,
          lastblock);
      if (CHECK_RETURN_VALS && (ret != lastblock))
        ERR("Error decoding last LZ4 block");
#endif
      /* bitshuffle here */
      int64_t bref;
      bref =
          bshuf_trans_byte_bitrow_elem(&output[(blocks_length - 1) * blocksize],
                                       &tmp[0], lastblock / itemsize, itemsize);
      bref = bshuf_shuffle_bit_eightelem(
          &tmp[0], &output[(blocks_length - 1) * blocksize],
          lastblock / itemsize, itemsize);
    }
  }

  /* ends: omp_bslz4 */
  return 0;
}
/* Definition for omp_get_threads_func */
int omp_get_threads_used(int num_threads) {

  if (num_threads == 0)
    num_threads = omp_get_max_threads();
  return num_threads;

  return 0;
}

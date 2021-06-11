
/* A curated collection of different BSLZ4 readers
   This is automatically generated code
   Edit this to change the original :
     C:\Users\wright\Work
   Folders\Documents\programming\github\jonwright\bslz4decoders\bslz4decoders\ccodes\codegen.py
   Created on :
     Wed Jun  9 15:51:51 2021
   Code generator written by Jon Wright.
*/

#include <stdint.h> /* uint32_t etc */
#include <stdio.h>  /* print error message before killing process(!?!?) */
#include <stdlib.h> /* malloc and friends */
#include <string.h> /* memcpy */

#ifdef USEIPP
#include <ippdc.h> /* for intel ... going to need a platform build system */
#endif

#include <lz4.h> /* assumes you have this already */

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
/* Signature for onecore_lz4_func */
int onecore_lz4(const char *, size_t, int, char *, size_t);
/* Signature for print_offsets_func */
int print_offsets(const char *, size_t, int);
/* Signature for read_starts_func */
int read_starts(const char *, size_t, int, int, uint32_t *, int);
/* Definition for onecore_lz4_func */
int onecore_lz4(const char *compressed, size_t compressed_length, int itemsize,
                char *output, size_t output_length) {
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
  /* begin: onecore_lz4 */

  int p = 12;
  for (int i = 0; i < blocks_length - 1; ++i) {
    int nbytes = (int)READ32BE(&compressed[p]);
#ifdef USEIPP
    IppStatus ret = ippsDecodeLZ4_8u(&compressed[p + 4], nbytes,
                                     &output[i * blocksize], &bsize);
    if (CHECK_RETURN_VALS && (ret != ippStsNoErr))
      ERR("Error LZ4 block");
#else
    int ret = LZ4_decompress_safe(&compressed[p + 4], &output[i * blocksize],
                                  nbytes, blocksize);
    if (CHECK_RETURN_VALS && (ret != blocksize))
      ERR("Error LZ4 block");
#endif
    p = p + nbytes + 4;
  }
  /* last block, might not be full blocksize */
  {
    int lastblock = (int)total_output_length - blocksize * (blocks_length - 1);
    /* last few bytes are copied flat */
    int copied = lastblock % (8 * itemsize);
    lastblock -= copied;
    memcpy(&output[total_output_length - (size_t)copied],
           &compressed[compressed_length - (size_t)copied], (size_t)copied);

    int nbytes = (int)READ32BE(&compressed[p]);
#ifdef USEIPP
    int bsize = blocksize;
    IppStatus ret =
        ippsDecodeLZ4_8u(&compressed[p + 4], nbytes,
                         &output[(blocks_length - 1) * blocksize], &bsize);
    if (CHECK_RETURN_VALS && (ret != ippStsNoErr))
      ERR("Error LZ4 block");
#else
    int ret = LZ4_decompress_safe(&compressed[p + 4],
                                  &output[(blocks_length - 1) * blocksize],
                                  nbytes, lastblock);
    if (CHECK_RETURN_VALS && (ret != lastblock))
      ERR("Error decoding last LZ4 block");
#endif
  }

  /* ends: onecore_lz4 */
  return 0;
}
/* Definition for print_offsets_func */
int print_offsets(const char *compressed, size_t compressed_length,
                  int itemsize) {
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
  /* begin: print_starts */

  printf("total_output_length %ld\n", total_output_length);
  printf("blocks_length %d\n", blocks_length);
  for (int i = 0; i < blocks_length; i++)
    printf("%d %d, ", i, blocks[i]);
  printf("About to free and return\n");

  /* ends: print_starts */
  /* begin: create_starts */
  free(blocks);
  /* end: create_starts */
  return 0;
}
/* Definition for read_starts_func */
int read_starts(const char *compressed, size_t compressed_length, int itemsize,
                int blocksize, uint32_t *blocks, int blocks_length) {
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
  return 0;
}
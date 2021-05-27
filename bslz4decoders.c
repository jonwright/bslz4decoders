
/* A curated collection of different BSLZ4 readers
   This is automatically generated code
   Edit this to change the original :
     /home/jon/Documents/bslz4_codecs/codegen.py
   Created on :
     Thu May 27 21:03:42 2021
   Code generator written by Jon Wright.
*/

#include <hdf5.h>   /* to grab chunks independently of h5py api (py27 issue) */
#include <lz4.h>    /* assumes you have this already */
#include <stdint.h> /* uint32_t etc */
#include <stdio.h>  /* print error message before killing process(!?!?) */
#include <stdlib.h> /* malloc and friends */
#include <string.h> /* memcpy */

#ifdef USEIPP
#include <ippdc.h> /* for intel ... going to need a platform build system */
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
/* Signature for h5_read_direct */
size_t h5_read_direct(int64_t, int, char *, size_t);
/* Signature for omp_lz4_make_starts_func */
int omp_lz4(const char *, size_t, int, char *, size_t);
/* Signature for omp_lz4_with_starts_func */
int omp_lz4_blocks(const char *, size_t, int, int, uint32_t *, int, char *,
                   size_t);
/* Signature for onecore_lz4_func */
int onecore_lz4(const char *, size_t, int, char *, size_t);
/* Signature for print_offsets_func */
int print_offsets(const char *, size_t, int);
/* Signature for read_starts_func */
int read_starts(const char *, size_t, int, int, uint32_t *, int);
/* Definition for h5_read_direct */
size_t h5_read_direct(int64_t dataset_id, int frame, char *chunk,
                      size_t chunk_length) {

  {
    /* see:
       https://support.hdfgroup.org/HDF5/doc/HL/RM_HDF5Optimized.html#H5DOread_chunk

       ... assuming this is h5py.dataset.id.id :
        hid_t dataset;
        if((dataset = H5Dopen2(hname, dsetname, H5P_DEFAULT)) < 0)
            ERR("Failed to open h5file");
    */
    hsize_t offset[3];
    offset[0] = (hsize_t)frame; /* assumes 3D frame-by-frame chunks */
    offset[1] = 0;
    offset[2] = 0;

    /* Get the size of the compressed chunk to return */
    hsize_t chunk_nbytes;
    herr_t ret;
    ret = H5Dget_chunk_storage_size(dataset_id, offset, &chunk_nbytes);
    if (chunk_nbytes > chunk_length) {
      fprintf(stderr, "Chunk does not fit into your arg");
      return 0;
    }
    if (ret < 0) {
      fprintf(stderr, "Problem getting storage size for the chunk");
      return 0;
    }
    /* Use H5DOread_chunk() to read the chunk back
       ... becomes H5Dread_chunk in later library versions */
    uint32_t read_filter_mask;
    ret = H5Dread_chunk(dataset_id, H5P_DEFAULT, offset, &read_filter_mask,
                        chunk);
    if (ret < 0) {
      fprintf(stderr, "error reading chunk");
      return 0;
    }
    if (read_filter_mask != 0) {
      fprintf(stderr, "chunk was filtered");
      return 0;
    }
    return chunk_nbytes;
  }

} /* Definition for omp_lz4_make_starts_func */
int omp_lz4(const char *compressed, size_t compressed_length, int itemsize,
            char *output, size_t output_length) {
  /* begin: chunks_2_blocks */

  size_t total_output_length;
  total_output_length = READ64BE(compressed);
  int blocksize;
  blocksize = (int)READ32BE((compressed + 8));
  if (blocksize == 0) {
    blocksize = 8192;
  }

  /* ends: chunks_2_blocks */
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
  /* begin: omp_lz4 */

  int error = 0;
#pragma omp parallel for shared(error)
  for (int i = 0; i < blocks_length - 1; i++) {
#ifdef USEIPP
    int bsize = blocksize;
    IppStatus ret = ippsDecodeLZ4_8u(&compressed[blocks[i] + 4u],
                                     (int)READ32BE(compressed + blocks[i]),
                                     &output[i * blocksize], &bsize);
    if (CHECK_RETURN_VALS && (ret != ippStsNoErr))
      ERR("Error LZ4 block");
#else
    int ret =
        LZ4_decompress_safe(compressed + blocks[i] + 4u, output + i * blocksize,
                            (int)READ32BE(compressed + blocks[i]), blocksize);
    if (CHECK_RETURN_VALS && (ret != blocksize))
      error = 1;
#endif
  }
  if (error)
    ERR("Error decoding LZ4");
  /* last block, might not be full blocksize */
  {
    int lastblock = (int)output_length - blocksize * (blocks_length - 1);
    /* last few bytes are copied flat */
    int copied = lastblock % (8 * itemsize);
    lastblock -= copied;
    memcpy(&output[output_length - (size_t)copied],
           &compressed[compressed_length - (size_t)copied], (size_t)copied);
    int nbytes = (int)READ32BE(compressed + blocks[blocks_length - 1]);
#ifdef USEIPP
    int bsize = lastblock;
    IppStatus ret =
        ippsDecodeLZ4_8u(&compressed[blocks[blocks_length - 1] + 4u],
                         (int)READ32BE(compressed + blocks[blocks_length - 1]),
                         &output[(blocks_length - 1) * blocksize], &bsize);
    if (CHECK_RETURN_VALS && (ret != ippStsNoErr))
      ERR("Error LZ4 block");
#else
    int ret = LZ4_decompress_safe(compressed + blocks[blocks_length - 1] + 4u,
                                  output + (blocks_length - 1) * blocksize,
                                  nbytes, lastblock);
    if (CHECK_RETURN_VALS && (ret != lastblock))
      ERR("Error decoding last LZ4 block");
#endif
  }

  /* ends: omp_lz4 */
  /* begin: create_starts */
  free(blocks);
  /* end: create_starts */
  return 0;
}
/* Definition for omp_lz4_with_starts_func */
int omp_lz4_blocks(const char *compressed, size_t compressed_length,
                   int itemsize, int blocksize, uint32_t *blocks,
                   int blocks_length, char *output, size_t output_length) {
  /* begin: omp_lz4 */

  int error = 0;
#pragma omp parallel for shared(error)
  for (int i = 0; i < blocks_length - 1; i++) {
#ifdef USEIPP
    int bsize = blocksize;
    IppStatus ret = ippsDecodeLZ4_8u(&compressed[blocks[i] + 4u],
                                     (int)READ32BE(compressed + blocks[i]),
                                     &output[i * blocksize], &bsize);
    if (CHECK_RETURN_VALS && (ret != ippStsNoErr))
      ERR("Error LZ4 block");
#else
    int ret =
        LZ4_decompress_safe(compressed + blocks[i] + 4u, output + i * blocksize,
                            (int)READ32BE(compressed + blocks[i]), blocksize);
    if (CHECK_RETURN_VALS && (ret != blocksize))
      error = 1;
#endif
  }
  if (error)
    ERR("Error decoding LZ4");
  /* last block, might not be full blocksize */
  {
    int lastblock = (int)output_length - blocksize * (blocks_length - 1);
    /* last few bytes are copied flat */
    int copied = lastblock % (8 * itemsize);
    lastblock -= copied;
    memcpy(&output[output_length - (size_t)copied],
           &compressed[compressed_length - (size_t)copied], (size_t)copied);
    int nbytes = (int)READ32BE(compressed + blocks[blocks_length - 1]);
#ifdef USEIPP
    int bsize = lastblock;
    IppStatus ret =
        ippsDecodeLZ4_8u(&compressed[blocks[blocks_length - 1] + 4u],
                         (int)READ32BE(compressed + blocks[blocks_length - 1]),
                         &output[(blocks_length - 1) * blocksize], &bsize);
    if (CHECK_RETURN_VALS && (ret != ippStsNoErr))
      ERR("Error LZ4 block");
#else
    int ret = LZ4_decompress_safe(compressed + blocks[blocks_length - 1] + 4u,
                                  output + (blocks_length - 1) * blocksize,
                                  nbytes, lastblock);
    if (CHECK_RETURN_VALS && (ret != lastblock))
      ERR("Error decoding last LZ4 block");
#endif
  }

  /* ends: omp_lz4 */
  return 0;
}
/* Definition for onecore_lz4_func */
int onecore_lz4(const char *compressed, size_t compressed_length, int itemsize,
                char *output, size_t output_length) {
  /* begin: chunks_2_blocks */

  size_t total_output_length;
  total_output_length = READ64BE(compressed);
  int blocksize;
  blocksize = (int)READ32BE((compressed + 8));
  if (blocksize == 0) {
    blocksize = 8192;
  }

  /* ends: chunks_2_blocks */
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
    int lastblock = (int)output_length - blocksize * (blocks_length - 1);
    /* last few bytes are copied flat */
    int copied = lastblock % (8 * itemsize);
    lastblock -= copied;
    memcpy(&output[output_length - (size_t)copied],
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
  /* begin: chunks_2_blocks */

  size_t total_output_length;
  total_output_length = READ64BE(compressed);
  int blocksize;
  blocksize = (int)READ32BE((compressed + 8));
  if (blocksize == 0) {
    blocksize = 8192;
  }

  /* ends: chunks_2_blocks */
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

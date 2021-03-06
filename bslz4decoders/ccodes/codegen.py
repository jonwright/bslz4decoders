

import subprocess, tempfile, os, time, sys

from ctools import *
# C-code generic header things and function writers ...

# the functions we have defined
FUNCS = {}

#
INCLUDES = """
/* A curated collection of different BSLZ4 readers
   This is automatically generated code
   Edit this to change the original :
     %s
   Created on :
     %s
   Code generator written by Jon Wright.
*/

#include <stdlib.h>   /* malloc and friends */
#include <stdint.h>   /* uint32_t etc */
#include <string.h>   /* memcpy */
#include <stdio.h>    /* print error message before killing process(!?!?) */

"""% ( __file__, time.ctime())

BSH = """
#include "bitshuffle_core.h"
"""

LZ4H = """ 
#ifdef USEIPP
#warning using ipp
#include <ippdc.h>    /* for intel ... going to need a platform build system */
#else
#warning not using ipp
#include <lz4.h>      /* assumes you have this already */
#endif
"""

H5H = """#include <hdf5.h>     /* to grab chunks independently of h5py api (py27 issue) */
"""


MACROS = """
/* see https://justine.lol/endian.html */
#define READ32BE(p) \\
  ( (uint32_t)(255 & (p)[0]) << 24 |\\
    (uint32_t)(255 & (p)[1]) << 16 |\\
    (uint32_t)(255 & (p)[2]) <<  8 |\\
    (uint32_t)(255 & (p)[3])       )
#define READ64BE(p) \\
  ( (uint64_t)(255 & (p)[0]) << 56 |\\
    (uint64_t)(255 & (p)[1]) << 48 |\\
    (uint64_t)(255 & (p)[2]) << 40 |\\
    (uint64_t)(255 & (p)[3]) << 32 |\\
    (uint64_t)(255 & (p)[4]) << 24 |\\
    (uint64_t)(255 & (p)[5]) << 16 |\\
    (uint64_t)(255 & (p)[6]) <<  8 |\\
    (uint64_t)(255 & (p)[7])       )

#define ERRVAL -1

#define ERR(s) \\
  { fprintf( stderr, \"ERROR %s\\n\", s); \\
    return ERRVAL; }

#define CHECK_RETURN_VALS 1
"""

# To interpret a chunk coming from hdf5. Includes the 12 byte header.
chunk_args = [  "const uint8_t * compressed",
                "size_t compressed_length",
                "int itemsize" ]
# To write output data
output_args = [ "uint8_t * output",
                "size_t output_length" ]
# Scratch space as shuffle is not inplace
# moved to stack ... # shuf_args = [ "uint8_t * tmp", "size_t tmp_length" ]

# Same as for chunks, but includes an array of blocks.
# These point to the 4 byte BE int which prefix each lz4 block.
blocklist_args = chunk_args +  [ "size_t blocksize", "uint32_t * blocks", "int blocks_length" ]

# What follows are a series of small fragments of C code that we will paste together
# later.
chunkdecoder = cfragments(
    # To convert an input to get the block start positions
    total_length = cfrag( """
    size_t total_output_length;
    total_output_length = READ64BE( compressed );
    """),
    read_blocksize = cfrag( """
    int blocksize;
    blocksize = (int) READ32BE( (compressed+8) );
    if (blocksize == 0) { blocksize = 8192; }
    """ ),
    # To compute the number of blocks : you cannot now pass this in
    blocks_length = cfrag( """
    int blocks_length;
    blocks_length = (int)( (total_output_length + (size_t) blocksize - 1) / (size_t) blocksize );
    """ ),
    create_starts = cfrag( """
    uint32_t  * blocks;
    blocks = (uint32_t *) malloc( ((size_t) blocks_length) * sizeof( uint32_t ) );
    if (blocks == NULL) {
       ERR("small malloc failed");
    }
    """ , # Defers the free to the end of the block:
    "   free( blocks );" ),
    read_starts = cfrag( """
    blocks[0] = 12;
    for( int i = 1; i < blocks_length ; i++ ){
        int nbytes = (int) READ32BE( ( compressed + blocks[i-1] ) );
        blocks[i] = (uint32_t)(nbytes + 4) + blocks[i-1];
        if ( blocks[i] >= compressed_length ){
            ERR("Overflow reading starts");
        }
    }
    """ ),
    print_starts = cfrag( """
    printf("total_output_length %ld\\n", total_output_length);
    printf("blocks_length %d\\n", blocks_length);
    for( int i = 0; i < blocks_length ; i++ )
        printf("%d %d, ", i, blocks[i]);
    printf("About to free and return\\n");
    """ ),
) #### end of chunkdecoder

# Now we start to build some functions:

FUNCS['print_offsets_func'] = cfunc(
    "int print_offsets", chunk_args,
        chunkdecoder(   "total_length",
                        "read_blocksize" ,
                        "blocks_length" ,
                        "create_starts",
                        "read_starts" ,
                        "print_starts" ) )

FUNCS['read_starts_func'] = cfunc(
    "int read_starts", blocklist_args, chunkdecoder( "read_starts" ))

# Now to decode with lz4 (no shuffle yet)
#    LZ4LIB_API int LZ4_decompress_safe
#           (const char* src, char* dst, int compressedSize, int dstCapacity);
lz4decoders = cfragments(
    # requires the blocks to be first created
    # this is a mess, needs a refactor / cleanup
    omp_bslz4_blocked = cfrag("""
    /* allow user to set the number of threads */
    if ( num_threads == 0 ) num_threads = omp_get_max_threads();
    if ( blocksize > 8192 ) return -99; /* fixed the stack space */
    int error=0;
    {
    int i; /* msvc does not let you put this inside the for */
#pragma omp parallel num_threads(num_threads) shared(error)
    {
    char tltmp[8192]; /* thread local */
#pragma omp for 
    for(  i = 0; i < blocks_length-1; i++ ){
#ifdef USEIPP
      int bsize = blocksize;
      IppStatus ret = ippsDecodeLZ4_8u( (Ipp8u*) &compressed[blocks[i] + 4u], (int) READ32BE( compressed + blocks[i] ),
                                         &output[i * blocksize], &bsize );
      if ( CHECK_RETURN_VALS && ( ret != ippStsNoErr ) ) error = 1;
#else
        int ret = LZ4_decompress_safe(  (char*)( compressed + blocks[i] + 4u),
                                        (char*)(output + i * blocksize),
                                           (int) READ32BE( compressed + blocks[i] ),
                                           blocksize );
        if ( CHECK_RETURN_VALS && (ret != (int) blocksize)) error = 1;
#endif
      /* bitshuffle here */
      int64_t bref;
      bref = bshuf_trans_byte_bitrow_elem( &output[i * blocksize], &tltmp[0], blocksize/itemsize, itemsize );
      bref =  bshuf_shuffle_bit_eightelem( &tltmp[0], &output[i * blocksize], blocksize/itemsize, itemsize );
    }
    }
 /* end parallel loop */
    if (error) ERR("Error decoding LZ4");
    /* last block, might not be full blocksize */
    {
      int lastblock = (int) total_output_length - blocksize * (blocks_length - 1);
      /* last few bytes are copied flat */
      int copied = lastblock % ( 8 * itemsize );
      lastblock -= copied;
      memcpy( &output[ total_output_length - (size_t) copied ],
              &compressed[ compressed_length - (size_t) copied ], (size_t) copied );
      int nbytes = (int) READ32BE( compressed + blocks[blocks_length - 1]);
#ifdef USEIPP
      int bsize = lastblock;
      IppStatus ret = ippsDecodeLZ4_8u( (Ipp8u*) &compressed[ blocks[blocks_length-1] + 4u],
                                        (int) READ32BE( compressed + blocks[blocks_length-1] ),
                                         (Ipp8u*) &output[(blocks_length-1) * blocksize], &bsize );
      if ( CHECK_RETURN_VALS && ( ret != ippStsNoErr ) ) ERR("Error LZ4 block");
#else
      int ret = LZ4_decompress_safe( (char* )compressed + blocks[blocks_length-1] + 4u,
                                     (char*) (output + (blocks_length-1) * blocksize),
                                     nbytes,
                                     lastblock );
      if ( CHECK_RETURN_VALS && ( ret != lastblock ) ) ERR("Error decoding last LZ4 block");
#endif
      /* bitshuffle here */
      int64_t bref;
      char tltmp[8192]; /* thread local */
      bref = bshuf_trans_byte_bitrow_elem( &output[(blocks_length-1) * blocksize], &tltmp[0], lastblock/itemsize, itemsize );
      bref =  bshuf_shuffle_bit_eightelem( &tltmp[0], &output[(blocks_length-1) * blocksize], lastblock/itemsize, itemsize );
      }
    return 0;
    }
    """),
    # this one is like the onescore, but parallel. All threads read all the offsets
    # we only decompress "our" work
    omp_bslz4 = cfrag("""
    /* allow user to set the number of threads */
    if ( num_threads == 0 ) num_threads = omp_get_max_threads();
    if ( blocksize > 8192 ) return -99; /* fixed the stack space */
    int error=0;
    {
#pragma omp parallel num_threads(num_threads) shared(error)
    {
    char tltmp[8192]; /* thread local */
    int nt = omp_get_num_threads();
    int id = omp_get_thread_num();

    int p = 12;
    int i = 0;
    for( i = 0; i < blocks_length - 1 ; i++ ){
       int nbytes = (int) READ32BE( &compressed[p] );
       if( (i % nt) == id) { /* do the decompression */
#ifdef USEIPP
         int bsize = blocksize;
         IppStatus ret = ippsDecodeLZ4_8u( (Ipp8u*) &compressed[p + 4], nbytes,
                                           (Ipp8u*) &output[i * blocksize], &bsize );
         if ( CHECK_RETURN_VALS && ( ret != ippStsNoErr ) ) error = 1;
#else
         int ret = LZ4_decompress_safe( (char*) &compressed[p + 4],
                                        (char*) &output[i * blocksize],
                                        nbytes,
                                        blocksize );
         if ( CHECK_RETURN_VALS && ( ret != blocksize ) ) error = 1;
#endif
         /* bitshuffle here */
         int64_t bref;
         bref = bshuf_trans_byte_bitrow_elem( &output[i * blocksize], &tltmp[0], blocksize/itemsize, itemsize );
         bref =  bshuf_shuffle_bit_eightelem( &tltmp[0], &output[i * blocksize], blocksize/itemsize, itemsize );
       } /* ends thread local work */
       p = p + nbytes + 4;
    } /* ends for loop */
    i = blocks_length - 1;
    /* last block, might not be full blocksize */
    if( (i % nt) == id) { /* do the decompression */
      int lastblock = (int) total_output_length - blocksize * (blocks_length - 1);
      /* last few bytes are copied flat */
      int copied = lastblock % ( 8 * itemsize );
      lastblock -= copied;
      memcpy( &output[ total_output_length - (size_t) copied ],
              &compressed[ compressed_length - (size_t) copied ], (size_t) copied );

      int nbytes = (int) READ32BE( &compressed[p] );
#ifdef USEIPP
      int bsize = lastblock;
      IppStatus ret = ippsDecodeLZ4_8u( (Ipp8u*) &compressed[p + 4], nbytes,
                                        (Ipp8u*) &output[(blocks_length-1)* blocksize], &bsize );
      if ( CHECK_RETURN_VALS && ( ret != ippStsNoErr ) ) error = 1;
#else
      int ret = LZ4_decompress_safe( (char*) &compressed[p + 4],
                                     (char*) &output[(blocks_length-1) * blocksize],
                                     nbytes,
                                     lastblock );
      if ( CHECK_RETURN_VALS && ( ret != lastblock ) ) error = 1;
#endif
      /* bitshuffle here */
      int64_t bref;
      bref = bshuf_trans_byte_bitrow_elem( &output[(blocks_length-1) * blocksize], &tltmp[0], lastblock/itemsize, itemsize );
      bref =  bshuf_shuffle_bit_eightelem( &tltmp[0], &output[(blocks_length-1) * blocksize], lastblock/itemsize, itemsize );
    }
    } /* end of parallel section */
    if(error) ERR("Error decoding LZ4");
    return 0;
    }
    """),
    # Also do the bitshuffle inside this loop (e.g. from L1)
    onecore_bslz4 = cfrag("""
    int p = 12;
    if(  blocksize > 8192 ){
       return -101;
    }
    char tmp[8192];
    for( int i = 0; i < blocks_length - 1 ; ++i ){
       int nbytes = (int) READ32BE( &compressed[p] );
#ifdef USEIPP
      int bsize = blocksize;
      IppStatus ret = ippsDecodeLZ4_8u( (Ipp8u*) &compressed[p + 4], nbytes,
                                        (Ipp8u*) &output[i * blocksize], &bsize );
      if ( CHECK_RETURN_VALS && ( ret != ippStsNoErr ) ) ERR("Error LZ4 block");
#else
       int ret = LZ4_decompress_safe( (char*) &compressed[p + 4],
                                      (char*) &output[i * blocksize],
                                      nbytes,
                                      blocksize );
      if ( CHECK_RETURN_VALS && ( ret != blocksize ) ) ERR("Error LZ4 block");
#endif
      /* bitshuffle here */
      int64_t bref;
      bref = bshuf_trans_byte_bitrow_elem( &output[i * blocksize], &tmp[0], blocksize/itemsize, itemsize );
      bref =  bshuf_shuffle_bit_eightelem( &tmp[0], &output[i * blocksize], blocksize/itemsize, itemsize );
      p = p + nbytes + 4;
    }
    /* last block, might not be full blocksize */
    {
      int lastblock = (int) total_output_length - blocksize * (blocks_length - 1);
      /* last few bytes are copied flat */
      int copied = lastblock % ( 8 * itemsize );
      lastblock -= copied;
      memcpy( &output[ total_output_length - (size_t) copied ],
              &compressed[ compressed_length - (size_t) copied ], (size_t) copied );

      int nbytes = (int) READ32BE( &compressed[p] );
#ifdef USEIPP
      int bsize = lastblock;
      IppStatus ret = ippsDecodeLZ4_8u( (Ipp8u*) &compressed[p + 4], nbytes,
                                        (Ipp8u*) &output[(blocks_length-1)* blocksize], &bsize );
      if ( CHECK_RETURN_VALS && ( ret != ippStsNoErr ) ) ERR("Error LZ4 block");
#else
      int ret = LZ4_decompress_safe( (char*) &compressed[p + 4],
                                     (char*) &output[(blocks_length-1) * blocksize],
                                     nbytes,
                                     lastblock );
      if ( CHECK_RETURN_VALS && ( ret != lastblock ) ) ERR("Error decoding last LZ4 block");
#endif
      /* bitshuffle here */
      int64_t bref;
      bref = bshuf_trans_byte_bitrow_elem( &output[(blocks_length-1) * blocksize], &tmp[0], lastblock/itemsize, itemsize );
      bref =  bshuf_shuffle_bit_eightelem( &tmp[0], &output[(blocks_length-1) * blocksize], lastblock/itemsize, itemsize );
    }
    return 0;
    """),
    
)


FUNCS['onecore_bslz4_func'] = cfunc(
    "int onecore_bslz4", chunk_args + output_args,
    (chunkdecoder+lz4decoders)(
        "total_length" ,
        "read_blocksize" ,
        "blocks_length",
        "onecore_bslz4") )

FUNCS['omp_bslz4_make_starts_func'] = cfunc(
    "int omp_bslz4", chunk_args + output_args + ['int num_threads',],
    (chunkdecoder+lz4decoders)(
        "total_length" ,
        "read_blocksize" ,
        "blocks_length",
        "omp_bslz4" ) )

FUNCS['omp_bslz4_with_starts_func'] = cfunc(
    "int omp_bslz4_blocks", blocklist_args + output_args + ['int num_threads',],
    (chunkdecoder+lz4decoders)( "total_length" ,"omp_bslz4_blocked" ) )

FUNCS['omp_get_threads_func'] = cfunc(
    'int omp_get_threads_used', ['int num_threads',],
    """
    if ( num_threads == 0 ) num_threads = omp_get_max_threads();
    return num_threads;
    """
)

# This next one is very, very, flaky. It needs to be compiled with the same
# h5 as used to open the dataset, which might have been h5py. Dont do that!

# the reason to have this stuff was to verify any issues in h5py. There was some
# problem with python2.7 giving back bytes as "array( [ 1, 2, ...] )" instead
# of a binary blob.

# If you use h5py. datasetid. get_chunk_info() then you can skip all this.
FUNCS['h5_open_file'] = cfunc(
    "size_t h5_open_file", #  hid_t is int64_t (today, somwhere)
    ["char* hname"],
    """
    hid_t file;
    file = H5Fopen( hname,  H5F_ACC_RDONLY, H5P_DEFAULT );
    return file;
    """ )
FUNCS['h5_close_file'] = cfunc(
    "int h5_close_file", #  hid_t is int64_t (today, somwhere)
    ["int64_t hfile" ],
    """
    return H5Fclose( hfile );
    """ )
FUNCS['h5_open_dset'] = cfunc(
    "size_t h5_open_dset", #  hid_t is int64_t (today, somwhere)
    ["int64_t h5file", "char* dsetname" ],
    """
    hid_t dataset;
    if((dataset = H5Dopen2(h5file, dsetname, H5P_DEFAULT)) < 0)
        ERR("Failed to open datset");
    return dataset;
    """ )
FUNCS['h5_close_dset'] = cfunc(
    "int h5_close_dset", #  hid_t is int64_t (today, somwhere)
    ["int64_t dset"],
    """
    return H5Dclose( dset );
    """ )
FUNCS['h5_dsinfo'] = cfunc( # read itemsize, classtype, signed, total_bytes, ndims, dims (skip byteorder)
    "size_t h5_dsinfo",
    ["int64_t dataset_id", "int64_t * dsinfo", "int dsinfo_length"],
    """
    hid_t h5t = H5Dget_type( dataset_id );
    if (h5t < 0) return h5t;
    dsinfo[0] = H5Tget_size( h5t );
    dsinfo[1] = H5Tget_class( h5t );
    if (dsinfo[1] == H5T_INTEGER){
        dsinfo[2] = H5Tget_sign( h5t );
    } else {
        dsinfo[2] = 0;
    }
    H5Tclose( h5t );
    hid_t h5s = H5Dget_space( dataset_id );
    if (h5s < 0) return h5s;
    dsinfo[4] = H5Sget_simple_extent_ndims( h5s );
    if ( (dsinfo[4] + 4) > dsinfo_length ){
       H5Sclose( h5s );
       return -1;
    }
    int err = H5Sget_simple_extent_dims( h5s, &dsinfo[5], NULL );
    H5Sclose( h5s );
    int64_t nb = 1;
    for( int i = 0; i < dsinfo[4]; i++ ) nb *= dsinfo[5+i];
    dsinfo[3] = nb;
    return err;
    """
    )
FUNCS['h5_chunk_size'] = cfunc(
    "size_t h5_chunk_size", # name, args, body
    ["int64_t dataset_id", "int frame" ],
    """
{
    hsize_t offset[3];
    offset[0] = (hsize_t) frame;  /* assumes 3D frame-by-frame chunks */
    offset[1] = 0;
    offset[2] = 0;

    /* Get the size of the compressed chunk to return */
    hsize_t chunk_nbytes;
    herr_t ret;
    ret = H5Dget_chunk_storage_size(dataset_id, offset, &chunk_nbytes);
    if( ret == 0 )
        return chunk_nbytes;
    return ret;
}
""")
FUNCS['h5_read_direct'] = cfunc(
    "size_t h5_read_direct", # name, args, body
                             # hid_t is int64_t (today, somwhere)
                            # hsize_t in unsigned long long == uint64_t (today, somewhere)
    ["int64_t dataset_id", "int frame", "uint8_t * chunk", "size_t chunk_length" ],
    """
{
/* see:
   https://support.hdfgroup.org/HDF5/doc/HL/RM_HDF5Optimized.html#H5DOread_chunk

   ... assuming this is h5py.dataset.id.id :
    hid_t dataset;
    if((dataset = H5Dopen2(hname, dsetname, H5P_DEFAULT)) < 0)
        ERR("Failed to open h5file");
*/
    hsize_t offset[3];
    offset[0] = (hsize_t) frame;  /* assumes 3D frame-by-frame chunks */
    offset[1] = 0;
    offset[2] = 0;

    /* Get the size of the compressed chunk to return */
    hsize_t chunk_nbytes;
    herr_t ret;
    ret = H5Dget_chunk_storage_size(dataset_id, offset, &chunk_nbytes);
    if ( chunk_nbytes > chunk_length ) {
        fprintf(stderr, "Chunk does not fit into your arg");
        return 0;
        }
    if ( ret < 0 ) {
        fprintf(stderr,"Problem getting storage size for the chunk");
        return 0;
        }
    /* Use H5DOread_chunk() to read the chunk back
       ... becomes H5Dread_chunk in later library versions */
    uint32_t read_filter_mask;
    ret = H5Dread_chunk(dataset_id, H5P_DEFAULT, offset, &read_filter_mask, chunk);
    if ( ret < 0 ) {
        fprintf( stderr, "error reading chunk");
        return 0;
        }
    if ( read_filter_mask != 0 ){
        fprintf(stderr, "chunk was filtered");
        return 0;
     }
    return chunk_nbytes;
}
""")


def main( testcompile = False ):

    h5funcs = {name:FUNCS[name] for name in FUNCS if name.startswith("h5")}
    write_pyf("h5chunk", "h5chunk.c", h5funcs )
    write_funcs("h5chunk.c", h5funcs, INCLUDES + H5H, MACROS )
    for name in h5funcs:
        FUNCS.pop( name )

    ompfuncs= {name:FUNCS[name] for name in FUNCS if name.startswith("omp")}
    write_pyf("ompdecoders" , "ompdecoders.c", ompfuncs )
    write_pyf("ippompdecoders" , "ompdecoders.c", ompfuncs )
    write_funcs("ompdecoders.c" , ompfuncs, INCLUDES + LZ4H + "#include <omp.h>\n", MACROS )
    for name in ompfuncs:
        FUNCS.pop( name )

    write_pyf("decoders" , "decoders.c", FUNCS )
    write_pyf("ippdecoders" , "decoders.c", FUNCS )
    write_funcs("decoders.c" , FUNCS, INCLUDES + LZ4H + BSH, MACROS )


    if testcompile:
        for series in h5funcs, ompfuncs, FUNCS:
            test_funcs_compile(series)




if __name__=="__main__":
    main( len(sys.argv)==2 )

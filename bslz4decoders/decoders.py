

import struct
import bitshuffle
import numpy as np

from bslz4decoders.ccodes.h5chunk import h5_dsinfo
from bslz4decoders.ccodes.decoders import read_starts, onecore_bslz4
from bslz4decoders.ccodes.ompdecoders import omp_bslz4, omp_bslz4_blocks, omp_get_threads_used

try:
    from bslz4decoders.ccodes.ippdecoders    import onecore_bslz4 as ipponecore_bslz4 
    from bslz4decoders.ccodes.ippompdecoders import omp_bslz4            as ippomp_bslz4
    from bslz4decoders.ccodes.ippompdecoders import omp_bslz4_blocks     as ippomp_bslz4_blocks
    from bslz4decoders.ccodes.ippompdecoders import omp_get_threads_used as ippomp_get_threads_used
    GOTIPP = True
except ImportError:
    GOTIPP = False

"""
We are aiming to duplicate this interface from bitshuffle :

>>> help(bitshuffle.decompress_lz4)
Help on built-in function decompress_lz4 in module bitshuffle.ext:

decompress_lz4(...)
    Decompress a buffer using LZ4 then bitunshuffle it yielding an array.

    Parameters
    ----------
    arr : numpy array
        Input data to be decompressed.
    shape : tuple of integers
        Shape of the output (decompressed array). Must match the shape of the
        original data array before compression.
    dtype : numpy dtype
        Datatype of the output array. Must match the data type of the original
        data array before compression.
    block_size : positive integer
        Block size in number of elements. Must match value used for
        compression.

    Returns
    -------
    out : numpy array with shape *shape* and data type *dtype*
        Decompressed data.
"""

class BSLZ4ChunkConfig:
    """ Metadata needed for decoding a chunk

    shape = shape of this chunk
    bpp = bytes per pixel
    output_nbytes = shape * bpp
    blocksize = transpose blocksize for bitshuffle
    npytype = numpy dtype == convenience
    """
    __slots__ = [ "shape", "dtype", "blocksize", "output_nbytes", "bpp" ]

    def __init__(self, shape, dtype, blocksize=8192, output_nbytes=None ):
        self.shape = shape
        self.dtype = dtype
        self.bpp = dtype.itemsize
        self.blocksize = blocksize
        if output_nbytes is not None:
            self.output_nbytes = output_nbytes
        else:
            self.output_nbytes = shape[0]*shape[1]*dtype.itemsize

    def get_blocks( self, chunk, blocks=None ):
        """
        allow blocks to be pre-allocated (e.g. pinned memory)
        sets self.blocksize only if blocks is None
        """
        if blocks is None:
            # We do this in python as it doesn't seem worth making a call back
            # ... otherwise need to learn to call free on a numpy array
            total_bytes, self.blocksize = struct.unpack_from("!QL", chunk, 0)
            if self.blocksize == 0:
                self.blocksize = 8192
            nblocks =  (total_bytes + self.blocksize - 1) // self.blocksize
            assert self.output_nbytes == total_bytes, "chunk config mismatch:"+repr(self)
            blocks = np.empty( nblocks, np.uint32 )
        read_starts( chunk, self.dtype.itemsize, self.blocksize, blocks )
        return blocks
    
    def last_blocksize( self ):
        last = self.output_nbytes % self.blocksize
        tocopy = last % ( self.dtype.itemsize * 8 )
        last -= tocopy
        return last
    
    def tocopy( self ):
        return self.output_nbytes % ( self.dtype.itemsize * 8 )

    def __repr__(self):
        return "%s %s %d %d"%( repr(self.shape), repr(self.dtype),
                            self.blocksize, self.output_nbytes)

    
class BSLZ4ChunkConfigDirect( BSLZ4ChunkConfig ):
    def __init__(self, dsid, blocksize=8192):
        self.blocksize = blocksize
        dsinfo = np.zeros( 16, np.int64 )
        err = h5_dsinfo( dsid, dsinfo )
        self.bpp, classtype, signed, output_nbytes, ndims = dsinfo[:5]
        assert ndims == 3
        shape = dsinfo[5:5+ndims]
        self.shape = shape[1], shape[2]
        self.output_nbytes = self.bpp * shape[1] * shape[2] # per frame
        self.nframes = shape[0]
        #            H5T_INTEGER          = 0,   /*integer types                              */
        #            H5T_FLOAT            = 1,   /*floating-point types                       */
        #  H5T_SGN_ERROR        = -1,  /*error                                      */
        #  H5T_SGN_NONE         = 0,   /*this is an unsigned type                   */
        #  H5T_SGN_2            = 1,   /*two's complement                           */
        #  H5T_NSGN             = 2  
        if classtype == 0: 
            if signed < 0:
                raise Exception("H5T_SGN_ERROR")
            # taking NSGN as not signed, but did not find out what it really means yet
            self.dtype = np.dtype( 'uiu'[signed] + str( self.bpp ) ) 
        elif classtype == 1:
            self.dtype = np.dtype( 'f' + str( self.bpp ) )
        
    
    
def decompress_bitshuffle( chunk, config, output = None ):
    """  Generic bitshuffle decoder depending on the
    bitshuffle library from https://github.com/kiyo-masui/bitshuffle

    input: chunk compressed data
           config, gives the shape and dtype
    returns: decompressed data
    """
    r = bitshuffle.decompress_lz4( chunk[12:],
                                   config.shape,
                                   np.dtype(config.dtype),
                                   config.blocksize // config.dtype.itemsize )
    if output is not None:
        output[:] = r
    else:
        output = r
    return output



def decompress_onecore( chunk, config, output = None ):
    """  One core decoding from our ccodes
    """
    if output is None:
        output = np.empty( config.shape, config.dtype )
    # tmp = np.empty( config.blocksize, np.uint8 )
    err = onecore_bslz4( np.asarray(chunk) ,
                         config.dtype.itemsize, output.view( np.uint8 ) )
    if err:
        raise Exception("Decoding error")
    # TODO: put the bitshuffle into C !
    return output.view(config.dtype).reshape( config.shape )


def decompress_omp( chunk, config, output = None, num_threads=0):
    """  Openmp decoding from our ccodes module
    """
    if output is None:
        output = np.empty( config.shape, config.dtype )
    if num_threads == 0:
        num_threads = omp_get_threads_used( num_threads )
    # tmp = np.empty( config.blocksize * num_threads, np.uint8 )
    err = omp_bslz4( np.asarray(chunk) , config.dtype.itemsize, output.view( np.uint8 ),
                     num_threads )
    if err:
        raise Exception("Decoding error")
    return output.view(config.dtype).reshape( config.shape )


def decompress_omp_blocks( chunk, config,
                           offsets=None,
                           output = None,
                           num_threads = 0 ):
    """  Openmp decoding from our ccodes module
    (In the long run - we are expecting the offsets to be cached sonewhere)
    """
    achunk = np.asarray( chunk )
    if output is None:
        output = np.empty( config.shape, config.dtype )
    if offsets is None:
        offsets = config.get_blocks( achunk )
    if num_threads == 0:        
        num_threads = omp_get_threads_used( num_threads )
    # tmp = np.empty( config.blocksize * num_threads, np.uint8 )
    err = omp_bslz4_blocks( achunk , config.dtype.itemsize,
                            config.blocksize, offsets, output.view( np.uint8 ),
                            num_threads )
    if err:
        raise Exception("Decoding error")
    return output.view(config.dtype).reshape( config.shape )


if GOTIPP:
    def decompress_ipponecore( chunk, config, output = None ):
        """  One core decoding from our ccodes
        """
        if output is None:
            output = np.empty( config.shape, config.dtype )
        # tmp = np.empty( config.blocksize, np.uint8 )
        err = ipponecore_bslz4( np.asarray(chunk) ,
                                config.dtype.itemsize, output.view( np.uint8 ) )
        if err:
            raise Exception("Decoding error")
        # TODO: put the bitshuffle into C !
        return output.view(config.dtype).reshape( config.shape )


    def decompress_ippomp( chunk, config, output = None, num_threads=0):
        """  Openmp decoding from our ccodes module
        todo: cache num_threads
        """
        if output is None:
            output = np.empty( config.shape, config.dtype )
        if num_threads == 0:            
            num_threads = omp_get_threads_used( num_threads )
        # tmp = np.empty( config.blocksize * num_threads, np.uint8 )
        err = ippomp_bslz4( np.asarray(chunk) , config.dtype.itemsize, output.view( np.uint8 ),
                            num_threads )
        if err:
            raise Exception("Decoding error")
        return output.view(config.dtype).reshape( config.shape )


    def decompress_ippomp_blocks( chunk, config,
                                  offsets=None,
                                  output = None,
                                  num_threads = 0 ):
        """  Openmp decoding from our ccodes module
        (In the long run - we are expecting the offsets to be cached sonewhere)
        todo: cache num_threads and tmp
        """
        achunk = np.asarray( chunk )
        if output is None:
            output = np.empty( config.shape, config.dtype )
        if offsets is None:
            offsets = config.get_blocks( achunk )
        if num_threads == 0:
            num_threads = omp_get_threads_used( num_threads )
        # tmp = np.empty( config.blocksize * num_threads, np.uint8 )
        err = ippomp_bslz4_blocks( achunk , config.dtype.itemsize,
                                   config.blocksize, offsets, output.view( np.uint8 ),
                                   num_threads )
        if err:
            raise Exception("Decoding error")
        return output.view(config.dtype).reshape( config.shape )

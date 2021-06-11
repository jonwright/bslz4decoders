

import struct
import bitshuffle
import numpy as np

from bslz4decoders.ccodes.decoders import read_starts, onecore_lz4
from bslz4decoders.ccodes.ompdecoders import omp_lz4, omp_lz4_blocks


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
    """ Wrapper over a binary blob that comes from a hdf5 file """

    __slots__ = [ "shape", "dtype", "blocksize", "output_nbytes" ]

    def __init__(self, shape, dtype, blocksize=8192, output_nbytes=0 ):
        self.shape = shape
        self.dtype = dtype
        self.blocksize = blocksize
        if output_nbytes:
            self.output_nbytes = output_nbytes
        else:
            self.output_nbytes = shape[0]*shape[1]*dtype.itemsize

    def get_blocks( self, chunk ):
        # We do this in python as it doesn't seem worth making a call back
        # ... otherwise need to learn to call free on a numpy array
        total_bytes, blocksize = struct.unpack_from("!QL", chunk, 0)
        if blocksize == 0:
            blocksize = 8192
        nblocks =  (total_bytes + blocksize - 1) // blocksize
        assert self.output_nbytes == total_bytes, "chunk config mismatch:"+repr(self)
        blocks = np.empty( nblocks, np.uint32 )
        read_starts( chunk, self.dtype.itemsize, blocksize, blocks )
        # does not mess about with self.blocksize
        return blocksize, blocks

    def __repr__(self):
        return "%s %s %d %d"%( repr(self.shape), repr(self.dtype),
                            self.blocksize, self.output_nbytes)

def decompress_bitshuffle( chunk, config, output = None ):
    """  Generic bitshuffle decoder depending on the
    bitshuffle library from https://github.com/kiyo-masui/bitshuffle

    input: chunk compressed data
           config, gives the shape and dtype
    returns: decompressed data
    """
    # FIXME: this bombs on windows.
    if output is not None:
        o = output.ravel()
        tb, bs = struct.unpack_from("!QL", chunk, 0)
        print("about to shuffle", config.shape, config.dtype, config.blocksize, tb, bs, o.nbytes)
        print(len(chunk), type(chunk))
        r = bitshuffle.decompress_lz4( chunk[12:], config.shape,
                                          config.dtype, config.blocksize )
        print("back from shuffle")
        o[:] = r
    else:
        output = bitshuffle.decompress_lz4( chunk[12:], config.shape,
                                          config.dtype, config.blocksize )
        print("back from shuffle")
    return output




def decompress_onecore( chunk, config, output = None ):
    """  One core decoding from our ccodes
    """
    if output is None:
        output = np.empty( config.shape, config.dtype )
    err = onecore_lz4( np.asarray(chunk) ,
                    config.dtype.itemsize, output.view( np.uint8 ) )
    if err:
        raise Exception("Decoding error")
    # TODO: put the bitshuffle into C !
    return bitshuffle.bitunshuffle( output.view(config.dtype) ).reshape( config.shape )


def decompress_omp( chunk, config, output = None ):
    """  Openmp decoding from our ccodes module
    """
    if output is None:
        output = np.empty( config.shape, config.dtype )
    err = omp_lz4( np.asarray(chunk) , config.dtype.itemsize, output.view( np.uint8 ) )
    if err:
        raise Exception("Decoding error")
    # TODO: put the bitshuffle into C !
    return bitshuffle.bitunshuffle( output.view(config.dtype) ).reshape( config.shape )


def decompress_omp_blocks( chunk, config,
                           offsets=None, blocksize=None,
                           output = None ):
    """  Openmp decoding from our ccodes module
    (In the long run - we are expecting the offsets to be cached sonewhere)
    """
    achunk = np.asarray( chunk )
    if output is None:
        output = np.empty( config.shape, config.dtype )
    if offsets is None:
        blocksize, offsets = config.get_blocks( achunk )

    if blocksize is None:
        blocksize = config.blocksize
    err = omp_lz4_blocks( achunk , config.dtype.itemsize,
                          blocksize, offsets, output.view( np.uint8 ) )
    if err:
        raise Exception("Decoding error")
    # TODO: put the bitshuffle into C !
    return bitshuffle.bitunshuffle( output.view(config.dtype) ).reshape( config.shape )
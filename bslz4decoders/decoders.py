

import struct
import bitshuffle

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

class BSLZ4Chunk:
    """ Wrapper over a binary blob that comes from a hdf5 file """

    __slots__ = [ "chunk", "shape", "dtype", "blocksize", "output_nbytes" ]

    def __init__(self, shape, dtype, chunk = None ):
        self.blocksize = 8192
        self.shape = shape
        self.dtype = dtype
        self.chunk = chunk
        if chunk:
            self.output_nbytes, self.blocksize = struct.unpack( "!QL", chunk[:12] )
            if self.blocksize == 0:
                self.blocksize = 8192
        else:
            self.output_nbytes = shape[0]*shape[1]*dtype.itemsize


def decompress_bitshuffle( chunk ):
    """  Generic bitshuffle decoder depending on the
    bitshuffle library from https://github.com/kiyo-masui/bitshuffle

    input: chunk = instance of the BSLZ4 chunk type above (carries shape/dtype)
    returns: decompressed data
    """
    return bitshuffle.decompress_lz4( chunk.chunk, chunk.shape,
                                        chunk.dtype, chunk.blocksize )



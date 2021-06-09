

import hdf5plugin, h5py
import numpy as np
import bslz4decoders, bitshuffle, timeit
from bslz4decoders import read_chunks
from testcases import testcases


def runtest_lz4chunkdecoders( decoder, frame = 0, rpt = 10 ):
    for h5name, dset in testcases:
        ref = read_chunks.get_frame_h5py( h5name, dset, frame )
        t0 = timeit.default_timer()
        for _ in range(rpt):
            ref = read_chunks.get_frame_h5py( h5name, dset, frame )
        t1 = timeit.default_timer()
        ref = bitshuffle.bitshuffle( ref )
        out = None
        t1 = timeit.default_timer()
        for _ in range(rpt):
            chunk, shp, dtyp = read_chunks.get_chunk(
                h5name, dset, frame )
            if out is None:
                out = np.empty( shp[1]*shp[2], dtyp )
            decoder( chunk, dtyp.itemsize, out.view( np.uint8 ) )
            decoded = out
        t2 = timeit.default_timer()
        if not (decoded == ref.ravel()).all():
            print("Fail!")
            print(decoded)
            print(ref.ravel())
        else:
            print(" h5py %.3f ms"%((t1-t0)*1e3), end=' ')
            print(" lz4only %.3f ms"%((t2-t1)*1e3), end= ' ')
        print( " ", h5name, dset )


def runtest_lz4blockdecoders( decoder, frame = 0, rpt = 10 ):
    for h5name, dset in testcases:
        ref = read_chunks.get_frame_h5py( h5name, dset, frame )
        blocks = None
        t0 = timeit.default_timer()
        for _ in range(rpt):
            ref = read_chunks.get_frame_h5py( h5name, dset, frame )
        t1 = timeit.default_timer()
        ref = bitshuffle.bitshuffle( ref )
        t1 = timeit.default_timer()
        for _ in range(rpt):
            chunk, shp, dtyp = read_chunks.get_chunk(
                h5name, dset, frame )
            if blocks is None:
                blocksize, blocks = read_chunks.get_blocks( chunk, shp, dtyp )
                out = np.empty( shp[1]*shp[2], dtyp )
            decoder( chunk, dtyp.itemsize, blocksize, blocks, out.view( np.uint8 ) )
            decoded = out
        t2 = timeit.default_timer()
        if not (decoded == ref.ravel()).all():
            print("Fail!")
            print(decoded)
            print(ref.ravel())
        else:
            print(" h5py %.3f ms"%((t1-t0)*1e3), end=' ')
            print(" lz4only %.3f ms"%((t2-t1)*1e3), end= ' ')
        print( " ", h5name, dset )

def testonecore():
    from bslz4decoders.ccodes.decoders import onecore_lz4
    runtest_lz4chunkdecoders(onecore_lz4)

def testomp():
    from bslz4decoders.ccodes.ompdecoders import omp_lz4
    runtest_lz4chunkdecoders(omp_lz4)

def testompblocked():
    from bslz4decoders.ccodes.ompdecoders import omp_lz4_blocks
    runtest_lz4blockdecoders(omp_lz4_blocks)


if __name__=="__main__":
    print("onecore")
    testonecore()
    print("omp reading offsets")
    testomp()
    print("omp given offsets")
    testompblocked()
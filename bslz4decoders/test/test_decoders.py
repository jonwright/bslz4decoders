

import sys, timeit
import numpy as np
from bslz4decoders import read_chunks, decoders
from bslz4decoders.test.testcases import testcases as TESTCASES
RPT = 10

def runtest_lz4chunkdecoders( decoder, frame = 0, rpt = 10 ):
    for h5name, dset in TESTCASES:
        ref = read_chunks.get_frame_h5py( h5name, dset, frame )
        t0 = timeit.default_timer()
        for _ in range(RPT):
            ref = read_chunks.get_frame_h5py( h5name, dset, frame )
        t1 = timeit.default_timer()
        out = None
        t1 = timeit.default_timer()
        for _ in range(RPT):
            config, chunk = read_chunks.get_chunk( h5name, dset, frame )
            out = np.empty( config.shape, config.dtype )
            decoded = decoder( chunk, config, output = out )
        t2 = timeit.default_timer()
        if not (decoded == ref).all():
            print("Fail!")
            print(decoded)
            print(ref.ravel())
        else:
            print(" h5py %.3f ms"%((t1-t0)*1e3), end=' ')
            print(" here %.3f ms"%((t2-t1)*1e3), end= ' ')
        print( " ", h5name, dset )


def runtest_lz4blockdecoders( decoder, frame = 0 ):
    for h5name, dset in TESTCASES:
        ref = read_chunks.get_frame_h5py( h5name, dset, frame )
        blocks = None
        t0 = timeit.default_timer()
        for _ in range(RPT):
            ref = read_chunks.get_frame_h5py( h5name, dset, frame )
        t1 = timeit.default_timer()
        for _ in range(RPT):
            config, chunk = read_chunks.get_chunk( h5name, dset, frame )
            blocks = config.get_blocks( chunk )
            out = np.empty( config.shape, config.dtype )
            decoded = decoder( chunk, config, output=out )
        t2 = timeit.default_timer()
        if not (decoded == ref).all():
            print("Fail!")
            print(decoded)
            print(ref.ravel())
        else:
            print(" h5py %.3f ms"%((t1-t0)*1e3), end=' ')
            print(" here %.3f ms"%((t2-t1)*1e3), end= ' ')
        print( " ", h5name, dset )


def testchunkdecoders():
    for func in ( # "decompress_bitshuffle",
                  "decompress_onecore",
                  "decompress_omp" ):
        print(func)
        runtest_lz4chunkdecoders( getattr( decoders, func ) )

def testblocked():
    for func in ( "decompress_omp_blocks", ):
        print(func)
        runtest_lz4chunkdecoders( getattr( decoders, func ) )


if __name__=="__main__":

    if len(sys.argv) > 2:
        TESTCASES = [ (sys.argv[1], sys.argv[2]), ]

    if len(sys.argv) > 3:
        RPT = int(sys.argv[3])

    testchunkdecoders()
    testblocked()

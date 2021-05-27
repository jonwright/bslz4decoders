

import hdf5plugin, h5py
import numpy as np
import bslz4decoders, bitshuffle, timeit
import read_chunks
from testcases import testcases


def runtest_lz4chunkdecoders( decoder, frame = 0, rpt = 10 ):
    for h5name, dset in testcases:
        print( " ",h5name, dset )
        t0 = timeit.default_timer()
        for _ in range(rpt):
            ref = read_chunks.get_frame_h5py( h5name, dset, frame )
        t1 = timeit.default_timer()
        for _ in range(rpt):
            chunk, shp, dtyp = read_chunks.get_chunk(
                h5name, dset, frame )
            out = np.empty( shp[1]*shp[2], dtyp )
            decoder( chunk, dtyp.itemsize, out.view( np.uint8 ) )
            decoded = bitshuffle.bitunshuffle( out )
        t2 = timeit.default_timer()
        if not (decoded == ref.ravel()).all():
            print("Fail!")
            print(decoded)
            print(ref.ravel())
        else:
            print("    h5py %.6f"%(t1-t0))
            print("    obfu %.6f"%(t2-t1))


def runtest_lz4blockdecoders( decoder, frame = 0, rpt = 10 ):
    for h5name, dset in testcases:
        print( " ", h5name, dset )
        blocks = None
        t0 = timeit.default_timer()
        for _ in range(rpt):
            ref = read_chunks.get_frame_h5py( h5name, dset, frame )
        t1 = timeit.default_timer()
        for _ in range(rpt):
            chunk, shp, dtyp = read_chunks.get_chunk(
                h5name, dset, frame )
            if blocks is None:
                blocksize, blocks = read_chunks.get_blocks( chunk, shp, dtyp ) 
                out = np.empty( shp[1]*shp[2], dtyp )
            decoder( chunk, dtyp.itemsize, blocksize, blocks, out.view( np.uint8 ) )
            decoded = bitshuffle.bitunshuffle( out )
        t2 = timeit.default_timer()
        if not (decoded == ref.ravel()).all():
            print("Fail!")
            print(decoded)
            print(ref.ravel())
        else:
            print("    h5py %.6f"%(t1-t0))
            print("    obfu %.6f"%(t2-t1))            

def testonecore():
    runtest_lz4chunkdecoders(bslz4decoders.onecore_lz4)

def testomp():
    runtest_lz4chunkdecoders(bslz4decoders.omp_lz4)

def testompblocked():
    runtest_lz4blockdecoders(bslz4decoders.omp_lz4_blocks)

def testipp():
    runtest_lz4chunkdecoders(bslz4decoders.onecore_ipp)

if __name__=="__main__":
    print("onecore")
    testonecore()
    print("ipp")
    testipp()
    print("omp reading offsets")
    testomp()
    print("omp given offsets")
    testompblocked()



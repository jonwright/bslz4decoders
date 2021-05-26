

import hdf5plugin, h5py, numpy as np, bslz4decoders, bitshuffle
import read_chunks

testcases = [
   ("bslz4testcases.h5", "data_uint8"),
   ("bslz4testcases.h5", "data_uint16"),
   ("bslz4testcases.h5", "data_uint32"),
]

def runtest_lz4chunkdecoders( decoder ):
    for h5name, dset in testcases:
        # reference data
        ref = read_chunks.get_frame_h5py( h5name, dset, 0 )
        # compressed data
        (filtered, byteschunk), shp, dtyp = read_chunks.get_chunk( h5name, dset, 0 )
        chunk = np.frombuffer(byteschunk, np.uint8)
        # allocate output
        out = np.empty( shp[0]*shp[1], dtyp )
        # decode it for lz4
        decoder( chunk, dtyp.itemsize, out.view( np.uint8 ) )
        # bitshuffle it
        decoded = bitshuffle.bitunshuffle( out )
        if not (decoded == ref.ravel()).all():
            print("Fail!")
            print(decoded)
            print(ref.ravel())

def testonecore():
    runtest_lz4chunkdecoders(bslz4decoders.onecore_lz4)

def testomp():
    runtest_lz4chunkdecoders(bslz4decoders.omp_lz4)

def runtest_lz4blockdecoders( decoder ):
    for h5name, dset in testcases:
        # reference data
        ref = read_chunks.get_frame_h5py( h5name, dset, 0 )
        (filtered, byteschunk), shp, dtyp = read_chunks.get_chunk( h5name, dset, 0 )
        chunk = np.frombuffer(byteschunk, np.uint8)
        out = np.empty( shp[0]*shp[1], dtyp )
        blocksize, blocks = read_chunks.get_blocks( chunk, shp, dtyp )
        decoder( chunk, dtyp.itemsize, blocksize, blocks, out.view( np.uint8 ) )
        decoded = bitshuffle.bitunshuffle( out )
        if not (decoded == ref.ravel()).all():
            print("Fail!")
            print(decoded)
            print(ref.ravel())

def testompblocked():
    runtest_lz4blockdecoders( bslz4decoders.omp_lz4_blocks )

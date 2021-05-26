

import hdf5plugin, h5py, numpy as np, bslz4decoders, bitshuffle
import read_chunks

testcases = [
   ("bslz4testcases.h5", "data_uint8"),
   ("bslz4testcases.h5", "data_uint16"),
   ("bslz4testcases.h5", "data_uint32"),
]

def runtest_lz4chunkdecoders( decoder ):
    for h5name, dset in testcases:
        ref = read_chunks.get_frame_h5py( h5name, dset, 0 )
        (filtered, byteschunk), shp, dtyp = read_chunks.get_chunk( h5name, dset, 0 )
        chunk = np.frombuffer(byteschunk, np.uint8)
        out = np.empty( shp[1]*shp[2], dtyp )
        decoder( chunk, dtyp.itemsize, out.view( np.uint8 ) )
        decoded = bitshuffle.bitunshuffle( out )
        if not (decoded == ref.ravel()).all():
            print("Fail!")
            print(decoded)
            print(ref.ravel())

def testonecore():
    runtest_lz4chunkdecoders(bslz4decoders.onecore_lz4)

def testomp():
    runtest_lz4chunkdecoders(bslz4decoders.omp_lz4)




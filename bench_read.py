

import sys, hdf5plugin, h5py, timeit, numpy as np
import concurrent.futures

def read_serial( hname, dsetname, blocks=10):
    start = timeit.default_timer()
    bytesread = 0
    with h5py.File( hname , "r" ) as h5f:
        dset = h5f[dsetname]
        output = np.empty( (blocks, dset.shape[1], dset.shape[2]),
                           dset.dtype )
        for i in range(0, len(dset), blocks):
            last = min( i + blocks, len(dset) )
            dset.read_direct( output, np.s_[ i : last ], np.s_[ 0: last - i ] )
            bytesread += output[0].nbytes*(last-i)
    end = timeit.default_timer()
    dt = end - start
    gb = bytesread / 1e9 # generous
    print("%.1f GB, %.3f s, %.3f GB/s" % ( gb, dt, gb / dt ) )
            
if __name__=="__main__":
    
    if len(sys.argv) == 1:
        hname = "bslz4testcases.h5"
        dsets = [ "data_uint%d"%(i) for i in (8,16,32) ]
    else:
        hname = sys.argv[1]
        dsets = [ d for d in sys.argv[2:] ]
    for d in dsets:
        read_serial( hname, d )



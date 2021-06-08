

import sys, hdf5plugin, h5py, timeit, numpy as np
import concurrent.futures

def read_serial( hname, dsetname ):
    starto = timeit.default_timer()
    bytesread = 0
    with h5py.File( hname , "r" ) as h5f:
        dset = h5f[dsetname]
        output = np.empty( (1, dset.shape[1], dset.shape[2]), dset.dtype )
        start = timeit.default_timer()
        for i in range(0, len(dset)):
            dset.read_direct( output, np.s_[ i : i+1 ], np.s_[ 0 : 1 ] )
            bytesread += output[0].nbytes
        end = timeit.default_timer()
    endo = timeit.default_timer()
    dt = end - start
    dto = endo - starto
    mb = bytesread / 1e6 # generous
    print("%6.1f MB, %6.3f ms, %6.1f MB/s %6.3f ms overhead %s" % ( mb, 1e3*dt, mb / dt, 1e3*(dto-dt), dsetname ) )

if __name__=="__main__":
    if len(sys.argv) == 1:
        import testcases
        cases = testcases.testcases
    else:
        hname = sys.argv[1]
        cases = [ (hname, d) for d in sys.argv[2:] ]
    for hname, dset in cases:
        read_serial( hname, dset )



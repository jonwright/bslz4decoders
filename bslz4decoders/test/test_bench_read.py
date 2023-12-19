

import sys, hdf5plugin, h5py, timeit, numpy as np

from bslz4decoders.read_chunks import iter_h5chunks, iter_chunks
from bslz4decoders import decoders

def read_simple( hname, dsetname, mem=None, membuf=None ):
    with h5py.File( hname , "r" ) as h5f:
        dset = h5f[dsetname]
        output = dset[:]
    return output
    
def read_direct( hname, dsetname, mem = None, membuf=None ):
    with h5py.File( hname , "r" ) as h5f:
        dset = h5f[dsetname]
        if mem is None:
            mem = alloc_output( dset.shape, dset.dtype )
        dset.read_direct( mem )
    return mem


def h5py_chunks( decoder, hname, dsetname, mem = None, membuf = None ):
    # calls iter_chunks that uses h5py direct_chunk_read (and mallocs)
    for i, (config, chunk) in enumerate( iter_chunks( hname, dsetname )):
        decoder( chunk, config, output = mem[i] )
    return mem


def hdf5_chunks( decoder, hname, dsetname, mem = None, membuf = None ):
    # calls into hdf5 using an existing memory buffer (membuf)
    for i, (config, chunk) in enumerate( iter_h5chunks( hname, dsetname, memory=membuf )):
        decoder( chunk, config, output = mem[i] )
    return mem


def measure(ref, fun, *args ):
    for i in range(3):
        start = timeit.default_timer()
        data = fun( *args )
        end = timeit.default_timer()
        dt = end - start
        mb = data.nbytes / 1e6 # generous
        print(fun.__name__, i, "%6.1f MB, %6.3f ms, %6.1f MB/s" % ( mb, 1e3*dt, mb / dt ) )
        if not (ref == data).all():
            print('Error!')


if __name__=="__main__":
    try:
        hname = sys.argv[1]
        dset = sys.argv[2]
    except IndexError:
        import testcases
        hname, dset = testcases.testcases[0]
        
    print(hname, dset)
    o0 = read_simple( hname, dset )
    obuf = np.zeros(4096*4096*4,np.uint8)
    omem = np.zeros_like(o0)
    for chunkreader in (h5py_chunks, hdf5_chunks):
        for decompressor in decoders.decompressors:
             def fun( *args ):
                return chunkreader( decompressor, *args )
             fun.__name__ = f'{chunkreader.__name__}|{decompressor.__name__}'
             measure(o0, fun, hname, dset, omem, obuf )

    for fun in [ read_simple, read_direct ]:
        measure(o0, fun, hname, dset, omem, obuf )
    
    



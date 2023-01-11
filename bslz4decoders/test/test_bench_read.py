

import sys, hdf5plugin, h5py, timeit, numpy as np

from bslz4decoders.read_chunks import iter_h5chunks
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


def _read_h5chunk( decoder, hname, dsetname, mem = None, membuf = None ):
    for i, (config, chunk) in enumerate( iter_h5chunks( hname, dsetname, memory=membuf )):
        decoder( chunk, config, output = mem[i] )
    return mem

def read_h5chunk_decompress_onecore( hname, dsetname, mem = None, membuf = None ):
    return _read_h5chunk( decoders.decompress_onecore, hname, dsetname, mem = mem, membuf = membuf )

def read_h5chunk_decompress_omp( hname, dsetname, mem = None, membuf = None ):
    return _read_h5chunk( decoders.decompress_omp, hname, dsetname,  mem = mem, membuf = membuf )

def read_h5chunk_decompress_omp_blocks( hname, dsetname, mem = None, membuf = None ):
    return _read_h5chunk( decoders.decompress_omp_blocks, hname, dsetname,  mem = mem, membuf = membuf )

def read_h5chunk_decompress_ipponecore( hname, dsetname, mem = None, membuf = None ):
    return _read_h5chunk( decoders.decompress_ipponecore, hname, dsetname,  mem = mem, membuf = membuf )

def read_h5chunk_decompress_ippomp( hname, dsetname, mem = None, membuf = None ):
    return _read_h5chunk( decoders.decompress_ippomp, hname, dsetname,  mem = mem, membuf = membuf )

def read_h5chunk_decompress_ippomp_blocks( hname, dsetname, mem = None, membuf = None ):
    return _read_h5chunk( decoders.decompress_ippomp_blocks, hname, dsetname,  mem = mem, membuf = membuf )


def measure(fun, *args ):
    for i in range(3):
        start = timeit.default_timer()
        data = fun( *args )
        end = timeit.default_timer()
        dt = end - start
        mb = data.nbytes / 1e6 # generous
        print(fun.__name__, i, "%6.1f MB, %6.3f ms, %6.1f MB/s" % ( mb, 1e3*dt, mb / dt ) )
    return data


if __name__=="__main__":
    try:
        hname = sys.argv[1]
        dset = sys.argv[2]
    except IndexError:
        import testcases
        
    o0 = read_simple( hname, dset )
    obuf = np.zeros(4096*4096*4,np.uint8)
    for fun in (read_simple, 
                read_direct, 
                read_h5chunk_decompress_onecore,
                read_h5chunk_decompress_ipponecore,
                read_h5chunk_decompress_omp,
                read_h5chunk_decompress_ippomp,
                read_h5chunk_decompress_omp_blocks,
                read_h5chunk_decompress_ippomp_blocks) :
        omem = np.zeros_like(o0)
        o1 = measure( fun, hname, dset, omem, obuf )
        for i,(f0, f1) in enumerate(zip(o0, o1)):
            if not (f0 == f1).all():
                print('error in frame',i)
                print(f0)
                print(f1)
    
    



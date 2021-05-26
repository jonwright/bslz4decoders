

import h5py, hdf5plugin, struct
import numpy as np, bslz4decoders

def get_frame_h5py( h5name, dset, frame ):
    with h5py.File(h5name, "r") as h5f:
        frm = h5f[dset][frame]
    return frm

def get_chunk( h5name, dset, frame ):
    with h5py.File(h5name, "r") as h5f:
        dset = h5f[dset]
        assert len(dset.chunks) == 3
        assert dset.chunks[0] == 1
        assert dset.chunks[1] == dset.shape[1]
        assert dset.chunks[2] == dset.shape[2]
        return dset.id.read_direct_chunk( (frame, 0, 0) ), (dset.shape[1], dset.shape[2]), dset.dtype

def get_blocks( chunk, shp, dtyp ):
    total_output_size, blocksize = struct.unpack_from( "!QL", chunk, 0 )
    print(total_output_size, blocksize)
    assert total_output_size == shp[0]*shp[1]*dtyp.itemsize
    if blocksize == 0:
        blocksize = 8192
    nblocks = (total_output_size + blocksize - 1 ) // ( blocksize )
    blocks = np.empty( nblocks, dtype = np.uint32 )
    bslz4decoders.read_starts( chunk, dtyp.itemsize, blocksize, blocks )
    return blocksize, blocks

def get_frames_h5py( h5name, dset ):
    with h5py.File(h5name, "r") as h5f:
        for frm in h5f[dset]:
            yield frm

def get_chunks( h5name, dset ):
    with h5py.File(h5name, "r") as h5f:
        dset = h5f[dset]
        assert len(dset.chunks) == 3
        assert dset.chunks[0] == 1
        assert dset.chunks[1] == dset.shape[1]
        assert dset.chunks[2] == dset.shape[2]
        for frame in range(dset.shape[0]):
            yield dset.id.read_direct_chunk( (frame, 0, 0) ), (dset.shape[1], dset.shape[2]), dset.dtype

def bench( func, *args ):
    start = timeit.default_timer()
    func(*args)
    end = timeit.default_timer()
    print("%.6f /s"%(end-start), func.__name__, args)

def benchiter( func, *args ):
    start = timeit.default_timer()
    frms = [frm for frm in func(*args)]
    end = timeit.default_timer()
    print("%.6f /s"%(end-start), func.__name__, args)



if __name__=="__main__":
    hname = "bslz4testcases.h5"
    get_chunk( hname, 'data_uint8', 0 )
    import timeit, sys
    for s in [8,16,32]:
        print()
        bench( get_frame_h5py, "bslz4testcases.h5", "data_uint%d"%(s), 0 )
        benchiter( get_frames_h5py, "bslz4testcases.h5", "data_uint%d"%(s) )
        bench( get_chunk, "bslz4testcases.h5", "data_uint%d"%(s), 0 )
        benchiter( get_chunks, "bslz4testcases.h5", "data_uint%d"%(s) )


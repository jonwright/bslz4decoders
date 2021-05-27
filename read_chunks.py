

import h5py, hdf5plugin, bslz4decoders, numpy as np

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
        # inefficient
        buffer = np.empty( dset.dtype.itemsize*dset.shape[1]*dset.shape[2]+1024,
                           np.uint8 )
        csize = bslz4decoders.h5_read_direct( dset.id.id, frame, buffer )
        return buffer[:csize].copy(), dset.shape, dset.dtype        

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
        # inefficient
        buffer = np.empty( dset.dtype.itemsize*dset.shape[1]*dset.shape[2]+1024,
                           np.uint8 ) 
        for frame in range(dset.shape[0]):
            csize = bslz4decoders.h5_read_direct( dset.id.id, frame, buffer )
            yield buffer[:csize], dset.shape, dset.dtype

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
    import sys
    if len(sys.argv) == 1:
        hname = "bslz4testcases.h5"
        dsets = [ "data_uint%d"%(i) for i in (8,16,32) ]
    else:
        hname = sys.argv[1]
        dsets = [ d for d in sys.argv[2:] ]
    import timeit, sys
    for d in dsets:
        print()
        bench( get_frame_h5py, hname, d, 0 )
        benchiter( get_frames_h5py, hname, d )
        bench( get_chunk, hname, d, 0 )
        benchiter( get_chunks, hname, d )


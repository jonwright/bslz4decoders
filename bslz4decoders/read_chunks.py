

import h5py, hdf5plugin, numpy as np, struct

from bslz4decoders.ccodes import decoders, h5chunk

def get_frame_h5py( h5name, dset, frame ):
    """ returns h5name::/dset[frame] """
    with h5py.File(h5name, "r") as h5f:
        frm = h5f[dset][frame]
    return frm

def get_chunk( h5name, dset, frame, useh5py=True ):
    """ returns the chunk behind h5name::/dset[frame] """
    with h5py.File(h5name, "r") as h5f:
        dset = h5f[dset]
        assert len(dset.chunks) == 3
        assert dset.chunks[0] == 1
        assert dset.chunks[1] == dset.shape[1]
        assert dset.chunks[2] == dset.shape[2]
        if useh5py:
            filterlist, buffer = dset.id.read_direct_chunk( (frame, 0, 0 ) )
            buffer = np.frombuffer( buffer, dtype=np.uint8 )
        else:        # inefficient
            raise Exception("Not ready yet!")
            buffer = np.empty( dset.dtype.itemsize*dset.shape[1]*dset.shape[2]+1024,
                               np.uint8 )
            csize = h5chunk.h5_read_direct( dset.id.id, frame, buffer )
            buffer = buffer[:csize].copy()
        return buffer, dset.shape, dset.dtype

def get_frames_h5py( h5name, dset ):
    with h5py.File(h5name, "r") as h5f:
        for frm in h5f[dset]:
            yield frm

def get_chunks( h5name, dset, useh5py=True ):
    with h5py.File(h5name, "r") as h5f:
        dset = h5f[dset]
        assert len(dset.chunks) == 3
        assert dset.chunks[0] == 1
        assert dset.chunks[1] == dset.shape[1]
        assert dset.chunks[2] == dset.shape[2]
        if not useh5py:
            buffer = np.empty( dset.dtype.itemsize*dset.shape[1]*dset.shape[2]+1024,
                               np.uint8 )
        for frame in range(dset.shape[0]):
            if useh5py:
                filterlist, buffer = dset.id.read_direct_chunk( (frame, 0, 0 ) )
                buffer = np.frombuffer( buffer, dtype=np.uint8 )
            else:        # inefficient
                raise Exception("Not ready yet!")
                csize = h5chunk.h5_read_direct( dset.id.id, frame, buffer )
                buffer = buffer[:csize].copy()
            yield buffer, dset.shape, dset.dtype


def get_blocks( chunk, shape, dtyp ):
    # We do this in python as it doesn't seem worth making a call back
    # ... otherwise need to learn to call free on a numpy array
    total_bytes, blocksize = struct.unpack_from("!QL", chunk, 0)
    if blocksize == 0:
        blocksize = 8192
    nblocks =  (total_bytes + blocksize - 1) // blocksize
    blocks = np.empty( nblocks, np.uint32 )
    decoders.read_starts( chunk, dtyp.itemsize, blocksize, blocks )
    return blocksize, blocks

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
    from bslz4decoders.test.testcases import testcases
    # hum - wrong folder
    if len(sys.argv) == 1:
        hname = "bslz4testcases.h5"
        dsets = [ "data_uint%d"%(i) for i in (8,16,32) ]
    else:
        hname = sys.argv[1]
        dsets = [ d for d in sys.argv[2:] ]
    import timeit, sys
    for hname, d in testcases:
        print()
        bench( get_chunk, hname, d, 0 )
        bench( get_frame_h5py, hname, d, 0 )
        benchiter( get_chunks, hname, d )
        benchiter( get_frames_h5py, hname, d )

        chunk, shape, dtyp = get_chunk( hname, d, 0 )
        bench( get_blocks, chunk, shape, dtyp )


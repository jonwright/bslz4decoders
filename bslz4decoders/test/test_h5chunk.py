


from bslz4decoders.test.testcases import testcases as TESTCASES
from bslz4decoders.ccodes import h5chunk
from bslz4decoders.read_chunks import iter_h5chunks
import timeit
import h5py, numpy as np

def ref_chunk( h5name, dsetname, frame ):
    """ return the chunk behind h5name::/dset[frame] """
   
    dset = h5py.File(h5name, "r")[dsetname]
#        assert len(dset.chunks) == 3
#        assert dset.chunks[0] == 1
#        assert dset.chunks[1] == dset.shape[1]
#        assert dset.chunks[2] == dset.shape[2]
    filterlist, buffer = dset.id.read_direct_chunk( (frame, 0, 0 ) )
    return buffer

def h5chunk_chunk( h5name, dsetname, frame):
    """ error check please """
    hfid = dsid = None
    try:
        hfid = h5chunk.h5_open_file( h5name )
        assert hfid>0, hfid
        dsid = h5chunk.h5_open_dset( hfid, dsetname )
        assert dsid>0, dsids
        nbytes = h5chunk.h5_chunk_size(dsid, frame)
        chunk = np.empty( nbytes, np.uint8 )
        err = h5chunk.h5_read_direct( dsid, frame, chunk )
    except:
        raise Exception("Error reading %s %s %d"%(h5name, dsetname, frame))
    finally:
        if dsid is not None:
            h5chunk.h5_close_dset( dsid )
        if hfid is not None:
            h5chunk.h5_close_file( hfid )
    return chunk


def test_iterh5chunk():
    for h5name, dset in TESTCASES:
        for i, (cfg, chunk) in enumerate( iter_h5chunks( h5name, dset )):
            ref = ref_chunk( h5name, dset, i )
            assert (np.frombuffer( ref, np.uint8 ) == chunk).all()
        print(h5name, dset, i)
    print("iter_h5chunks was OK?")

def test_h5chunk():
    for h5name, dset in TESTCASES:
        for i in range(2):
            ref = ref_chunk( h5name, dset, i)
            chk = h5chunk_chunk( h5name, dset, i)
        assert (np.frombuffer( ref, np.uint8 ) == chk).all()

def bench_h5chunk():
    for h5name, dset in TESTCASES:
        for i in range(2):
            gl = { 'h5name':h5name, 'dset': dset, 'i': i,
                    'ref_chunk':ref_chunk, 'h5chunk_chunk': h5chunk_chunk }
            print("h5py %.3f ms here %.3f ms"% (
                timeit.timeit( "ref_chunk( h5name, dset, i)" , number = 100,
                globals=gl)*10,
                timeit.timeit( "h5chunk_chunk( h5name, dset, i)" , number = 100,
                globals=gl)*10), h5name, dset, i)



if __name__=="__main__":
    test_h5chunk() # warm up first
    test_iterh5chunk()
    bench_h5chunk()

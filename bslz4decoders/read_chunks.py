

import h5py, hdf5plugin, numpy as np, struct

from bslz4decoders.ccodes import decoders
from bslz4decoders.decoders import BSLZ4ChunkConfig

# All the reading from h5py should be done in a single thread
# We should have multiple threads that are reading from the same
# file at the same time. Some kind of protection might be
# needed..., perhaps:
import threading
# ... assert threading.current_thread() is threading.main_thread()
#

def get_frame_h5py( h5name, dsetname, frame ):
    """ returns h5name::/dset[frame] """
    frm = h5py.File(h5name, "r")[dsetname][frame]
    return frm

def get_frames_h5py( h5name, dsetname,  firstframe=0, lastframe=None, stepframe=1 ):
    """ Grabs one big blob of data
    It is going to fill your memory if you ask for too many
    """
    dset = h5py.File(h5name, "r")[dsetname]
    if lastframe is None:
        assert len(dset.shape) == 3
        lastframe = len(dset)
    return dset[ firstframe : lastframe : stepframe ]

def iter_frames_h5py( h5name, dsetname,  firstframe=0, lastframe=None, stepframe=1 ):
    """ Returns one frame at a time """
    dset = h5py.File(h5name, "r")[dsetname]
    if lastframe is None:
        assert len(dset.shape) == 3
        lastframe = len(dset)
    for i in range( firstframe, lastframe, stepframe ):
        yield dset[ i ]

def get_chunk( h5name, dsetname, frame ):
    """ return the chunk behind h5name::/dset[frame] """
    dset = h5py.File(h5name, "r")[dsetname]
    assert len(dset.chunks) == 3
    assert dset.chunks[0] == 1
    assert dset.chunks[1] == dset.shape[1]
    assert dset.chunks[2] == dset.shape[2]
    filterlist, buffer = dset.id.read_direct_chunk( (frame, 0, 0 ) )
    config = BSLZ4ChunkConfig( (dset.shape[1], dset.shape[2]), dset.dtype )
    return config, np.frombuffer( buffer, np.uint8 )

def get_chunks( h5name, dset, firstframe=0, lastframe=None, stepframe=1 ):
    """ return the series of chunks  """
    dset = h5py.File(h5name, "r")[dset]
    assert len(dset.chunks) == 3
    assert dset.chunks[0] == 1
    assert dset.chunks[1] == dset.shape[1]
    assert dset.chunks[2] == dset.shape[2]
    if lastframe is None:
        lastframe = dset.shape[0]
    chunks = []
    config =  BSLZ4ChunkConfig( (dset.shape[1], dset.shape[2]), dset.dtype )
    for frame in range(firstframe, lastframe, stepframe):
        filterlist, buffer = dset.id.read_direct_chunk( (frame, 0, 0 ) )
        chunks.append( np.frombuffer( buffer, np.uint8 ) )
    return config, chunks

def iter_chunks( h5name, dset, firstframe=0, lastframe=None, stepframe=1 ):
    """ return the series of chunks  """
    dset = h5py.File(h5name, "r")[dset]
    assert len(dset.chunks) == 3
    assert dset.chunks[0] == 1
    assert dset.chunks[1] == dset.shape[1]
    assert dset.chunks[2] == dset.shape[2]
    if lastframe is None:
        lastframe = dset.shape[0]
    config = BSLZ4ChunkConfig( ( dset.shape[1], dset.shape[2]), dset.dtype )
    for frame in range(firstframe, lastframe, stepframe):
        filterlist, buffer = dset.id.read_direct_chunk( (frame, 0, 0 ) )
        yield config, np.frombuffer( buffer, np.uint8 )

def queue_chunks( q, h5name, dset, firstframe=0, lastframe=None, stepframe=1 ):
    """ move this elsewhere ... """
    assert threading.current_thread() is threading.main_thread()
    for tup in iter_chunks( h5name, dset, firstframe, lastframe, stepframe ):
        q.put( tup )
    q.put(None)






import h5py, hdf5plugin, numpy as np, struct

from bslz4decoders.ccodes import decoders
from bslz4decoders.decoders import BSLZ4ChunkConfig,  BSLZ4ChunkConfigDirect
from bslz4decoders.ccodes import h5chunk

# All the reading from hdf5 should be done in a single thread
# We could have multiple threads that are reading from the same
# file at the same time. Some kind of protection might be
# needed..., perhaps:
import threading
# ... for now:
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
    # if we use frombuffer we get write-only memory that has to be copied in f2py
    # ... because const can be cast away in C I guess
    ar = np.fromstring( buffer, np.uint8 )
    return config, ar

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
        chunks.append( np.fromstring( buffer, np.uint8 ) )
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
        yield config, np.fromstring( buffer, np.uint8 )

def queue_chunks( q, h5name, dset, firstframe=0, lastframe=None, stepframe=1 ):
    """ move this elsewhere ... """
    assert threading.current_thread() is threading.main_thread()
    for tup in iter_chunks( h5name, dset, firstframe, lastframe, stepframe ):
        q.put( tup )
    q.put(None)


def iter_h5chunks( h5name, dsetname, firstframe=0, lastframe=None, stepframe=1,
                   memory = None ):
    """
    Allows you to malloc your buffer yourself (e.g. pinned)
    Otherwise it recycles the same buffer over and over (so not safe for threading)
    """
    hfid = dsid = None
    try:
        hfid = h5chunk.h5_open_file( h5name )
        assert hfid>0, hfid
        dsid = h5chunk.h5_open_dset( hfid, dsetname )
        config = BSLZ4ChunkConfigDirect( dsid )
        if lastframe is None:
            lastframe = config.nframes
        assert dsid>0, dsid
        if memory is None:
            chunk = np.empty( config.output_nbytes+4096, np.uint8 )
        else:
            chunk = memory
        for frame in range( firstframe, lastframe, stepframe ):
            # nbytes = h5chunk.h5_chunk_size(dsid, frame)
            nbytesread = h5chunk.h5_read_direct( dsid, frame, chunk )
            assert nbytesread > 0, "h5chunk.h5_read direct error "+ str(err)
            yield config, chunk[:nbytesread] # todo - does this copy on a gpu?
    except:
        raise #Exception("Error reading %s %s %d"%(h5name, dsetname, frame))
    finally:
        if dsid is not None:
            h5chunk.h5_close_dset( dsid )
        if hfid is not None:
            h5chunk.h5_close_file( hfid )


def relname( fname, relfname ):
    """ relative path name to locate child hdfs """
    return os.path.join( os.path.dirname( fname ), relname )

def iter_vds(h5name, dsname,
             # firstframe = 0, lastframe=None, stepframe=1,
             memory = None):
    """
    3D data only. Assumes simple mappings only.
    chunk_size == 2d frame_size, so chunks == (1, shape[1], shape[2])

    returns a sequence of (h5name, dset) for the real, underlying data
    """
    h = h5py.File( h5name, 'r')
    dataset = h[dsname]
    assert len(dataset.shape) == 3    # 3D series of frames
    assert dataset.is_virtual
    vds = d.virtual_sources() # list of [ VDSmap objects , ]
    # each one is a NamedTuple with vspace, file_name, dset_name, src_space
    #  vspace.shape = complete, output, dataset shape
    #  vspace.get_select_bounds() == pair of tuples with bbox corners
    #  src_space.shape = size of this item selection
    bounds = np.array( [ (vs.vspace.get_select_bounds(), vs.src_space.get_select_bounds()) for vs in vds ] )
    # vbounds.shape == num_files , 2==dest+src ,  2==start+end , 3==array index
    assert (bounds[:,:,0,1:2] == 0).all()          # check they are frames 0,0 to shape[1],shape[2]
    assert (bounds[:,:,1,1] == d.shape[1]-1).all()
    assert (bounds[:,:,1,2] == d.shape[2]-1).all()
    order = np.argsort( vbounds[:,0,0,0] ) # probably not needed ? 
    for i in order:
        vs = vds[i]
        fname = relname( h5name, vds[i].file_name )
        firstframe = bounds[i,1,0,0]
        lastframe  = bounds[i,1,1,0] + 1 # for use in python range/slicing
        if memory is None:
            iterator = iter_chunks
        else:
            iterator = iter_h5chunks # ouf
        for config, chunk in iterator( fname,
                                       vds[i].dset_name,
                                       firstframe=firstframe,
                                       lastframe=lastframe,
                                       memory=memory ):
            yield config, chunk
            #  fnum = frame_num - vspace_bounds[0][0] + source_bounds[0][0]
            #       2        - 2    + 2 == 2
            #      3605      - 3600 + 0 == 5
    


    


from bslz4decoders.lz4dc_cuda import BSLZ4CUDA
from bslz4decoders.ccodes.decoders import read_starts
from bslz4decoders.decoders import BSLZ4ChunkConfigDirect
from bslz4decoders.ccodes import h5chunk

from bslz4decoders.read_chunks import iter_h5chunks, iter_chunks, get_frame_h5py
from bslz4decoders.test.testcases import testcases as TESTCASES
import numpy as np
from timeit import default_timer
from pycuda import gpuarray
import pycuda.driver
import pycuda.autoinit
from pycuda.reduction import ReductionKernel

import pycuda.tools
import threading, queue


# summation kernels for reductions
gpu_sums = { 
        np.dtype('uint8')  : ReductionKernel(
            np.int64 , "0", "a+b", arguments="const unsigned char *in" ),
        np.dtype('uint16') : ReductionKernel(
            np.int64 , "0", "a+b", arguments="const unsigned short *in" ),
        np.dtype('uint32') : ReductionKernel(
            np.int64 , "0", "a+b", arguments="const unsigned int *in" ),
        np.dtype('int32')  : ReductionKernel(
            np.int64 , "0", "a+b", arguments="const int *in" ),
    }


class payload:
    
    def __init__(self, nbytes, blocksize=8192):
        self.gpumempool = pycuda.tools.DeviceMemoryPool()
        self.chunk = pycuda.driver.pagelocked_empty(
            nbytes, np.uint8,
            mem_flags=pycuda.driver.host_alloc_flags.DEVICEMAP)
        self.blocks = pycuda.driver.pagelocked_empty(
            (nbytes + 8192 - 1) // 8192, np.uint32,
            mem_flags=pycuda.driver.host_alloc_flags.DEVICEMAP)
        self.sums_d = gpuarray.empty( (1,), np.int64,
                                      allocator = self.gpumempool.allocate )
        self.sums = pycuda.driver.pagelocked_empty((1,), np.int64,
            mem_flags=pycuda.driver.host_alloc_flags.DEVICEMAP)
        self.dc = None
        
    def read_direct(self, dsid, frame ):
        self.frame = frame
        self.nbytesread = h5chunk.h5_read_direct( dsid, frame, self.chunk )
        assert self.nbytesread > 0, "h5chunk.h5_read direct error "+ str(err)
        
    def read_starts( self ):
        read_starts( self.chunk[:self.nbytesread],
                     self.config.dtype.itemsize,
                     self.config.blocksize,
                     self.blocks[:self.config.nblocks] )
        
    def rungpu(self):
        if self.dc is None:
            self.dc = BSLZ4CUDA( self.config.output_nbytes,
                                 self.config.dtype.itemsize,
                                 self.config.blocksize,
                                 allocator=self.gpumempool.allocate )
            self.out_gpu = gpuarray.empty( self.config.shape,
                                           dtype = self.config.dtype,
                                           allocator= self.gpumempool.allocate )  
        else:
            if self.config.output_nbytes != self.out_gpu.nbytes:
                self.out_gpu = gpuarray.empty( self.config.shape,
                                          dtype = self.config.dtype,
                                          allocator = self.gpumempool.allocate )
                self.dc.reset( self.config.output_nbytes,
                          self.config.dtype.itemsize,
                          self.config.blocksize )
        _ = self.dc( self.chunk[:self.nbytesread],
                     self.blocks[:self.config.nblocks], self.out_gpu )
        gpu_sums[ self.config.dtype ]( self.out_gpu,
                                       out = self.sums_d,
                                       stream = self.dc.stream,
                                       allocator = self.gpumempool.allocate )
        pycuda.driver.memcpy_dtoh_async( self.sums,
                                         self.sums_d.gpudata,
                                         stream = self.dc.stream )



            
    

        
        
class H5Reader( threading.Thread ):
    def __init__(self, h5q, meminq, memoutq):
        self.h5q = h5q
        self.meminq = meminq
        self.memoutq = memoutq
        threading.Thread.__init__(self)

    def run(self):
        while 1:
            h5name, dsetname = self.h5q.get()
            if h5name is None:
                break
            self.iter_h5chunks( h5name, dsetname )
        
    def iter_h5chunks( self, h5name, dsetname ):
        hfid = dsid = None
        try:
            hfid = h5chunk.h5_open_file( h5name )
            assert hfid>0, hfid
            dsid = h5chunk.h5_open_dset( hfid, dsetname )
            config = BSLZ4ChunkConfigDirect( dsid )
            assert dsid>0, dsid
            for frame in range( config.nframes ):
                p = self.meminq.get()
                p.config = config
                p.read_direct( dsid, frame )
                self.memoutq.put( p )
            self.memoutq.put( 0 )
        except:
            raise #Exception("Error reading %s %s %d"%(h5name, dsetname, frame))
        finally:
            if dsid is not None:
                h5chunk.h5_close_dset( dsid )
            if hfid is not None:
                h5chunk.h5_close_file( hfid )

class UnBlock( threading.Thread ):
    def __init__(self, inq, outq):
        self.inq = inq
        self.outq = outq
        threading.Thread.__init__(self)
    def run(self):
        while 1:
            p = self.inq.get()
            if p is None:
                break
            if p != 0:
                p.read_starts()
            self.outq.put(p)

            
nbytes = 2168*2064*4
nthread = 8

h5q = queue.Queue()      # filenames
emptyq = queue.Queue()   # chunks
blq = queue.Queue()      # add blocks
gpuq = queue.Queue()     # feed to gpu

reader = H5Reader( h5q, emptyq, blq )
reader.start()
blocker = UnBlock( blq, gpuq )
blocker.start()
buff = {}

for i in range(nthread):
    buff[i] = payload( nbytes )
    buff[i].name = i

    
def sum_reduce_cuda( hname, dset ):
    sums = {}
    for i in range(nthread):
        emptyq.put( buff[i] )
    h5q.put( ( hname, dset ) )
    ongpu = []
    while 1:
        p = gpuq.get()
        if p == 0:
            break
        p.rungpu()
        ongpu.append( p )
        if len(ongpu)>nthread//2:
            p = ongpu.pop(0)
            p.dc.stream.synchronize()
            sums[p.frame] = int(p.sums)
            emptyq.put(p)
    for i in range(len(ongpu)):
        p = ongpu.pop(0)
        p.dc.stream.synchronize()
        sums[p.frame] = int(p.sums)
    while emptyq.qsize():
        emptyq.get_nowait()
    return sums




def main():
    import sum_testcases
    try:
        for i in range(3):
            sum_testcases.run_sum_testcases( sum_reduce_cuda )
    finally:
        h5q.put( (None, None) )
        blq.put(None)
        reader.join()
        blocker.join()
        for i in range(nthread):
            b = buff.pop(i)
            del b
        
    
def bug():
    try:
        sum_reduce_cuda( 'eiger_0000.h5', 'entry_0000/measurement/data' )
    except:
        print("Got an error")
        raise
    finally:
        h5q.put( (None, None) )
        blq.put(None)
        reader.join()
        blocker.join()
        for i in range(nthread):
            b = buff.pop(i)
            del b

if __name__=="__main__":
#    bug()
    main()

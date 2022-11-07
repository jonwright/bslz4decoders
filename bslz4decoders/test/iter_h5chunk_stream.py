
from bslz4decoders.lz4dc_cuda import BSLZ4CUDA
from bslz4decoders.read_chunks import iter_h5chunks, iter_chunks, get_frame_h5py
from bslz4decoders.test.testcases import testcases as TESTCASES

from timeit import default_timer
import numpy as np, pylab as pl
from pycuda import gpuarray
import pycuda.driver
from pycuda.reduction import ReductionKernel

import pycuda.tools
memorypool = pycuda.tools.PageLockedMemoryPool()
gpumempool = pycuda.tools.DeviceMemoryPool()

def check_and_show( a1, a2 ):
    assert a1.shape == a2.shape, str(a1.shape)+str(a2.shape)
    assert a1.dtype == a2.dtype, str(a1.dtype)+str(a2.dtype)
    match = (a1 == a2).all()
    if not match:
        import pylab as pl
        f, a = pl.subplots(2,2,sharex=True, sharey=True)
        a[0][0].imshow(a1, interpolation='nearest')
        a[1][0].imshow(a2, interpolation='nearest')
        a[1][1].imshow(a1-a2, interpolation='nearest')
        pl.show()


def testcuda(hname, dset):
    gpu_sums = { 
        np.dtype('uint8')  : ReductionKernel( np.int64 , "0", "a+b", arguments="const unsigned char *in" ),
        np.dtype('uint16') : ReductionKernel( np.int64 , "0", "a+b", arguments="const unsigned short *in" ),
        np.dtype('uint32') : ReductionKernel( np.int64 , "0", "a+b", arguments="const unsigned int *in" ),
        np.dtype('int32') : ReductionKernel( np.int64 , "0", "a+b", arguments="const int *in" ),
    }
    out_gpu = None
    dc = None
    t0 = default_timer()*1e3
    print("Read  Decode Init GPUdc GPUsum ", end="")
    if __debug__: 
        print("DtoH Hsum H5pyRead Hsum")
    else:
        print()
        
    iterator = iter_h5chunks( hname, dset, memory = pycuda.driver.pagelocked_empty(
        (4096*4096*4,), np.uint8, mem_flags=pycuda.driver.host_alloc_flags.DEVICEMAP) )
    frm = 0
    nb = 0##
    import concurrent.futures#
    reader = concurrent.futures.ThreadPoolExecutor(1)
    ##
    def pipe( iterator ):
        for config, chunk in iterator:
            blocks = config.get_blocks( chunk )
            yield config, chunk, blocks
    
    p = pipe( iterator )
    datafuture = reader.submit( next, p )

    while 1:
        t = [ default_timer()*1e3, ]
        try:
            # todo : put this in a different thread. And only get next after
            # you finish reading the data.
            #            config,chunk = next( iterator )
            config, chunk, blocks = datafuture.result()
        except:
            break
        if dc is None:
            dc = BSLZ4CUDA( config.output_nbytes, config.dtype.itemsize, config.blocksize )
            out_gpu = gpuarray.empty( config.shape, dtype = config.dtype,
                                      allocator= gpumempool.allocate )  
        else:
            if config.output_nbytes != out_gpu.nbytes:
                out_gpu = gpuarray.empty( config.shape, dtype = config.dtype,
                                          allocator = gpumempool.allocate )
                                          
                dc.reset( config.output_nbytes, config.dtype.itemsize, config.blocksize )
        t.append( default_timer()*1e3 )
        _ = dc( chunk, blocks, out_gpu )
        datafuture = reader.submit( next, p )
        t.append( default_timer()*1e3 )
        sgpu = gpu_sums[ config.dtype ]( out_gpu, stream=dc.stream )
        t.append( default_timer()*1e3 )
        sgpu = sgpu.get()
        t.append( default_timer()*1e3 )#
        if __debug__:
            data = out_gpu.get()
            sdata = data.sum( dtype = np.int64)
            t.append( default_timer()*1e3 )
            ref = get_frame_h5py( hname, dset, frm )
            t.append( default_timer()*1e3 )
            sref = ref.ravel().sum(dtype = np.int64)
            t.append( default_timer()*1e3 ) 
            check_and_show( ref, data )
            assert(sref == sdata),  " ".join((repr(sref),repr(sdata)))
            assert(sref == sgpu),  " ".join((repr(sref),repr(sdata)))
        dt = [t[i]-t[i-1] for i in range(1,len(t))]
        print(("%.3f ms, "*len(dt))%tuple(dt), hname, dset )
        frm += 1
        nb += config.output_nbytes
        if frm == 1:
            t1 = default_timer()*1e3
    print("Total time after startup",t[-1]-t0,"ms", t[-1]-t1,'ms after init')
    print("Total GB",nb/1e9,"speed ~ %.3f GB/s  asymptotic ~ %.3f GB/s"%((nb/1e6)/(t[-1]-t0),
                                                                         (nb - config.output_nbytes)/1e6/(t[-1]-t1)))
    print( frm, 'frames', 1e3*frm/(t[-1]-t0), 1e3*(frm-1)/(t[-1]-t1),'fps')

if __name__=="__main__":
    import sys
    testcuda(sys.argv[1], sys.argv[2])

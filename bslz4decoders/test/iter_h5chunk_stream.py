
from bslz4decoders.lz4dc_cuda import BSLZ4CUDA
from bslz4decoders.read_chunks import iter_h5chunks, iter_chunks, get_frame_h5py
from bslz4decoders.test.testcases import testcases as TESTCASES
import numpy as np
from timeit import default_timer
from pycuda import gpuarray
import pycuda.driver
from pycuda.reduction import ReductionKernel

import pycuda.tools

gpumempool = pycuda.tools.DeviceMemoryPool()

import concurrent.futures
# one spare thread, not one new one per run
executor = concurrent.futures.ThreadPoolExecutor(1) 

# summation kernels for reductions
gpu_sums = { 
        np.dtype('uint8')  : ReductionKernel( np.int64 , "0", "a+b", arguments="const unsigned char *in" ),
        np.dtype('uint16') : ReductionKernel( np.int64 , "0", "a+b", arguments="const unsigned short *in" ),
        np.dtype('uint32') : ReductionKernel( np.int64 , "0", "a+b", arguments="const unsigned int *in" ),
        np.dtype('int32')  : ReductionKernel( np.int64 , "0", "a+b", arguments="const int *in" ),
    }

      

def blocker( hname, dset, start, mem ):
    # To be called in threads. Reads chunks and computes blocks
    for config, chunk in iter_h5chunks( hname, dset, memory = mem, firstframe=start, stepframe=2):
        blocks = config.get_blocks( chunk )
        yield config, chunk, blocks

nbytes = 4096*4096*4
mem = ( pycuda.driver.pagelocked_empty( nbytes, np.uint8, 
                                        mem_flags=pycuda.driver.host_alloc_flags.DEVICEMAP),
        pycuda.driver.pagelocked_empty( nbytes, np.uint8, 
                                        mem_flags=pycuda.driver.host_alloc_flags.DEVICEMAP) )
    
def readerblocker( hname, dset ):
    global mem
    # have 2 reading objects. One will read while the other computes decompression.
    # try to avoid reading into the data that is being transferred
    readers = ( blocker( hname, dset, 0, mem[0]), blocker( hname, dset, 1, mem[1]) )
    j = 0
    q = executor.submit( next, readers[j] )
    while 1:
        try:
            result = q.result()
        except StopIteration:
            return
        j = (j+1)%2
        q = executor.submit( next, readers[j] )
        yield result
        


def sum_reduce_cuda(hname, dset):
    """ Do the reduction on the GPU """
    dc = None        
    out_gpu = None        
    frm = 0
    sums = {}
    t0 = default_timer()
    for config, chunk, blocks in readerblocker( hname, dset ): # threads in her
        if dc is None:
            dc = BSLZ4CUDA( config.output_nbytes, config.dtype.itemsize, config.blocksize,
                            allocator=gpumempool.allocate )
            out_gpu = gpuarray.empty( config.shape, dtype = config.dtype,
                                      allocator= gpumempool.allocate )  
        else:
            if config.output_nbytes != out_gpu.nbytes:
                out_gpu = gpuarray.empty( config.shape, dtype = config.dtype,
                                          allocator = gpumempool.allocate )
                dc.reset( config.output_nbytes, config.dtype.itemsize, config.blocksize )
        _ = dc( chunk, blocks, out_gpu )
        sgpu = gpu_sums[ config.dtype ]( out_gpu, stream=dc.stream )
        sums[frm] = sgpu.get()
        if frm == 0:
            t1 = default_timer()
        frm += 1
    t2 = default_timer()
    print("GPU timing, first frame %.3f ms, next ones, %.3f ms %.1f maxfps"%((t1-t0)*1e3, (t2-t1)*1e3/(frm-1),
                                                                 (frm-1)/(t2-t1)))
                                                                
    return sums


if __name__=="__main__":
    import sum_testcases
    for i in range(2):
        sum_testcases.run_sum_testcases( sum_reduce_cuda )
    

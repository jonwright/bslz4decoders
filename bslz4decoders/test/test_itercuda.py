
from bslz4decoders.lz4dc_cuda import BSLZ4CUDA
from bslz4decoders.read_chunks import iter_chunks, get_frame_h5py
from bslz4decoders.test.testcases import testcases as TESTCASES

from timeit import default_timer
import numpy as np, pylab as pl
from pycuda import gpuarray
from pycuda.reduction import ReductionKernel



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
    gpu_sums = { 1 : ReductionKernel( np.int64 , "0", "a+b", arguments="const unsigned char *in" ),
            2 : ReductionKernel( np.int64 , "0", "a+b", arguments="const unsigned short *in" ),
            4 : ReductionKernel( np.int64 , "0", "a+b", arguments="const unsigned int *in" ),
    }
    out_gpu = None
    dc = None
    print("Getchunk Init GPUdc GPUsum DtoH Hsum H5pyRead Hsum")
    for frm, (config,chunk ) in enumerate( iter_chunks( hname, dset ) ):
        t = [ default_timer()*1e3, ]
        blocks = config.get_blocks( chunk )
        nbytes = config.output_nbytes
        t.append( default_timer()*1e3)
        if dc is None:
            dc = BSLZ4CUDA( nbytes, config.dtype.itemsize, config.blocksize )
            out_gpu = gpuarray.empty( config.shape, dtype = config.dtype )
        else:
            if nbytes != out_gpu.nbytes:
                out_gpu = gpuarray.empty( config.shape, dtype = config.dtype )
                dc.reset( config.output_nbytes, config.dtype.itemsize, config.blocksize )
        t.append( default_timer()*1e3 )
        _ = dc( chunk, blocks, out_gpu )
        t.append( default_timer()*1e3 )
        sgpu = gpu_sums[ config.dtype.itemsize ]( out_gpu ).get()
        t.append( default_timer()*1e3 )
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
            # print(sref, sdata, type(sref), type(sdata))

if __name__=="__main__":
    import sys
    testcuda(sys.argv[1], sys.argv[2])

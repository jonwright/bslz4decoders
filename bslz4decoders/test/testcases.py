
import os, h5py

def find_testcases():
    testcases = []
    for hname, dset in ( ( 'bslz4testcases.h5', None ),
                       ("/data/id11/nanoscope/blc12454/id11/WAu5um/WAu5um_DT3/scan0001/eiger_0000.h5",
                            "/entry_0000/ESRF-ID11/eiger/data"),
                        ):
        if os.path.exists(hname):
            if dset is None:
                with h5py.File( hname, 'r' ) as h:
                    for ds in list(h['/']):
                        testcases.append( (hname, ds) )
            else:
                testcases.append( (hname, dset) )
    return testcases


testcases = find_testcases()

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


def testcuda():
    from bslz4decoders.lz4dc_cuda import BSLZ4CUDA
    from bslz4decoders.read_chunks import get_chunk, get_blocks, get_frame_h5py
    from timeit import default_timer
    import numpy
    import numpy as np
    from pycuda import gpuarray
    from pycuda.reduction import ReductionKernel

    gpu_sums = { 1 : ReductionKernel( np.int64 , "0", "a+b", arguments="const unsigned char *in" ),
            2 : ReductionKernel( np.int64 , "0", "a+b", arguments="const unsigned short *in" ),
            4 : ReductionKernel( np.int64 , "0", "a+b", arguments="const unsigned int *in" ),
    }
    out_gpu = None
    frm = 0
    dc = None
    print("Getchunk Init GPUdc GPUsum DtoH Hsum H5pyRead Hsum")
    for hname, dset in testcases[::-1]:
        t = [ default_timer()*1e3, ]
        chunk, shp, dtyp = get_chunk( hname, dset, frm)
        blocksize, blocks = get_blocks( chunk, shp, dtyp )
        nbytes = shp[1]*shp[2]*dtyp.itemsize
        t.append( default_timer()*1e3)
        if dc is None:
            dc = BSLZ4CUDA( shp[1]*shp[2]*dtyp.itemsize, dtyp.itemsize, blocksize )
            out_gpu = gpuarray.empty( (shp[1]*shp[2]), dtype = dtyp )
            #out_gpu = numpy.empty( (shp[1]*shp[2]), dtype = dtyp )
        else:
            if nbytes != out_gpu.nbytes:
                out_gpu = gpuarray.empty( (shp[1]*shp[2]), dtype = dtyp )
                #out_gpu = numpy.empty( (shp[1]*shp[2]), dtype = dtyp )
                dc.reset( shp[1]*shp[2]*dtyp.itemsize, dtyp.itemsize, blocksize )
        t.append( default_timer()*1e3 )
        _ = dc( chunk, blocks, out_gpu )
        t.append( default_timer()*1e3 )
        sgpu = gpu_sums[ dtyp.itemsize ]( out_gpu ).get()
        t.append( default_timer()*1e3 )
        data = out_gpu.get()
        sdata = data.sum( dtype = numpy.int64)
        t.append( default_timer()*1e3 )
        ref = get_frame_h5py( hname, dset, frm )
        t.append( default_timer()*1e3 )
        sref = ref.ravel().sum(dtype = numpy.int64)
        t.append( default_timer()*1e3 )
        dt = [t[i]-t[i-1] for i in range(1,len(t))]
        print(("%.3f ms, "*len(dt))%tuple(dt), hname, dset )
        # print(sref, sdata, type(sref), type(sdata))
        check_and_show( ref, data.reshape((shp[1],shp[2])))
        assert(sref == sdata),  " ".join((repr(sref),repr(sdata)))
        assert(sref == sgpu),  " ".join((repr(sref),repr(sdata)))



if __name__=="__main__":
    testcuda()

from bslz4decoders.lz4dc_cuda import BSLZ4CUDA
from bslz4decoders.read_chunks import get_chunk, get_frame_h5py
from bslz4decoders.test.testcases import testcases as TESTCASES

from timeit import default_timer
import numpy as np
from pycuda import gpuarray
from pycuda.scan import InclusiveScanKernel
from pycuda.elementwise import ElementWiseKernel


class ticktock:
    def __init__(self):
        self.t = [ default_timer()*1e3, ]
    def __call__(self):
        self.t.append( default_timer()*1e3 )

gpu_cumsum_u32 = InclusiveScanKernel(np.int32, "a+b", arguments="const unsigned int *in" )


gpu_threshold_u32 = ElementWiseKernel("unsigned int *in, unsigned int *out , unsigned int threshold",
                                      "out[i] = (in[i] > threshold ) ? 1 : 0"
                                      "threshold_u32" )
gpu_threshold_u16 = ElementWiseKernel("unsigned short *in, unsigned int *out , unsigned short threshold",
                                      "out[i] = (in[i] > threshold ) ? 1 : 0"
                                      "threshold_u16" )

wgpu = """
__global__ write_gpu%s( %s *image, unsigned int *idx, unsigned int *msk,
                      %s *dest, unsigned int *ij ){
   int i,j
   i = blockIdx.x * blockDim.x + threadIdx.x;
   if(msk[i]>0){
      j = idx[i];
      dest[j] = image[i];
      ij[j]   = i;
   }
}
"""



write_gpu_mod = SourceModule( wgpu%( "u32", "unsigned int", "unsigned int") +
                              wgpu%( "u16", "unsigned short", "unsigned short")  )

def process_thread( qin ):

    gpu_img = None # the image data output
    dc = None
    while 1:
        args = qin.get()
        if args is None:
            break
        config, chunk = args
        blocks = config.get_blocks( chunk ) # parse data on cpu ...
        if dc is None:
            dc = BSLZ4CUDA( config.output_nbytes, config.dtype.itemsize, config.blocksize )
            out_gpu = gpuarray.empty( config.shape, dtype = config.dtype )
            msk_gpu = gpuarray.empty( config.shape, dtype = np.uint32 )
        # decompress
        dc( chunk, blocks, out_gpu )
        # threshold
        if config.dtype == np.uint32:
            gpu_threshold_u32( out_gpu, msk_gpu, THRESHOLD )
        elif  config.dtype == np.uint16:
            gpu_threshold_u16( out_gpu, msk_gpu, THRESHOLD )
        else:
            print("no threshold for that datatype")
        # find output index destinations
        idx_gpu = gpu_cumsum_u32( msk_gpu )
        # write out to the destination 
        write_gpu( out_gpu, msk_gpu, idx_gpu, sparsedata_gpu, sparse_idx_gpu )
        
        
        
def bench(h5name, dset):
    inputq = queue.Queue(maxsize=8)
    processor = threading.Thread( target=process_thread, args=( cq, ) )
    processor.start()
    read_chunks.queue_chunks( inputq, h5name, dset )

                dc.reset( config.output_nbytes, config.dtype.itemsize, blocksize )
        t()
        _ = dc( chunk, blocks, out_gpu )
        t()
        sgpu = gpu_sums[ config.dtype.itemsize ]( out_gpu ).get()
        t()
        data = out_gpu.get()
        sdata = data.sum( dtype = np.int64)
        t()
        ref = get_frame_h5py( hname, dset, frm )
        t()
        sref = ref.ravel().sum(dtype = np.int64)
        t()
        dt = [t[i]-t[i-1] for i in range(1,len(t))]
        print(("%.3f ms, "*len(dt))%tuple(dt), hname, dset )
        # print(sref, sdata, type(sref), type(sdata))
        check_and_show( ref, data )
        assert(sref == sdata),  " ".join((repr(sref),repr(sdata)))
        assert(sref == sgpu),  " ".join((repr(sref),repr(sdata)))



    processor.join()
        

if __name__=="__main__":
    import sys
    hname = sys.argv[1]
    dset = sys.argv[2]
    bench( hname, dset )





import os, sys
import numpy as np, h5py
import timeit
from bslz4decoders.read_chunks import get_chunk

import pycuda.autoinit, pycuda.driver, pycuda.gpuarray, pycuda.compiler


def get_sources():
    folder =  os.path.join( os.path.split(__file__)[0], "cuda")
    names  = "nvcomp_extract.cu", "h5lz4dc.cu", "shuffles.cu"
    lines = [ open( os.path.join( folder, name ), 'r' ).read() for name in  names ]
    return " ".join( lines )


class BSLZ4CUDA:

    modsrc = get_sources()

    def __init__(self, total_output_bytes, bpp, blocksize, allocator=None ):
        """ cache / reuse memory """
        try:
            self.mod = pycuda.compiler.SourceModule( self.modsrc )
        except:
            open('kernel.cu', 'w').write(self.modsrc)
            raise
        self.h5lz4dc   = self.mod.get_function("h5lz4dc")
        self.h5lz4dc.prepare( [np.intp, np.intp, np.uint32, np.uint32, np.intp] )
        self.shuf_end = self.mod.get_function("simple_shuffle_end")
        self.shuf_end.prepare( [np.intp, np.intp, np.uint32, np.uint32, np.uint32, np.uint32 ] )
        self.copybytes = self.mod.get_function("copybytes")
        self.copybytes.prepare( [np.intp, np.uint32, np.intp, np.uint32])
        self.stream = pycuda.driver.Stream()
        self.allocator = allocator
        self.reset(  total_output_bytes, bpp, blocksize )


    def reset(self, total_output_bytes, bpp, blocksize):

        self.total_output_bytes = total_output_bytes
        self.bpp = bpp
        self.blocksize = np.uint32( blocksize )
        self.shuf_8192 = self.mod.get_function( "shuf_8192_%d"%(bpp*8) )
        self.shuf_8192.prepare( [np.intp, np.intp] )

        # number of full 8 kB blocks (rounds down)
        self.nblocks = self.total_output_bytes // 8192
        self.tblocks = (self.total_output_bytes + 8191)//8192
        # block size for shuffles
        # shuffle blocking:
        self.shblock = (32,32,1)   # various transpose examples
        self.shgrid  = ( self.nblocks , 2 , 1 )
        # LZ4 blocking :
        # a single block is 32 threads and this handles 2 chunks
        # ... to be investigated in more detail perhaps
        self.lz4block = (32,2,1)
        self.lz4grid = ((self.tblocks+1)//2, 1, 1)
        self.output_d = pycuda.gpuarray.empty( self.total_output_bytes, dtype = np.uint8, allocator=self.allocator)
        self.chunk_d = pycuda.gpuarray.empty( self.total_output_bytes, dtype = np.uint8, allocator=self.allocator)
        self.blocks_d = pycuda.gpuarray.empty( self.tblocks, np.uint32, allocator=self.allocator) 
        # 32 bit offsets into the compressed for blocks
        self.shuf_d = None #  hold the final unshuffled data, can be an output arg
        # last few bytes are copied:
        self.bytes_to_copy  = ( self.total_output_bytes % ( bpp * 8 ) )
        # print("bytes to copy", self.bytes_to_copy)
        self.evt = pycuda.driver.Event()

    def __call__(self, chunk, blocks, outarg=None ):
        """
        chunk = bslz4 compressed data
        blocks = block start intex (12, ... )
        return out = output array (can be numpy or __cuda_array_interface__)
        """
        assert len(blocks) == self.tblocks
        if outarg is None: # allocate device and return array
            if self.shuf_d is None:
                self.shuf_d = pycuda.gpuarray.empty( self.total_output_bytes, dtype = np.uint8, allocator=self.allocator )
            outd = self.shuf_d
            out = np.empty( self.total_output_bytes, np.uint8 )
        else:
            if hasattr( outarg, "__cuda_array_interface__" ): # return a CUDA array
                outd = outarg
            else: # hope that was a numpy array
                out = outarg.ravel().view( np.uint8 )
                assert out.nbytes == self.total_output_bytes, "out is the wrong size"
                assert len(out.shape) == 1, "use flat/ravelled array for output (no padding)"
                if self.shuf_d is None:
                    self.shuf_d = pycuda.gpuarray.empty( self.total_output_bytes, dtype = np.uint8, allocator=self.allocator)
                outd = self.shuf_d
        # the compressed data
        pycuda.driver.memcpy_htod_async( self.chunk_d.gpudata, chunk, self.stream )
        # 32 bit offsets into the compressed for blocks :
        pycuda.driver.memcpy_htod_async( self.blocks_d.gpudata, blocks, self.stream )
        # now decompress
        self.h5lz4dc.prepared_async_call(
            self.lz4grid,
            self.lz4block,
            self.stream,
            self.chunk_d.gpudata,
            self.blocks_d.gpudata,
            self.tblocks,
            self.blocksize,
            self.output_d.gpudata,
            )
        self.shuf_8192.prepared_async_call( self.shgrid, self.shblock, self.stream,
                                            self.output_d.gpudata, outd.gpudata )
        pos = self.nblocks*self.blocksize
        todo = self.total_output_bytes - pos
        if todo > self.bytes_to_copy:
            self.shuf_end.prepared_async_call((int((todo+31)//32),1,1), (32,1,1), self.stream,
                                             self.output_d.gpudata, outd.gpudata, self.blocksize,
                                             todo,
                                             self.bpp,
                                             pos)
        if hasattr( outarg, "__cuda_array_interface__"):
            if self.bytes_to_copy > 0:
                self.copybytes.prepared_async_call( (1,1,1),( self.bytes_to_copy, 1, 1 ),self.stream,
                                                    self.chunk_d.gpudata, chunk.nbytes - self.bytes_to_copy,
                                                    outd.gpudata, self.total_output_bytes - self.bytes_to_copy)
            return outd
        else:
            try:
                self.shuf_d.get( out )
            except:
                print(self.shuf_d.shape, self.shuf_d.dtype, out.shape, out.dtype)
                raise
            if self.bytes_to_copy > 0:
                out[-self.bytes_to_copy:] = chunk[-self.bytes_to_copy:]
            return out
        raise Exception("failed")

def testcase( hname, dset, frm):
    print("Reading", hname, "::", dset, "[", frm, "]")
    config, chunk  = get_chunk( hname, dset, frm )
    blocksize, blocks = config.get_blocks( chunk )

    output = np.empty( config.output_nbytes , np.uint8 )
    decompressor = BSLZ4CUDA( config.output_nbytes, config.dtype.itemsize, config.blocksize )
    decomp = decompressor( chunk, blocks, outarg=output ).view(config.dtype).reshape(config.shape)

    ref = h5py.File( hname, 'r' )[dset][frm]

    if (ref==decomp).all():
        print("Test passes!!!")
    else:
        print("FAILS!!!")
        print("decomp:",        decomp)
#        for i in range(0,8192,64):
#          print(i,decomp.ravel()[i:i+64])
        print("ref:")
        print(ref)
        err = abs(ref-decomp).ravel()
        ierr  = np.argmax( err)
        print(ierr, decomp.ravel()[ierr], ref.ravel()[ierr] )
        print(ref.ravel()[-10:])
        print(decomp.ravel()[-10:])
#        sys.exit()
        import pylab as pl
        pl.imshow(ref,aspect='auto',interpolation='nearest')
        pl.figure()
        pl.imshow(decomp, aspect='auto', interpolation='nearest')
        pl.figure()
        pl.imshow(ref-decomp, aspect='auto', interpolation='nearest')
        pl.show()



if __name__=="__main__":

    hname = sys.argv[1]
    dset  = sys.argv[2]
    frm   = int(sys.argv[3])
    testcase( hname, dset, frm )

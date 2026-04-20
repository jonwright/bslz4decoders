



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

# Only compile this once on import!
modsrc = get_sources()
mod = pycuda.compiler.SourceModule( modsrc )
h5lz4dc   = mod.get_function("h5lz4dc")
h5lz4dc.prepare( [np.intp, np.intp, np.uint32, np.uint32, np.intp] )
h5lz4dc_batch = mod.get_function("h5lz4dc_batch")
h5lz4dc_batch.prepare( [np.intp, np.intp, np.intp, np.uint32, np.intp] )
h5lz4dc_shuffle_batch = mod.get_function("h5lz4dc_shuffle_batch")
h5lz4dc_shuffle_batch.prepare( [np.intp, np.intp, np.intp, np.intp, np.uint32, np.uint32, np.intp] )
shuf_end = mod.get_function("simple_shuffle_end")
shuf_end.prepare( [np.intp, np.intp, np.uint32, np.uint32, np.uint32, np.uint32 ] )
copybytes = mod.get_function("copybytes")
copybytes.prepare( [np.intp, np.uint32, np.intp, np.uint32])
shuf_8192 = {}
for bpp in (1,2,4):
    shuf_8192[bpp] = mod.get_function( "shuf_8192_%d"%(bpp*8) )
    shuf_8192[bpp].prepare( [np.intp, np.intp] )

        
class BSLZ4CUDA:
        
    def __init__(self, total_output_bytes, bpp, blocksize, allocator=pycuda.driver.mem_alloc ):
        """ cache / reuse memory """
        self.stream = pycuda.driver.Stream()
        self.allocator = allocator
        self.reset(  total_output_bytes, bpp, blocksize )


    def reset(self, total_output_bytes, bpp, blocksize):

        self.total_output_bytes = total_output_bytes
        self.bpp = bpp
        self.blocksize = np.uint32( blocksize )
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
        h5lz4dc.prepared_async_call(
            self.lz4grid,
            self.lz4block,
            self.stream,
            self.chunk_d.gpudata,
            self.blocks_d.gpudata,
            self.tblocks,
            self.blocksize,
            self.output_d.gpudata,
            )
        shuf_8192[self.bpp].prepared_async_call( self.shgrid, self.shblock, self.stream,
                                                      self.output_d.gpudata, outd.gpudata )
        pos = self.nblocks*self.blocksize
        todo = self.total_output_bytes - pos
        if todo > self.bytes_to_copy:
            shuf_end.prepared_async_call((int((todo+31)//32),1,1), (32,1,1), self.stream,
                                             self.output_d.gpudata, outd.gpudata, self.blocksize,
                                             todo,
                                             self.bpp,
                                             pos)
        if hasattr( outarg, "__cuda_array_interface__"):
            if self.bytes_to_copy > 0:
                copybytes.prepared_async_call( (1,1,1),( self.bytes_to_copy, 1, 1 ),self.stream,
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

    def batch_call(self, chunks_list, blocks_list, outputs_list):
        """
        Process multiple images in a single fused kernel call.
        
        chunks_list: list of compressed chunk arrays (numpy or pagelocked)
        blocks_list: list of block start arrays
        outputs_list: list of output arrays (numpy or GPU arrays)
        
        All images must have same bpp and blocksize (configured in __init__).
        Returns list of output arrays.
        """
        num_images = len(chunks_list)
        assert len(blocks_list) == num_images
        assert len(outputs_list) == num_images
        
        # Allocate device memory for pointer arrays
        compressed_ptrs_h = np.empty(num_images, dtype=np.intp)
        blocks_ptrs_h = np.empty(num_images, dtype=np.intp)
        output_ptrs_h = np.empty(num_images, dtype=np.intp)
        num_blocks_h = np.empty(num_images, dtype=np.uint32)
        total_bytes_h = np.empty(num_images, dtype=np.uint32)
        
        # Upload chunks and blocks, collect pointers
        chunks_d = []
        blocks_d = []
        outputs_d = []
        
        for i in range(num_images):
            # Upload compressed chunk
            chunk_d = pycuda.gpuarray.to_gpu(chunks_list[i])
            chunks_d.append(chunk_d)
            compressed_ptrs_h[i] = int(chunk_d.gpudata)
            
            # Upload block starts
            block_d = pycuda.gpuarray.to_gpu(blocks_list[i])
            blocks_d.append(block_d)
            blocks_ptrs_h[i] = int(block_d.gpudata)
            num_blocks_h[i] = len(blocks_list[i])
            
            # Handle output
            if hasattr(outputs_list[i], "__cuda_array_interface__"):
                output_d = outputs_list[i]
            else:
                output_d = pycuda.gpuarray.to_gpu(outputs_list[i])
            outputs_d.append(output_d)
            output_ptrs_h[i] = int(output_d.gpudata)
            total_bytes_h[i] = self.total_output_bytes
        
        # Upload pointer arrays
        compressed_ptrs_d = pycuda.gpuarray.to_gpu(compressed_ptrs_h)
        blocks_ptrs_d = pycuda.gpuarray.to_gpu(blocks_ptrs_h)
        output_ptrs_d = pycuda.gpuarray.to_gpu(output_ptrs_h)
        num_blocks_d = pycuda.gpuarray.to_gpu(num_blocks_h)
        total_bytes_d = pycuda.gpuarray.to_gpu(total_bytes_h)
        
        # Launch fused kernel
        max_blocks = int(num_blocks_h.max())
        lz4grid = ((max_blocks + 1) // 2, 1, num_images)
        
        h5lz4dc_shuffle_batch.prepared_call(
            lz4grid,
            self.lz4block,
            compressed_ptrs_d.gpudata,
            blocks_ptrs_d.gpudata,
            num_blocks_d.gpudata,
            total_bytes_d.gpudata,
            self.blocksize,
            np.uint32(self.bpp),
            output_ptrs_d.gpudata
        )
        
        # Download results for numpy outputs
        results = []
        for i in range(num_images):
            if not hasattr(outputs_list[i], "__cuda_array_interface__"):
                outputs_d[i].get(outputs_list[i])
            results.append(outputs_list[i])
        
        return results

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

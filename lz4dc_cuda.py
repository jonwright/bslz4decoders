



import sys
import numpy as np, bitshuffle, hdf5plugin, h5py

from read_chunks import get_chunk, get_blocks

import pycuda.autoinit, pycuda.driver, pycuda.gpuarray, pycuda.compiler

class BSLZ4CUDA:

    modsrc = " ".join( [ open(srcfile, 'r').read() for srcfile in
        ( "nvcomp_extract.cu", "h5lz4dc.cu", "shuffles.cu" ) ] )

    def __init__(self, total_output_bytes, bpp, blocksize ):
        """ cache / reuse memory """
        try:
            self.mod = pycuda.compiler.SourceModule( self.modsrc )
        except:
            open('kernel.cu', 'w').write(self.modsrc)
            raise
        self.h5lz4dc   = self.mod.get_function("h5lz4dc")
        self.shuf_end = self.mod.get_function("simple_shuffle_end")
        self.copybytes = self.mod.get_function("copybytes")

        self.reset(  total_output_bytes, bpp, blocksize )

    def reset(self, total_output_bytes, bpp, blocksize):

        self.total_output_bytes = total_output_bytes
        self.bpp = bpp
        self.blocksize = np.uint32( blocksize )
        self.shuf_8192 = self.mod.get_function( "shuf_8192_%d"%(bpp*8) )

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
        self.output_d = pycuda.gpuarray.empty( self.total_output_bytes, dtype = np.uint8 )
        self.blocks_d = pycuda.gpuarray.empty( self.tblocks, np.uint32 )      # 32 bit offsets into the compressed for blocks
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
                self.shuf_d = pycuda.gpuarray.empty( self.total_output_bytes, dtype = np.uint8 )
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
                    self.shuf_d = pycuda.gpuarray.empty( self.total_output_bytes, dtype = np.uint8 )
                outd = self.shuf_d

        chunk_d = pycuda.driver.mem_alloc( chunk.nbytes )        # the compressed data
        pycuda.driver.memcpy_htod( chunk_d, chunk )         # compressed data on the device
        self.blocks_d[:] = blocks[:]      # 32 bit offsets into the compressed for blocks
        self.h5lz4dc( chunk_d,
                      self.blocks_d,
                      np.uint32( self.tblocks ),
                      self.blocksize,
                      self.output_d,
                      block=self.lz4block, grid=self.lz4grid )
        self.shuf_8192( self.output_d, outd, block = self.shblock, grid = self.shgrid )
        pos = self.nblocks*self.blocksize
        todo = self.total_output_bytes - pos
        if todo > self.bytes_to_copy:
            self.shuf_end( self.output_d,
                           outd,
                           self.blocksize,
                           np.uint32(todo),
                           np.uint32(self.bpp),
                           np.uint32(pos),
                           block = (32,1,1), grid = (int((todo+31)//32),1,1) )
        if hasattr( outarg, "__cuda_array_interface__"):
            if self.bytes_to_copy > 0:
                self.copybytes(  chunk_d, np.uint32( chunk.nbytes - self.bytes_to_copy ),
                                 outd, np.uint32( self.total_output_bytes - self.bytes_to_copy),
                                 block = ( self.bytes_to_copy, 1, 1 ), grid = (1,1,1) )
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
    chunk, shape, dtyp  = get_chunk( hname, dset, frm )
    total_output_elem  = shape[1]*shape[2]
    total_output_bytes = total_output_elem*dtyp.itemsize
    blocksize, blocks = get_blocks( chunk, shape, dtyp )

    output = np.empty( total_output_elem, dtyp )
    decompressor = BSLZ4CUDA( total_output_bytes, dtyp.itemsize, blocksize )
    decomp = decompressor( chunk, blocks, outarg=output ).view( dtyp ).reshape( (shape[1], shape[2]) )

#    outputd = pycuda.gpuarray.empty( total_output_elem, dtyp )
#    decomp2 = decompressor( chunk, blocks, outarg=outputd ).get().view( dtyp ).reshape( (shape[1], shape[2]) )




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

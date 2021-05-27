

import subprocess, tempfile, os, time, sys

# C-code generic header things and function writers ... 

FUNCS = {}

INCLUDES = """
/* A curated collection of different BSLZ4 readers
   This is automatically generated code
   Edit this to change the original : 
     %s
   Created on : 
     %s
   Code generator written by Jon Wright.
*/

#include <stdlib.h>   /* malloc and friends */
#include <stdint.h>   /* uint32_t etc */
#include <string.h>   /* memcpy */
#include <stdio.h>    /* print error message before killing process(!?!?) */
#include <lz4.h>      /* assumes you have this already */
#include <ippdc.h>    /* for intel ... going to need a platform build system */
#include <hdf5.h>     /* to grab chunks independently of h5py api (py27 issue) */ 

"""% ( __file__, time.ctime())

MACROS = """
/* see https://justine.lol/endian.html */
#define READ32BE(p) \\
  ( (uint32_t)(255 & (p)[0]) << 24 |\\
    (uint32_t)(255 & (p)[1]) << 16 |\\
    (uint32_t)(255 & (p)[2]) <<  8 |\\
    (uint32_t)(255 & (p)[3])       )
#define READ64BE(p) \\
  ( (uint64_t)(255 & (p)[0]) << 56 |\\
    (uint64_t)(255 & (p)[1]) << 48 |\\
    (uint64_t)(255 & (p)[2]) << 40 |\\
    (uint64_t)(255 & (p)[3]) << 32 |\\
    (uint64_t)(255 & (p)[4]) << 24 |\\
    (uint64_t)(255 & (p)[5]) << 16 |\\
    (uint64_t)(255 & (p)[6]) <<  8 |\\
    (uint64_t)(255 & (p)[7])       )
    
#define ERR(s) \\
  { fprintf( stderr, \"ERROR %s\\n\", s); \\
    return -1; }

#define CHECK_RETURN_VALS 1
"""

class compiler:
    def compile(self, cname, oname):
        ret = subprocess.run( " ".join([self.CC,] + self.CFLAGS + 
                             ["-c", cname,"-o", oname]),
                                  shell = True, capture_output=True )
        return ret

UFL = "-Wmissing-prototypes -Wshadow -Wconversion"
UFL = "-std=c99 -fopenmp -Wall -Wall -Wstrict-prototypes -Wshadow -Wmissing-prototypes -Wconversion".split()

class GCC(compiler):
    CC = 'gcc'
    CFLAGS = UFL + [ '-O2', '-fsanitize=undefined']

class CLANG(compiler):
    CC = 'clang'
    CFLAGS = UFL + [ '-O2' ]

class ICC(compiler):
    # module load Intel/2020 at ESRF
    CC = 'icc'
    CFLAGS = ['-qopenmp', '-O2', '-Wall' ]

class cfrag:
    """ A fragment of C code to insert. May have a defer to 
    put at the end of a logical block """
    def __init__(self, fragment, defer=None):
        self.fragment = fragment
        self.defer = defer

TMAP = {
    'char': 'integer(kind=1)',
    'int16_t': 'integer(kind=2)',
    'int32_t': 'integer(kind=4)',
    'int64_t': 'integer(kind=8)',
    'uint16_t': 'integer(kind=-2)',
    'uint32_t': 'integer(kind=-4)',
    'uint64_t': 'integer(kind=-8)',
    'int': 'integer', # doubtful 
    'size_t': 'integer', # surely wrong
}

class cfunc:
    """ A C function, name, arguments, body """
    def __init__(self, name, args, body):
        self.name = name
        self.args = args
        self.body = body
    def signature(self):
        types = [" ".join( a.split()[:-1]) for a in self.args ]
        return "%s ( %s );\n" % ( self.name, ", ".join( types ) )
    def definition(self):
        if self.name.startswith("void "):
            return "%s ( %s ){\n%s\n}\n" % ( self.name, ", ".join( self.args ), self.body )
        if self.name.startswith("int "):      # returning an error status
            return "%s ( %s ){\n%s\n    return 0;\n}\n" % (
                 self.name, ", ".join( self.args ), self.body )
        if self.name.startswith("size_t "):   # returning a memory size
            return "%s ( %s ){\n%s\n}" % (
                 self.name, ", ".join( self.args ), self.body )
        raise Exception("add a return type case!!!")
    def pyf(self):
        anames = []
        tnames = []
        atypes = []
        for a in self.args:
            tokens = a.split( )
            anames.append( tokens[-1] )
        fname = self.name.split()[-1]
        sub = self.name.startswith("void")
        lines = ['\n']
        if sub:
            lines.append( "subroutine %s(%s)"%(fname, ",".join(anames)) )
        else:
            lines.append( "function %s(%s)"%(fname, ",".join(anames)) )
        lines.append( 'intent(c) %s'%(fname) )
        lines.append( 'intent(c)' )
        for argu in self.args:
#            lines.append('! %s'%(argu))
            tokens = argu.split( )
            a = tokens[-1]
            m = " ".join([ t for t in tokens[:-1] if t not in ('const','*')])
            t = " ".join(tokens[:-1])
            if t.find("*")>0:
                if t.find("const")>=0:
                    intent = 'intent(in)'
                else:
                    intent = 'intent(inout)'
                dim = "dimension( %s_length)"%(a)        
                decl = ' , '.join( (TMAP[m], intent, dim ))
                decl += ":: %s"%( a )
            elif a.endswith('_length'):
                decl = TMAP[m] + ' , intent( hide ), depend( %s ) :: %s'%(
                   a.replace('_length',''), a )
            else:
                decl = TMAP[m] + ' :: %s'%(a)
            lines.append( decl  )
        if sub:
            lines.append( "end subroutine %s%(fname)" )
        else:
            ftype = " ".join( self.name.split()[:-1] )
            lines.append( TMAP[ftype] + " :: " + fname )
            lines.append( "end function %s"%(fname) )
        return lines +["\n"]
                
    def testcompile(self, cc=GCC() ):
        with tempfile.NamedTemporaryFile( mode = 'w', suffix = '.c', dir=os.getcwd() ) as tmp:
            tmp.write( INCLUDES )
            tmp.write( MACROS )
            tmp.write( self.signature() )
            tmp.write( self.definition() )
            tmp.flush()
            ofile = tmp.name[:-1] + "o"
            ret = cc.compile( tmp.name, ofile )
        if os.path.exists( ofile ):
            os.remove( ofile )
            print("Compilation OK",cc.CC)
            if len(ret.stderr):
                print(ret.stderr.decode())
        else:
            print("Compilation failed")
            print(ret.stderr.decode())

class cfragments:
    def __init__(self, **kwds):
        self.__dict__.update( kwds )
    def __add__(self, other):
        return cfragments( **{**self.__dict__, **other.__dict__} )
    def __call__(self, *names):
        lines = []
        for name in names:
            lines.append("/* begin: %s */"%(name))
            lines.append( self.__dict__[name].fragment )
            lines.append("/* ends: %s */"%(name))
        for name in names:
            if self.__dict__[name].defer is not None:
                lines.append("/* begin: %s */"%(name))
                lines.append( self.__dict__[name].defer )
                lines.append("/* end: %s */"%(name))
        return "\n".join(lines)

# To interpret a chunk coming from hdf5. Includes the 12 byte header.
chunk_args = [  "const char * compressed",
                "size_t compressed_length",
                "int itemsize" ]
# To write output data
output_args = [ "char * output",
                "size_t output_length" ]
# Same as for chunks, but includes an array of blocks.
# These point to the 4 byte BE int which prefix each lz4 block.
blocklist_args = chunk_args +  [ "int blocksize", "uint32_t * blocks", "int blocks_length" ]

chunkdecoder = cfragments(
# To convert an input to get the block start positions
chunks_2_blocks = cfrag( """
   size_t total_output_length;
   total_output_length = READ64BE( compressed );
   int blocksize;
   blocksize = (int) READ32BE( (compressed+8) );
   if (blocksize == 0) { blocksize = 8192; }
""" ),
# To compute the number of blocks : you cannot now pass this in
blocks_length = cfrag( """
   int blocks_length;
   blocks_length = (int)( (total_output_length + (size_t) blocksize - 1) / (size_t) blocksize );
""" ),
create_starts = cfrag( """
   uint32_t  * blocks;
   blocks = (uint32_t *) malloc( ((size_t) blocks_length) * sizeof( uint32_t ) );
   if (blocks == NULL) {
       ERR("small malloc failed");
   }
""" , "   free( blocks );" ),
read_starts = cfrag( """
   blocks[0] = 12;
   for( int i = 1; i < blocks_length ; i++ ){
       int nbytes = (int) READ32BE( ( compressed + blocks[i-1] ) );
       blocks[i] = (uint32_t)(nbytes + 4) + blocks[i-1];
       if ( blocks[i] >= compressed_length ){
           ERR("Overflow reading starts");
       }

   }
""" ),
print_starts = cfrag( """
   printf("total_output_length %ld\\n", total_output_length);
   printf("blocks_length %d\\n", blocks_length);
   for( int i = 0; i < blocks_length ; i++ )
       printf("%d %d, ", i, blocks[i]);
""" ),
) #### end of chunkdecoder


FUNCS['print_offsets_func'] = cfunc( "int print_offsets", chunk_args, 
                    chunkdecoder( "chunks_2_blocks", "blocks_length" , "create_starts",
                            "read_starts" , "print_starts" ) )

FUNCS['read_starts_func'] = cfunc( "int read_starts",
                            blocklist_args,
                            chunkdecoder( "read_starts" ))

#    LZ4LIB_API int LZ4_decompress_safe (const char* src, char* dst, int compressedSize, int dstCapacity);
lz4decoders = cfragments(
    # requires the blocks to be first created
    omp_lz4 = cfrag("""
    int error=0;
#pragma omp parallel for shared(error)
    for( int i = 0; i < blocks_length-1; i++ ){
        int ret = LZ4_decompress_safe(  compressed + blocks[i] + 4u,
                                           output + i * blocksize,
                                           (int) READ32BE( compressed + blocks[i] ),
                                           blocksize );
        if ( CHECK_RETURN_VALS && (ret != blocksize)) error = 1;
    }
    if (error) ERR("Error decoding LZ4");
    /* last block, might not be full blocksize */
    {
      int lastblock = (int) output_length - blocksize * (blocks_length - 1);
      /* last few bytes are copied flat */
      int copied = lastblock % ( 8 * itemsize );
      lastblock -= copied;
      memcpy( &output[ output_length - (size_t) copied ], 
              &compressed[ compressed_length - (size_t) copied ], (size_t) copied );
      int nbytes = (int) READ32BE( compressed + blocks[blocks_length - 1]);
      int ret = LZ4_decompress_safe( compressed + blocks[blocks_length-1] + 4u,
                                     output + (blocks_length-1) * blocksize,
                                     nbytes,
                                     lastblock );
      if ( CHECK_RETURN_VALS && ( ret != lastblock ) ) ERR("Error decoding last LZ4 block");
    }
    """),
    # without using the blocks
    onecore_lz4 = cfrag("""
    int p = 12;
    for( int i = 0; i < blocks_length - 1 ; ++i ){
       int nbytes = (int) READ32BE( &compressed[p] );
       int ret = LZ4_decompress_safe( &compressed[p + 4],
                                      &output[i * blocksize],
                                      nbytes,
                                      blocksize );
      if ( CHECK_RETURN_VALS && ( ret != blocksize ) ) ERR("Error LZ4 block");
      p = p + nbytes + 4;
    }
    /* last block, might not be full blocksize */
    {
      int lastblock = (int) output_length - blocksize * (blocks_length - 1);
      /* last few bytes are copied flat */
      int copied = lastblock % ( 8 * itemsize );
      lastblock -= copied;
      memcpy( &output[ output_length - (size_t) copied ], 
              &compressed[ compressed_length - (size_t) copied ], (size_t) copied );

      int nbytes = (int) READ32BE( &compressed[p] );
      int ret = LZ4_decompress_safe( &compressed[p + 4],
                                     &output[(blocks_length-1) * blocksize],
                                     nbytes,
                                     lastblock );
      if ( CHECK_RETURN_VALS && ( ret != lastblock ) ) ERR("Error decoding last LZ4 block");
    }
   """),
    # todo omp task based ?
    onecore_ipp = cfrag("""
   int p = 12;
    for( int i = 0; i < blocks_length - 1 ; ++i ){
      int nbytes = (int) READ32BE( &compressed[p] );
      int bsize = blocksize;
      IppStatus ret = ippsDecodeLZ4_8u( &compressed[p + 4], nbytes,
                                         &output[i * blocksize], &bsize );
      if ( CHECK_RETURN_VALS && ( ret != ippStsNoErr ) ) ERR("Error LZ4 block");
      p = p + nbytes + 4;
    }
    /* last block, might not be full blocksize */
    {
      int lastblock = (int) output_length - blocksize * (blocks_length - 1);
      /* last few bytes are copied flat */
      int copied = lastblock % ( 8 * itemsize );
      lastblock -= copied;
      memcpy( &output[ output_length - (size_t) copied ], 
              &compressed[ compressed_length - (size_t) copied ], (size_t) copied );
      int nbytes = (int) READ32BE( &compressed[p] );
      int bsize = lastblock;
      IppStatus ret = ippsDecodeLZ4_8u( &compressed[p + 4], nbytes,
                                         &output[(blocks_length-1)* blocksize], &bsize );
      if ( CHECK_RETURN_VALS && ( ret != ippStsNoErr ) ) ERR("Error LZ4 block");
    }
    """),
)


FUNCS['onecore_lz4_func'] = cfunc( "int onecore_lz4", chunk_args + output_args, 
             (chunkdecoder+lz4decoders)(
                 "chunks_2_blocks", "blocks_length", "onecore_lz4") )

FUNCS['omp_lz4_make_starts_func'] = cfunc( "int omp_lz4", chunk_args + output_args, 
             (chunkdecoder+lz4decoders)( "chunks_2_blocks", "blocks_length" , 
             "create_starts", "read_starts" , "omp_lz4" ) ) 

FUNCS['omp_lz4_with_starts_func'] = cfunc( "int omp_lz4_blocks", blocklist_args + output_args, 
             lz4decoders( "omp_lz4" ) ) 

FUNCS['onecore_ipp_func'] = cfunc("int onecore_ipp", chunk_args + output_args,
             (chunkdecoder+lz4decoders)(
                 "chunks_2_blocks", "blocks_length", "onecore_ipp") )                                  

FUNCS['h5_read_direct'] = cfunc("size_t h5_read_direct", # name, args, body
                                # hid_t is int64_t (today, somwhere)
                                # hsize_t in unsigned long long == uint64_t (today, somewhere)
               ["int64_t dataset_id", "int frame", "char * chunk", "size_t chunk_length" ],
                                """
{
/* see: 
   https://support.hdfgroup.org/HDF5/doc/HL/RM_HDF5Optimized.html#H5DOread_chunk 

   ... assuming this is h5py.dataset.id.id :
    hid_t dataset;
    if((dataset = H5Dopen2(hname, dsetname, H5P_DEFAULT)) < 0)
        ERR("Failed to open h5file");
*/
    hsize_t offset[3];
    offset[0] = frame;  /* assumes 3D frame-by-frame chunks */
    offset[1] = 0;
    offset[2] = 0;

    /* Get the size of the compressed chunk to return */
    hsize_t chunk_nbytes;
    herr_t ret;
    ret = H5Dget_chunk_storage_size(dataset_id, offset, &chunk_nbytes);
    if ( chunk_nbytes > chunk_length )  ERR("Chunk does not fit into your arg");
    if ( ret < 0 ) ERR("Problem getting storage for the chunk");
    /* Use H5DOread_chunk() to read the chunk back 
       ... becomes H5Dread_chunk in later library versions */
    uint32_t read_filter_mask;
    ret = H5Dread_chunk(dataset_id, H5P_DEFAULT, offset, &read_filter_mask, chunk);
    if ( ret < 0 ) ERR("error reading chunk");
    if ( read_filter_mask != 0 ) ERR("chunk was filtered"); 
    return chunk_nbytes; 
}
""")

def write_funcs(fname, funcs):
    ks = sorted(funcs.keys())
    with open(fname, 'w') as cfile:
        cfile.write(INCLUDES)
        cfile.write(MACROS)
        for k in ks:
            print(k)
            cfile.write("/* Signature for %s */\n"%(k))
            cfile.write( funcs[k].signature() )
        for k in ks:
            print(k)
            cfile.write("/* Definition for %s */\n"%(k))
            cfile.write( funcs[k].definition() )
    os.system('clang-format -i %s'%(fname))

def write_pyf(modname, fname, funcs):
    ks = sorted(funcs.keys())
    with open(modname+'.pyf','w') as pyf:
        pyf.write("""
python module %s

interface
        """%(modname))
        for k in ks:
            pyf.write("\n".join(funcs[k].pyf()))
        pyf.write("end interface\n")
        pyf.write("end module %s\n"%(modname))


def test_funcs_compile():
    for func in FUNCS.keys():
        print(func)
        for compiler in GCC(), CLANG():
            FUNCS[func].testcompile(cc=compiler)

if __name__=="__main__":
    cfile = sys.argv[1] + ".c"
    f2pyfile = sys.argv[1]
    write_pyf(f2pyfile , cfile, FUNCS )
    write_funcs( cfile, FUNCS )
    test_funcs_compile()
    for compiler in GCC(), CLANG():
        print(compiler.CC, compiler.CFLAGS)
        ret = compiler.compile( cfile, "/dev/null" )
        print(ret.stdout.decode())
        print(ret.stderr.decode())

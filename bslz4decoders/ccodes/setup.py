

"""
Setup script
Re-builds the C code via
   python codegen.py
"""
if 0:
    import codegen
    codegen.main()

# import setuptools
import os, sys, platform, os.path

from distutils.core import setup, Extension
from distutils.command import build_ext
import numpy, numpy.f2py



f2pypath = os.path.split( numpy.f2py.__file__)[0]
fortraninc = os.path.join( f2pypath, 'src' )
fortranobj = os.path.join( fortraninc, 'fortranobject.c' )

bsinc = os.path.join( os.path.dirname(__file__), "bitshuffle_extract" )
bsobj = os.path.join( bsinc, "bitshuffle_core.c" )

assert os.path.exists( fortranobj )

copt =  {
    'msvc': ['/openmp', '/O2'] ,
    'unix': ['-fopenmp', '-O2', '-DF2PY_REPORT_ON_ARRAY_COPY' ] ,
    'mingw32': ['-fopenmp', '-O2'] ,
 }

lopt =  { k : [a for a in l] for k,l in copt.items() }
lopt['msvc'] = []

# might try:
# set CFLAGS=/arch:AVX2 for msvc
# CFLAGS=-march=native -mtune=native
# LDFLAGS=-march=native -mtune=native

# CFLAGS='-march=native' LDFLAGS='-march=native' \
#    python setup.py  build_ext --force --inplace \
#    -IPP=/home/esrf/wright/intel/oneapi/ipp/latest/lib/intel64


#  LDSHARED="icc -shared -ipp -ipp-link=static -xHost"                   \
#   CFLAGS="-DUSEIPP -ipp -ipp-link=static -O3 -std=c99 -fopenmp -xHost" \
#    CC=icc python setup.py build_ext --inplace --force
#  ... ->




class build_ext_subclass( build_ext.build_ext ):
    def build_extensions(self):
        """ attempt to defer the numpy import until later """
        c = self.compiler.compiler_type
        CF = [] ; LF=[]
        if "CFLAGS" in os.environ:
            CF = os.environ.get("CFLAGS").split(" ")
        if "LDFLAGS" in os.environ:
            LF = os.environ.get("LDFLAGS").split(" ")
        for e in self.extensions:
            if c in copt:
               e.extra_compile_args = copt[ c ] + CF
               e.extra_link_args = lopt[ c ] + LF
        print("Customised compiler",c,e.extra_compile_args,
                    e.extra_link_args)
        build_ext.build_ext.build_extensions(self)


def compile_paths( places ):
    incdirs = [ numpy.get_include(), fortraninc, bsinc ]
    libdirs = [ ]
    for place in places:
        for root in [os.path.join( place, "Library" ), place ]:
            i = os.path.join( root, "include" )
            l = os.path.join( root, "lib")
            if os.path.exists( i ) and os.path.exists( l ):
                incdirs.append( i )
                libdirs.append( l )
    return incdirs, libdirs


places = [ os.environ[var] for var in ("IPPROOT", "CONDA_PREFIX") if var in os.environ ]

if os.path.exists( "/nobackup/scratch/HDF5/HDF5-1.10.5" ): # scisoft15
    places.append( "/nobackup/scratch/HDF5/HDF5-1.10.5" )

incdirs, libdirs = compile_paths( places )

print(incdirs)
print(libdirs)



if platform.system() == 'Windows':
    LZ4 = "liblz4"
else:
    LZ4 = "lz4"

    
ext_modules = [ Extension( "h5chunk",
                           sources = ["h5chunk.c",
                                      "h5chunkmodule.c",
                                      fortranobj],
                           include_dirs  = incdirs,
                           libraries = ['hdf5'],
                           library_dirs  = libdirs ),
                Extension( "ompdecoders",
                           sources = ["ompdecoders.c",
                                      "ompdecodersmodule.c",
                                      fortranobj,
                                      bsobj],
                           include_dirs  = incdirs,
                           libraries = [LZ4],
                           library_dirs  = libdirs ),
                Extension( "decoders",
                           sources = ["decoders.c",
                                      "decodersmodule.c",
                                      fortranobj,
                                      bsobj],
                           include_dirs  = incdirs,
                           libraries = [LZ4],
                           library_dirs  = libdirs ),
]




ippdc = ( 'libippdc.a', 'libippcore.a' )
for arg in sys.argv:
    if arg.startswith('-IPP'):
        if arg.find("=")>=0:
            ipproot = arg.split("=")[1]
            ippdc = [ os.path.join( ipproot, a ) for a in ippdc ]
        else:
            ippdc = [] # intel compiler probably
        break
else:
    ippdc = None


if ippdc is not None: 
    ipp_modules = [
        Extension( "ippdecoders",
                   sources = ["decoders.c", "ippdecodersmodule.c", fortranobj, bsobj],
                   define_macros = [('USEIPP', '1')],
                   include_dirs  = incdirs,
                   extra_objects = ippdc,
                   library_dirs  = libdirs ),
        Extension( "ippompdecoders",
                   sources = ["ompdecoders.c", "ippompdecodersmodule.c", fortranobj, bsobj],
                   define_macros = [('USEIPP', '1')],
                   include_dirs  = incdirs,
                   extra_objects = ippdc,
                   library_dirs  = libdirs )
    ]
    ext_modules += ipp_modules


    
setup( name = "ccodes" ,
       ext_modules = ext_modules,
       cmdclass = { 'build_ext' : build_ext_subclass },
)

if ippdc is None and platform.machine() == 'x86_64':
    print("Intel IPP was not used")
    print("Add -IPP=/location/of/intel/oneapi/ipp/latest/lib/intel64 to your setup.py command" )



"""
Setup script
Re-builds the C code via
   python codegen.py
"""
import codegen
codegen.main()

# For pip / bdist_wheel etc
import setuptools
import os, sys, platform, os.path


from distutils.core import setup, Extension
from distutils.command import build_ext

import numpy, numpy.f2py



f2pypath = os.path.split( numpy.f2py.__file__)[0]
fortranobj = os.path.join( f2pypath, 'src', 'fortranobject.c' )
fortraninc = os.path.join( f2pypath, 'src' )

assert os.path.exists( fortranobj )

copt =  {
    'msvc': ['/openmp', '/O2'] ,
    'unix': ['-fopenmp', '-O2'], #, '-DF2PY_REPORT_ON_ARRAY_COPY=100'] ,
    'mingw32': ['-fopenmp', '-O2'] ,
 }
lopt =  { k : [a for a in l] for k,l in copt.items() }
lopt['msvc'] = []

# might try:
# set CFLAGS=/arch:AVX2 for msvc
# CFLAGS=-march=native -mtune=native
# LDFLAGS=-march=native -mtune=native

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

if "CONDA_PREFIX" in os.environ:
    incdirs = [ numpy.get_include(),
        fortraninc,
        os.path.join( os.environ['CONDA_PREFIX'], "Library", "include" ), ]
    libdirs = [os.path.join( os.environ['CONDA_PREFIX'], "Library", "lib" ),]

LZ4 = "liblz4"

ext_modules = [ Extension( "h5chunk",
                      sources = ["h5chunk.c", "h5chunkmodule.c", fortranobj],
                      include_dirs  = incdirs,
#                     define_macros = [('MAJOR_VERSION', '1')],
                      libraries = ['hdf5'],
                      library_dirs  = libdirs ),
               Extension( "ompdecoders",
                      sources = ["ompdecoders.c", "ompdecodersmodule.c", fortranobj],
                      include_dirs  = incdirs,
                      libraries = [LZ4],
                      library_dirs  = libdirs ),

               Extension( "decoders",
                      sources = ["decoders.c", "decodersmodule.c", fortranobj],
                      include_dirs  = incdirs,
                      libraries = [LZ4],
                      library_dirs  = libdirs ),
                  ]

setup( name = "ccodes" , ext_modules = ext_modules )



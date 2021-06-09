
"""
A bunch of things to copy paste together fragments of C-code

This is like a literate style but without the readability.
"""

import os, numpy.f2py


class compiler:
    """ to test if things can compile """
    def compile(self, cname, oname):
        ret = subprocess.run( " ".join([self.CC,] + self.CFLAGS +
                             ["-c", cname,"-o", oname]),
                                  shell = True, capture_output=True )
        return ret

# universal compiler flags:
UFL = "-Wmissing-prototypes -Wshadow -Wconversion"
UFL = "-std=c99 -fopenmp -Wall -Wall -Wstrict-prototypes -Wshadow -Wmissing-prototypes -Wconversion".split()

if "CONDA_PREFIX" in os.environ:
    UFL.append( "-I%s/include"%(os.environ['CONDA_PREFIX']) )
else:
    for p in ("/usr/include/hdf5/serial",):
        if os.path.exists( os.path.join(p, 'hdf5.h') ):
            UFL.append( "-I%s"%(p) )
            break

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

# Type maps for f2py wrappers
TMAP = {
    'char': 'integer(kind=1)',
    'int16_t': 'integer(kind=2)',
    'int32_t': 'integer(kind=4)',
    'int64_t': 'integer(kind=8)',
    'uint16_t': 'integer(kind=-2)',
    'uint32_t': 'integer(kind=-4)',
    'uint64_t': 'integer(kind=-8)',
    'int': 'integer(kind=4)', # doubtful
    'size_t': 'integer(kind=8)', # surely wrong
}

class cfunc:
    """ A C function, name, arguments, body """
    def __init__(self, name, args, body):
        self.name = name
        self.args = args
        self.body = body
    def signature(self):
        """ for a header"""
        types = [" ".join( a.split()[:-1]) for a in self.args ]
        return "%s ( %s );\n" % ( self.name, ", ".join( types ) )
    def definition(self):
        """ for the main source"""
        if self.name.startswith("void "):
            return "%s ( %s ){\n%s\n}\n" % ( self.name, ", ".join( self.args ), self.body )
        if self.name.startswith("int "):      # returning an error status
            return "%s ( %s ){\n%s\n    return 0;\n}\n" % (
                 self.name, ", ".join( self.args ), self.body )
        if self.name.startswith("size_t "):   # returning a memory size
            return "%s ( %s ){\n%s\n}" % (
                 self.name, ", ".join( self.args ), self.body )
        raise Exception("add a return type case!!! : "+  str(self.name) )
    def pyf(self):
        """ generate a pyf file to make a wrapper """
        anames = []
        tnames = []
        atypes = []
        for a in self.args:
            tokens = a.split( )
            anames.append( tokens[-1] )
        fname = self.name.split()[-1]
        sub = self.name.startswith("void") # no return value
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
                # This is disgusting. Strings have "name" in the variable name.
                # ... how did we ever end up in this mess. Could we go for
                # pointer "*" versus "[]" or put "ndarray" in the names instead?
                if a.find("name") < 0:
                    dim = "dimension( %s_length)"%(a)
                    decl = ' , '.join( (TMAP[m], intent, dim ))
                    decl += ":: %s"%( a )
                else:
                    decl = TMAP[m] + " :: %s"%( a )
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
        """ See if the thing can be compiled """
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
    """ a series of fragments """
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


CLANG_FORMAT = "clang-format -i "
if os.path.exists(r"C:/program files/LLVM/bin/clang-format.exe"):
    CLANG_FORMAT = r'"C:/program files/LLVM/bin/clang-format.exe" -i '


def write_funcs(fname, funcs, includes, macros):
    """ Write a series of functions into a file """
    ks = sorted(funcs.keys())
    with open(fname, 'w') as cfile:
        cfile.write(includes)
        cfile.write(macros)
        for k in ks:
            print(k)
            cfile.write("/* Signature for %s */\n"%(k))
            cfile.write( funcs[k].signature() )
        for k in ks:
            print(k)
            cfile.write("/* Definition for %s */\n"%(k))
            cfile.write( funcs[k].definition() )
    os.system(CLANG_FORMAT + fname)



def write_pyf(modname, fname, funcs):
    """ Write a pyf file """
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
    numpy.f2py.run_main( [modname+".pyf"] )


def test_funcs_compile(funcs):
    for func in funcs.keys():
        print(func)
        for compiler in GCC(), CLANG():
            funcs[func].testcompile(cc=compiler)


import sys
import timeit, sys
from bslz4decoders import read_chunks


def bench( func, *args ):
    start = timeit.default_timer()
    func(*args)
    end = timeit.default_timer()
    print("%.6f /s"%(end-start), func.__name__, args)

def benchiter( func, *args ):
    start = timeit.default_timer()
    for frm in func(*args):
        pass
    end = timeit.default_timer()
    print("%.6f /s"%(end-start), func.__name__, args)


def testbench( testcases = None ):
    if testcases is None:
        from bslz4decoders.test.testcases import testcases
    for hname, d in testcases:
        print()
        cnf, buf = read_chunks.get_chunk( hname, d, 0 )
        bench( cnf.get_blocks, buf )
        bench( read_chunks.get_chunk, hname, d, 0 )
        bench( read_chunks.get_frame_h5py, hname, d, 0 )
        benchiter( read_chunks.iter_chunks, hname, d )
        benchiter( read_chunks.iter_frames_h5py, hname, d )
        benchiter( read_chunks.get_chunks, hname, d )
        benchiter( read_chunks.get_frames_h5py, hname, d )

if __name__=="__main__":
    if len(sys.argv) == 3:
        testcases = [ (sys.argv[1], sys.argv[2]), ]
    testbench( testcases )

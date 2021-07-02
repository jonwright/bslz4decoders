
import timeit, sys
from bslz4decoders import read_chunks


def bench_parsechunks( h5name, dset ):
    n = 0
    t = [timeit.default_timer(),]
    for config, chunk in read_chunks.iter_chunks( h5name, dset ):
        blocks = config.get_blocks( chunk )
    t += [timeit.default_timer(),]
    for config, chunk in read_chunks.iter_chunks( h5name, dset ):
        n += 1
    t += [timeit.default_timer(),]
    for config, chunk in read_chunks.iter_chunks( h5name, dset ):
        blocks = config.get_blocks( chunk )
    t += [timeit.default_timer(),]
    print( h5name, dset, n )
    print("First read + decode %.2f ms %.2f us/frm"%(1e3*(t[1]-t[0]),1e6*(t[1]-t[0])/n))
    print("Second read  only   %.2f ms %.2f us/frm"%(1e3*(t[2]-t[1]),1e6*(t[2]-t[1])/n))
    print("Third read + decode %.2f ms %.2f us/frm"%(1e3*(t[3]-t[2]),1e6*(t[3]-t[2])/n))

if __name__=="__main__":
 
    if len(sys.argv) == 1:
        import testcases
        cases = testcases.testcases
    else:
        hname = sys.argv[1]
        cases = [ (hname, d) for d in sys.argv[2:] ]
    for h5name, dset in cases:
        bench_parsechunks( h5name, dset )




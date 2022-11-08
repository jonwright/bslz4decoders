
import os
os.environ['OMP_NUM_THREADS']='1'
import multiprocessing
import h5py, hdf5plugin
import sys, os
from timeit import default_timer


def sum_reduce( args ):
    h5name, dset, jobid, njobs = args
    results = {}
    with h5py.File( open(h5name, 'rb'), 'r') as hin:
        frames = hin[dset]
        for i in range( jobid, frames.shape[0], njobs  ):
            results[i] = frames[i].sum(dtype=int)
    return results
    
    
def main( h5name, dset, njobs ):
    args = [ (h5name, dset, i, njobs) for i in range(njobs) ]
    with multiprocessing.Pool( njobs ) as workers:
        print('Using',njobs,'workers')
        sums = {}
        for results in workers.imap_unordered( sum_reduce, args ):
            sums.update( results )
    return sums


if __name__=="__main__":
    
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        njobs = int( os.getenv('SLURM_CPUS_PER_TASK') )
    else:
        njobs = os.cpu_count()
        
    import sum_testcases
    
    i = 1
    while i < njobs:
        func = lambda hname, dset: main( hname, dset, i )
        t0 = default_timer()
        d1 = sum_testcases.run_sum_testcases( func )
        t1 = default_timer()
        i = i*2
    d1 = sum_testcases.run_sum_testcases( lambda hname, dset: main( hname, dset, njobs ) )



import queue
import threading
from bslz4decoders import decoders, read_chunks
from bslz4decoders.test.testcases import testcases as TESTCASES


results = []

def dc( qin ):
    while 1:
        args = qin.get()
        if args is None:
            break
        config, buffer = args # frame num?
        frm = decoders.decompress_omp( buffer, config )
        results.append( frm[10,20] )
    
    
def test_q(h5name, dset):
    global results
    print('with a q') 
    results = []
    cq = queue.Queue(maxsize=8)
    dct = threading.Thread( target=dc, args=( cq, ) )
    dct.start()
    read_chunks.queue_chunks( cq, h5name, dset )
    dct.join()
    print(results)

def test_q2(h5name, dset):
    global results
    print('no threads')
    results = []
    print(h5name, dset)
    for config, buffer in read_chunks.iter_chunks( h5name, dset):
        frm = decoders.decompress_omp( buffer, config )
        results.append( frm[10,20] )
    print(results)

    
def test_q3(h5name, dset):
    global results
    results = []
    import concurrent.futures, collections
    with concurrent.futures.ThreadPoolExecutor( 1 ) as p:
        CAP = 10
        q = collections.deque( maxlen=CAP )
        for config, buffer in read_chunks.iter_chunks( h5name, dset):
            if len(q) == CAP or (len(q) and q[0].done()):
                frm = q.popleft().result()
                results.append( frm[10,20] )
            q.append( p.submit( decoders.decompress_omp, buffer, config)) 
        while len(q):
            frm = q.popleft().result()  
            results.append( frm[10,20] )
    print(results)
           
                               
if __name__=="__main__":
    import timeit
    h5name, dset = TESTCASES[-1]
    t0 = timeit.default_timer()
    test_q(h5name, dset)
    t1 = timeit.default_timer()
    test_q2(h5name, dset)
    t2 = timeit.default_timer()
    test_q3(h5name, dset)
    t3 = timeit.default_timer()
    print('threaded',t1-t0,'not',t2-t1,'cthreaded',t3-t2)
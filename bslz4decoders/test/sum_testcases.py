

import numpy as np, h5py, hdf5plugin
import timeit
import os

cases =  [ ("/data/id11/nanoscope/blc12454/id11/WAu5um/WAu5um_DT3/scan0001/eiger_0000.h5", 
            "/entry_0000/ESRF-ID11/eiger/data"),
           ("/data/id11/jon/hdftest/kevlar.h5", "/entry/data/data" ),
           ("hplc.h5","/entry_0000/measurement/data"),
         ]


def getsums( h5name, dset ):
    
    if not os.path.exists(h5name):
        p,h = os.path.split(h5name)
        if os.path.exists(h):
            h5name = h
        else:
            print("Missing",h5name)
        
    with h5py.File( h5name, 'r') as hin:
        frames = hin[dset]
        sums = np.empty( len(frames), dtype=np.int64)
        for i in range(len(frames)):
            sums[i] = frames[i].sum( dtype = np.int64 )
    return sums, h5name


def run_sum_testcases( func ):
    import h5py
    with h5py.File( 'sum_testcases.h5', 'r') as hin:
        for name in list(hin):
            grp = hin[name]
            hname = grp['hname'][()]
            dset = grp['dset'][()]
            refdata = grp['sums'][()]
            print(hname, dset)
            t0 = timeit.default_timer()
            sums = func( hname, dset )
            t1 = timeit.default_timer()
            for i in range(len(refdata)):
                if sums[i] != refdata[i]:
                    print( i, sums[i], refdata[i])
                assert sums[i] == refdata[i], (i,'got',sums[i],'want',refdata[i])
            dt = t1 - t0
            print("Matches reference sum data %.3f ms/frame  %.1f fps"%(1e3*dt/len(sums), len(sums)/dt),
                  refdata.shape)
        
        
        
if __name__=="__main__":

    with h5py.File('sum_testcases.h5', 'w') as hout:
        i = 0
        for i, ( h5name, dset ) in enumerate( cases ):
            print(h5name , dset )
            grp = hout.require_group(str(i))
            grp['sums'], grp['hname'] = getsums( h5name, dset )
            grp['dset'] = dset

        

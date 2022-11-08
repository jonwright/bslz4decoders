
import os, h5py

def find_testcases():
    testcases = []
    for hname, dset in ( ( 'bslz4testcases.h5', None ),
                       ("/data/id11/nanoscope/blc12454/id11/WAu5um/WAu5um_DT3/scan0001/eiger_0000.h5",
                            "/entry_0000/ESRF-ID11/eiger/data"),
                         ("/data/id11/jon/hdftest/kevlar.h5", "/entry/data/data" ),
                        ("hplc.h5","/entry_0000/measurement/data"),
                        ):
        if os.path.exists(hname):
            if dset is None:
                with h5py.File( hname, 'r' ) as h:
                    for ds in list(h['/']):
                        testcases.append( (hname, ds) )
            else:
                testcases.append( (hname, dset) )
    return testcases


testcases = find_testcases()


    



if __name__=="__main__":
    for h,d in testcases:
        print(h,d)

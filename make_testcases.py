

import hdf5plugin, h5py, numpy as np

def write_array( h5name, dsetname, ary ):
    assert len(ary.shape) == 3
    with h5py.File( h5name, "a" ) as h5f:
        dset = h5f.create_dataset( dsetname, data = ary,
           chunks = (1, ary.shape[1], ary.shape[2]),
           **hdf5plugin.Bitshuffle( nelems=0, lz4=True) )


def make_testcases( ):
    hname = "bslz4testcases.h5"
    shp = (6, 123, 457)  # with awkward shape
    nelem = np.prod(shp)
    for dtyp, label in ( ( np.uint8, 'data_uint8' ),
                         ( np.uint16, 'data_uint16' ),
                         ( np.uint32, 'data_uint32' ) ):
        ary = np.arange( nelem, dtype = dtyp ).reshape( shp )
        write_array( hname, label, ary )


if __name__=="__main__":
    make_testcases()




        
import os

testcases = [ (hname, dataset) for (hname, dataset) in [
   ("bslz4testcases.h5", "data_uint8"),
   ("bslz4testcases.h5", "data_uint16"),
   ("bslz4testcases.h5", "data_uint32"),
   ("/data/id11/nanoscope/blc12454/id11/WAu5um/WAu5um_DT3/scan0001/eiger_0000.h5",
    "/entry_0000/ESRF-ID11/eiger/data"),
] if os.path.exists( hname ) ]


# Some source code for reading bslz4 data

by Jon Wright, 2021.

If you are here to read your data you probably want 
[hdf5plugin](https://pypi.org/project/hdf5plugin/) or
[bitshuffle](https://pypi.org/project/bitshuffle/) instead.

These are research / experimental codes to see if we can read and reduce 
data from ID11-ESRF more quickly.
You should be seeing about 1 GB/s per core already using the standard 
methods above.

[Wishlish](doc/wishlist.md)

This is not yet ready for general use, but the gpu kernels seem to run.
The interests are:

- getting data decompressed directly inside a GPU
- reading an ROI without decompressing a full image (for CPU)
- learning about what is happening when reading compressed data

To build:
```
  git clone https://github.com/jonwright/bslz4decoders
  # compile some C extensions for CPU work:
  cd bslz4decoders/ccodes
  # python codegen.py              # if your numpy is different 
  python setup.py build_ext --inplace
  cd ../..
  python -m pip install -e .
```

To see it do something:
```
   cd bslz4decoders/test
   python make_testcases.py
   python bench_read.py      # single core serial cpu 
   python testcases.py       # gpu test cases
```

To see how long it takes to read some frame:
```
nvprof --openacc-profiling off                       \
    python -m bslz4decoders.lz4dc_cuda               \
        eiger_0000.h5 /entry_0000/measurement/data 0
```

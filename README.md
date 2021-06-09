

# Some code for decoding bslz4 data

by Jon Wright, 2021.

If you are here to read your data you probably want [hdf5plugin](https://pypi.org/project/hdf5plugin/) or
[bitshuffle](https://pypi.org/project/bitshuffle/) instead.

These are research / experimental codes to see if we can read and reduce data from ID11-ESRF more quickly.
You should be seeing about 1 GB/s per core already using the standard methods above.

[Wishlish](doc/wishlist.md)


To build:
```
  git clone https://github.com/jonwright/bslz4decoders
  # compile the extensions:
  cd bslz4decoders/ccodes
  python setup.py build_ext --inplace
  cd ../..
  python -m pip install -e .
```

To see something:
```
   cd bslz4decoders/test
   python make_testcases.py
   python bench_read.py      # single core simple IO speed
   python testcases.py       # gpu test cases
```

To run a cuda decoder on your data:



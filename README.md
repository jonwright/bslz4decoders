

# Some code for decoding bslz4 data

by Jon Wright, 2021.

If you are here to read your data you probably want [hdf5plugin](https://pypi.org/project/hdf5plugin/) or
[bitshuffle](https://pypi.org/project/bitshuffle/).

These are research / experimental codes to see if we can reduce data from ID11-ESRF more quickly.
You should be getting about 1 GB/s per core already using the standard methods above.

[Wishlish](doc/wishlist.md)

[About bslz4](doc/about_bslz4.md)

```
  git clone https://github.com/jonwright/bslz4encoders
  cd bslz4encoders/csrc
  make
  # python codegen.py bslz4decoders
  # CFLAGS="-fopenmp" f2py -c bslz4decoders.pyf bslz4decoders.c -llz4 -lhdf5
```

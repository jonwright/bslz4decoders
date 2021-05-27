

Some code for decoding bslz4 data

Jon Wright, 2021

[Wishlish](wishlist.md)

[About bslz4](about_bslz4.md)

```
  git clone https://github.com/jonwright/bslz4encoders
  git submodule init
  git submodule update
  cd bitshuffle
  git apply ../bitshuffle.src.bitshuffle_core.patch

  make
  # python codegen.py bslz4decoders
  # CFLAGS="-fopenmp" f2py -c bslz4decoders.pyf bslz4decoders.c -llz4

  make test
```

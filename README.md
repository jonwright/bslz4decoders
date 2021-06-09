

Some code for decoding bslz4 data

Jon Wright, 2021

[Wishlish](doc/wishlist.md)

[About bslz4](doc/about_bslz4.md)

```
  git clone https://github.com/jonwright/bslz4encoders
  cd bslz4encoders/csrc
  make
  # python codegen.py bslz4decoders
  # CFLAGS="-fopenmp" f2py -c bslz4decoders.pyf bslz4decoders.c -llz4
```

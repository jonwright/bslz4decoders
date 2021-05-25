

Some code for decoding bslz4 data

Jon Wright, 2021

[Wishlish](wishlist.md)

[About bslz4](about_bslz4.md)

  python codegen.py bslz4decoders
  CFLAGS="-fopenmp" f2py -c bslz4decoders.pyf bslz4decoders.c -llz4

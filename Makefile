

bslz4decoders.cpython-39-aarch64-linux-gnu.so: bslz4decoders.c bslz4decoders.pyf
	CFLAGS="-fopenmp" f2py -c bslz4decoders.pyf bslz4decoders.c -llz4

bslz4decoders.pyf: codegen.py
	python codegen.py bslz4decoders

bslz4decoders.c: codegen.py
	python codegen.py bslz4decoders




# CFLAGS="-std=c99 -fopenmp" python -m numpy.f2py \
# -c bslz4decoders.pyf bslz4decoders.c \
# -I/data/id11/jon/conda_x86_64/envs/hdf5bench/include
# -L/data/id11/jon/conda_x86_64/envs/hdf5bench/lib
# -llz4 -lippdc -lippcore -lhdf5

bslz4decoders.cpython-39-aarch64-linux-gnu.so: bslz4decoders.c bslz4decoders.pyf
	CFLAGS="-fopenmp" f2py -c bslz4decoders.pyf bslz4decoders.c -llz4

bslz4decoders.pyf: codegen.py
	python codegen.py bslz4decoders

bslz4decoders.c: codegen.py
	python codegen.py bslz4decoders



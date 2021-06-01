



#bslz4decoders.cpython-38-x86_64-linux-gnu.so : bslz4decoders.c bslz4decoders.pyf
#	LDSHARED="icc -shared" CC="icc" CFLAGS="-std=c99 -fopenmp -xHost -fast -ipp" python -m numpy.f2py \
#	-c bslz4decoders.pyf bslz4decoders.c \
#	-I/data/id11/jon/conda_x86_64/envs/hdf5bench/include \
#	-L/data/id11/jon/conda_x86_64/envs/hdf5bench/lib \
#	-I/usr/include/x86_64-linux-gnu -llz4 -lhdf5

# python -m numpy.f2py -c bslz4decoders.pyf bslz4decoders.c
# -Ic:\Users\wright\.conda\envs\hdf5bench\Library\include
# -Lc:\Users\wright\.conda\envs\hdf5bench\Library\lib -lhdf5 -lliblz4


#bslz4decoders.cpython-38-x86_64-linux-gnu.so : bslz4decoders.c bslz4decoders.pyf
#	CFLAGS="-std=c99 -fopenmp -march=native" python -m numpy.f2py \
#	-c bslz4decoders.pyf bslz4decoders.c \
#	-I/data/id11/jon/conda_x86_64/envs/hdf5bench/include \
#	-L/data/id11/jon/conda_x86_64/envs/hdf5bench/lib \
#	-llz4 -lippdc -lippcore -lhdf5




bslz4decoders.so: bslz4decoders.c bslz4decoders.pyf
	CFLAGS='-fopenmp -march=native -std=c99' python -m numpy.f2py -c bslz4decoders.pyf bslz4decoders.c -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -lhdf5 -llz4

bslz4decoders.pyf: codegen.py
	python codegen.py bslz4decoders

bslz4decoders.c: codegen.py
	python codegen.py bslz4decoders

bslz4testcases.h5: make_testcases.py
	rm bslz4testcases.h5
	python make_testcases.py

test: bslz4testcases.h5 testcases.py 
	py.test --ignore=bitshuffle
	python bench_read.py
	python test_decoders.py
	python read_chunks.py

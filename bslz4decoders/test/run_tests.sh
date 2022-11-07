

python3 test_readchunks.py
python3 test_decoders.py
python3 test_q.py
python3 test_h5chunk.py
python3 test_bench_read.py
python3 test_cuda.py
python3 test_itercuda.py hplc.h5 /entry_0000/measurement/data
python3 test_iter_h5chunk_stream.py hplc.h5 /entry_0000/measurement/data
python3 -O iter_h5chunk_stream.py hplc.h5 /entry_0000/measurement/data

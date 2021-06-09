
A repository with various tests and ideas for reading bslz4 data.

Things that could be nice to have:
- reading an ROI from the middle of an image
- faster reading
- faster sparsification

Things that we are not doing in here:
- compression for writing
- general case blocksize or itemsize

test datasets
- basler id11, different roi
- eiger id11, 8, 16, 32 bits
- eiger elsewhere
- dectris and lima written data
- frelon id11 (16 bits)
- different characteristics (masked, low poisson rate, imaging, etc)
- testing framework to find data, run codecs, timing, correctness

test codecs
- bslz4 from bitshuffle
- bslz4 from hdf5plugin (should be the same)
- compiled as scalar / sse2 / avx2 / neon ?
- routines from blosc ?
- other shuffles from the versions like hackers delight and chess wiki (etc)
- lz4 from git head of lz4 project
- lz4 from intel ipp
- lz4 from CUDA nvcomp
- lz4 from Justine Tunney

Missing codecs, to be found or created
- simplified, non optimised bitshuffle code?
- cuda/opencl/glsl bitshuffles
- opencl lz4
- openmp with pre-known block sizes
- zero-skipping bitshuffles

Benchmarking
- amount of IO
- time to read one frame
- time to read a lot (>100) frames
- place where the data is available for a next step (gpu vs cpu and transfers)
- use of MPI and different computers

Potential speeds
- stream benchmarks (
        https://github.com/jeffhammond/STREAM ,
        https://github.com/UoB-HPC/BabelStream ,
        https://github.com/jodavies/opencl-stream )
- nvlink 150 GB/s to GPU inside a P9 ?
- CPU to memory 170 GB/s inside a P9 ?
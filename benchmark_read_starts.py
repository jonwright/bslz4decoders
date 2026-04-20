
import sys
import time
import numpy as np
import h5py
import numba as nb

from bslz4decoders.decoders import BSLZ4ChunkConfig
from bslz4decoders.ccodes.decoders import read_starts as c_read_starts

# Numba version
@nb.njit(cache=True)
def read_starts_numba(compressed, itemsize, blocksize, blocks):
    """Numba JIT compiled version of read_starts"""
    blocks[0] = 12
    compressed_length = len(compressed)
    for i in range(1, len(blocks)):
        pos = blocks[i-1]
        nbytes = (compressed[pos] << 24 | compressed[pos+1] << 16 |
                  compressed[pos+2] << 8 | compressed[pos+3])
        blocks[i] = nbytes + 4 + blocks[i-1]
        if blocks[i] >= compressed_length:
            raise ValueError("Overflow reading starts")
    return 0

# Pure Python version for comparison
def read_starts_python(compressed, itemsize, blocksize, blocks):
    """Pure Python version"""
    blocks[0] = 12
    compressed_length = len(compressed)
    for i in range(1, len(blocks)):
        pos = int(blocks[i-1])
        nbytes = (int(compressed[pos]) << 24 | int(compressed[pos+1]) << 16 |
                  int(compressed[pos+2]) << 8 | int(compressed[pos+3]))
        blocks[i] = nbytes + 4 + blocks[i-1]
        if blocks[i] >= compressed_length:
            raise ValueError("Overflow reading starts")
    return 0


def benchmark_read_starts(h5name, dsetname, num_warmup=10, num_iterations=100):
    """
    Benchmark read_starts implementations
    
    Parameters:
    -----------
    h5name : str
        HDF5 file path
    dsetname : str
        Dataset name
    num_warmup : int
        Number of warmup iterations (for JIT compilation)
    num_iterations : int
        Number of timed iterations
    """
    
    print(f"Benchmarking: {h5name} :: {dsetname}")
    print("=" * 80)
    
    # Open dataset and get info
    with h5py.File(h5name, 'r') as f:
        dset = f[dsetname]
        nframes = dset.shape[0]
        dtype = dset.dtype
        frame_shape = dset.shape[1:]
        
        # Get chunk info from first frame
        chunk_info = dset.id.read_direct_chunk((0, 0, 0))
        chunk = np.frombuffer(chunk_info[1], dtype=np.uint8)
        
        # Create config
        config = BSLZ4ChunkConfig(frame_shape, dtype)
        config.get_blocks(chunk)  # This sets blocksize
        
        print(f"Dataset shape: {dset.shape}")
        print(f"Frame shape: {frame_shape}")
        print(f"Dtype: {dtype}")
        print(f"Number of frames: {nframes}")
        print(f"Blocksize: {config.blocksize}")
        print(f"Number of blocks per frame: {config.nblocks}")
        print()
        
        # Collect all chunks
        print("Reading all chunks from HDF5...")
        chunks = []
        for frame in range(nframes):
            chunk_info = dset.id.read_direct_chunk((frame, 0, 0))
            chunk = np.frombuffer(chunk_info[1], dtype=np.uint8)
            chunks.append(chunk)
        print(f"Read {len(chunks)} chunks")
        print()
        
        # Prepare blocks arrays
        blocks_c = np.empty(config.nblocks, dtype=np.uint32)
        blocks_numba = np.empty(config.nblocks, dtype=np.uint32)
        blocks_python = np.empty(config.nblocks, dtype=np.uint32)
        
        # Warmup - JIT compilation
        print(f"Warming up (JIT compilation)... {num_warmup} iterations")
        for i in range(min(num_warmup, len(chunks))):
            chunk = chunks[i % len(chunks)]
            c_read_starts(chunk, dtype.itemsize, config.blocksize, blocks_c)
            read_starts_numba(chunk, dtype.itemsize, config.blocksize, blocks_numba)
            read_starts_python(chunk, dtype.itemsize, config.blocksize, blocks_python)
        print("Warmup complete")
        print()
        
        # Verify all implementations give same results
        print("Verifying correctness...")
        chunk = chunks[0]
        c_read_starts(chunk, dtype.itemsize, config.blocksize, blocks_c)
        read_starts_numba(chunk, dtype.itemsize, config.blocksize, blocks_numba)
        read_starts_python(chunk, dtype.itemsize, config.blocksize, blocks_python)
        
        if not np.array_equal(blocks_c, blocks_numba):
            print("ERROR: Numba results don't match C!")
            print("C:", blocks_c[:10])
            print("Numba:", blocks_numba[:10])
            return
        
        if not np.array_equal(blocks_c, blocks_python):
            print("ERROR: Python results don't match C!")
            print("C:", blocks_c[:10])
            print("Python:", blocks_python[:10])
            return
        
        print("✓ All implementations produce identical results")
        print()
        
        # Benchmark C version
        print(f"Benchmarking C version... {num_iterations} iterations")
        times_c = []
        for iteration in range(num_iterations):
            chunk = chunks[iteration % len(chunks)]
            t0 = time.perf_counter()
            c_read_starts(chunk, dtype.itemsize, config.blocksize, blocks_c)
            t1 = time.perf_counter()
            times_c.append(t1 - t0)
        
        # Benchmark Numba version
        print(f"Benchmarking Numba version... {num_iterations} iterations")
        times_numba = []
        for iteration in range(num_iterations):
            chunk = chunks[iteration % len(chunks)]
            t0 = time.perf_counter()
            read_starts_numba(chunk, dtype.itemsize, config.blocksize, blocks_numba)
            t1 = time.perf_counter()
            times_numba.append(t1 - t0)
        
        # Benchmark Python version
        print(f"Benchmarking Python version... {num_iterations} iterations")
        times_python = []
        for iteration in range(num_iterations):
            chunk = chunks[iteration % len(chunks)]
            t0 = time.perf_counter()
            read_starts_python(chunk, dtype.itemsize, config.blocksize, blocks_python)
            t1 = time.perf_counter()
            times_python.append(t1 - t0)
        
        print()
        print("=" * 80)
        print("RESULTS")
        print("=" * 80)
        
        times_c = np.array(times_c) * 1e6  # Convert to microseconds
        times_numba = np.array(times_numba) * 1e6
        times_python = np.array(times_python) * 1e6
        
        print(f"{'Implementation':<15} {'Mean (μs)':<12} {'Std (μs)':<12} {'Min (μs)':<12} {'Max (μs)':<12} {'Speedup':<10}")
        print("-" * 80)
        
        mean_c = np.mean(times_c)
        mean_numba = np.mean(times_numba)
        mean_python = np.mean(times_python)
        
        print(f"{'C':<15} {mean_c:>11.3f} {np.std(times_c):>11.3f} "
              f"{np.min(times_c):>11.3f} {np.max(times_c):>11.3f} {'1.00x':<10}")
        
        print(f"{'Numba':<15} {mean_numba:>11.3f} {np.std(times_numba):>11.3f} "
              f"{np.min(times_numba):>11.3f} {np.max(times_numba):>11.3f} "
              f"{mean_c/mean_numba:<10.2f}x")
        
        print(f"{'Python':<15} {mean_python:>11.3f} {np.std(times_python):>11.3f} "
              f"{np.min(times_python):>11.3f} {np.max(times_python):>11.3f} "
              f"{mean_c/mean_python:<10.2f}x")
        
        print()
        print(f"Numba is {mean_numba/mean_c:.2f}x slower than C")
        print(f"Python is {mean_python/mean_c:.2f}x slower than C")
        print(f"Numba is {mean_python/mean_numba:.2f}x faster than Python")
        print()
        
        # Calculate percentage of total decompression time
        # Rough estimate: assume decompression takes ~1ms per frame
        estimated_decomp_time = 1000  # microseconds
        print(f"Estimated impact on total decompression time:")
        print(f"  C:      {mean_c/estimated_decomp_time*100:.2f}% of frame decode time")
        print(f"  Numba:  {mean_numba/estimated_decomp_time*100:.2f}% of frame decode time")
        print(f"  Python: {mean_python/estimated_decomp_time*100:.2f}% of frame decode time")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python benchmark_read_starts.py <h5file> <dataset>")
        sys.exit(1)
    
    h5name = sys.argv[1]
    dsetname = sys.argv[2]
    
    benchmark_read_starts(h5name, dsetname, num_warmup=10, num_iterations=1000)

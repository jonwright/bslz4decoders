
#include "cuda_runtime.h"
#include "nvcomp/lz4.h"

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#define READ32BE(p) \
  ( (uint32_t)(255 & (p)[0]) << 24 |\
    (uint32_t)(255 & (p)[1]) << 16 |\
    (uint32_t)(255 & (p)[2]) <<  8 |\
    (uint32_t)(255 & (p)[3])       )
#define READ64BE(p) \
  ( (uint64_t)(255 & (p)[0]) << 56 |\
    (uint64_t)(255 & (p)[1]) << 48 |\
    (uint64_t)(255 & (p)[2]) << 40 |\
    (uint64_t)(255 & (p)[3]) << 32 |\
    (uint64_t)(255 & (p)[4]) << 24 |\
    (uint64_t)(255 & (p)[5]) << 16 |\
    (uint64_t)(255 & (p)[6]) <<  8 |\
    (uint64_t)(255 & (p)[7])       )


int main(int argc, char* argv[]){

    if( argc != 4){
        printf("Usage: filename bytes_per_pixel output\n");
        exit(1);
    }

    cudaStream_t stream = NULL;
    /* Read in the compressed data */
    struct stat statbuf;
    int fd = open( argv[1], O_RDONLY );
    fstat( fd, &statbuf );
    int compressed_bytes = statbuf.st_size;
    char* h_compressed = (char *) malloc( compressed_bytes * sizeof( char ) ); 
    ssize_t read_in = read( fd, h_compressed, compressed_bytes );
    close( fd );
    if (read_in != compressed_bytes) exit(-1);
    printf("Read in %ld bytes from %s\n", read_in, argv[1] );

    int bpp = atoi( argv[2] );
    printf("Assuming %d bytes per pixel for block design\n", bpp);

    /* total output size */
    size_t total_output_bytes = READ64BE( &h_compressed[0] );
    size_t chunk_size = READ32BE( &h_compressed[8] );
    if ( chunk_size == 0 ) chunk_size = 8192;
    size_t num_chunks = ( total_output_bytes + chunk_size - 1 ) / chunk_size;
    printf("Total bytes to be decompressed %ld, chunk_size %ld, num_chunks %ld\n", 
            total_output_bytes, chunk_size, num_chunks);
    /* Size of each chunk uncompressed */
    size_t * uncomp_sizes;
    cudaMallocHost((void**)&uncomp_sizes, sizeof(*uncomp_sizes)*num_chunks);
    for (size_t i = 0; i < num_chunks; ++i) 
        uncomp_sizes[i] = chunk_size;
    int lastchunk = total_output_bytes % chunk_size;
    int tocopy = lastchunk % ( 8*bpp );
    lastchunk -= tocopy;
    if (lastchunk > 0)
        uncomp_sizes[ num_chunks-1 ] = lastchunk;
    // copy uncompressed chunk sizes to the GPU
    size_t * d_uncomp_sizes;
    cudaMalloc((void**)&d_uncomp_sizes, sizeof(*d_uncomp_sizes));
    cudaMemcpyAsync(d_uncomp_sizes, uncomp_sizes, sizeof(*d_uncomp_sizes)*num_chunks, 
        cudaMemcpyHostToDevice, stream);
    /* Copy the compressed data to the GPU */
    void * d_compressed_data;
    cudaMalloc( &d_compressed_data, compressed_bytes );
    cudaMemcpyAsync(d_compressed_data, h_compressed, compressed_bytes, 
        cudaMemcpyHostToDevice, stream);  
    // sizes and start positions of each compressed chunk
    size_t* h_comp_ptrs = (size_t *) malloc( sizeof( size_t )*num_chunks );
    size_t* h_comp_sizes = (size_t *) malloc( sizeof( size_t )*num_chunks );

    void ** d_comp_ptrs;
    cudaMalloc((void**)&d_comp_ptrs, sizeof( *d_comp_ptrs )*num_chunks );    
    size_t * d_comp_sizes;
    cudaMalloc( &d_comp_sizes, sizeof(d_comp_sizes)*num_chunks);
    
    int p = 12;
    for( int i = 0; i < num_chunks; i++){
        h_comp_ptrs[i] = (size_t) d_compressed_data + p + 4;
        size_t nbytes = READ32BE( &h_compressed[p] );
        h_comp_sizes[i] = nbytes;
        p += 4 + nbytes;
    }
    cudaMemcpy(d_comp_sizes, h_comp_sizes, sizeof(*d_comp_sizes)*num_chunks,
              cudaMemcpyHostToDevice);
    cudaMemcpy(d_comp_ptrs, h_comp_ptrs, sizeof(*d_comp_ptrs)*num_chunks,
              cudaMemcpyHostToDevice);
    /* write output space on GPU */
    void *d_out_data;
    cudaMalloc( &d_out_data, total_output_bytes ); // decompressed data
    /* Pointers to chunks in the write space on host */
    void ** data_ptrs;
    cudaMallocHost( (void**)& data_ptrs, sizeof( *data_ptrs )*num_chunks );
    for( int i=0; i<num_chunks; i++ )
        data_ptrs[i] = ((char*) d_out_data) + chunk_size*i;
    // copy output pointers to the GPU
    void ** d_decomp_output;
    cudaMalloc( (void**) &d_decomp_output, sizeof(*d_decomp_output)*num_chunks);
    cudaMemcpy(d_decomp_output, data_ptrs, sizeof(*d_decomp_output)*num_chunks,
              cudaMemcpyHostToDevice);
    /* Temp space on GPU */
    size_t temp_bytes;
    nvcompBatchedLZ4DecompressGetTempSize(num_chunks, chunk_size, &temp_bytes);
    void * d_decomp_temp;
    cudaMalloc(&d_decomp_temp, temp_bytes);
    
    nvcompBatchedLZ4DecompressAsync(
        d_comp_ptrs, 
        d_comp_sizes, 
        d_uncomp_sizes, 
        chunk_size, 
        num_chunks,
        d_decomp_temp, 
        temp_bytes, 
        d_decomp_output, 
        stream);

    cudaStreamSynchronize( stream );
    /* destination storage on host */
    char* h_data = (char *) malloc( total_output_bytes * sizeof(char) );
    cudaMemcpy(h_data, d_out_data, total_output_bytes * sizeof(char), cudaMemcpyDeviceToHost);

    /* Do not forget to copy the last bytes !!! */
    for( int i=tocopy ; i>0; i--)
        h_data[ total_output_bytes - i ] = h_compressed[ compressed_bytes - i];
    /* write to a file */
    FILE* fout = fopen( argv[3], "w" );
    fwrite( h_data, sizeof(char), total_output_bytes, fout );
    fflush( fout );
    fclose( fout );
    printf("Wrote %ld\n",total_output_bytes);
    cudaFreeHost( uncomp_sizes );
    cudaFree( d_uncomp_sizes );
    cudaFree( d_compressed_data);
    cudaFreeHost( h_comp_ptrs );
    cudaFree(d_comp_ptrs);
    cudaFree(d_comp_sizes);
    cudaFreeHost( h_comp_sizes);
    cudaFree( d_out_data);
    cudaFreeHost( data_ptrs);
    cudaFree( d_decomp_output);
    cudaFree( d_decomp_temp);
    free( h_data );
    free( h_compressed );
}

## Bitshuffled data

Within the hdf5 bslz4 format the data come in blocks. These blocks are usually 8192
bytes long, except for the last block. Each individual block is bit transposed. This
means the original data are:

 - 8192 bytes * 8 = 65536 bits
 - For 32-bit data this is a bit matrix of size [2048][32]
 - For 16 bit data [4096][16]
 - etc

Making the transpose of this gives a block of data which are:
 - 32-bit data [32][2048] = [32][ 256 bytes] = [32][ 64 u32]
 - 16-bit data [16][4096] = [16][ 512 bytes] = [16][128 u32]
 -  8-bit data  [8][8192] = [ 8][1024 bytes] = [ 8][256 u32]

If the last block has fewer values than needed (2048 for 32 bit) then the block length
for that block is reduced. The last block length is chosen to have a multiple of 8 elements
in order to get aligned output data. Any left over values that are not a multiple
of 8 are copied directly to the output. For 8 bit data the blocksize should be a multiple
of 8 (bytes), for 16 bits the multiple is 16, for 32-bits it should be 32 (bytes).

Some pictures would be useful here.

## Random notes

One strange thing, transpose != untranpose. How is that (see above) ? The matrix is rectangular !

 - usual block size is 8192 bytes
 - this is either (1024,8) or (2048,4) or (4096,2) or (8192,1) elements
 - the transpose is of the whole block
 - Case of (2048,4) -> (4, 2048)...
 - ... The first 2048 bits (==256 bytes), (==64 uint32) are the first bit of each output
 - Case of (4092, 2) -> (2, 4096)...
 - ... The first 4096 bits (==512 bytes), (==256 uint16) are the first bit of each output

## code from the bitshuffle library

bshuf_untrans_bit_elem : undo the transpose

calls
 1 bshuf_trans_byte_bitrow_[VERSION]
 2 bshuf_shuffle_bit_eightelem_[VERSION]

VERSION = AVX/SSE/NEON/scal

 AVX calls SSE for byte_bitrow if elem_size%4
 SSE calls scal for shuffle_bit_eightelem if elem_size%2

trans_byte is quite involved.

shuffle_bit is using movemask_epi8 + slli with a loop over 8 bits.

Scalar code from
```
int64_t bshuf_trans_byte_bitrow_scal(const void* in, void* out, const size_t size,
         const size_t elem_size) {
    size_t ii, jj, kk, nbyte_row;
    const char *in_b;              // ?? change signature
    char *out_b;


    in_b = (const char*) in;
    out_b = (char*) out;

    nbyte_row = size / 8;          // ?? does many blocks here. What if it is one block?
                                   // ... so size would be 8,16,32 and fixed.

    CHECK_MULT_EIGHT(size);        // ?? skip

// ?? Seems we could just unroll the whole thing for 1,2,4 byte sizes and one block
// ?? Do an outer loop which includes the bit moves too.

    for (jj = 0; jj < elem_size; jj++) {             // [1,2,4]
        for (ii = 0; ii < nbyte_row; ii++) {         // 1
            for (kk = 0; kk < 8; kk++) {
                out_b[ii * 8 * elem_size + jj * 8 + kk] = \
                        in_b[(jj * 8 + kk) * nbyte_row + ii];
            }
        }
    }
    return size * elem_size;
}
```

```

/* Transpose 8x8 bit array packed into a single quadword *x*.
 * *t* is workspace. */
#define TRANS_BIT_8X8(x, t) {                                               \
        t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AALL;                          \
        x = x ^ t ^ (t << 7);                                               \
        t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCLL;                         \
        x = x ^ t ^ (t << 14);                                              \
        t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0LL;                         \
        x = x ^ t ^ (t << 28);                                              \
    }

/* Transpose 8x8 bit array along the diagonal from upper right
   to lower left */
#define TRANS_BIT_8X8_BE(x, t) {                                            \
        t = (x ^ (x >> 9)) & 0x0055005500550055LL;                          \
        x = x ^ t ^ (t << 9);                                               \
        t = (x ^ (x >> 18)) & 0x0000333300003333LL;                         \
        x = x ^ t ^ (t << 18);                                              \
        t = (x ^ (x >> 36)) & 0x000000000F0F0F0FLL;                         \
        x = x ^ t ^ (t << 36);                                              \
    }


/* Shuffle bits within the bytes of eight element blocks. */
int64_t bshuf_shuffle_bit_eightelem_scal(const void* in, void* out, \
        const size_t size, const size_t elem_size) {

    const char *in_b;   // change signature
    char *out_b;
    uint64_t x, t;
    size_t ii, jj, kk;
    size_t nbyte, out_index;


    uint64_t e=1;  // probably this is at compile time?
    const int little_endian = *(uint8_t *) &e == 1;
    const size_t elem_skip = little_endian ? elem_size : -elem_size;
    const uint64_t elem_offset = little_endian ? 0 : 7 * elem_size;

    CHECK_MULT_EIGHT(size); // skip

    in_b = (const char*) in; // signature
    out_b = (char*) out;

    nbyte = elem_size * size;   // ?? fix size

     // ?? looping over the 8 bits
    for (jj = 0; jj < 8 * elem_size; jj += 8) {
        for (ii = 0; ii + 8 * elem_size - 1 < nbyte; ii += 8 * elem_size) {
            x = *((uint64_t*) &in_b[ii + jj]);  // one 8x8 byte block
            if (little_endian) {
                TRANS_BIT_8X8(x, t);
            } else {
                TRANS_BIT_8X8_BE(x, t);
            }
            for (kk = 0; kk < 8; kk++) {
                out_index = ii + jj / 8 + elem_offset + kk * elem_skip;
                *((uint8_t*) &out_b[out_index]) = x;
                x = x >> 8;
            }
        }
    }
    return size * elem_size;
}
```

## Found on the internet

From https://stackoverflow.com/questions/6930667/what-is-the-fastest-way-to-transpose-the-bits-in-an-8x8-block-on-bits

```
uint8_t get_byte(uint64_t matrix, unsigned col)
{
    const uint64_t column_mask = 0x8080808080808080ull;
    const uint64_t magic       = 0x2040810204081ull;

    return ((matrix << (7 - col)) & column_mask) * magic  >> 56;
}

// You may need to change the endianness if you address the data in a different way
uint64_t block8x8 = ((uint64_t)byte[7] << 56) | ((uint64_t)byte[6] << 48)
                  | ((uint64_t)byte[5] << 40) | ((uint64_t)byte[4] << 32)
                  | ((uint64_t)byte[3] << 24) | ((uint64_t)byte[2] << 16)
                  | ((uint64_t)byte[1] <<  8) |  (uint64_t)byte[0];

for (int i = 0; i < 8; i++)
    byte_out[i] = get_byte(block8x8, i);
```
Also:

https://github.com/dsnet/matrix-transpose/blob/master/matrix_transpose.c

There is also code in hackers delight for 32x32 unrolled.

CUDA code :

https://stackoverflow.com/questions/46615703/efficiently-transposing-a-large-dense-binary-matrix

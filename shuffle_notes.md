

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
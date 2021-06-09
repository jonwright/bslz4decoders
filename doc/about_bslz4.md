
## About bslz4

### bitshuffle + lz4 blocks in hdf5

The plugin used for Eiger data is recording the data in blocks. The first 12 bytes give the total size of the data (BE 64) and the blocksize (BE 32). Then the binary follows with a 4 byte (big endian!) integer giving the block length and then an lz4 compressed block.

In the case of bslz4 format, we know the lengths of the input lz4 compressed data as well as the uncompressed data that are going to be written. This should make it feasible to detect and avoid runaway or other overwrite errors.

The block start positions and lengths are not stored in a convenient way as you have to scan through the data to find them. Making some kind of index to help for this would be one-off job.

## Description of the lz4 decompression

You get a byte value which is two nibbles (0-15). One tells you how many literals follow, the other tells you how much to use for a copy to output. If the value is 15 then you take the next byte and add it on. You keep doing that while the next byte is 255. Then you copy as many literals as it said (could be zero). Now there is a 2 byte (16 bit) offset, which tells you where to start copying from the output to append and make new output. Now you look for the 255 add-ons for the copying. Finally you copy from output to append, taking into account that the length to append can wrap.

There are some details about max and min numbers of copies and ends of things for generic decompressors.

Part of the interest to look at this is to see if there is a way to "skip" the top end of the values for data which are mainly ones and zeros.

## Description of the bitshuffle transformation

See [shuffle_notes.md](shuffle_notes.md)

For the moment we are looking at integer data for 8, 16, 32 bit images which are the cases we need. In the regular data, each pixel is stored as a binary number, usually in little endian format. The pixels come one after the other. After doing a bitshuffle, the first numbers to come out store the first bit of the numbers to follow. The next blocks of number is the next bit, an so on. The data are transposed in blocks and rectangular block can be shuffled independently. The trailing numbers which are not a multiple of (8|16|32) are passed through unchanged.

Little endian means the small part of the number comes first in memory. For 8 bit numbers there is no effect. For 16 bit (2 byte) numbers it means the first value in memory is the part (<255) and the second part is (>255).

To undo a bitshuffle we need to know if the data are 8, 16 or 32 bits (or 1, 2 or 4 bytes).
In the original code the implementation is highly optimised.


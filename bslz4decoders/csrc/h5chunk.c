
/* A curated collection of different BSLZ4 readers
   This is automatically generated code
   Edit this to change the original :
     codegen.py
   Created on :
     Wed Jun  9 15:00:20 2021
   Code generator written by Jon Wright.
*/

#include <stdint.h> /* uint32_t etc */
#include <stdio.h>  /* print error message before killing process(!?!?) */
#include <stdlib.h> /* malloc and friends */
#include <string.h> /* memcpy */

#ifdef USEIPP
#include <ippdc.h> /* for intel ... going to need a platform build system */
#endif

#include <hdf5.h> /* to grab chunks independently of h5py api (py27 issue) */
/* Signature for h5_chunk_size */
size_t h5_chunk_size(int64_t, int);
/* Signature for h5_close_dset */
size_t h5_open_dset(int64_t, char *);
/* Signature for h5_close_file */
int h5_close_file(int64_t);
/* Signature for h5_open_dset */
size_t h5_open_dset(int64_t, char *);
/* Signature for h5_open_file */
size_t h5_open_file(char *);
/* Signature for h5_read_direct */
size_t h5_read_direct(int64_t, int, char *, size_t);
/* Definition for h5_chunk_size */
size_t h5_chunk_size(int64_t dataset_id, int frame) {

  {
    hsize_t offset[3];
    offset[0] = (hsize_t)frame; /* assumes 3D frame-by-frame chunks */
    offset[1] = 0;
    offset[2] = 0;

    /* Get the size of the compressed chunk to return */
    hsize_t chunk_nbytes;
    herr_t ret;
    ret = H5Dget_chunk_storage_size(dataset_id, offset, &chunk_nbytes);
    return ret;
  }

} /* Definition for h5_close_dset */
size_t h5_open_dset(int64_t h5file, char *dsetname) {

  hid_t dataset;
  if ((dataset = H5Dopen2(h5file, dsetname, H5P_DEFAULT)) < 0)
    ERR("Failed to open datset");
  return dataset;

} /* Definition for h5_close_file */
int h5_close_file(int64_t hfile) {

  return H5Fclose(hfile);

  return 0;
}
/* Definition for h5_open_dset */
size_t h5_open_dset(int64_t h5file, char *dsetname) {

  hid_t dataset;
  if ((dataset = H5Dopen2(h5file, dsetname, H5P_DEFAULT)) < 0)
    ERR("Failed to open datset");
  return dataset;

} /* Definition for h5_open_file */
size_t h5_open_file(char *hname) {

  hid_t file;
  file = H5Fopen(hname, H5F_ACC_RDONLY, H5P_DEFAULT);
  return file;

} /* Definition for h5_read_direct */
size_t h5_read_direct(int64_t dataset_id, int frame, char *chunk,
                      size_t chunk_length) {

  {
    /* see:
       https://support.hdfgroup.org/HDF5/doc/HL/RM_HDF5Optimized.html#H5DOread_chunk

       ... assuming this is h5py.dataset.id.id :
        hid_t dataset;
        if((dataset = H5Dopen2(hname, dsetname, H5P_DEFAULT)) < 0)
            ERR("Failed to open h5file");
    */
    hsize_t offset[3];
    offset[0] = (hsize_t)frame; /* assumes 3D frame-by-frame chunks */
    offset[1] = 0;
    offset[2] = 0;

    /* Get the size of the compressed chunk to return */
    hsize_t chunk_nbytes;
    herr_t ret;
    ret = H5Dget_chunk_storage_size(dataset_id, offset, &chunk_nbytes);
    if (chunk_nbytes > chunk_length) {
      fprintf(stderr, "Chunk does not fit into your arg");
      return 0;
    }
    if (ret < 0) {
      fprintf(stderr, "Problem getting storage size for the chunk");
      return 0;
    }
    /* Use H5DOread_chunk() to read the chunk back
       ... becomes H5Dread_chunk in later library versions */
    uint32_t read_filter_mask;
    ret = H5Dread_chunk(dataset_id, H5P_DEFAULT, offset, &read_filter_mask,
                        chunk);
    if (ret < 0) {
      fprintf(stderr, "error reading chunk");
      return 0;
    }
    if (read_filter_mask != 0) {
      fprintf(stderr, "chunk was filtered");
      return 0;
    }
    return chunk_nbytes;
  }
}
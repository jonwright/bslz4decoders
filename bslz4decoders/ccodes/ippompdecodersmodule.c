/* File: ippompdecodersmodule.c
 * This file is auto-generated with f2py (version:2).
 * f2py is a Fortran to Python Interface Generator (FPIG), Second Edition,
 * written by Pearu Peterson <pearu@cens.ioc.ee>.
 * Generation date: Mon Nov  7 18:59:26 2022
 * Do not edit this file directly unless you know what you are doing!!!
 */

#ifdef __cplusplus
extern "C" {
#endif

/*********************** See f2py2e/cfuncs.py: includes ***********************/
#include "Python.h"
#include <stdarg.h>
#include "fortranobject.h"
#include <math.h>

/**************** See f2py2e/rules.py: mod_rules['modulebody'] ****************/
static PyObject *ippompdecoders_error;
static PyObject *ippompdecoders_module;

/*********************** See f2py2e/cfuncs.py: typedefs ***********************/
typedef unsigned char unsigned_char;
#ifdef _WIN32
typedef __int64 long_long;
#else
typedef long long long_long;
typedef unsigned long long unsigned_long_long;
#endif


/****************** See f2py2e/cfuncs.py: typedefs_generated ******************/
/*need_typedefs_generated*/

/********************** See f2py2e/cfuncs.py: cppmacros **********************/
#if defined(PREPEND_FORTRAN)
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) _##F
#else
#define F_FUNC(f,F) _##f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) _##F##_
#else
#define F_FUNC(f,F) _##f##_
#endif
#endif
#else
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F
#else
#define F_FUNC(f,F) f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F##_
#else
#define F_FUNC(f,F) f##_
#endif
#endif
#endif
#if defined(UNDERSCORE_G77)
#define F_FUNC_US(f,F) F_FUNC(f##_,F##_)
#else
#define F_FUNC_US(f,F) F_FUNC(f,F)
#endif

#define rank(var) var ## _Rank
#define shape(var,dim) var ## _Dims[dim]
#define old_rank(var) (PyArray_NDIM((PyArrayObject *)(capi_ ## var ## _tmp)))
#define old_shape(var,dim) PyArray_DIM(((PyArrayObject *)(capi_ ## var ## _tmp)),dim)
#define fshape(var,dim) shape(var,rank(var)-dim-1)
#define len(var) shape(var,0)
#define flen(var) fshape(var,0)
#define old_size(var) PyArray_SIZE((PyArrayObject *)(capi_ ## var ## _tmp))
/* #define index(i) capi_i ## i */
#define slen(var) capi_ ## var ## _len
#define size(var, ...) f2py_size((PyArrayObject *)(capi_ ## var ## _tmp), ## __VA_ARGS__, -1)

#define CHECKSCALAR(check,tcheck,name,show,var)\
    if (!(check)) {\
        char errstring[256];\
        sprintf(errstring, "%s: "show, "("tcheck") failed for "name, var);\
        PyErr_SetString(ippompdecoders_error,errstring);\
        /*goto capi_fail;*/\
    } else 
#ifdef DEBUGCFUNCS
#define CFUNCSMESS(mess) fprintf(stderr,"debug-capi:"mess);
#define CFUNCSMESSPY(mess,obj) CFUNCSMESS(mess) \
    PyObject_Print((PyObject *)obj,stderr,Py_PRINT_RAW);\
    fprintf(stderr,"\n");
#else
#define CFUNCSMESS(mess)
#define CFUNCSMESSPY(mess,obj)
#endif

#ifndef max
#define max(a,b) ((a > b) ? (a) : (b))
#endif
#ifndef min
#define min(a,b) ((a < b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a,b) ((a > b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a,b) ((a < b) ? (a) : (b))
#endif


/************************ See f2py2e/cfuncs.py: cfuncs ************************/
static int f2py_size(PyArrayObject* var, ...)
{
  npy_int sz = 0;
  npy_int dim;
  npy_int rank;
  va_list argp;
  va_start(argp, var);
  dim = va_arg(argp, npy_int);
  if (dim==-1)
    {
      sz = PyArray_SIZE(var);
    }
  else
    {
      rank = PyArray_NDIM(var);
      if (dim>=1 && dim<=rank)
        sz = PyArray_DIM(var, dim-1);
      else
        fprintf(stderr, "f2py_size: 2nd argument value=%d fails to satisfy 1<=value<=%d. Result will be 0.\n", dim, rank);
    }
  va_end(argp);
  return sz;
}

static int int_from_pyobj(int* v,PyObject *obj,const char *errmess) {
    PyObject* tmp = NULL;
    if (PyInt_Check(obj)) {
        *v = (int)PyInt_AS_LONG(obj);
        return 1;
    }
    tmp = PyNumber_Int(obj);
    if (tmp) {
        *v = PyInt_AS_LONG(tmp);
        Py_DECREF(tmp);
        return 1;
    }
    if (PyComplex_Check(obj))
        tmp = PyObject_GetAttrString(obj,"real");
    else if (PyString_Check(obj) || PyUnicode_Check(obj))
        /*pass*/;
    else if (PySequence_Check(obj))
        tmp = PySequence_GetItem(obj,0);
    if (tmp) {
        PyErr_Clear();
        if (int_from_pyobj(v,tmp,errmess)) {Py_DECREF(tmp); return 1;}
        Py_DECREF(tmp);
    }
    {
        PyObject* err = PyErr_Occurred();
        if (err==NULL) err = ippompdecoders_error;
        PyErr_SetString(err,errmess);
    }
    return 0;
}

static int long_long_from_pyobj(long_long* v,PyObject *obj,const char *errmess) {
    PyObject* tmp = NULL;
    if (PyLong_Check(obj)) {
        *v = PyLong_AsLongLong(obj);
        return (!PyErr_Occurred());
    }
    if (PyInt_Check(obj)) {
        *v = (long_long)PyInt_AS_LONG(obj);
        return 1;
    }
    tmp = PyNumber_Long(obj);
    if (tmp) {
        *v = PyLong_AsLongLong(tmp);
        Py_DECREF(tmp);
        return (!PyErr_Occurred());
    }
    if (PyComplex_Check(obj))
        tmp = PyObject_GetAttrString(obj,"real");
    else if (PyString_Check(obj) || PyUnicode_Check(obj))
        /*pass*/;
    else if (PySequence_Check(obj))
        tmp = PySequence_GetItem(obj,0);
    if (tmp) {
        PyErr_Clear();
        if (long_long_from_pyobj(v,tmp,errmess)) {Py_DECREF(tmp); return 1;}
        Py_DECREF(tmp);
    }
    {
        PyObject* err = PyErr_Occurred();
        if (err==NULL) err = ippompdecoders_error;
        PyErr_SetString(err,errmess);
    }
    return 0;
}


/********************* See f2py2e/cfuncs.py: userincludes *********************/
/*need_userincludes*/

/********************* See f2py2e/capi_rules.py: usercode *********************/


/* See f2py2e/rules.py */
extern int omp_bslz4(unsigned_char*,long_long,int,unsigned_char*,long_long,int);
extern int omp_bslz4_blocks(unsigned_char*,long_long,int,long_long,unsigned*,int,unsigned_char*,long_long,int);
extern int omp_get_threads_used(int);
/*eof externroutines*/

/******************** See f2py2e/capi_rules.py: usercode1 ********************/


/******************* See f2py2e/cb_rules.py: buildcallback *******************/
/*need_callbacks*/

/*********************** See f2py2e/rules.py: buildapi ***********************/

/********************************* omp_bslz4 *********************************/
static char doc_f2py_rout_ippompdecoders_omp_bslz4[] = "\
omp_bslz4 = omp_bslz4(compressed,itemsize,output,num_threads)\n\nWrapper for ``omp_bslz4``.\
\n\nParameters\n----------\n"
"compressed : input rank-1 array('B') with bounds (compressed_length)\n"
"itemsize : input int\n"
"output : in/output rank-1 array('B') with bounds (output_length)\n"
"num_threads : input int\n"
"\nReturns\n-------\n"
"omp_bslz4 : int";
/* extern int omp_bslz4(unsigned_char*,long_long,int,unsigned_char*,long_long,int); */
static PyObject *f2py_rout_ippompdecoders_omp_bslz4(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           int (*f2py_func)(unsigned_char*,long_long,int,unsigned_char*,long_long,int)) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
/*decl*/

  int omp_bslz4_return_value=0;
  unsigned_char *compressed = NULL;
  npy_intp compressed_Dims[1] = {-1};
  const int compressed_Rank = 1;
  PyArrayObject *capi_compressed_tmp = NULL;
  int capi_compressed_intent = 0;
  PyObject *compressed_capi = Py_None;
  long_long compressed_length = 0;
  int itemsize = 0;
  PyObject *itemsize_capi = Py_None;
  unsigned_char *output = NULL;
  npy_intp output_Dims[1] = {-1};
  const int output_Rank = 1;
  PyArrayObject *capi_output_tmp = NULL;
  int capi_output_intent = 0;
  PyObject *output_capi = Py_None;
  long_long output_length = 0;
  int num_threads = 0;
  PyObject *num_threads_capi = Py_None;
  static char *capi_kwlist[] = {"compressed","itemsize","output","num_threads",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
    "OOOO|:ippompdecoders.omp_bslz4",\
    capi_kwlist,&compressed_capi,&itemsize_capi,&output_capi,&num_threads_capi))
    return NULL;
/*frompyobj*/
  /* Processing variable compressed */
  ;
  capi_compressed_intent |= F2PY_INTENT_IN|F2PY_INTENT_C;
  capi_compressed_tmp = array_from_pyobj(NPY_UBYTE,compressed_Dims,compressed_Rank,capi_compressed_intent,compressed_capi);
  if (capi_compressed_tmp == NULL) {
    PyObject *exc, *val, *tb;
    PyErr_Fetch(&exc, &val, &tb);
    PyErr_SetString(exc ? exc : ippompdecoders_error,"failed in converting 1st argument `compressed' of ippompdecoders.omp_bslz4 to C/Fortran array" );
    npy_PyErr_ChainExceptionsCause(exc, val, tb);
  } else {
    compressed = (unsigned_char *)(PyArray_DATA(capi_compressed_tmp));

  /* Processing variable itemsize */
    f2py_success = int_from_pyobj(&itemsize,itemsize_capi,"ippompdecoders.omp_bslz4() 2nd argument (itemsize) can't be converted to int");
  if (f2py_success) {
  /* Processing variable output */
  ;
  capi_output_intent |= F2PY_INTENT_INOUT|F2PY_INTENT_C;
  capi_output_tmp = array_from_pyobj(NPY_UBYTE,output_Dims,output_Rank,capi_output_intent,output_capi);
  if (capi_output_tmp == NULL) {
    PyObject *exc, *val, *tb;
    PyErr_Fetch(&exc, &val, &tb);
    PyErr_SetString(exc ? exc : ippompdecoders_error,"failed in converting 3rd argument `output' of ippompdecoders.omp_bslz4 to C/Fortran array" );
    npy_PyErr_ChainExceptionsCause(exc, val, tb);
  } else {
    output = (unsigned_char *)(PyArray_DATA(capi_output_tmp));

  /* Processing variable num_threads */
    f2py_success = int_from_pyobj(&num_threads,num_threads_capi,"ippompdecoders.omp_bslz4() 4th argument (num_threads) can't be converted to int");
  if (f2py_success) {
  /* Processing variable compressed_length */
  compressed_length = len(compressed);
  CHECKSCALAR(len(compressed)>=compressed_length,"len(compressed)>=compressed_length","hidden compressed_length","omp_bslz4:compressed_length=%ld",compressed_length) {
  /* Processing variable output_length */
  output_length = len(output);
  CHECKSCALAR(len(output)>=output_length,"len(output)>=output_length","hidden output_length","omp_bslz4:output_length=%ld",output_length) {
/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
  Py_BEGIN_ALLOW_THREADS
  omp_bslz4_return_value = (*f2py_func)(compressed,compressed_length,itemsize,output,output_length,num_threads);
  Py_END_ALLOW_THREADS
if (PyErr_Occurred())
  f2py_success = 0;
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_call_clock();
#endif
/*end of callfortranroutine*/
    if (f2py_success) {
/*pyobjfrom*/
/*end of pyobjfrom*/
    CFUNCSMESS("Building return value.\n");
    capi_buildvalue = Py_BuildValue("i",omp_bslz4_return_value);
/*closepyobjfrom*/
/*end of closepyobjfrom*/
    } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
  } /*CHECKSCALAR(len(output)>=output_length)*/
  /* End of cleaning variable output_length */
  } /*CHECKSCALAR(len(compressed)>=compressed_length)*/
  /* End of cleaning variable compressed_length */
  } /*if (f2py_success) of num_threads*/
  /* End of cleaning variable num_threads */
  if((PyObject *)capi_output_tmp!=output_capi) {
    Py_XDECREF(capi_output_tmp); }
  }  /*if (capi_output_tmp == NULL) ... else of output*/
  /* End of cleaning variable output */
  } /*if (f2py_success) of itemsize*/
  /* End of cleaning variable itemsize */
  if((PyObject *)capi_compressed_tmp!=compressed_capi) {
    Py_XDECREF(capi_compressed_tmp); }
  }  /*if (capi_compressed_tmp == NULL) ... else of compressed*/
  /* End of cleaning variable compressed */
/*end of cleanupfrompyobj*/
  if (capi_buildvalue == NULL) {
/*routdebugfailure*/
  } else {
/*routdebugleave*/
  }
  CFUNCSMESS("Freeing memory.\n");
/*freemem*/
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_clock();
#endif
  return capi_buildvalue;
}
/****************************** end of omp_bslz4 ******************************/

/****************************** omp_bslz4_blocks ******************************/
static char doc_f2py_rout_ippompdecoders_omp_bslz4_blocks[] = "\
omp_bslz4_blocks = omp_bslz4_blocks(compressed,itemsize,blocksize,blocks,output,num_threads)\n\nWrapper for ``omp_bslz4_blocks``.\
\n\nParameters\n----------\n"
"compressed : input rank-1 array('B') with bounds (compressed_length)\n"
"itemsize : input int\n"
"blocksize : input long\n"
"blocks : in/output rank-1 array('I') with bounds (blocks_length)\n"
"output : in/output rank-1 array('B') with bounds (output_length)\n"
"num_threads : input int\n"
"\nReturns\n-------\n"
"omp_bslz4_blocks : int";
/* extern int omp_bslz4_blocks(unsigned_char*,long_long,int,long_long,unsigned*,int,unsigned_char*,long_long,int); */
static PyObject *f2py_rout_ippompdecoders_omp_bslz4_blocks(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           int (*f2py_func)(unsigned_char*,long_long,int,long_long,unsigned*,int,unsigned_char*,long_long,int)) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
/*decl*/

  int omp_bslz4_blocks_return_value=0;
  unsigned_char *compressed = NULL;
  npy_intp compressed_Dims[1] = {-1};
  const int compressed_Rank = 1;
  PyArrayObject *capi_compressed_tmp = NULL;
  int capi_compressed_intent = 0;
  PyObject *compressed_capi = Py_None;
  long_long compressed_length = 0;
  int itemsize = 0;
  PyObject *itemsize_capi = Py_None;
  long_long blocksize = 0;
  PyObject *blocksize_capi = Py_None;
  unsigned *blocks = NULL;
  npy_intp blocks_Dims[1] = {-1};
  const int blocks_Rank = 1;
  PyArrayObject *capi_blocks_tmp = NULL;
  int capi_blocks_intent = 0;
  PyObject *blocks_capi = Py_None;
  int blocks_length = 0;
  unsigned_char *output = NULL;
  npy_intp output_Dims[1] = {-1};
  const int output_Rank = 1;
  PyArrayObject *capi_output_tmp = NULL;
  int capi_output_intent = 0;
  PyObject *output_capi = Py_None;
  long_long output_length = 0;
  int num_threads = 0;
  PyObject *num_threads_capi = Py_None;
  static char *capi_kwlist[] = {"compressed","itemsize","blocksize","blocks","output","num_threads",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
    "OOOOOO|:ippompdecoders.omp_bslz4_blocks",\
    capi_kwlist,&compressed_capi,&itemsize_capi,&blocksize_capi,&blocks_capi,&output_capi,&num_threads_capi))
    return NULL;
/*frompyobj*/
  /* Processing variable compressed */
  ;
  capi_compressed_intent |= F2PY_INTENT_IN|F2PY_INTENT_C;
  capi_compressed_tmp = array_from_pyobj(NPY_UBYTE,compressed_Dims,compressed_Rank,capi_compressed_intent,compressed_capi);
  if (capi_compressed_tmp == NULL) {
    PyObject *exc, *val, *tb;
    PyErr_Fetch(&exc, &val, &tb);
    PyErr_SetString(exc ? exc : ippompdecoders_error,"failed in converting 1st argument `compressed' of ippompdecoders.omp_bslz4_blocks to C/Fortran array" );
    npy_PyErr_ChainExceptionsCause(exc, val, tb);
  } else {
    compressed = (unsigned_char *)(PyArray_DATA(capi_compressed_tmp));

  /* Processing variable itemsize */
    f2py_success = int_from_pyobj(&itemsize,itemsize_capi,"ippompdecoders.omp_bslz4_blocks() 2nd argument (itemsize) can't be converted to int");
  if (f2py_success) {
  /* Processing variable blocksize */
    f2py_success = long_long_from_pyobj(&blocksize,blocksize_capi,"ippompdecoders.omp_bslz4_blocks() 3rd argument (blocksize) can't be converted to long_long");
  if (f2py_success) {
  /* Processing variable blocks */
  ;
  capi_blocks_intent |= F2PY_INTENT_INOUT|F2PY_INTENT_C;
  capi_blocks_tmp = array_from_pyobj(NPY_UINT,blocks_Dims,blocks_Rank,capi_blocks_intent,blocks_capi);
  if (capi_blocks_tmp == NULL) {
    PyObject *exc, *val, *tb;
    PyErr_Fetch(&exc, &val, &tb);
    PyErr_SetString(exc ? exc : ippompdecoders_error,"failed in converting 4th argument `blocks' of ippompdecoders.omp_bslz4_blocks to C/Fortran array" );
    npy_PyErr_ChainExceptionsCause(exc, val, tb);
  } else {
    blocks = (unsigned *)(PyArray_DATA(capi_blocks_tmp));

  /* Processing variable output */
  ;
  capi_output_intent |= F2PY_INTENT_INOUT|F2PY_INTENT_C;
  capi_output_tmp = array_from_pyobj(NPY_UBYTE,output_Dims,output_Rank,capi_output_intent,output_capi);
  if (capi_output_tmp == NULL) {
    PyObject *exc, *val, *tb;
    PyErr_Fetch(&exc, &val, &tb);
    PyErr_SetString(exc ? exc : ippompdecoders_error,"failed in converting 5th argument `output' of ippompdecoders.omp_bslz4_blocks to C/Fortran array" );
    npy_PyErr_ChainExceptionsCause(exc, val, tb);
  } else {
    output = (unsigned_char *)(PyArray_DATA(capi_output_tmp));

  /* Processing variable num_threads */
    f2py_success = int_from_pyobj(&num_threads,num_threads_capi,"ippompdecoders.omp_bslz4_blocks() 6th argument (num_threads) can't be converted to int");
  if (f2py_success) {
  /* Processing variable compressed_length */
  compressed_length = len(compressed);
  CHECKSCALAR(len(compressed)>=compressed_length,"len(compressed)>=compressed_length","hidden compressed_length","omp_bslz4_blocks:compressed_length=%ld",compressed_length) {
  /* Processing variable blocks_length */
  blocks_length = len(blocks);
  CHECKSCALAR(len(blocks)>=blocks_length,"len(blocks)>=blocks_length","hidden blocks_length","omp_bslz4_blocks:blocks_length=%d",blocks_length) {
  /* Processing variable output_length */
  output_length = len(output);
  CHECKSCALAR(len(output)>=output_length,"len(output)>=output_length","hidden output_length","omp_bslz4_blocks:output_length=%ld",output_length) {
/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
  Py_BEGIN_ALLOW_THREADS
  omp_bslz4_blocks_return_value = (*f2py_func)(compressed,compressed_length,itemsize,blocksize,blocks,blocks_length,output,output_length,num_threads);
  Py_END_ALLOW_THREADS
if (PyErr_Occurred())
  f2py_success = 0;
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_call_clock();
#endif
/*end of callfortranroutine*/
    if (f2py_success) {
/*pyobjfrom*/
/*end of pyobjfrom*/
    CFUNCSMESS("Building return value.\n");
    capi_buildvalue = Py_BuildValue("i",omp_bslz4_blocks_return_value);
/*closepyobjfrom*/
/*end of closepyobjfrom*/
    } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
  } /*CHECKSCALAR(len(output)>=output_length)*/
  /* End of cleaning variable output_length */
  } /*CHECKSCALAR(len(blocks)>=blocks_length)*/
  /* End of cleaning variable blocks_length */
  } /*CHECKSCALAR(len(compressed)>=compressed_length)*/
  /* End of cleaning variable compressed_length */
  } /*if (f2py_success) of num_threads*/
  /* End of cleaning variable num_threads */
  if((PyObject *)capi_output_tmp!=output_capi) {
    Py_XDECREF(capi_output_tmp); }
  }  /*if (capi_output_tmp == NULL) ... else of output*/
  /* End of cleaning variable output */
  if((PyObject *)capi_blocks_tmp!=blocks_capi) {
    Py_XDECREF(capi_blocks_tmp); }
  }  /*if (capi_blocks_tmp == NULL) ... else of blocks*/
  /* End of cleaning variable blocks */
  } /*if (f2py_success) of blocksize*/
  /* End of cleaning variable blocksize */
  } /*if (f2py_success) of itemsize*/
  /* End of cleaning variable itemsize */
  if((PyObject *)capi_compressed_tmp!=compressed_capi) {
    Py_XDECREF(capi_compressed_tmp); }
  }  /*if (capi_compressed_tmp == NULL) ... else of compressed*/
  /* End of cleaning variable compressed */
/*end of cleanupfrompyobj*/
  if (capi_buildvalue == NULL) {
/*routdebugfailure*/
  } else {
/*routdebugleave*/
  }
  CFUNCSMESS("Freeing memory.\n");
/*freemem*/
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_clock();
#endif
  return capi_buildvalue;
}
/************************** end of omp_bslz4_blocks **************************/

/**************************** omp_get_threads_used ****************************/
static char doc_f2py_rout_ippompdecoders_omp_get_threads_used[] = "\
omp_get_threads_used = omp_get_threads_used(num_threads)\n\nWrapper for ``omp_get_threads_used``.\
\n\nParameters\n----------\n"
"num_threads : input int\n"
"\nReturns\n-------\n"
"omp_get_threads_used : int";
/* extern int omp_get_threads_used(int); */
static PyObject *f2py_rout_ippompdecoders_omp_get_threads_used(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           int (*f2py_func)(int)) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
/*decl*/

  int omp_get_threads_used_return_value=0;
  int num_threads = 0;
  PyObject *num_threads_capi = Py_None;
  static char *capi_kwlist[] = {"num_threads",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
    "O|:ippompdecoders.omp_get_threads_used",\
    capi_kwlist,&num_threads_capi))
    return NULL;
/*frompyobj*/
  /* Processing variable num_threads */
    f2py_success = int_from_pyobj(&num_threads,num_threads_capi,"ippompdecoders.omp_get_threads_used() 1st argument (num_threads) can't be converted to int");
  if (f2py_success) {
/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
  Py_BEGIN_ALLOW_THREADS
  omp_get_threads_used_return_value = (*f2py_func)(num_threads);
  Py_END_ALLOW_THREADS
if (PyErr_Occurred())
  f2py_success = 0;
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_call_clock();
#endif
/*end of callfortranroutine*/
    if (f2py_success) {
/*pyobjfrom*/
/*end of pyobjfrom*/
    CFUNCSMESS("Building return value.\n");
    capi_buildvalue = Py_BuildValue("i",omp_get_threads_used_return_value);
/*closepyobjfrom*/
/*end of closepyobjfrom*/
    } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
  } /*if (f2py_success) of num_threads*/
  /* End of cleaning variable num_threads */
/*end of cleanupfrompyobj*/
  if (capi_buildvalue == NULL) {
/*routdebugfailure*/
  } else {
/*routdebugleave*/
  }
  CFUNCSMESS("Freeing memory.\n");
/*freemem*/
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_clock();
#endif
  return capi_buildvalue;
}
/************************ end of omp_get_threads_used ************************/
/*eof body*/

/******************* See f2py2e/f90mod_rules.py: buildhooks *******************/
/*need_f90modhooks*/

/************** See f2py2e/rules.py: module_rules['modulebody'] **************/

/******************* See f2py2e/common_rules.py: buildhooks *******************/

/*need_commonhooks*/

/**************************** See f2py2e/rules.py ****************************/

static FortranDataDef f2py_routine_defs[] = {
  {"omp_bslz4",-1,{{-1}},0,(char *)omp_bslz4,(f2py_init_func)f2py_rout_ippompdecoders_omp_bslz4,doc_f2py_rout_ippompdecoders_omp_bslz4},
  {"omp_bslz4_blocks",-1,{{-1}},0,(char *)omp_bslz4_blocks,(f2py_init_func)f2py_rout_ippompdecoders_omp_bslz4_blocks,doc_f2py_rout_ippompdecoders_omp_bslz4_blocks},
  {"omp_get_threads_used",-1,{{-1}},0,(char *)omp_get_threads_used,(f2py_init_func)f2py_rout_ippompdecoders_omp_get_threads_used,doc_f2py_rout_ippompdecoders_omp_get_threads_used},

/*eof routine_defs*/
  {NULL}
};

static PyMethodDef f2py_module_methods[] = {

  {NULL,NULL}
};

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "ippompdecoders",
  NULL,
  -1,
  f2py_module_methods,
  NULL,
  NULL,
  NULL,
  NULL
};

PyMODINIT_FUNC PyInit_ippompdecoders(void) {
  int i;
  PyObject *m,*d, *s, *tmp;
  m = ippompdecoders_module = PyModule_Create(&moduledef);
  Py_SET_TYPE(&PyFortran_Type, &PyType_Type);
  import_array();
  if (PyErr_Occurred())
    {PyErr_SetString(PyExc_ImportError, "can't initialize module ippompdecoders (failed to import numpy)"); return m;}
  d = PyModule_GetDict(m);
  s = PyString_FromString("$Revision: $");
  PyDict_SetItemString(d, "__version__", s);
  Py_DECREF(s);
  s = PyUnicode_FromString(
    "This module 'ippompdecoders' is auto-generated with f2py (version:2).\nFunctions:\n"
"  omp_bslz4 = omp_bslz4(compressed,itemsize,output,num_threads)\n"
"  omp_bslz4_blocks = omp_bslz4_blocks(compressed,itemsize,blocksize,blocks,output,num_threads)\n"
"  omp_get_threads_used = omp_get_threads_used(num_threads)\n"
".");
  PyDict_SetItemString(d, "__doc__", s);
  Py_DECREF(s);
  ippompdecoders_error = PyErr_NewException ("ippompdecoders.error", NULL, NULL);
  /*
   * Store the error object inside the dict, so that it could get deallocated.
   * (in practice, this is a module, so it likely will not and cannot.)
   */
  PyDict_SetItemString(d, "_ippompdecoders_error", ippompdecoders_error);
  Py_DECREF(ippompdecoders_error);
  for(i=0;f2py_routine_defs[i].name!=NULL;i++) {
    tmp = PyFortranObject_NewAsAttr(&f2py_routine_defs[i]);
    PyDict_SetItemString(d, f2py_routine_defs[i].name, tmp);
    Py_DECREF(tmp);
  }



/*eof initf2pywraphooks*/
/*eof initf90modhooks*/

/*eof initcommonhooks*/


#ifdef F2PY_REPORT_ATEXIT
  if (! PyErr_Occurred())
    on_exit(f2py_report_on_exit,(void*)"ippompdecoders");
#endif
  return m;
}
#ifdef __cplusplus
}
#endif

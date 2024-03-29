/* File: h5chunkmodule.c
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
#include <stdarg.h>
#include "Python.h"
#include "fortranobject.h"
#include <string.h>
#include <math.h>

/**************** See f2py2e/rules.py: mod_rules['modulebody'] ****************/
static PyObject *h5chunk_error;
static PyObject *h5chunk_module;

/*********************** See f2py2e/cfuncs.py: typedefs ***********************/
typedef char * string;
#ifdef _WIN32
typedef __int64 long_long;
#else
typedef long long long_long;
typedef unsigned long long unsigned_long_long;
#endif

typedef unsigned char unsigned_char;

/****************** See f2py2e/cfuncs.py: typedefs_generated ******************/
/*need_typedefs_generated*/

/********************** See f2py2e/cfuncs.py: cppmacros **********************/
\
#define FAILNULL(p) do {                                            \
    if ((p) == NULL) {                                              \
        PyErr_SetString(PyExc_MemoryError, "NULL pointer found");   \
        goto capi_fail;                                             \
    }                                                               \
} while (0)

#define STRINGMALLOC(str,len)\
    if ((str = (string)malloc(sizeof(char)*(len+1))) == NULL) {\
        PyErr_SetString(PyExc_MemoryError, "out of memory");\
        goto capi_fail;\
    } else {\
        (str)[len] = '\0';\
    }

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
        PyErr_SetString(h5chunk_error,errstring);\
        /*goto capi_fail;*/\
    } else 
#define STRINGFREE(str) do {if (!(str == NULL)) free(str);} while (0)

#define STRINGCOPYN(to,from,buf_size)                           \
    do {                                                        \
        int _m = (buf_size);                                    \
        char *_to = (to);                                       \
        char *_from = (from);                                   \
        FAILNULL(_to); FAILNULL(_from);                         \
        (void)strncpy(_to, _from, sizeof(char)*_m);             \
        _to[_m-1] = '\0';                                      \
        /* Padding with spaces instead of nulls */              \
        for (_m -= 2; _m >= 0 && _to[_m] == '\0'; _m--) {      \
            _to[_m] = ' ';                                      \
        }                                                       \
    } while (0)


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
        if (err==NULL) err = h5chunk_error;
        PyErr_SetString(err,errmess);
    }
    return 0;
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
        if (err==NULL) err = h5chunk_error;
        PyErr_SetString(err,errmess);
    }
    return 0;
}

static int string_from_pyobj(string *str,int *len,const string inistr,PyObject *obj,const char *errmess) {
    PyArrayObject *arr = NULL;
    PyObject *tmp = NULL;
#ifdef DEBUGCFUNCS
fprintf(stderr,"string_from_pyobj(str='%s',len=%d,inistr='%s',obj=%p)\n",(char*)str,*len,(char *)inistr,obj);
#endif
    if (obj == Py_None) {
        if (*len == -1)
            *len = strlen(inistr); /* Will this cause problems? */
        STRINGMALLOC(*str,*len);
        STRINGCOPYN(*str,inistr,*len+1);
        return 1;
    }
    if (PyArray_Check(obj)) {
        if ((arr = (PyArrayObject *)obj) == NULL)
            goto capi_fail;
        if (!ISCONTIGUOUS(arr)) {
            PyErr_SetString(PyExc_ValueError,"array object is non-contiguous.");
            goto capi_fail;
        }
        if (*len == -1)
            *len = (PyArray_ITEMSIZE(arr))*PyArray_SIZE(arr);
        STRINGMALLOC(*str,*len);
        STRINGCOPYN(*str,PyArray_DATA(arr),*len+1);
        return 1;
    }
    if (PyString_Check(obj)) {
        tmp = obj;
        Py_INCREF(tmp);
    }
    else if (PyUnicode_Check(obj)) {
        tmp = PyUnicode_AsASCIIString(obj);
    }
    else {
        PyObject *tmp2;
        tmp2 = PyObject_Str(obj);
        if (tmp2) {
            tmp = PyUnicode_AsASCIIString(tmp2);
            Py_DECREF(tmp2);
        }
        else {
            tmp = NULL;
        }
    }
    if (tmp == NULL) goto capi_fail;
    if (*len == -1)
        *len = PyString_GET_SIZE(tmp);
    STRINGMALLOC(*str,*len);
    STRINGCOPYN(*str,PyString_AS_STRING(tmp),*len+1);
    Py_DECREF(tmp);
    return 1;
capi_fail:
    Py_XDECREF(tmp);
    {
        PyObject* err = PyErr_Occurred();
        if (err==NULL) err = h5chunk_error;
        PyErr_SetString(err,errmess);
    }
    return 0;
}


/********************* See f2py2e/cfuncs.py: userincludes *********************/
/*need_userincludes*/

/********************* See f2py2e/capi_rules.py: usercode *********************/


/* See f2py2e/rules.py */
extern long_long h5_chunk_size(long_long,int);
extern int h5_close_dset(long_long);
extern int h5_close_file(long_long);
extern long_long h5_dsinfo(long_long,long_long*,int);
extern long_long h5_open_dset(long_long,string,size_t);
extern long_long h5_open_file(string,size_t);
extern long_long h5_read_direct(long_long,int,unsigned_char*,long_long);
/*eof externroutines*/

/******************** See f2py2e/capi_rules.py: usercode1 ********************/


/******************* See f2py2e/cb_rules.py: buildcallback *******************/
/*need_callbacks*/

/*********************** See f2py2e/rules.py: buildapi ***********************/

/******************************* h5_chunk_size *******************************/
static char doc_f2py_rout_h5chunk_h5_chunk_size[] = "\
h5_chunk_size = h5_chunk_size(dataset_id,frame)\n\nWrapper for ``h5_chunk_size``.\
\n\nParameters\n----------\n"
"dataset_id : input long\n"
"frame : input int\n"
"\nReturns\n-------\n"
"h5_chunk_size : long";
/* extern long_long h5_chunk_size(long_long,int); */
static PyObject *f2py_rout_h5chunk_h5_chunk_size(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           long_long (*f2py_func)(long_long,int)) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
/*decl*/

  long_long h5_chunk_size_return_value=0;
  long_long dataset_id = 0;
  PyObject *dataset_id_capi = Py_None;
  int frame = 0;
  PyObject *frame_capi = Py_None;
  static char *capi_kwlist[] = {"dataset_id","frame",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
    "OO|:h5chunk.h5_chunk_size",\
    capi_kwlist,&dataset_id_capi,&frame_capi))
    return NULL;
/*frompyobj*/
  /* Processing variable dataset_id */
    f2py_success = long_long_from_pyobj(&dataset_id,dataset_id_capi,"h5chunk.h5_chunk_size() 1st argument (dataset_id) can't be converted to long_long");
  if (f2py_success) {
  /* Processing variable frame */
    f2py_success = int_from_pyobj(&frame,frame_capi,"h5chunk.h5_chunk_size() 2nd argument (frame) can't be converted to int");
  if (f2py_success) {
/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
  Py_BEGIN_ALLOW_THREADS
  h5_chunk_size_return_value = (*f2py_func)(dataset_id,frame);
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
    capi_buildvalue = Py_BuildValue("L",h5_chunk_size_return_value);
/*closepyobjfrom*/
/*end of closepyobjfrom*/
    } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
  } /*if (f2py_success) of frame*/
  /* End of cleaning variable frame */
  } /*if (f2py_success) of dataset_id*/
  /* End of cleaning variable dataset_id */
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
/**************************** end of h5_chunk_size ****************************/

/******************************* h5_close_dset *******************************/
static char doc_f2py_rout_h5chunk_h5_close_dset[] = "\
h5_close_dset = h5_close_dset(dset)\n\nWrapper for ``h5_close_dset``.\
\n\nParameters\n----------\n"
"dset : input long\n"
"\nReturns\n-------\n"
"h5_close_dset : int";
/* extern int h5_close_dset(long_long); */
static PyObject *f2py_rout_h5chunk_h5_close_dset(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           int (*f2py_func)(long_long)) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
/*decl*/

  int h5_close_dset_return_value=0;
  long_long dset = 0;
  PyObject *dset_capi = Py_None;
  static char *capi_kwlist[] = {"dset",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
    "O|:h5chunk.h5_close_dset",\
    capi_kwlist,&dset_capi))
    return NULL;
/*frompyobj*/
  /* Processing variable dset */
    f2py_success = long_long_from_pyobj(&dset,dset_capi,"h5chunk.h5_close_dset() 1st argument (dset) can't be converted to long_long");
  if (f2py_success) {
/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
  Py_BEGIN_ALLOW_THREADS
  h5_close_dset_return_value = (*f2py_func)(dset);
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
    capi_buildvalue = Py_BuildValue("i",h5_close_dset_return_value);
/*closepyobjfrom*/
/*end of closepyobjfrom*/
    } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
  } /*if (f2py_success) of dset*/
  /* End of cleaning variable dset */
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
/**************************** end of h5_close_dset ****************************/

/******************************* h5_close_file *******************************/
static char doc_f2py_rout_h5chunk_h5_close_file[] = "\
h5_close_file = h5_close_file(hfile)\n\nWrapper for ``h5_close_file``.\
\n\nParameters\n----------\n"
"hfile : input long\n"
"\nReturns\n-------\n"
"h5_close_file : int";
/* extern int h5_close_file(long_long); */
static PyObject *f2py_rout_h5chunk_h5_close_file(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           int (*f2py_func)(long_long)) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
/*decl*/

  int h5_close_file_return_value=0;
  long_long hfile = 0;
  PyObject *hfile_capi = Py_None;
  static char *capi_kwlist[] = {"hfile",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
    "O|:h5chunk.h5_close_file",\
    capi_kwlist,&hfile_capi))
    return NULL;
/*frompyobj*/
  /* Processing variable hfile */
    f2py_success = long_long_from_pyobj(&hfile,hfile_capi,"h5chunk.h5_close_file() 1st argument (hfile) can't be converted to long_long");
  if (f2py_success) {
/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
  Py_BEGIN_ALLOW_THREADS
  h5_close_file_return_value = (*f2py_func)(hfile);
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
    capi_buildvalue = Py_BuildValue("i",h5_close_file_return_value);
/*closepyobjfrom*/
/*end of closepyobjfrom*/
    } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
  } /*if (f2py_success) of hfile*/
  /* End of cleaning variable hfile */
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
/**************************** end of h5_close_file ****************************/

/********************************* h5_dsinfo *********************************/
static char doc_f2py_rout_h5chunk_h5_dsinfo[] = "\
h5_dsinfo = h5_dsinfo(dataset_id,dsinfo)\n\nWrapper for ``h5_dsinfo``.\
\n\nParameters\n----------\n"
"dataset_id : input long\n"
"dsinfo : in/output rank-1 array('q') with bounds (dsinfo_length)\n"
"\nReturns\n-------\n"
"h5_dsinfo : long";
/* extern long_long h5_dsinfo(long_long,long_long*,int); */
static PyObject *f2py_rout_h5chunk_h5_dsinfo(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           long_long (*f2py_func)(long_long,long_long*,int)) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
/*decl*/

  long_long h5_dsinfo_return_value=0;
  long_long dataset_id = 0;
  PyObject *dataset_id_capi = Py_None;
  long_long *dsinfo = NULL;
  npy_intp dsinfo_Dims[1] = {-1};
  const int dsinfo_Rank = 1;
  PyArrayObject *capi_dsinfo_tmp = NULL;
  int capi_dsinfo_intent = 0;
  PyObject *dsinfo_capi = Py_None;
  int dsinfo_length = 0;
  static char *capi_kwlist[] = {"dataset_id","dsinfo",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
    "OO|:h5chunk.h5_dsinfo",\
    capi_kwlist,&dataset_id_capi,&dsinfo_capi))
    return NULL;
/*frompyobj*/
  /* Processing variable dataset_id */
    f2py_success = long_long_from_pyobj(&dataset_id,dataset_id_capi,"h5chunk.h5_dsinfo() 1st argument (dataset_id) can't be converted to long_long");
  if (f2py_success) {
  /* Processing variable dsinfo */
  ;
  capi_dsinfo_intent |= F2PY_INTENT_INOUT|F2PY_INTENT_C;
  capi_dsinfo_tmp = array_from_pyobj(NPY_LONGLONG,dsinfo_Dims,dsinfo_Rank,capi_dsinfo_intent,dsinfo_capi);
  if (capi_dsinfo_tmp == NULL) {
    PyObject *exc, *val, *tb;
    PyErr_Fetch(&exc, &val, &tb);
    PyErr_SetString(exc ? exc : h5chunk_error,"failed in converting 2nd argument `dsinfo' of h5chunk.h5_dsinfo to C/Fortran array" );
    npy_PyErr_ChainExceptionsCause(exc, val, tb);
  } else {
    dsinfo = (long_long *)(PyArray_DATA(capi_dsinfo_tmp));

  /* Processing variable dsinfo_length */
  dsinfo_length = len(dsinfo);
  CHECKSCALAR(len(dsinfo)>=dsinfo_length,"len(dsinfo)>=dsinfo_length","hidden dsinfo_length","h5_dsinfo:dsinfo_length=%d",dsinfo_length) {
/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
  Py_BEGIN_ALLOW_THREADS
  h5_dsinfo_return_value = (*f2py_func)(dataset_id,dsinfo,dsinfo_length);
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
    capi_buildvalue = Py_BuildValue("L",h5_dsinfo_return_value);
/*closepyobjfrom*/
/*end of closepyobjfrom*/
    } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
  } /*CHECKSCALAR(len(dsinfo)>=dsinfo_length)*/
  /* End of cleaning variable dsinfo_length */
  if((PyObject *)capi_dsinfo_tmp!=dsinfo_capi) {
    Py_XDECREF(capi_dsinfo_tmp); }
  }  /*if (capi_dsinfo_tmp == NULL) ... else of dsinfo*/
  /* End of cleaning variable dsinfo */
  } /*if (f2py_success) of dataset_id*/
  /* End of cleaning variable dataset_id */
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
/****************************** end of h5_dsinfo ******************************/

/******************************** h5_open_dset ********************************/
static char doc_f2py_rout_h5chunk_h5_open_dset[] = "\
h5_open_dset = h5_open_dset(h5file,dsetname)\n\nWrapper for ``h5_open_dset``.\
\n\nParameters\n----------\n"
"h5file : input long\n"
"dsetname : input string(len=-1)\n"
"\nReturns\n-------\n"
"h5_open_dset : long";
/* extern long_long h5_open_dset(long_long,string,size_t); */
static PyObject *f2py_rout_h5chunk_h5_open_dset(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           long_long (*f2py_func)(long_long,string,size_t)) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
/*decl*/

  long_long h5_open_dset_return_value=0;
  long_long h5file = 0;
  PyObject *h5file_capi = Py_None;
  string dsetname = NULL;
  int slen(dsetname);
  PyObject *dsetname_capi = Py_None;
  static char *capi_kwlist[] = {"h5file","dsetname",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
    "OO|:h5chunk.h5_open_dset",\
    capi_kwlist,&h5file_capi,&dsetname_capi))
    return NULL;
/*frompyobj*/
  /* Processing variable h5file */
    f2py_success = long_long_from_pyobj(&h5file,h5file_capi,"h5chunk.h5_open_dset() 1st argument (h5file) can't be converted to long_long");
  if (f2py_success) {
  /* Processing variable dsetname */
  slen(dsetname) = -1;
  f2py_success = string_from_pyobj(&dsetname,&slen(dsetname),"",dsetname_capi,"string_from_pyobj failed in converting 2nd argument `dsetname' of h5chunk.h5_open_dset to C string");
  if (f2py_success) {
/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
  Py_BEGIN_ALLOW_THREADS
  h5_open_dset_return_value = (*f2py_func)(h5file,dsetname,slen(dsetname));
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
    capi_buildvalue = Py_BuildValue("L",h5_open_dset_return_value);
/*closepyobjfrom*/
/*end of closepyobjfrom*/
    } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
    STRINGFREE(dsetname);
  }  /*if (f2py_success) of dsetname*/
  /* End of cleaning variable dsetname */
  } /*if (f2py_success) of h5file*/
  /* End of cleaning variable h5file */
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
/**************************** end of h5_open_dset ****************************/

/******************************** h5_open_file ********************************/
static char doc_f2py_rout_h5chunk_h5_open_file[] = "\
h5_open_file = h5_open_file(hname)\n\nWrapper for ``h5_open_file``.\
\n\nParameters\n----------\n"
"hname : input string(len=-1)\n"
"\nReturns\n-------\n"
"h5_open_file : long";
/* extern long_long h5_open_file(string,size_t); */
static PyObject *f2py_rout_h5chunk_h5_open_file(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           long_long (*f2py_func)(string,size_t)) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
/*decl*/

  long_long h5_open_file_return_value=0;
  string hname = NULL;
  int slen(hname);
  PyObject *hname_capi = Py_None;
  static char *capi_kwlist[] = {"hname",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
    "O|:h5chunk.h5_open_file",\
    capi_kwlist,&hname_capi))
    return NULL;
/*frompyobj*/
  /* Processing variable hname */
  slen(hname) = -1;
  f2py_success = string_from_pyobj(&hname,&slen(hname),"",hname_capi,"string_from_pyobj failed in converting 1st argument `hname' of h5chunk.h5_open_file to C string");
  if (f2py_success) {
/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
  Py_BEGIN_ALLOW_THREADS
  h5_open_file_return_value = (*f2py_func)(hname,slen(hname));
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
    capi_buildvalue = Py_BuildValue("L",h5_open_file_return_value);
/*closepyobjfrom*/
/*end of closepyobjfrom*/
    } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
    STRINGFREE(hname);
  }  /*if (f2py_success) of hname*/
  /* End of cleaning variable hname */
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
/**************************** end of h5_open_file ****************************/

/******************************* h5_read_direct *******************************/
static char doc_f2py_rout_h5chunk_h5_read_direct[] = "\
h5_read_direct = h5_read_direct(dataset_id,frame,chunk)\n\nWrapper for ``h5_read_direct``.\
\n\nParameters\n----------\n"
"dataset_id : input long\n"
"frame : input int\n"
"chunk : in/output rank-1 array('B') with bounds (chunk_length)\n"
"\nReturns\n-------\n"
"h5_read_direct : long";
/* extern long_long h5_read_direct(long_long,int,unsigned_char*,long_long); */
static PyObject *f2py_rout_h5chunk_h5_read_direct(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           long_long (*f2py_func)(long_long,int,unsigned_char*,long_long)) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
/*decl*/

  long_long h5_read_direct_return_value=0;
  long_long dataset_id = 0;
  PyObject *dataset_id_capi = Py_None;
  int frame = 0;
  PyObject *frame_capi = Py_None;
  unsigned_char *chunk = NULL;
  npy_intp chunk_Dims[1] = {-1};
  const int chunk_Rank = 1;
  PyArrayObject *capi_chunk_tmp = NULL;
  int capi_chunk_intent = 0;
  PyObject *chunk_capi = Py_None;
  long_long chunk_length = 0;
  static char *capi_kwlist[] = {"dataset_id","frame","chunk",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
    "OOO|:h5chunk.h5_read_direct",\
    capi_kwlist,&dataset_id_capi,&frame_capi,&chunk_capi))
    return NULL;
/*frompyobj*/
  /* Processing variable dataset_id */
    f2py_success = long_long_from_pyobj(&dataset_id,dataset_id_capi,"h5chunk.h5_read_direct() 1st argument (dataset_id) can't be converted to long_long");
  if (f2py_success) {
  /* Processing variable frame */
    f2py_success = int_from_pyobj(&frame,frame_capi,"h5chunk.h5_read_direct() 2nd argument (frame) can't be converted to int");
  if (f2py_success) {
  /* Processing variable chunk */
  ;
  capi_chunk_intent |= F2PY_INTENT_INOUT|F2PY_INTENT_C;
  capi_chunk_tmp = array_from_pyobj(NPY_UBYTE,chunk_Dims,chunk_Rank,capi_chunk_intent,chunk_capi);
  if (capi_chunk_tmp == NULL) {
    PyObject *exc, *val, *tb;
    PyErr_Fetch(&exc, &val, &tb);
    PyErr_SetString(exc ? exc : h5chunk_error,"failed in converting 3rd argument `chunk' of h5chunk.h5_read_direct to C/Fortran array" );
    npy_PyErr_ChainExceptionsCause(exc, val, tb);
  } else {
    chunk = (unsigned_char *)(PyArray_DATA(capi_chunk_tmp));

  /* Processing variable chunk_length */
  chunk_length = len(chunk);
  CHECKSCALAR(len(chunk)>=chunk_length,"len(chunk)>=chunk_length","hidden chunk_length","h5_read_direct:chunk_length=%ld",chunk_length) {
/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
  Py_BEGIN_ALLOW_THREADS
  h5_read_direct_return_value = (*f2py_func)(dataset_id,frame,chunk,chunk_length);
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
    capi_buildvalue = Py_BuildValue("L",h5_read_direct_return_value);
/*closepyobjfrom*/
/*end of closepyobjfrom*/
    } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
  } /*CHECKSCALAR(len(chunk)>=chunk_length)*/
  /* End of cleaning variable chunk_length */
  if((PyObject *)capi_chunk_tmp!=chunk_capi) {
    Py_XDECREF(capi_chunk_tmp); }
  }  /*if (capi_chunk_tmp == NULL) ... else of chunk*/
  /* End of cleaning variable chunk */
  } /*if (f2py_success) of frame*/
  /* End of cleaning variable frame */
  } /*if (f2py_success) of dataset_id*/
  /* End of cleaning variable dataset_id */
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
/*************************** end of h5_read_direct ***************************/
/*eof body*/

/******************* See f2py2e/f90mod_rules.py: buildhooks *******************/
/*need_f90modhooks*/

/************** See f2py2e/rules.py: module_rules['modulebody'] **************/

/******************* See f2py2e/common_rules.py: buildhooks *******************/

/*need_commonhooks*/

/**************************** See f2py2e/rules.py ****************************/

static FortranDataDef f2py_routine_defs[] = {
  {"h5_chunk_size",-1,{{-1}},0,(char *)h5_chunk_size,(f2py_init_func)f2py_rout_h5chunk_h5_chunk_size,doc_f2py_rout_h5chunk_h5_chunk_size},
  {"h5_close_dset",-1,{{-1}},0,(char *)h5_close_dset,(f2py_init_func)f2py_rout_h5chunk_h5_close_dset,doc_f2py_rout_h5chunk_h5_close_dset},
  {"h5_close_file",-1,{{-1}},0,(char *)h5_close_file,(f2py_init_func)f2py_rout_h5chunk_h5_close_file,doc_f2py_rout_h5chunk_h5_close_file},
  {"h5_dsinfo",-1,{{-1}},0,(char *)h5_dsinfo,(f2py_init_func)f2py_rout_h5chunk_h5_dsinfo,doc_f2py_rout_h5chunk_h5_dsinfo},
  {"h5_open_dset",-1,{{-1}},0,(char *)h5_open_dset,(f2py_init_func)f2py_rout_h5chunk_h5_open_dset,doc_f2py_rout_h5chunk_h5_open_dset},
  {"h5_open_file",-1,{{-1}},0,(char *)h5_open_file,(f2py_init_func)f2py_rout_h5chunk_h5_open_file,doc_f2py_rout_h5chunk_h5_open_file},
  {"h5_read_direct",-1,{{-1}},0,(char *)h5_read_direct,(f2py_init_func)f2py_rout_h5chunk_h5_read_direct,doc_f2py_rout_h5chunk_h5_read_direct},

/*eof routine_defs*/
  {NULL}
};

static PyMethodDef f2py_module_methods[] = {

  {NULL,NULL}
};

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "h5chunk",
  NULL,
  -1,
  f2py_module_methods,
  NULL,
  NULL,
  NULL,
  NULL
};

PyMODINIT_FUNC PyInit_h5chunk(void) {
  int i;
  PyObject *m,*d, *s, *tmp;
  m = h5chunk_module = PyModule_Create(&moduledef);
  Py_SET_TYPE(&PyFortran_Type, &PyType_Type);
  import_array();
  if (PyErr_Occurred())
    {PyErr_SetString(PyExc_ImportError, "can't initialize module h5chunk (failed to import numpy)"); return m;}
  d = PyModule_GetDict(m);
  s = PyString_FromString("$Revision: $");
  PyDict_SetItemString(d, "__version__", s);
  Py_DECREF(s);
  s = PyUnicode_FromString(
    "This module 'h5chunk' is auto-generated with f2py (version:2).\nFunctions:\n"
"  h5_chunk_size = h5_chunk_size(dataset_id,frame)\n"
"  h5_close_dset = h5_close_dset(dset)\n"
"  h5_close_file = h5_close_file(hfile)\n"
"  h5_dsinfo = h5_dsinfo(dataset_id,dsinfo)\n"
"  h5_open_dset = h5_open_dset(h5file,dsetname)\n"
"  h5_open_file = h5_open_file(hname)\n"
"  h5_read_direct = h5_read_direct(dataset_id,frame,chunk)\n"
".");
  PyDict_SetItemString(d, "__doc__", s);
  Py_DECREF(s);
  h5chunk_error = PyErr_NewException ("h5chunk.error", NULL, NULL);
  /*
   * Store the error object inside the dict, so that it could get deallocated.
   * (in practice, this is a module, so it likely will not and cannot.)
   */
  PyDict_SetItemString(d, "_h5chunk_error", h5chunk_error);
  Py_DECREF(h5chunk_error);
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
    on_exit(f2py_report_on_exit,(void*)"h5chunk");
#endif
  return m;
}
#ifdef __cplusplus
}
#endif

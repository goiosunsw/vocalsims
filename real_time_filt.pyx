import numpy 
cimport numpy
cimport cython

def circ_filt(a,b,in_buff,out_buff,idx=0):
    """
    filters input and output signals from in_buff and out_buff
    using linear coefficients a (for output) and b (for input) 
    """
    return numpy.sum(b*in_buff) - numpy.sum(a[1:]*out_buff)

ctypedef numpy.double_t DTYPE_t 

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef DTYPE_t circ_filt_and_insert_2(numpy.ndarray[DTYPE_t,ndim=1] a,
                         numpy.ndarray[DTYPE_t,ndim=1] b,
                         numpy.ndarray[DTYPE_t,ndim=1] in_buff_orig,
                         numpy.ndarray[DTYPE_t,ndim=1] out_buff_orig,
                         DTYPE_t in_smpl,
                         int idx=0,
                         int n=1):
    """
    filters input and output signals from in_buff and out_buff
    using linear coefficients a (for output) and b (for input) 

    also inserts incoming and outgoing samples in double-length buffers
    """
    cdef DTYPE_t out

    in_buff_orig[idx] = in_smpl
    in_buff_orig[idx+n] = in_smpl
    in_buff = in_buff_orig[idx+n:idx+n-len(b):-1]
    out_buff = out_buff_orig[idx+n-1:idx+n-len(a):-1]
    out = sum(b*in_buff) - sum(a[1:]*out_buff)
    out /= a[0]
    out_buff_orig[idx] = out
    out_buff_orig[idx+n] = out
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef DTYPE_t circ_filt_and_insert(numpy.ndarray[DTYPE_t,ndim=1] a,
                         numpy.ndarray[DTYPE_t,ndim=1] b,
                         numpy.ndarray[DTYPE_t,ndim=1] in_buff,
                         numpy.ndarray[DTYPE_t,ndim=1] out_buff,
                         DTYPE_t in_smpl,
                         int idx=0,
                         int n=1):
    """
    filters input and output signals from in_buff and out_buff
    using linear coefficients a (for output) and b (for input) 

    also inserts incoming and outgoing samples in double-length buffers
    """
    cdef DTYPE_t out
    cdef int na = a.shape[0]
    cdef int nb = b.shape[0]
    cdef int ii 

    in_buff[idx] = in_smpl
    in_buff[idx+n] = in_smpl
    out = 0
    for ii in range(nb):
        out += b[ii]*in_buff[idx+n-ii]
    for ii in range(1,na):
        out -= a[ii]*out_buff[idx+n-ii]
    out /= a[0]
    out_buff[idx] = out
    out_buff[idx+n] = out
    return out

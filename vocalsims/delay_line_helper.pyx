import numpy 
cimport numpy
cimport cython

ctypedef numpy.double_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def delay_ticker(numpy.ndarray[DTYPE_t,ndim=1] buff, 
                           DTYPE_t in_smpl, 
                           int counter, 
                           int n, 
                           int delay):
    idx = counter % n
    buff[idx] = buff[idx+n] = in_smpl
    return buff[idx+n-delay]
import RealTimeFilters as rtf
from numpy.random import randn
from numpy import zeros, sqrt, sum
import scipy.signal as sig

plots = True

b,a=sig.butter(15,.1)
print(b)
print(a)

N=50000
x = randn(N)
y = zeros(N)

def test_fun(N=N):
    filt = rtf.RealTimeFilter(b,a)
    for ii in range(N):
        y[ii] = filt.tick(x[ii])
    
def test_fun_native(N=N):
    filt = rtf.RealTimeFilter(b,a)
    filt.set_native()
    for ii in range(N):
        y[ii] = filt.tick(x[ii])

import cProfile
print("*** Running cython ***")
cProfile.run("test_fun()")
y_cython = y
print("*** Running native ***")
cProfile.run("test_fun_native()")
y_native = y

print("Square sum of signals:")
print(sqrt(sum((y_native)**2/N)))

print("Difference between implementations:")
print(sqrt(sum((y_native-y_cython)**2/N)))

print("Difference to scipy.signal")
import scipy.signal as sig
y_scipy = sig.lfilter(b,a,x)
print(sqrt(sum((y_scipy-y_cython)**2/N)))


if plots:
    import matplotlib.pyplot as pl
    pl.plot(x)
    pl.plot(y_native)
    pl.plot(y_cython)
    pl.plot(y_scipy)
    pl.show()
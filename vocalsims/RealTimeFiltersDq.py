from collections import deque
import numpy as np

class RealTimeFilter(object):
    """
    Implements a LTI filter to be used in real time,
    one sample at a time
    
    Filter is defined in the finite difference form:

    a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]
                          - a[1]*y[n-1] - ... - a[N]*y[n-N]

    Create as:
        `filt = RealTimeFilter(b,a)`

        providing `b` and `a` as iterables

    Attributes
    ----------
    b: list with in-signal coefficients
    a: list with out-signal coefficients 

    Methods
    -------
 
    reset(): set delay lines to 0

    tick(x): insert new sample and output filtered sample

    set_coeffs(b,a): set coefficients without resetting dealy lines

    """

    def __init__(self,b=[1],a=[1]):
        """
        Create new filter with coefficients a and b

        `a` and `b` should be provided as lists
        """

        self.set_coeffs(b,a)
        self.init_delay_lines()
        self.counter = 0

    
    def init_delay_lines(self):
        self.in_buff = deque(self.n)
        self.out_buff = deque(self.n)

    def set_coeffs(self, b, a, reset=True):
        self.a = a
        self.b = b 
        if reset:
            na = len(self.a)
            nb = len(self.b)
            self.n = max(na,nb)
            self.init_delay_lines()

    def increment_counter(self):
        self.counter += 1

    def tick(self, in_smpl):
        """
        Input a new sample and output one sample
        """
        self.increment_counter()
        idx = self.counter % self.n
        out = self._cfilt(self.a, self.b, self.in_buff, self.out_buff, in_smpl, idx=idx, n=self.n)
        return out

    def next_output(self, in_smpl):
        """
        return next filter output without introducing a new input sample
        """
        idx = (self.counter + 1) % self.n
        in_buff = self.in_buff[idx+self.n-1:idx+self.n-len(self.b):-1]
        in_buff = np.concatenate(([in_smpl], in_buff))
        out_buff = self.out_buff[idx+self.n-1:idx+self.n-len(self.a):-1]
        #import pdb
        #pdb.set_trace()
        out = np.sum(self.b*in_buff) - sum(self.a[1:]*out_buff)
        out /= self.a[0]
        return out


    def reset(self, in_val=0, out_val=0):
        """
        Resets the circular buffers to 0
        """
        self.init_delay_lines()
        self.counter = 0
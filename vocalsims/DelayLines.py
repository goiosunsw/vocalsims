import numpy as np
from .delay_line_helper import delay_ticker

class IntegerDelayLine(object):
    """
    Implements a delay line for use in real time
    
    Create as:
        `dl = IntegerDelayLine(delay)`
        delay: (int)


    Methods
    -------
 
    reset(): set delay line to 0

    tick(x): insert new sample and output filtered sample

    """

    def __init__(self, delay=1):
        """
        Create new delay with an integer delay del 
        """

        self.n = (delay)+1
        self.delay = delay
        self.init_delay_line()
        self.counter = 0
    
    def init_delay_line(self):
        self.buff = np.zeros(self.n*2)

    def increment_counter(self):
        self.counter += 1

    def tick(self, in_smpl):
        """
        Input a new sample and output one sample
        """

        self.increment_counter()
        return delay_ticker(self.buff,in_smpl,self.counter,self.n,self.delay)

    def next_output(self):
        """
        return next filter output without introducing a new input sample
        """
        idx = (self.counter + 1) % self.n
        out = self.buff[idx+self.n-self.delay]
        return out

    def get_line(self):
        idx = self.counter % self.n
        return self.buff[idx+self.n:idx+self.n-self.delay:-1]

    def reset(self, in_val=0, out_val=0):
        """
        Resets the circular buffers to 0
        """
        self.init_delay_line()
        self.counter = 0

    def get_delay(self):
        return self.n-1


import numpy as np
from collections import deque

class IntegerDelayLine(object):
    """
    Implements a delay line for use in real time
    
    Create as:
        `dl = IntegerDelayLineDq(delay)`
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

        self.n = (delay)
        self.delay = delay
        self.init_delay_line()
        self.counter = 0
    
    def init_delay_line(self):
        self.buff = deque([0.0]*(self.n+1),maxlen=self.n+1) 

    def increment_counter(self):
        self.counter += 1

    def tick(self, in_smpl):
        """
        Input a new sample and output one sample
        """

        self.increment_counter()
        self.buff.append(in_smpl)
        return self.buff[0]

    def next_output(self):
        """
        return next filter output without introducing a new input sample
        """
        return self.buff[0]

    def get_line(self):
        return self.buff

    def reset(self, in_val=0, out_val=0):
        """
        Resets the circular buffers to 0
        """
        self.init_delay_line()
        self.counter = 0

    def get_delay(self):
        return self.n-1


import numpy as np

'''
Simple delay buffer in ring buffer design.
Buffer lenghts constitues delay with which elements are read.
'''

class DelayBuffer:
  
  def __init__(self, shape, dtype=float):
    # assuming last dimension is temporal
    self.data = np.zeros(shape, dtype=dtype)
    self.length = shape[-1]
    self.lengthAxis = len(shape)-1
    self.pointer = 0
    self.dtype = dtype

  def write(self, dataframe):
    self.data[..., self.pointer] = dataframe
    self.pointer = (self.pointer + 1) % self.length

  def read(self):
    return self.data[..., self.pointer]

  def clear(self):
    self.data = np.zeros(np.shape(self.data), dtype=self.dtype)
    self.pointer = 0
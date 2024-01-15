import numpy as np
import math

class Wiener:
    def __init__(self, time_length:int, n_realisations:int, dt:float, vector_dimension:int=3,seed:int=0):
        self.time_length = time_length
        self.n_realisations = n_realisations
        self.dt = dt
        self.vector_dimension = vector_dimension
        np.random.seed(seed)
        self.W = np.cumsum(np.random.normal(size=(time_length,vector_dimension,n_realisations),scale=math.sqrt(dt)),axis=0)
    
    def sample(self, dt:float, realisation:int=0):
        if dt < self.dt:
            raise ValueError("dt must be greater than the original dt")
        elif dt == self.dt:
            return self.W[:,:,realisation]
        else:
            return self.W[::int(dt/self.dt),:,realisation]

class Time():
    def __init__(self, t_end, n):
        self.t_end = t_end
        self.n = n
        self.t = np.linspace(0, t_end, n)
        self.dt = t_end/n

    def sample(self, dt):
        if dt < self.dt:
            raise ValueError("dt must be greater than the original dt")
        elif dt == self.dt:
            return self.t
        else:
            return self.t[::int(dt/self.dt)]
    
    
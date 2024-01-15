import numpy as np

class Vector_Model:
    def __init__(self,mu, alpha, p):
        self.mu = mu
        self.alpha = alpha
        self.p = p

    def f(self, t, x):
        return [self.mu*x[0], -(x[1]-x[2]), (x[1]-x[2])/self.alpha]

    def g(self, t, x):
        return [x[0]*x[1], self.p*x[1], 0]

    def g_prime(self, t, x):
        return [x[1],self.p,0]

    def euler_method(self, x_0, W,t):
        dt = t[1]-t[0] 
        self.dt = dt
        x = np.zeros((len(t),3))
        x[0,:] = x_0

        for time in range(len(t)-1):
            for i in range(3):
                x[time+1,i] = x[time,i] + self.f(time,x[time,:])[i]*dt + self.g(time,x[time,:])[i]*(W[time+1,i]-W[time,i])
        return x[:,0]

    def milstein_method(self, x_0, W,t):
        dt = t[1]-t[0]
        self.dt = dt
        x = np.zeros((len(t),3))
        x[0,:] = x_0

        for time in range(len(t)-1):
            for i in range(3):
                x[time+1,i] = x[time,i] + self.f(time,x[time,:])[i]*dt + self.g(time,x[time,:])[i]*(W[time+1,i]-W[time,i]) + 0.5*self.g(time,x[time,:])[i]*self.g_prime(time,x[time,:])[i]*((W[time+1,i]-W[time,i])**2 - dt)
        return x[:,0]

class Black_Scholes:
    def __init__(self, mu,sigma_0):
        self.mu = mu
        self.sigma_0 = sigma_0
    
    def f(self, t, x):
        return self.mu*x
    
    def g(self, t, x):
        return x*self.sigma_0
    
    def g_prime(self, t, x):
        return self.sigma_0
    
    def euler_method(self, x_0, W,t):
        dt = t[1]-t[0]
        self.dt = dt
        x = np.zeros(len(t))
        if type(x_0) == list:
            x[0] = x_0[0]
        else:
            x[0] = x_0

        for time in range(len(t)-1):
            x[time+1] = x[time] + self.f(time,x[time])*dt + self.g(time,x[time])*(W[time+1,0]-W[time,0])
        return x
    
    def milstein_method(self, x_0, W,t):
        dt = t[1]-t[0]
        self.dt = dt
        x = np.zeros(len(t))
        if type(x_0) == list:
            x[0] = x_0[0]
        else:
            x[0] = x_0

        for time in range(len(t)-1):
            x[time+1] = x[time] + self.f(time,x[time])*dt + self.g(time,x[time])*(W[time+1,0]-W[time,0]) + 0.5*self.g(time,x[time])*self.g_prime(time,x[time])*((W[time+1,0]-W[time,0])**2 - dt)
        return x
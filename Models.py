import numpy as np
import matplotlib.pyplot as plt

class Vector_Model:
    def __init__(self,mu, alpha, p):
        self.mu = mu
        self.alpha = alpha
        self.p = p
        self.name = "Vector Model"

    def f(self, t, x):
        return np.array([self.mu*x[0], -(x[1]-x[2]), (x[1]-x[2])/self.alpha])

    def g(self, t, x):
        return np.array([x[0]*x[1], self.p*x[1], 0])

    def g_prime(self, t, x):
        return np.array([x[1],self.p,0])

    def euler_method(self, x_0, W,t):
        """
        Implements the Euler method for solving the Vector Equation.

        Parameters:
        - x_0: Initial values of the state variables (length=(n_state_variables)).
        - W: Array of Wiener Process (shape=(n_timesteps,n_state_variables)).
        - t: Array of time points (length=(n_timesteps)).

        Returns:
        - x[:,0]: Array of first state variable values at each time point (shape=(n_timesteps)).
        """
        dt = t[1]-t[0] 
        self.dt = dt
        x = np.zeros((len(t),3))
        x[0,:] = x_0

        for time in range(len(t)-1):
            x[time+1,:] = x[time,:] + self.f(time,x[time,:])*dt + self.g(time,x[time,:])*(W[time+1,:]-W[time,:])
        return x[:,0]

    def milstein_method(self, x_0, W, t):
        """
        Implements the Milstein method for solving the Vector Equation.

        Parameters:
        - x_0: Initial values of the state variables (length=(n_state_variables)).
        - W: Array of Wiener Process (shape=(n_timesteps,n_state_variables)).
        - t: Array of time points (length=(n_timesteps)).

        Returns:
        - x[:,0]: Array of first state variable values at each time point (shape=(n_timesteps)).
        """
        dt = t[1] - t[0]
        self.dt = dt
        x = np.zeros((len(t), 3))
        x[0, :] = x_0

        for time in range(len(t) - 1):
                x[time + 1, :] = x[time, :] + self.f(time, x[time, :]) * dt + self.g(time, x[time, :]) * (
                            W[time + 1, :] - W[time, :]) + 0.5 * self.g(time, x[time, :]) * self.g_prime(time, x[time, :]) * (
                                             (W[time + 1, :] - W[time, :]) ** 2 - dt)
        return x[:, 0]

class Black_Scholes:
    def __init__(self, mu,sigma_0):
        self.mu = mu
        self.sigma_0 = sigma_0
        self.name = "Black Scholes"
    
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
    
if __name__ == "__main__":
    """Example of how to use the Vector_Model and the Black Scholes classes."""
    
    #Parameters
    mu = 0.01
    alpha = 0.5
    p = 0

    #Initial Conditions
    sigma_0 = 0.2
    xi_0 = 0.1
    S_0 = 50
    X_0 = [S_0,xi_0,sigma_0]

    #initialize the model
    model = Black_Scholes(mu, sigma_0) #Vector_Model(mu, alpha, p)

    #Time
    t_end = 1
    dt = 10e-2
    t = np.arange(0,t_end+dt,dt)
    print(t.shape)

    #Wiener Process
    W = np.cumsum(np.random.normal(size=(len(t)+1,3),scale=np.sqrt(dt)),axis=0)

    #Simulate the SDE using the Euler and Milstein methods
    x_euler = model.euler_method(X_0, W,t)
    x_milstein = model.milstein_method(X_0, W,t)

    #Plot the results
    plt.plot(t,x_milstein,label="milstein")
    plt.plot(t,x_euler,label="euler")
    plt.grid()
    plt.xlabel("t")
    plt.ylabel("X")
    plt.legend()
    plt.title(model.name)
    plt.show()
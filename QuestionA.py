from Models import Black_Scholes, Vector_Model
import numpy as np
import matplotlib.pyplot as plt

def plot_tracks(alpha, p,t, W):
    #initialize the model
    model = Vector_Model(mu, alpha, p)

    #Simulate the SDE using the Euler and Milstein methods
    x_euler = model.euler_method(X_0, W,t)

    #Plot the results
    plt.plot(t,x_euler,label=f"alpha={alpha}, p={p}")


if __name__ == "__main__":
    #Parameters
    mu = 0.01
    dt = 10e-3
    t_end = 1
    t = np.arange(0,t_end+dt,dt)

    #Initial Conditions
    sigma_0 = 0.2
    xi_0 = 0.1
    S_0 = 50
    X_0 = [S_0,xi_0,sigma_0]

    #Wiener Process
    np.random.seed(0)
    W = np.cumsum(np.random.normal(size=(len(t)+1,3),scale=np.sqrt(dt)),axis=0)
    
    for alpha in [0.5,1,2]:
        for p in [0,0.5,1]:
            plot_tracks(alpha, p,t,W)
    
    plt.grid()
    plt.xlabel("t")
    plt.ylabel("X")
    plt.legend()
    plt.show()
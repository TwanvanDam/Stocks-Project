from Models import Black_Scholes, Vector_Model
from WienerProcesses import Wiener, Time
import numpy as np
import math
import matplotlib.pyplot as plt

def h(x):
    return x*x

def Weak_convergence(n_list, n_realisations, model, X_0, W, t_end=1):
    t = Time(t_end, n_list[0])
    euler_error = []
    milstein_error = []

    for size in n_list:
        t_sample = t.sample(t_end/size)
        x_euler_avg = np.zeros(len(t_sample))
        x_milstein_avg = np.zeros(len(t_sample))

        for i in range(n_realisations):
            W_sample = W.sample(t_end/size, i)
            x_euler = model.euler_method(X_0, W_sample, t_sample)
            x_euler_avg += x_euler
            x_milstein = model.milstein_method(X_0, W_sample, t_sample)
            x_milstein_avg += x_milstein
        
        x_euler_avg /= n_realisations
        x_milstein_avg /= n_realisations
        if size == n[0]:
            ground_truth_euler = h(x_euler_avg)
            ground_truth_milstein = h(x_milstein_avg)

        else:
            euler_error.append(np.mean(np.abs(h(x_euler_avg)-ground_truth_euler[::int((t_end/size)/(t_end/n_list[0]))])))
            milstein_error.append(np.mean(np.abs(h(x_milstein_avg)-ground_truth_milstein[::int((t_end/size)/(t_end/n_list[0]))])))
    
    plt.loglog(dt_list[1:], euler_error, label="Euler")
    plt.loglog(dt_list[1:], milstein_error, label="Milstein")
    plt.grid()
    plt.xlabel("dt")
    plt.ylabel("Error")
    plt.title("Weak Convergence x^2")
    plt.legend()
    plt.savefig("./Plots/Weak_convergence.png",dpi=300)
    plt.show()
    return


if __name__ == "__main__":
    #Parameters
    mu = 0.01
    alpha = 0.5
    p = 0

    #Initial Conditions
    sigma_0 = 0.2
    xi_0 = 0.1
    S_0 = 50
    X_0 = [S_0,xi_0,sigma_0]
    
    model =  Black_Scholes(mu, sigma_0) #Vector_Model(mu, alpha, p)

    #Time parameters
    t_end = 1
    order = 6
    n_realisations = 10
    n = [int(10**i) for i in range(order,0,-1)]
    dt_list = [1/n[i] for i in range(len(n))]
    
    W = Wiener(int(n[0]), n_realisations, dt_list[0], seed=0)

    Weak_convergence(n, n_realisations, model, X_0, W, t_end)

import numpy as np
import math
import matplotlib.pyplot as plt
from Models import Black_Scholes, Vector_Model
from WienerProcesses import Wiener

def Plot_different_timesteps(n_list, model, X_0, W, t_end=1, Euler=True):
    dt_list = [t_end/n_list[i] for i in range(len(n_list))]
    for size in n_list:
        dt = t_end/size
        W_sample = W.sample(dt)
        t = np.linspace(0, t_end, int(size))
        
        if Euler==True:
            x = model.euler_method(X_0, W_sample,t)
        else:
            x = model.milstein_method(X_0, W_sample,t)
        plt.plot(t,x,label=f"dt={t_end/size}")
    plt.xlabel("t")
    plt.ylabel("X")
    plt.title("Different timesteps")
    plt.grid()
    plt.legend()
    plt.savefig("./Plots/Different_timesteps.png",dpi=300)
    plt.show()
    return
        
def Strong_convergence(n_list, n_realisations, model, X_0, W, t_end=1):
    dt_list = [t_end/n_list[i] for i in range(len(n_list))]

    plot_list_euler = np.zeros((len(n)-1,n_realisations))
    plot_list_milstein = np.zeros((len(n)-1,n_realisations))
    
    for j in range(n_realisations):
        print(f"{j+1}/{n_realisations}")
        for size in n_list:
            dt = t_end/size
            W_sample = W.sample(dt,j)
            t = np.linspace(0, t_end, int(size))
            
            x_euler = model.euler_method(X_0, W_sample,t)
            x_milstein = model.milstein_method(X_0, W_sample,t)
            #plt.plot(t,x_milstein)

            if size == n_list[0]:
                ground_truth_euler = x_euler
                ground_truth_milstein = x_milstein
            else:
                error_euler = np.mean(np.abs(ground_truth_euler[::int(n[0]/size)] - x_euler))
                plot_list_euler[n.index(size)-1,j] = error_euler

                error_milstein = np.mean(np.abs(ground_truth_milstein[::int(n[0]/size)] - x_milstein))
                plot_list_milstein[n.index(size)-1,j] = error_milstein
    plt.loglog(dt_list[1:],np.mean(plot_list_euler,axis=1),'.-',label="euler")
    plt.loglog(dt_list[1:],np.mean(plot_list_milstein,axis=1),'.-',label="milstein")
    plt.xlabel("dt")
    plt.ylabel("Error")
    plt.title("Strong Convergence")
    plt.grid()
    plt.legend()
    plt.savefig("./Plots/Strong_Convergence.png",dpi=300)
    plt.show()


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

    model = Vector_Model(mu, alpha, p) #Black_Scholes(mu, sigma_0)
    #Time
    order = 6
    n = [int(10**i) for i in range(order,0,-1)]
    dt_list = [1/n[i] for i in range(len(n))]
    print("Using these timesteps for the strong convergence test:")
    print(dt_list)
    
    t_end = 1
    dt = dt_list[0]

    n_realisations = 2
    #ensure all methods use the same random numbers to ensure fair comparison
    W = Wiener(int(n[0]), n_realisations, dt,seed=0)

    #Plot_different_timesteps(n, model, X_0, W, t_end=t_end, Euler=True)
    Strong_convergence(n, n_realisations, model, X_0, W, t_end=t_end)

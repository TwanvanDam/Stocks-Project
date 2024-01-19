import numpy as np
import matplotlib.pyplot as plt
from Models import Black_Scholes, Vector_Model
import alive_progress

def Plot_different_timesteps(n_list, model, X_0, W, t_end=1, Euler=True):
    for size in n_list:
        sampling_rate = int(n_list[0]/size)
        W_sample = W[::sampling_rate,:,0]
        t = np.linspace(0, t_end, int(size)+1)
        
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
    plt.savefig("./Plots/Different_timesteps_" + model.name +".png",dpi=300)
    plt.show()
    return
        
def Strong_convergence(n_list, n_realisations, model, X_0, W, t_end=1):
    dt_list = [t_end/n_list[i] for i in range(len(n_list))]

    plot_list_euler = np.zeros((len(n)-1,n_realisations))
    plot_list_milstein = np.zeros((len(n)-1,n_realisations))
    
    with alive_progress.alive_bar(n_realisations) as bar:
        for j in range(n_realisations):
            for n_timesteps in n_list:
                sampling_rate = int(n_list[0]/n_timesteps)
                W_sample = W[::sampling_rate,:,j]
                t = np.linspace(0, t_end, n_timesteps+1)
                
                x_euler = model.euler_method(X_0, W_sample,t)
                x_milstein = model.milstein_method(X_0, W_sample,t)

                if n_timesteps == n_list[0]:
                    ground_truth_euler = x_euler
                    ground_truth_milstein = x_milstein
                else:
                    error_euler = np.abs(ground_truth_euler[-1] - x_euler[-1])
                    plot_list_euler[n.index(n_timesteps)-1,j] = error_euler

                    error_milstein = np.abs(ground_truth_milstein[-1] - x_milstein[-1])
                    plot_list_milstein[n.index(n_timesteps)-1,j] = error_milstein
            bar()
    starting_point_slope05 = np.mean(plot_list_euler,axis=1)[0]/((dt_list[1])**0.5)
    print((dt_list[1])**0.5)
    print(np.mean(plot_list_euler,axis=1)[0])
    print(starting_point_slope05)
    starting_point_slope1 = np.mean(plot_list_milstein,axis=1)[0]/dt_list[1]
    
    plt.loglog(dt_list[1:],np.mean(plot_list_euler,axis=1),'.-',label="Euler-Maruyama")
    plt.loglog(dt_list[1:],np.mean(plot_list_milstein,axis=1),'.-',label="Milstein")
    plt.plot(dt_list[1:],np.array(dt_list[1:])/starting_point_slope1,'k--',label="slope=1")
    plt.plot(dt_list[1:],np.power(dt_list[1:],0.5)*starting_point_slope05,'k-.',label="slope=0.5")
    plt.xlabel("dt")
    plt.ylabel("Error")
    plt.title("Strong Convergence")
    plt.grid()
    plt.legend()
    plt.savefig("./Plots/Strong_Convergence_" + model.name +".png",dpi=300)
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

    model = Vector_Model(mu, alpha, p)  #Black_Scholes(mu, sigma_0)  
    #Time
    order = 6
    t_end = 1

    n = [int(10**i) for i in range(order,0,-1)]
    dt_list = [t_end/n[i] for i in range(len(n))]
    
    print("Using these timesteps for the strong convergence test:")
    print(dt_list)

    n_realisations = 5
    #ensure all methods use the same random numbers to ensure fair comparison
    np.random.seed(0)
    W = np.cumsum(np.random.normal(size=(n[0]+1,3,n_realisations),scale=np.sqrt(dt_list[0])),axis=0)

    #Plot_different_timesteps(n, model, X_0, W, t_end=t_end, Euler=True)
    Strong_convergence(n, n_realisations, model, X_0, W, t_end=t_end)

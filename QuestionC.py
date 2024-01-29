import numpy as np
import matplotlib.pyplot as plt
from Models import Black_Scholes
        
def Strong_convergence_BS(n_list:list, n_realisations:int, model, X_0:float, W:np.ndarray, t_end=1):
    """
    - n_list : list of timesteps to use
    - n_realisations : number of realisations
    - model : model with embedded methods and parameters
    - X_0 : initial conditions
    - W : Wiener process has shape (n_list[0],n_realisations)
    """
    
    dt_list = [t_end/n_list[i] for i in range(len(n_list))]

    plot_list_euler = np.zeros((len(n),n_realisations))
    plot_list_milstein = np.zeros((len(n),n_realisations))
    
    for j in range(n_realisations):
        print(f"Realisation {j+1}/{n_realisations}")
        for n_timesteps in n_list:
            #sample a smaller Wiener Process
            sampling_rate = int(n_list[0]/n_timesteps)
            W_sample = W[::sampling_rate,j]
            
            #generate time array
            t = np.linspace(0, t_end, n_timesteps+1)
            
            #compute the solutions using euler and milstein method
            x_euler = model.euler_method(X_0, W_sample,t)
            x_milstein = model.milstein_method(X_0, W_sample,t)

            #compute the exact solution
            x_exact = model.exact_solution(X_0, W_sample,t)
            
            #compute the error
            error_euler = np.abs(x_exact[-1] - x_euler[-1])
            error_milstein = np.abs(x_exact[-1] - x_milstein[-1])
            
            #plot list has size (len(n),n_realisations) and stores the error for every timestep for every realization
            plot_list_euler[n.index(n_timesteps),j] = error_euler
            plot_list_milstein[n.index(n_timesteps),j] = error_milstein
    
    #used to allign the lines to determine the slope
    starting_point_slope05 = np.mean(plot_list_euler,axis=1)[0]/((dt_list[0])**0.5)
    starting_point_slope1 = np.mean(plot_list_milstein,axis=1)[0]/dt_list[0]
    
    #plot the mean of the error for every timestep
    plt.loglog(dt_list,np.mean(plot_list_euler,axis=1),'.-',label="Euler-Maruyama")
    plt.loglog(dt_list,np.mean(plot_list_milstein,axis=1),'.-',label="Milstein")
    
    #Plot the lines with slopes of 0.5 and 1
    plt.plot(dt_list,np.array(dt_list)*starting_point_slope1,'k--',label=r"$\Delta t$")
    plt.plot(dt_list,np.power(dt_list,0.5)*starting_point_slope05,'k-.',label=r"$\Delta t^{0.5}$")
    
    plt.xlabel("dt")
    plt.ylabel("Error")
    plt.title("Strong Convergence")
    plt.grid()
    plt.legend()
    plt.savefig("./Plots/StrongeConvergence_" + model.name +".png",dpi=300)
    plt.show()
    return 

if __name__ == "__main__":  
    #Parameters
    mu = 0.1
    sigma_0 = 0.2
    
    #Initial Conditions
    S_0 = 50

    #initialize the model
    model = Black_Scholes(mu, sigma_0)
    
    order = 5 #How fine are the timesteps. order = 5 => smallest dt = 10^-5
    t_end = 1
    
    n_realisations = 500
    np.random.seed(0)

    #list with the number of timesteps to use
    n = [int(10**i) for i in range(order,0,-1)]

    #list with the dt's to use
    dt_list = [t_end/n[i] for i in range(len(n))]
    
    print("Using these timesteps for the strong convergence test:")
    print(dt_list)

    #Wiener Process initialized with the smallest dt. We can sample from this W to obtain a wiener process for bigger dt's
    #Wiener Process should start at zero
    W = np.cumsum(np.vstack((np.zeros((1,n_realisations)),np.random.normal(size=(n[0],n_realisations),scale=np.sqrt(dt_list[0])))),axis=0)
    
    Strong_convergence_BS(n, n_realisations, model, S_0, W, t_end=t_end)

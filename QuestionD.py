from Models import Black_Scholes, Vector_Model
import numpy as np
import matplotlib.pyplot as plt
import alive_progress

def h(x,exponent):
    return np.power(x,exponent)

def Weak_convergence(n_list, n_realisations, model, X_0, W, t_end=1,exponent=2):
    """
    Computes the weak convergence of a numerical method for solving stochastic differential equations.

    Parameters:
    - n_list (list): List of number of time steps for each iteration.
    - n_realisations (int): Number of realisations.
    - model (object): Object representing the SDE model.
    - X_0 (float): Initial value.
    - W (ndarray): Array of Brownian motion values.
    - t_end (float): End time. Default is 1.
    - exponent (float): Exponent for computing the error. Default is 2.

    Returns:
    None
    """
    total_iterations = sum(n_list)*n_realisations           #for the progress bar
    dt_list = [t_end/n_list[i] for i in range(len(n_list))] #list of time steps
    euler_error = []                                        #list of errors for Euler
    milstein_error = []                                     #list of errors for Milstein     
    
    with alive_progress.alive_bar(total_iterations) as bar:
        for n_timesteps in n_list:
            sampling_rate = int(n_list[0]/n_timesteps) #sampling rate for the Wiener Process
            t = np.linspace(0, t_end, n_timesteps+1)   #time points
            x_euler_avg = np.zeros(len(t))             #average of the realisations
            x_milstein_avg = np.zeros(len(t))

            for i in range(n_realisations):
                W_sample = W[::sampling_rate,:,i]
                x_euler = model.euler_method(X_0, W_sample, t)          #compute the realisations
                x_euler_avg += x_euler
                x_milstein = model.milstein_method(X_0, W_sample, t)    #compute the realisations
                x_milstein_avg += x_milstein
                bar(n_timesteps)                                        #update the progress bar
            
            x_euler_avg /= n_realisations
            x_milstein_avg /= n_realisations

            if n_timesteps == n_list[0]: #save the ground truth
                ground_truth_euler = h(x_euler_avg,exponent)
                ground_truth_milstein = h(x_milstein_avg,exponent)

            else: #compute the error
                euler_error.append(np.abs(h(x_euler_avg[-1],exponent)-ground_truth_euler[-1]))
                milstein_error.append(np.abs(h(x_milstein_avg[-1],exponent)-ground_truth_milstein[-1]))

    
    """Make the plots"""
    correction_1 = euler_error[0]/(dt_list[1]**0.5) #used for aligning the slopes
    correction_2 = milstein_error[0]/dt_list[1]

    plt.plot(dt_list[1:], euler_error, label="Euler")
    plt.plot(dt_list[1:], milstein_error, label="Milstein")
    plt.plot(dt_list[1:],np.array(dt_list[1:])*correction_2,'k--',label="slope=1")
    plt.plot(dt_list[1:],np.power(dt_list[1:],0.5)*correction_1,'k-.',label="slope=0.5")
    plt.loglog()
    plt.grid()
    plt.xlabel("dt")
    plt.ylabel("Error")
    plt.title(f"Weak Convergence x^{exponent} for {model.name}")
    plt.legend()
    plt.savefig("./Plots/Weak_convergence_" + model.name +".png",dpi=300)
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

    #initialize the model
    model = Vector_Model(mu, alpha, p)  #Black_Scholes(mu, sigma_0)  
    
    #Time
    order = 5
    t_end = 1

    #set the number of timesteps 
    n = [int(10**i) for i in range(order,0,-1)]
    dt_list = [t_end/n[i] for i in range(len(n))]
    
    print("Using these timesteps for the weak convergence test:")
    print(dt_list)

    n_realisations = 5
    
    np.random.seed(0)
    #Wiener Process
    W = np.cumsum(np.random.normal(size=(n[0]+1,3,n_realisations),scale=np.sqrt(dt_list[0])),axis=0)

    Weak_convergence(n, n_realisations, model, X_0, W, t_end,exponent=2)

from Models import Black_Scholes
import numpy as np
import matplotlib.pyplot as plt
import alive_progress

def h(x,exponent):
    return np.power(x,exponent)

def Weak_convergence_BS(n_list:list, n_realisations:int, model, X_0:float, W, t_end=1,exponent=[1]):
    """
    Computes the weak convergence of a numerical method for solving stochastic differential equations.

    Parameters:
    - n_list: List of number of time steps for each iteration.
    - n_realisations : Number of realisations.
    - model : Object representing the SDE model.
    - X_0 : Initial value.
    - W : Array of the Wiener Process.
    - t_end : End time. Default is 1.
    - exponent : Exponent for computing the error. Default is 2.
    """
    
    dt_list = [t_end/n_list[i] for i in range(len(n_list))] #list of time steps
    euler_error = np.zeros((len(n_list),len(exponent)))                                        #list of errors for Euler
    milstein_error = np.zeros((len(n_list),len(exponent)))                                     #list of errors for Milstein     
    
    for n_timesteps in n_list:
        sampling_rate = int(n_list[0]/n_timesteps) #sampling rate for the Wiener Process
        t = np.linspace(0, t_end, n_timesteps+1)
        
        x_euler_avg = np.zeros((len(exponent)))        #array to store average of every method
        x_milstein_avg = np.zeros((len(exponent)))  
        x_exact_avg = np.zeros((len(exponent)))  

        for i in range(n_realisations):
            if n_timesteps == n_list[0]:
                print(f"Realisation {i+1}/{n_realisations}")
            W_sample = W[::sampling_rate,i]
            
            #Calculate realizations
            x_euler = model.euler_method(X_0, W_sample, t)    
            x_milstein = model.milstein_method(X_0, W_sample, t)  
            x_exact = model.exact_solution(X_0,W_sample,t)
            
            #Store the average for every h function
            for j in range(len(exponent)):
                x_euler_avg[j] += h(x_euler[-1],exponent[j])/n_realisations
                x_milstein_avg[j] += h(x_milstein[-1],exponent[j])/n_realisations
                x_exact_avg[j] += h(x_exact[-1],exponent[j])/n_realisations

        #Determine the error by comparing the values to the exact values
        euler_error[n_list.index(n_timesteps),:] = np.abs(x_euler_avg - x_exact_avg)
        milstein_error[n_list.index(n_timesteps),:] = np.abs(x_milstein_avg-x_exact_avg)

    
    """Make the plots"""
    for j in range(len(exponent)):
        #used for aligning the slopes
        correction_1 = euler_error[0,j]/(dt_list[0]**0.5) 
        correction_2 = milstein_error[0,j]/dt_list[0]
        correction_3 = euler_error[0,j]/dt_list[0]

        #Plot the errors
        plt.plot(dt_list, euler_error[:,j], label="Euler")
        plt.plot(dt_list, milstein_error[:,j], label="Milstein")
        
        #plot the slopes
        plt.plot(dt_list,np.array(dt_list)*correction_2,'k--',label=r"$\Delta t$")
        plt.plot(dt_list,np.power(dt_list,0.5)*correction_1,'k-.',label=r"$\Delta t^{0.5}$")
        plt.plot(dt_list,np.array(dt_list)*correction_3,'k--')
        
        plt.loglog()
        plt.grid()
        plt.xlabel("dt")
        plt.ylabel("Error")
        plt.title(f"Weak Convergence x^{exponent[j]}")
        plt.legend()
        plt.savefig(f"./Plots/Weak_convergence_{exponent[j]}_{model.name}.png",dpi=300)
        plt.show()
    return


if __name__ == "__main__":
    #Parameters
    mu = 0.1
    sigma_0 = 0.2

    S_0 = 50

    #initialize the model
    model = Black_Scholes(mu, sigma_0)
    
    #Time
    order = 5
    t_end = 1

    #set the number of timesteps 
    n = [int(10**i) for i in range(order,0,-1)]
    dt_list = [t_end/n[i] for i in range(len(n))]
    
    print("Using these timesteps for the weak convergence test:")
    print(dt_list)

    n_realisations = 500
    
    np.random.seed(0)
    #Wiener Process
    W = np.cumsum(np.vstack((np.zeros((1,n_realisations)),np.random.normal(size=(n[0],n_realisations),scale=np.sqrt(dt_list[0])))),axis=0)
    Weak_convergence_BS(n, n_realisations, model, S_0, W, t_end,exponent=[1,2,3])

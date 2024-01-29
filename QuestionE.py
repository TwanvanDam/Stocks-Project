import numpy as np
import matplotlib.pyplot as plt
from Models import Vector_Model
        
def Strong_convergence_Vector(n_list:list, n_realisations:int, model, X_0:list, W:np.ndarray, t_end=1):
    """
    - n_list : list of timesteps to use
    - n_realisations : number of realisations
    - model : model with embedded methods and parameters
    - X_0 : initial conditions
    - W : Wiener process has shape (n_list[0],3,n_realisations)
    """
    #list of the different values used for dt
    dt_list = [t_end/n_list[i] for i in range(len(n_list))]

    #Stores the error for every value all the different timesteps and different realizations
    plot_list_euler = np.zeros((len(n),n_realisations))
    plot_list_milstein = np.zeros((len(n),n_realisations))
    
    for j in range(n_realisations):
        print(f"Realisation {j+1}/{n_realisations}")
        t = np.linspace(0,t_end,n_list[0]+1)
        
        #Exact solution is obtained by using the biggest number of timesteps
        x_exact_euler = model.euler_method(X_0,W[:,:,j], t)
        x_exact_milstein = model.milstein_method(X_0,W[:,:,j], t)
        
        #loop over remaining timesteps
        for n_timesteps in n_list[1:]:
            #sample a smaller Wiener Process
            sampling_rate = int(n_list[0]/n_timesteps)
            W_sample = W[::sampling_rate,:,j]
            
            #generate time array
            t = np.linspace(0, t_end, n_timesteps+1)
            
            #compute the solutions using euler and milstein method
            x_euler = model.euler_method(X_0, W_sample,t)
            x_milstein = model.milstein_method(X_0, W_sample,t)

            #compute the error
            error_euler = np.abs(x_exact_euler[-1] - x_euler[-1])
            error_milstein = np.abs(x_exact_milstein[-1] - x_milstein[-1])
            
            #plot list has size (len(n),n_realisations) and stores the error for every timestep for every realization
            plot_list_euler[n.index(n_timesteps),j] = error_euler
            plot_list_milstein[n.index(n_timesteps),j] = error_milstein
    
        #used to allign the lines to determine the slope
    starting_point_slope1 = np.mean(plot_list_euler,axis=1)[1]/((dt_list[1])**0.5)

    #Calculate and plot the mean of the error for every timestep
    plt.loglog(dt_list[1:],np.mean(plot_list_euler,axis=1)[1:],'.-',label="Euler-Maruyama")
    plt.loglog(dt_list[1:],np.mean(plot_list_milstein,axis=1)[1:],'.-',label="Milstein")
    
    #Plot the lines with slopes of 0.5
    plt.plot(dt_list[1:],np.power(dt_list[1:],0.5)*starting_point_slope1,'k-.',label=r"$\Delta t^{0.5}$")
    
    plt.xlabel("dt")
    plt.ylabel("Error")
    plt.title("Strong Convergence")
    plt.grid()
    plt.legend()
    plt.savefig(f"./Plots/P={p}Alpha={alpha}/StrongConvergence{n_realisations}" + model.name +".png",dpi=300)
    plt.show()
    return 

def h(x,power):
    return np.power(x,power)

def Weak_convergence_Vector(n_list:list, n_realisations:int, model, X_0:float, W, t_end=1,exponent:list=[1]):
    """
    Computes the weak convergence of a numerical method for solving stochastic differential equations.

    Parameters:
    - n_list: List of number of time steps for each iteration.
    - n_realisations : Number of realisations.
    - model : Object representing the SDE model.
    - X_0 : Initial value.
    - W : Array of the Wiener Process has size (n_list[0],3,n_realizations).
    - t_end : End time. Default is 1.
    - exponent : Exponent for computing the error. Default is 1.
    """
    
    #list of time steps, the first will be used as exact solution
    dt_list = [t_end/n_list[i] for i in range(len(n_list))] 
    
    #list to store the errors
    euler_error = np.zeros((len(n_list),len(exponent)))                                        #list of errors for Euler
    milstein_error = np.zeros((len(n_list),len(exponent)))                                     #list of errors for Milstein     
    
    #list to store the average of h(solution)
    x_exact_euler_avg = np.zeros((n_realisations,len(exponent)))  
    x_exact_milstein_avg = np.zeros((n_realisations,len(exponent)))  

    t =  np.linspace(0, t_end, n_list[0]+1)
    
    #First calculate all "exact" solutions
    for i in range(n_realisations):
        print(f"Calculating {i+1}/{n_realisations} exact solutions")
        W_sample = W[:,:,i]
        
        x_exact_euler = model.euler_method(X_0,W_sample,t)[-1]
        x_exact_milstein = model.milstein_method(X_0,W_sample,t)[-1]
        #print(x_exact_euler,i)

        for j in range(len(exponent)):
            x_exact_euler_avg[i,j] = h(x_exact_euler,exponent[j])
            x_exact_milstein_avg[i,j] = h(x_exact_milstein,exponent[j])

    for n_timesteps in n_list[1:]:
        #sampling rate for the Wiener Process
        sampling_rate = int(n_list[0]/n_timesteps) 
        t = np.linspace(0, t_end, n_timesteps+1)
        
        #array to store average of the realisations for this n_timesteps for the different h functions
        x_euler_avg = np.zeros((n_realisations,len(exponent)))        
        x_milstein_avg = np.zeros((n_realisations,len(exponent)))  
        
        for i in range(n_realisations):
            W_sample = W[::sampling_rate,:,i]
            
            x_euler = model.euler_method(X_0, W_sample, t)[-1]    
            x_milstein = model.milstein_method(X_0, W_sample, t)[-1]
            
            for j in range(len(exponent)):
                #if j == 0:
                 #   print(x_euler-x_exact_euler_avg[i,0])
                x_euler_avg[i,j] = h(x_euler,exponent[j])
                x_milstein_avg[i,j] = h(x_milstein,exponent[j])
        #Calculate the errors for this timestep
        euler_error[n_list.index(n_timesteps),:] = np.abs(np.mean(x_euler_avg,axis=0) - np.mean(x_exact_euler_avg,axis=0))
        milstein_error[n_list.index(n_timesteps),:] = np.abs(np.mean(x_milstein_avg,axis=0)-np.mean(x_exact_milstein_avg,axis=0))
    
    """Make the plots"""
    for j in range(len(exponent)):
        #used for aligning the slopes
        correction_1 = euler_error[1,j]/(dt_list[1]**0.5) 
        correction_2 = milstein_error[1,j]/(dt_list[1]**0.5)

        plt.plot(dt_list[1:], euler_error[1:,j], label="Euler")
        plt.plot(dt_list[1:], milstein_error[1:,j], label="Milstein")
        plt.plot(dt_list[1:],np.power(dt_list[1:],0.5)*correction_1,'k-.',label=r"$\Delta t^{0.5}$")
        plt.plot(dt_list[1:],np.power(dt_list[1:],0.5)*correction_2,'k-.')
        plt.loglog()
        plt.grid()
        plt.xlabel("dt")
        plt.ylabel("Error")
        plt.title(f"Weak Convergence x^{exponent[j]}")
        plt.legend()
        plt.savefig(f"./Plots/P={p}Alpha={alpha}/Weak_convergence_{exponent[j]}_{n_realisations}_{model.name}.png",dpi=300)
        plt.show()
    return


if __name__ == "__main__":  
    #Parameters
    mu = 0.10
    p = 0.5
    alpha = 1.2
    
    #Initial Conditions
    S_0 = 50
    sigma_0 = 0.20
    xi_0 = 0.20

    X_0 = [S_0,sigma_0,xi_0]

    #initialize the model
    model = Vector_Model(mu,alpha,p)
    
    order = 5 #How fine are the timesteps. order = 5 => smallest dt = 10^-5
    t_end = 1
    
    n_realisations = 100
    np.random.seed(0)

    #list with the number of timesteps to use
    n = [int(10**i) for i in range(order+1,0,-1)]

    #list with the dt's to use
    dt_list = [t_end/n[i] for i in range(len(n))]
    
    print(f"The exact solution will be approximated by {dt_list[0]}")
    print(f"Using these timesteps for the strong convergence test: {dt_list[1:]}")

    #Wiener Process initialized with the smallest dt. We can sample from this W to obtain a wiener process for bigger dt's
    #Wiener Process should start at zero
    W = np.cumsum(np.vstack((np.zeros((1,3,n_realisations)),np.random.normal(size=(n[0],3,n_realisations),scale=np.sqrt(dt_list[0])))),axis=0)
    
    Weak_convergence_Vector(n, n_realisations, model, X_0, W, t_end=t_end,exponent=[1,2,3])

    Strong_convergence_Vector(n, n_realisations, model, X_0, W, t_end=t_end)
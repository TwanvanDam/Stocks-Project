import numpy as np
import matplotlib.pyplot as plt

def Wiener_process(Time_array,seed=0):
    np.random.seed(seed)
    dt = Time_array[1]-Time_array[0]
    W = np.cumsum(np.random.normal(0,dt,size=(Time_array.shape)))
    return W


if __name__ == "__main__":
    t = np.linspace(0,1,10000)
    for i in range(100):
        W = Wiener_process(t,i)
        plt.plot(t[::10],W[::10])
    plt.show()


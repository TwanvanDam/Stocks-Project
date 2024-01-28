import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

Nt = 1000
Nt_w = 1000
dt = 1/Nt
dt_w = 1/Nt_w
t = np.arange(0, 1, dt)
t_w = np.arange(0, 1, 1/Nt_w)

p = 0.5
alpha = 0.2

W_1 = np.zeros(Nt_w)
W_2 = np.zeros(Nt_w)

S = np.zeros(Nt)
S[0] = 50

sigma = np.zeros(Nt)
sigma[0] = 0.2

xi = np.zeros(Nt)
xi[0] = 0.2

mu = 0.1

for i in range(1, Nt_w):
    W_1[i] = W_1[i-1] + np.sqrt(dt_w)* np.random.randn()
for i in range(1, Nt_w):
    W_2[i] = W_2[i-1] + np.sqrt(dt_w)* np.random.randn()


def euler(W_1, W_2, S, sigma, xi, mu, dt, p, alpha):
    for i in range(1, Nt):
        S[i] = S[i-1] + mu * S[i-1] * dt + sigma[i-1] * S[i-1] * (W_1[i*int((Nt_w/Nt))] - W_1[(i-1)*int((Nt_w/Nt))])
        sigma[i] = sigma[i-1] - (sigma[i-1] - xi[i-1])*dt + p * sigma[i-1] * (W_2[i*int((Nt_w/Nt))] - W_2[(i-1)*int((Nt_w/Nt))])
        xi[i] = xi[i-1] + 1/alpha * (sigma[i-1]-xi[i-1])*dt
    return S, sigma, xi

def milstein(W_1, W_2, S, sigma, xi, mu, dt, p, alpha):
    for i in range(1, Nt):
        S[i] = S[i-1] + mu * S[i-1] * dt + sigma[i-1] * S[i-1] * (W_1[i*int((Nt_w/Nt))] - W_1[(i-1)*int((Nt_w/Nt))]) + 0.5 * sigma[i-1] * S[i-1] * sigma[i-1] * ((W_1[i*int((Nt_w/Nt))] - W_1[(i-1)*int((Nt_w/Nt))])**2 - dt)
        sigma[i] = sigma[i-1] - (sigma[i-1] - xi[i-1])*dt + p * sigma[i-1] * (W_2[i*int((Nt_w/Nt))] - W_2[(i-1)*int((Nt_w/Nt))]) + 0.5 * p * sigma[i-1] * p *((W_2[i*int((Nt_w/Nt))] - W_2[(i-1)*int((Nt_w/Nt))])**2 - dt)
        xi[i] = xi[i-1] + 1/alpha * (sigma[i-1]-xi[i-1])*dt
    return S, sigma, xi


# show the wiener processes
plt.plot(t_w, W_1, label = "W_1")
plt.title('Simulated Wiener Process')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

plt.plot(t_w, W_2, label= "W_2")
plt.title('Simulated Wiener Process')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# show the stock price for euler and milstein
plt.figure(2)
plt.plot(t, euler(W_1, W_2, S, sigma, xi, mu, dt, p, alpha)[0], label = "euler" )
plt.plot(t, milstein(W_1, W_2, S, sigma, xi, mu, dt, p, alpha)[0], label = "milstein" )
plt.title('Simulated Stock price')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

#show the volatilities for euler and milstein schemes
plt.figure(3)
plt.plot(t, euler(W_1, W_2, S, sigma, xi, mu, dt, p, alpha)[1],label = "past dependent volatility euler" )
plt.plot(t, euler(W_1, W_2, S, sigma, xi, mu, dt, p, alpha)[2], label = 'long term volatility euler' )
plt.plot(t, milstein(W_1, W_2, S, sigma, xi, mu, dt, p, alpha)[1],label = "past dependent volatility milstein" )
plt.plot(t, milstein(W_1, W_2, S, sigma, xi, mu, dt, p, alpha)[2], label = 'long term volatility milstein' )
plt.title('Simulated volatility')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

#show the stock price for euler for different values of P
plt.figure(4)
plt.plot(t, euler(W_1, W_2, S, sigma, xi, mu, dt, 0.5, alpha)[0], label = "euler p=0.5" )
plt.plot(t, euler(W_1, W_2, S, sigma, xi, mu, dt, 1.5, alpha)[0], label = "euler p=1.5" )
plt.plot(t, euler(W_1, W_2, S, sigma, xi, mu, dt, 4.5, alpha)[0], label = "euler p=4.5" )
plt.title('Simulated Stock price')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# show the stock price for euler for different values of alpha
plt.figure(5)
plt.plot(t, euler(W_1, W_2, S, sigma, xi, mu, dt, p, 0.2)[0], label = "euler alpha=0.2" )
plt.plot(t, euler(W_1, W_2, S, sigma, xi, mu, dt, p, 1.2)[0], label = "euler alpha=1.2" )
plt.plot(t, euler(W_1, W_2, S, sigma, xi, mu, dt, p, 4.2)[0], label = "euler alpha=4.2" )
plt.title('Simulated Stock price')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# show the volatilities for euler for different values of P
plt.figure(6)
plt.plot(t, euler(W_1, W_2, S, sigma, xi, mu, dt, 0.5, alpha)[1],label = "past dependent volatility euler p=0.5" )
plt.plot(t, euler(W_1, W_2, S, sigma, xi, mu, dt, 1.5, alpha)[1], label = 'past dependent volatility euler p=1.5' )
plt.plot(t, euler(W_1, W_2, S, sigma, xi, mu, dt, 4.5, alpha)[1], label = 'past dependent volatility euler p=4.5' )
plt.plot(t, euler(W_1, W_2, S, sigma, xi, mu, dt, 0.5, alpha)[2],label = "long term volatility euler p=0.5" )
plt.plot(t, euler(W_1, W_2, S, sigma, xi, mu, dt, 1.5, alpha)[2], label = 'long term volatility euler p=1.5' )
plt.plot(t, euler(W_1, W_2, S, sigma, xi, mu, dt, 4.5, alpha)[2], label = 'long term volatility euler p=4.5' )
plt.title('Simulated volatility')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# show the volatilities for euler for different values of alpha
plt.figure(7)
plt.plot(t, euler(W_1, W_2, S, sigma, xi, mu, dt, p, 0.2)[1],label = "past dependent volatility euler alpha=0.2" )
plt.plot(t, euler(W_1, W_2, S, sigma, xi, mu, dt, p, 1.2)[1], label = 'past dependent volatility euler alpha=1.2' )
plt.plot(t, euler(W_1, W_2, S, sigma, xi, mu, dt, p, 4.2)[1], label = 'past dependent volatility euler alpha=4.2' )
plt.plot(t, euler(W_1, W_2, S, sigma, xi, mu, dt, p, 0.2)[2],label = "long term volatility euler alpha=0.2" )
plt.plot(t, euler(W_1, W_2, S, sigma, xi, mu, dt, p, 1.2)[2], label = 'long term volatility euler alpha=1.2' )
plt.plot(t, euler(W_1, W_2, S, sigma, xi, mu, dt, p, 4.2)[2], label = 'long term volatility euler alpha=4.2' )
plt.title('Simulated volatility')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

print(milstein(W_1, W_2, S, sigma, xi, mu, dt, p, alpha)[0][-1])
print(euler(W_1, W_2, S, sigma, xi, mu, dt, p, alpha)[0][-1])

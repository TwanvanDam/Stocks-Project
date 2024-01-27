
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

Nt = 1000
Nt_w = 10000
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

plt.figure(2)
plt.plot(t, euler(W_1, W_2, S, sigma, xi, mu, dt, p, alpha)[0], label = "euler" )
plt.plot(t, milstein(W_1, W_2, S, sigma, xi, mu, dt, p, alpha)[0], label = "milstein" )
plt.title('Simulated Stock price')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

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

print(milstein(W_1, W_2, S, sigma, xi, mu, dt, p, alpha)[0][-1])
print(euler(W_1, W_2, S, sigma, xi, mu, dt, p, alpha)[0][-1])
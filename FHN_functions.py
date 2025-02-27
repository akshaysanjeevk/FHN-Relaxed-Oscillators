import numpy as np
import matplotlib.pyplot as plt




def FHN(state, t, eps, a, b, I):
    u, v = state  
    dudt = u - (u**3) / 3 - v + I
    dvdt = eps * (u + a - b*v) 
    return [dudt, dvdt]

def FHN_String(state, t, eps, a, b, I, D):
    N = len(state) // 2  
    u = state[:N]  
    v = state[N:]  
    dudt = np.zeros(N)
    dvdt = np.zeros(N)
    for i in range(N):
        dudt[i] = u[i] - (u[i]**3) / 3 - v[i] + I
        if i > 0 and i < N - 1:  # Internal nodes
            dvdt[i] = eps * (u[i] + a - b * v[i]) + D * (v[i-1] + v[i+1] - 2 * v[i])
        elif i == 0:  # Left boundary
            dvdt[i] = eps * (u[i] + a - b * v[i]) + D * (v[1] - v[i])
        elif i == N - 1:  # Right boundary
            dvdt[i] = eps * (u[i] + a - b * v[i]) + D * (v[i-1] - v[i])
    
    return np.hstack((dudt, dvdt))

def FHN2(state, t, a, e, k, b):
    u, v = state
    dudt = u*(1 - u)*(u-a) - v
    dvdt = e*(k*u - v - b)
    return np.hstack((dudt, dvdt))

def FHN2_String(state, t, a, e, k, b, D):#string
    N = len(state) // 2  
    u = state[:N]  
    v = state[N:]  
    dudt = np.zeros(N)
    dvdt = np.zeros(N)
    for i in range(N):
        dudt[i] = u[i]*(1 - u[i])*(u[i] - a) - v[i]
        if i > 0 and i < N - 1:
            dvdt[i] = e * (k*u[i] - v[i] - b) + D*(v[i-1] + v[i+1] - 2*v[i])
        elif i == 0:
            dvdt[i] = D*(v[1] - v[i])  
        elif i == N - 1:
            dvdt[i] = D*(v[i-1] - v[i])  
    return np.hstack([dudt, dvdt])       
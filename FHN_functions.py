import numpy as np
import matplotlib.pyplot as plt

def FHNString(t, y, parms):#string
    N = parms['N']
    u = y[:N]
    v = y[N:]
    dudt = np.zeros(N)
    dvdt = np.zeros(N)
    for i in range(N):
        dudt[i] = u[i] * (1 - u[i]) * (u[i] - parms['a']) - v[i]
        if i > 0 and i < N - 1:
            dvdt[i] = parms['e'] * (parms['k'] * u[i] - v[i] - parms['b']) + parms['D'] * (v[i-1] + v[i+1] - 2 * v[i])
        elif i == 0:
            dvdt[i] = parms['D'] * (v[1] - v[i])  
        elif i == N - 1:
            dvdt[i] = parms['D'] * (v[i-1] - v[i])  
    return np.concatenate([dudt, dvdt])

def singleFHN(t, y, params):
    u=y[0]; v=y[1]
    dudt = u*(1 - u)*(u - params['a']) - v
    dvdt = params['e']*(params['k']*u - v - params['b'])
    return [dudt, dvdt]

def fitzhugh_nagumo(state, t, epsilon, a, b, I):
    u, v = state
    du_dt = u - (u**3) / 3 - v + I
    dv_dt = epsilon * (u + a - b * v)
    return [du_dt, dv_dt]


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
    
    return np.concatenate([dudt, dvdt])

        
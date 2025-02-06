import numpy as np
import matplotlib.pyplot as plt
import scipy as spy

plt.rcParams['text.usetex'] = True

import numpy as np

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


params = {
    'N' : 10,
    'a' : .139, 
    'e' : .001,
    'k' : .6,
    'b' : 70,
    'D' : .4e3
}

y0 = np.random.uniform(0.1, 0.9, 2*params['N'])  
# y0 = np.zeros(2*params['N'])
tspan = (0, 10)
sol = spy.integrate.solve_ivp(lambda t, y: FHNString(t, y, params), 
                tspan, y0, t_eval=np.linspace(0, 10, 100),
                method='RK45')


plt.figure(figsize=(8, 3))
plt.imshow(sol.y[:params['N'], :], aspect='auto', cmap='viridis')

cbar = plt.colorbar()
cbar.set_label('$u(t)$')
plt.yticks(ticks=np.arange(params['N']), labels=np.arange(1, params['N']+1))
plt.xlabel('Time')
plt.ylabel('Cells')


plt.show()
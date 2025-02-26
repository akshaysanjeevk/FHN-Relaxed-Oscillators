import numpy as np
import matplotlib.pyplot as plt
import scipy as spy

plt.rcParams['text.usetex'] = True

import FHN_functions as f




params = {
    'N' : 10,
    'a' : .139, 
    'e' : .001,
    'k' : .6,
    'b' : 70,
    'D' : 2e3
}

y0 = np.random.uniform(0.1, 0.9, 2*params['N'])  
# y0 = np.zeros(2*params['N'])
tspan = (0, 1e-4)
sol = spy.integrate.solve_ivp(lambda t, y: f.FHNString(t, y, params), 
                tspan, y0, t_eval=np.linspace(0, 1e-4, 100),
                method='RK45')


plt.figure(figsize=(8, 3))
plt.imshow(sol.y[:params['N'], :], aspect='auto', cmap='viridis')

cbar = plt.colorbar()
cbar.set_label('$u(t)$')
plt.yticks(ticks=np.arange(params['N']), labels=np.arange(1, params['N']+1))
plt.xlabel('Time')
plt.ylabel('Cells')


plt.show()
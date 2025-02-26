import numpy as np
import matplotlib.pyplot as plt
import scipy as spy

plt.rcParams['text.usetex'] = True

import FHN_functions as f

from scipy.integrate import odeint




eps = 1.08  
a = 0.7
b = 0.8
I = 0.5
D = 0.1 
N = 10   

state0 = np.random.rand(2 * N) * 0.1  

t = np.linspace(0, 100, 1000)

solution = odeint(f.FHN_String, state0, t, args=(eps, a, b, I, D))

u, v = solution[:N, :], solution[N:, :]
print(solution.shape)

plt.figure(figsize=(10, 3))
plt.imshow(u, aspect='auto', cmap='viridis')
cbar = plt.colorbar()
cbar.set_label('$u(t)$')
# plt.xlabel('Time')
# plt.ylabel('Cell')
plt.show()
# plt.savefig('SO.png', dpi=450)



import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spyint

plt.rcParams['text.usetex'] = True

def solveFHN(prm, dt = 'auto'):
    def core(t, y):
        u,v = y[0], y[1]
        dudt = u*(1-u)*(u-prm['a']) - v
        dvdt = prm['e']*(prm['k']*u-v-prm['b'])
        return [dudt, dvdt]
    if dt == 'auto':
        sol = spyint.solve_ivp(core, (0,prm['tmax']), prm['y0'], method='LSODA')
    else: 
        time = np.linspace(0, prm['tmax'], prm['tn'])
        sol = spyint.solve_ivp(core, (0,prm['tmax']), 1/prm['y0'], method='LSODA', t_eval=time)

    return sol.y, sol.t



def solveFHNodeint(prm):
    def core(y, t):
        u, v = y
        dudt = u*(1-u)*(u-prm['a']) - v
        dvdt = prm['e'] * (prm['k']*u - v - prm['b'])
        return [dudt, dvdt]
    time = np.linspace(0, prm['tmax'], prm['tn'])
    sol = spyint.odeint(core, prm['y0'], time)
    return sol.T, time

SO = {
    'N' : 10,
    'a' : .139, 
    'e' : .1e-3, 
    'k' : .6, 
    'b' : .06,  
    'D' : 0, # 1e3, 
    'tmax' : 300000, 
    'tn' : 1000000,
    'y0': [0,1]
    # 'y0' : np.zeros(20)
}

ysol, time = solveFHN(SO)
plt.figure(figsize=(10, 3))
plt.plot(time, ysol[0])
plt.plot(time, ysol[1])
plt.show()
print('Run completed')
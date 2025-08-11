import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spyint

plt.rcParams['text.usetex'] = True

def solveFHN(prm):
    def FHN(t, y):
        u,v = y
        dudt = u*(1-u)*(u-prm['a']) - v
        dvdt = prm['e']*(prm['k']*u-v-prm['b'])
        return [dudt, dvdt]
    
    t_vec = np.linspace(0, prm['tmax'], prm['tn'])
    
    sol = spy.solve_ivp(FHN, (0 , prm['tmax']), prm['y0'], method='RK45' )
    return sol.y, sol.t


def FHNString(prm):
    # N = y//2
    def core(t, y):
        u = y[:prm['N']]; v = y[prm['N']:]
        dudt = u*(1-u)*(u-prm['a']) - v 
        
        dvdt0 =  prm['D']*(v[1]-v[0]); dvdtn = prm['D']*(v[-2]-v[-1])
        coupling = v[0:-2] + v[2:] - 2*v[1:-1]
        dvdti = prm['e'] *(prm['k']*u[1:-1]  - v[1:-1] -prm['b']) +prm['D']*coupling
        
        # dvdt = np.concatenate([np.array(dvdt0), dvdti, np.array(dvdtn)])
        dvdt = np.hstack(([dvdt0], dvdti, [dvdtn]))
        return np.concatenate([dudt,dvdt])
    
    sol = spyint.solve_ivp(core, (0,prm['tmax']), prm['y0'], method='RK45')
    return sol.y, sol.t

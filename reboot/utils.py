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
        sol = spyint.solve_ivp(core, (0,prm['tmax']), prm['y0'], method='RK45')
    else: 
        time = np.linspace(0, prm['tmax'], prm['tn'])
        sol = spyint.solve_ivp(core, (0,prm['tmax']), prm['y0'], method='RK45', t_eval=time)
    return sol.y, sol.t


def FHNString(prm, dt='auto'):
    def core(t, y):
        u = y[:prm['N']]; v = y[prm['N']:]
        dudt = u*(1-u)*(u-prm['a']) - v 
        
        dvdt0 =  prm['D']*(v[1]-v[0]); dvdtn = prm['D']*(v[-2]-v[-1])
        coupling = v[0:-2] + v[2:] - 2*v[1:-1]
        dvdti = prm['e'] *(prm['k']*u[1:-1]  - v[1:-1] -prm['b']) +prm['D']*coupling
        
        # dvdt = np.concatenate([np.array(dvdt0), dvdti, np.array(dvdtn)])
        dvdt = np.hstack(([dvdt0], dvdti, [dvdtn]))
        return np.concatenate([dudt,dvdt])
    if dt == 'auto':
        sol = spyint.solve_ivp(core, (0,prm['tmax']), prm['y0'], method='RK45')
    else: 
        time = np.linspace(0, prm['tmax'], prm['tn'])
        sol = spyint.solve_ivp(core, (0,prm['tmax']), prm['y0'], method='RK45', t_eval=time)
    return sol.y, sol.t

def solve_plot(prm):
    ysol, t = FHNString(prm)

    plt.figure(figsize=(10, 3))
    plt.imshow(ysol[:prm['N']], cmap='inferno', aspect='auto', interpolation='nearest')
    cbar = plt.colorbar()   
    cbar.set_label('$u$')
    plt.ylabel('String')
    plt.xlabel('Time')
    # plt.title(str(prm))
    


def split_integration(prm, diffusion, tmax2 )
    sol0, t0 = solve_plot(prm)
    prm['D'] = diffusion; prm['y0'] = sol0[:,-1]
    prm['tmax'] = tmax2
    sol1, t1 = solve_plot(prm)
from functools import partial

import jax.numpy as np
from jax import vmap, jit, jvp, random
from jax.experimental.jet import jet
from tqdm import trange
import matplotlib.pyplot as plt

from pinns.nn import Siren


class ivps:
    name = 'ivps'
    T = 1.
    
    # solution, initial condition and pde must be overrided
    def u(self):
        raise NotImplementedError
    
    def u0(self):
        raise NotImplementedError
    
    def pde(self):
        raise NotImplementedError

    def loss_ic(self, params, x):
        t = np.zeros((1,))
        init_data = self.u0(x)
        init_pred = self.u(params, t,x)[0,...]
        loss_ic = np.mean( (init_pred - init_data)**2 )
        return loss_ic

    # Default: periodic on x
    def loss_bc(self, params, t):
        x = self.X * self.x_bd
        u = self.u(params, t,x)
        loss_bc = np.mean( (u[...,-1] - u[...,0])**2 )
        return loss_bc
    
    @partial(jit, static_argnums=(0,))
    def loss(self, params, t,x):
        loss = (self.pde(params, t,x).mean()
                + 1e3*self.loss_ic(params, x)
                + self.loss_bc(params, t))
        return loss
    
    
    def train(self, optimizer, domain_tr, key, params, nIter=5*10**4):
        print(self.equation)
        T, X = self.T, self.X
        x_L, x_R = self.x_bd
        domain = [*domain_tr]
        Nt, Nx = domain[0].size, domain[1].size
        state = optimizer.init_state(params)
        loss_log = []

        @jit
        def step(params, state, *args, **kwargs):
            params, state = optimizer.update(params, state, *args, **kwargs)
            return params, state

        for it in (pbar:= trange(1,nIter+1)):
            params, state = step(params, state, *domain)
            if it%100 == 0:
                loss = state.value
                loss_log.append(loss)
                # domain sampling
                key, *subkey = random.split(key, 3)
                domain[0] = T*random.uniform(subkey[0], (Nt,))
                domain[1] = X*random.uniform(subkey[1], (Nx,), minval=x_L, maxval=x_R)
                pbar.set_postfix({'pinn loss': f'{loss:.3e}'})
        
        self.opt_params, self.loss_log = params, loss_log
    

    def drawing(self, save=True):
        dir = f'figures/{self.name}'
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
        # loss log
        ax1.semilogy(self.loss_log, label='PINN Loss')
        ax1.set_xlabel('100 iterations')
        ax1.set_ylabel('Mean Squared Error')
        # Solution profile
        opt_params = self.opt_params
        domain = (self.T * np.linspace(0,1, 200),
                  self.X * np.linspace(*self.x_bd, 200))
        pred = self.u(opt_params, *domain)
        im = ax2.imshow(pred.T, origin='lower', cmap='jet', aspect='auto')
        ax2.axis('off')
        fig.colorbar(im)
        if save:
            fig.savefig(dir)
        else:
            fig.show()


class burgers(ivps):
    name = 'burgers'
    equation = 'u_t + uu_x = νu_xx/𝝅'
    X = 1.
    x_bd = np.array([-1, 1])

    def __init__(self, nu=1e-02):
        self.nu = nu
    
    def u0(self, x):
        return -np.sin(np.pi*x)

    def pde(self, params, t,x):
        u_t = jvp(lambda t: self.u(params, t,x), (t,), (np.ones(t.shape),))[1]
        u, (u_x, u_xx) = jet(lambda x: self.u(params, t,x), 
                             (x,), 
                             ((np.ones(x.shape), np.zeros(x.shape)),))
        pde = (u_t + u*u_x - self.nu*u_xx/np.pi)**2 
        return pde
    
    def loss_bc(self, params, t):
        x = self.X * self.x_bd
        # Dirichelt on x
        u = self.u(params, t,x)
        loss_bc = np.mean( u**2 )
        return loss_bc


class advection(ivps):
    name = 'advection'
    equation = 'u_t + βu_x = 0'
    T = 1.
    X = 2*np.pi
    x_bd = np.array([0, 1])

    def __init__(self, beta=30.):
        self.beta = beta

    def u0(self, x):
        return np.sin(x)
    
    def pde(self, params, t,x):
        _, u_t = jvp(lambda t: self.u(params, t,x), (t,), (np.ones(t.shape),))
        _, u_x = jvp(lambda x: self.u(params, t,x), (x,), (np.ones(x.shape),))
        pde = (u_t + self.beta*u_x)**2 
        return pde
    
    
class reaction(ivps):
    name = 'reaction'
    equation = 'u_t = ρu(1-u)'
    X = 2*np.pi
    x_bd = np.array([0, 1])

    def __init__(self, rho=5.):
        self.rho = rho
    
    def u0(self, x):
        exponent = 4*(x-np.pi)/np.pi
        return np.exp(-0.5*(exponent**2))
    
    def pde(self, params, t,x):
        u, u_t = jvp(lambda t: self.u(params, t,x), (t,), (np.ones(t.shape),))
        pde = (u_t - self.rho*u*(1-u))**2
        return pde


class reaction_diffusion(ivps):
    name = 'reaction_diffusion'
    equation = 'u_t = νu_xx + ρu(1-u)'
    X = 2*np.pi
    x_bd = np.array([0, 1])

    def __init__(self, nu=5., rho=5.):
        self.nu, self.rho = nu, rho
    
    def u0(self, x):
        exponent = 4*(x-np.pi)/np.pi
        return np.exp(-0.5*(exponent**2))
    
    def pde(self, params, t,x):
        _, u_t = jvp(lambda t: self.u(params, t,x), (t,), (np.ones(t.shape),))
        u, (_, u_xx) = jet(lambda x: self.u(params, t,x), 
                           (x,), 
                           ((np.ones(x.shape), np.zeros(x.shape)),))
        pde = (u_t -self.nu*u_xx - self.rho*u*(1-u))**2
        return pde

    
class allen_cahn(ivps):
    name = 'allen_cahn'
    equation = 'u_t = νu_xx + ρu(1-u^2)'
    X = 1.
    x_bd = np.array([-1, 1])

    def __init__(self, nu=1e-04, rho=5.):
        self.nu, self.rho = nu, rho
    
    def u0(self, x):
        return x**2 * np.cos(np.pi*x)
    
    def pde(self, params, t,x):
        _, u_t = jvp(lambda t: self.u(params, t,x), (t,), (np.ones(t.shape),))
        u, (_, u_xx) = jet(lambda x: self.u(params, t,x), 
                           (x,), 
                           ((np.ones(x.shape), np.zeros(x.shape)),))
        pde = (u_t - self.nu*u_xx + self.rho*u**3 - self.rho*u)**2
        return pde
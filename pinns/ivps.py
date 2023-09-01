from functools import partial

import jax.numpy as np
from jax import vmap, jit, jvp
from jax.experimental.jet import jet

from pinns.nn import Siren

class ivps:
    def __init__(self, width=64, depth=5, w0=8.):
        # architecture
        layers = [2] + [width for _ in range(depth-1)] + [1]
        self.init, self.apply = Siren(layers, w0)
        # vectorize
        self.u = vmap(vmap(self._u, (None,0,None)), (None,None,0), 1) #(Nt, Nx)

    def _u(self, params, t,x):
        input = np.hstack([t,x])
        _u = self.apply(params, input).squeeze()
        return _u

    def loss_ic(self, params, x):
        t = np.zeros((1,))
        init_data = self.u0(x)
        init_pred = self.u(params, t,x)[0,...]
        loss_ic = np.mean( (init_pred - init_data)**2 )
        return loss_ic

    def loss_bc(self, params, t):
        x = self.X * self.x_bd
        # Dirichlet on x
        u = self.u(params, t,x)
        loss_bc = np.mean( u**2 )
        return loss_bc
    
    @partial(jit, static_argnums=(0,))
    def loss(self, params, t,x):
        loss = (self.pde(params, t,x).mean()
                + 1e3*self.loss_ic(params, x)
                + self.loss_bc(params, t))
        return loss
    
    # initial condition and pde must be overrided
    def u0(self, x):
        raise NotImplementedError
    
    def pde(self, t,x):
        raise NotImplementedError


class burgers(ivps):
    name = 'burgers'
    T = 1.
    X = 1.
    x_bd = np.array([-1, 1])

    def __init__(self, width=64, depth=5, w0=8., nu=1e-02):
        super().__init__(width, depth, w0)
        self.nu = nu
    
    def pde(self, params, t,x):
        u_t = jvp(lambda t: self.u(params, t,x), (t,), (np.ones(t.shape),))[1]
        u, (u_x, u_xx) = jet(lambda x: self.u(params, t,x), 
                             (x,), 
                             ((np.ones(x.shape), np.zeros(x.shape)),))
        pde = (u_t + u*u_x - self.nu*u_xx/np.pi)**2 
        return pde
    
    # Initial function
    def u0(self, x):
        return -np.sin(np.pi*x)


class advection(ivps):
    name = 'advection'
    T = 1.
    X = 2*np.pi
    x_bd = np.array([0, 1])

    def __init__(self, width=64, depth=5, w0=8., beta=30.):
        super().__init__(width, depth, w0)
        # coefficient
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
    T = 1.
    X = 2*np.pi
    x_bd = np.array([0, 1])

    def __init__(self, width=64, depth=5, w0=8., rho=5.):
        super().__init__(width, depth, w0)
        # coefficient
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
    T = 1.
    X = 2*np.pi
    x_bd = np.array([0, 1])

    def __init__(self, width=64, depth=5, w0=8., nu=5., rho=5.):
        super().__init__(width, depth, w0)
        # coefficient
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
    T = 1.
    X = 1.
    x_bd = np.array([-1, 1])

    def __init__(self, width=64, depth=5, w0=8., nu=1e-04, rho=5.):
        super().__init__(width, depth, w0)
        # coefficients
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
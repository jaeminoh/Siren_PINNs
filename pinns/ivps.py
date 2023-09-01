from functools import partial

import jax.numpy as np
from jax import vmap, jit, jvp
from jax.experimental.jet import jet

from pinns.nn import Siren


class burgers:
    name = 'burgers'

    def __init__(self, width=64, depth=5, w0=8.):
        # architecture
        layers = [2] + [width for _ in range(depth-1)] + [1]
        self.init, self.apply = Siren(layers, w0)
        # domain
        self.T, self.X, self.x_bd = 1., 1., np.array([-1, 1])
        
        # vectorize
        self.u = vmap(vmap(self._u, (None,0,None)), (None,None,0), 1) #(Nt, Nx)
    
    def _u(self, params, t,x):
        input = np.hstack([t,x])
        _u = self.apply(params, input).squeeze()
        return _u
    
    def pde(self, params, t,x):
        u_t = jvp(lambda t: self.u(params, t,x), (t,), (np.ones(t.shape),))[1]
        u, (u_x, u_xx) = jet(lambda x: self.u(params, t,x), 
                             (x,), 
                             ((np.ones(x.shape), np.zeros(x.shape)),))
        pde = (u_t + u*u_x - 1e-2*u_xx/np.pi)**2 
        return pde
    
    def u0(self, x):
        # Initial function
        return -np.sin(np.pi*x)
        
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


class advection:
    name = 'advection'

    def __init__(self, width=64, depth=5, w0=8.):
        # architecture
        layers = [2] + [width for _ in range(depth-1)] + [1]
        self.init, self.apply = Siren(layers, w0)
        # domain
        self.T, self.X, self.x_bd = 1., 2*np.pi, np.array([0, 1])
        # coefficient
        self.beta = 30.
        # vectorize
        self.u = vmap(vmap(self._u, (None,0,None)), (None,None,0), 1) #(Nt, Nx)
    
    def u0(self, x):
        # Initial function
        return -np.sin(np.pi*x)
    
    def _u(self, params, t,x):
        input = np.hstack([t,x])
        _u = self.apply(params, input).squeeze()
        return _u
    
    def pde(self, params, t,x):
        _, u_t = jvp(lambda t: self.u(params, t,x), (t,), (np.ones(t.shape),))
        _, u_x = jvp(lambda x: self.u(params, t,x), (x,), (np.ones(x.shape),))
        pde = (u_t + self.beta*u_x)**2 
        return pde

    def u0(self, x):
        return np.sin(x)

    def loss_ic(self, params, x):
        t = np.array([0])
        init_data = self.u0(x)
        init_pred = self.u(params, t,x)[0,...]
        loss_ic = np.mean( (init_pred - init_data)**2 )
        return loss_ic
    
    def loss_bc(self, params, t):
        x = self.X*self.x_bd
        # Periodic on x
        u = self.u(params, t,x)
        loss_bc = np.mean( (u[...,-1] - u[...,0])**2 )
        return loss_bc
    
    @partial(jit, static_argnums=(0,))
    def loss(self, params, t,x):
        loss = (self.pde(params, t,x).mean()
                + 1e3*self.loss_ic(params, x)
                + self.loss_bc(params, t))
        return loss
    

class reaction:
    name = 'reaction'

    def __init__(self, width=64, depth=5, w0=8.):
        # architecture
        layers = [2] + [width for _ in range(depth-1)] + [1]
        self.init, self.apply = Siren(layers, w0)
        # domain
        self.T, self.X, self.x_bd = 1., 2*np.pi, np.array([0, 1])
        # coefficient
        self.rho = 5.
        # vectorize
        self.u = vmap(vmap(self._u, (None,0,None)), (None,None,0), 1) #(Nt, Nx)
    
    def _u(self, params, t,x):
        input = np.hstack([t,x])
        _u = self.apply(params, input).squeeze()
        return _u
    
    def pde(self, params, t,x):
        u, u_t = jvp(lambda t: self.u(params, t,x), (t,), (np.ones(t.shape),))
        pde = (u_t - self.rho*u*(1-u))**2
        return pde

    def u0(self, x):
        exponent = 4*(x-np.pi)/np.pi
        return np.exp(-0.5*(exponent**2))

    def loss_ic(self, params, x):
        t = np.zeros((1,))
        init_data = self.u0(x)
        init_pred = self.u(params, t,x)[0,...]
        loss_ic = np.mean( (init_pred - init_data)**2 )
        return loss_ic
    
    def loss_bc(self, params, t):
        x = self.X*self.x_bd
        # Periodic on x
        u = self.u(params, t,x)
        loss_bc = np.mean( (u[...,-1] - u[...,0])**2 )
        return loss_bc
    
    @partial(jit, static_argnums=(0,))
    def loss(self, params, t,x):
        loss = (self.pde(params, t,x).mean()
                + 1e3*self.loss_ic(params, x)
                + self.loss_bc(params, t))
        return loss


class reaction_diffusion:
    name = 'reaction_diffusion'

    def __init__(self, width=64, depth=5, w0=8.):
        # architecture
        layers = [2] + [width for _ in range(depth-1)] + [1]
        self.init, self.apply = Siren(layers, w0)
        # domain
        self.T, self.X, self.x_bd = 1., 2*np.pi, np.array([0,1])
        # coefficient
        self.nu, self.rho = 5., 5.
        # vectorize
        self.u = vmap(vmap(self._u, (None,0,None)), (None,None,0), 1) #(Nt, Nx)
    
    def _u(self, params, t,x):
        input = np.hstack([t,x])
        _u = self.apply(params, input).squeeze()
        return _u
    
    def pde(self, params, t,x):
        _, u_t = jvp(lambda t: self.u(params, t,x), (t,), (np.ones(t.shape),))
        u, (_, u_xx) = jet(lambda x: self.u(params, t,x), 
                           (x,), 
                           ((np.ones(x.shape), np.zeros(x.shape)),))
        pde = (u_t -self.nu*u_xx - self.rho*u*(1-u))**2
        return pde

    def u0(self, x):
        exponent = 4*(x-np.pi)/np.pi
        return np.exp(-0.5*(exponent**2))

    def loss_ic(self, params, x):
        t = np.zeros((1,))
        init_data = self.u0(x)
        init_pred = self.u(params, t,x)[0,...]
        loss_ic = np.mean( (init_pred - init_data)**2 )
        return loss_ic
    
    def loss_bc(self, params, t):
        x = self.X*self.x_bd
        # Periodic on x
        u = self.u(params, t,x)
        loss_bc = np.mean( (u[...,-1] - u[...,0])**2 )
        return loss_bc
    
    @partial(jit, static_argnums=(0,))
    def loss(self, params, t,x):
        loss = (self.pde(params, t,x).mean()
                + 1e3*self.loss_ic(params, x)
                + self.loss_bc(params, t))
        return loss
    

class allen_cahn:
    name = 'allen_cahn'

    def __init__(self, width=64, depth=5, w0=8.):
        # architecture
        layers = [2] + [width for _ in range(depth-1)] + [1]
        self.init, self.apply = Siren(layers, w0)
        # domain
        self.T, self.X, self.x_bd = 1., 1., np.array([-1, 1])
        # vectorize
        self.u = vmap(vmap(self._u, (None, 0,None)), (None, None,0), 1) #(Nt, Nx)
    
    def _u(self, params, t,x):
        input = np.hstack([t,x])
        _u = self.apply(params, input).squeeze()
        return _u
    
    def pde(self, params, t,x):
        _, u_t = jvp(lambda t: self.u(params, t,x), (t,), (np.ones(t.shape),))
        u, (_, u_xx) = jet(lambda x: self.u(params, t,x), 
                           (x,), 
                           ((np.ones(x.shape), np.zeros(x.shape)),))
        pde = (u_t - 1e-04*u_xx + 5*u**3 - 5*u)**2
        return pde

    def u0(self, x):
        return x**2 * np.cos(np.pi*x)

    def loss_ic(self, params, x):
        t = np.zeros((1,))
        init_data = self.u0(x)
        init_pred = self.u(params, t,x)[0,...]
        loss_ic = np.mean( (init_pred - init_data)**2 )
        return loss_ic
    
    def loss_bc(self, params, t):
        x = self.X*self.x_bd
        # Periodic on x
        u = self.u(params, t,x)
        loss_bc = np.mean( (u[...,-1] - u[...,0])**2 )
        return loss_bc
    
    @partial(jit, static_argnums=(0,))
    def loss(self, params, t,x):
        loss = (self.pde(params, t,x).mean()
                + 1e3*self.loss_ic(params, x)
                + self.loss_bc(params, t))
        return loss
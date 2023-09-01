from jax import random
import jax.numpy as np
from optax import adam
from jaxopt import OptaxSolver

from pinns import ivps, training


model = ivps.advection(width=64, depth=5, w0=8)
print(f'pde: {model.name}')

nIter = 1*10**5
lr = 1e-04
optimizer = OptaxSolver(fun=model.loss, opt=adam(lr))

Nt, Nx = 128, 128
domain_tr = (model.T * np.linspace(0,1, Nt),
             model.X * np.linspace(*model.x_bd, Nx))

init_key, train_key = random.split(random.PRNGKey(0))
init_params = model.init(init_key)

training.train(model, optimizer, domain_tr, train_key, init_params, nIter=nIter)
training.drawing(model, fname=model.name)
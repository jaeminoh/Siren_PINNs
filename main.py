from jax import random
import jax.numpy as np
from optax import adam, cosine_decay_schedule
from jaxopt import OptaxSolver

from pinns import ivps, training


model = ivps.burgers(width=64, depth=5, w0=8)
print(f'pde: {model.name}')

nIter = 1*10**5
lr = cosine_decay_schedule(1e-04, nIter)
optimizer = OptaxSolver(fun=model.loss, opt=adam(lr))

Nt, Nx = 128, 128
domain_tr = (model.T * np.linspace(0,1, Nt),
             model.X * np.linspace(*model.x_bd, Nx))

init_key, train_key = random.split(random.PRNGKey(0))
init_params = model.init(init_key)

model.opt_params, model.loss_log = training.train(model, optimizer, domain_tr, train_key, init_params, nIter=nIter)
training.drawing(model, fname=model.name)
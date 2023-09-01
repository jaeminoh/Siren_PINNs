import numpy as np
from tqdm import trange
from jax import jit, random
import matplotlib.pyplot as plt


def train(model, optimizer, domain_tr, key, params, nIter=5*10**4):
    T, X = model.T, model.X
    domain = [*domain_tr]
    Nt, Nx = domain[0].size, domain[1].size
    state = optimizer.init_state(params)
    model.loss_log = []

    @jit
    def step(params, state, *args, **kwargs):
        # optimizer: OptaxSolver
        params, state= optimizer.update(params, state, *args, **kwargs)
        return params, state

    for it in (pbar:= trange(1,nIter+1)):
        params, state = step(params, state, *domain)
        if it%100 == 0:
            loss = state.value
            model.loss_log.append(loss)
            # domain sampling
            key, *subkey = random.split(key, 3)
            domain[0] = T*random.uniform(subkey[0], (Nt,))
            domain[1] = X*random.uniform(subkey[1], (Nx,), minval=-1, maxval=1)
            model.opt_params = params
            pbar.set_postfix({'PINN Loss': f'{loss:.3e}'})


def drawing(model, save=True, fname='advection'):
    dir = f'figures/{fname}'
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
    # loss log
    ax1.semilogy(model.loss_log, label='PINN Loss')
    ax1.set_xlabel('100 iterations')
    ax1.set_ylabel('Mean Squared')
    # Solution profile
    opt_params = model.opt_params
    domain = (model.T * np.linspace(0,1, 200),
              model.X * np.linspace(*model.x_bd, 200))
    pred = model.u(opt_params, *domain)
    im = ax2.imshow(pred.T, origin='lower', cmap='jet', aspect='auto')
    ax2.axis('off')
    fig.colorbar(im)
    if save:
        fig.savefig(dir)
    else:
        fig.show()
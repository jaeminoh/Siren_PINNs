import jax.numpy as jnp
import jax.random as jr
import jaxopt
from optax import adam, cosine_decay_schedule

from pinns.ivps import advection
from pinns.nn import Siren


class SeparablePINN(advection):
    def __init__(self, width=64, depth=4, d_out=64, w0=8.0):
        super().__init__()
        layers = [1] + [width for _ in range(depth - 1)] + [d_out]
        self.init, self.apply = Siren(layers, w0)

    def u(self, params, t, x):  # (Nt, Nx)
        t, x = t.reshape(-1, 1), x.reshape(-1, 1)
        outputs = self.apply(params[0], t) @ self.apply(params[1], x).T
        return outputs


spinn = SeparablePINN()
*init_keys, train_key = jr.split(jr.key(0), 3)
init_params = [spinn.init(_key) for _key in init_keys]

nIter = 1 * 10**5
lr = cosine_decay_schedule(1e-03, nIter)


Nt, Nx = 128, 128
domain_tr = (
    spinn.T * jnp.linspace(0, 1, Nt),
    spinn.X * jnp.linspace(*spinn.x_bd, Nx),
)

optimizer = jaxopt.OptaxSolver(fun=spinn.loss, opt=adam(lr),)

spinn.train(optimizer, domain_tr, train_key, init_params, nIter=nIter)
spinn.drawing(save=True)

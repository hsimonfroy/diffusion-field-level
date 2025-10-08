from jax import numpy as jnp, random as jr, jit, grad, vmap, tree, value_and_grad, jvp, debug, jacrev, jacfwd

from diffrax import (diffeqsolve, ControlTerm, MultiTerm, ODETerm, VirtualBrownianTree,
                     ConstantStepSize, PIDController, SaveAt, Euler, Tsit5)

import flax.linen as nn
import numpy as np


def integ_sde(seed, t0, t1, dt0, y0, drift, diffusion, snapshots, pid=True):
    dim = y0.shape[-1]
    if snapshots is None: 
        saveat = SaveAt(t1=True)
    elif isinstance(snapshots, int): 
        saveat = SaveAt(ts=jnp.linspace(t0, t1, snapshots))
    else: 
        saveat = SaveAt(ts=jnp.asarray(snapshots))

    if pid:
        # See https://docs.kidger.site/diffrax/devdocs/srk_example/
        from lineax import DiagonalLinearOperator # NOTE diffrax >= 0.6.0
        from diffrax import ShARK, SpaceTimeLevyArea
        diffus = lambda t, y, args: DiagonalLinearOperator(diffusion(t, y, args))
        brownian_motion = VirtualBrownianTree(t0, t1, tol=1e-4, shape=(dim,), key=seed, levy_area=SpaceTimeLevyArea)
        terms = MultiTerm(ODETerm(drift), ControlTerm(diffus, brownian_motion))
        solver = ShARK() # NOTE diffrax >= 0.6.0
        controller = PIDController(rtol=1e-3, atol=1e-6, pcoeff=0.1, icoeff=0.3, dcoeff=0.) # ~2*1500 evals
        # controller = PIDController(rtol=1e-2, atol=1e-4, pcoeff=0.1, icoeff=0.3, dcoeff=0.) # ~2*250 evals
    else:
        from diffrax import WeaklyDiagonalControlTerm # NOTE diffrax <= 0.5.0
        brownian_motion = VirtualBrownianTree(t0, t1, tol=1e-4, shape=(dim,), key=seed)
        terms = MultiTerm(ODETerm(drift), WeaklyDiagonalControlTerm(diffusion, brownian_motion))
        solver = Euler()
        controller = ConstantStepSize()

    sol = diffeqsolve(terms, solver, t0, t1, dt0=dt0, y0=y0, stepsize_controller=controller, saveat=saveat,
                    #   max_steps=int(1e5),
                      )
    # debug.print("n_steps: {n}", n=sol.stats['num_steps'])
    return sol.ts, sol.ys


def integ_ode(t0, t1, dt0, y0, drift, snapshots, pid=True):

    if snapshots is None:
        saveat = SaveAt(t1=True)
    elif isinstance(snapshots, int):
        saveat = SaveAt(ts=jnp.linspace(t0, t1, snapshots))
    else:
        saveat = SaveAt(ts=jnp.asarray(snapshots))

    if pid:
        solver = Tsit5()
        controller = PIDController(rtol=1e-3, atol=1e-6, pcoeff=0., icoeff=1., dcoeff=0.)
    else:
        solver = Euler()
        controller = ConstantStepSize()

    terms = ODETerm(drift)
    sol = diffeqsolve(terms, solver, t0, t1, dt0=dt0, y0=y0, stepsize_controller=controller, saveat=saveat,
                    #   max_steps=int(1e5),
                      )
    return sol.ts, sol.ys



def hutchinson_divergence(seed, vf, x):
    def one_probe(seed):
        r = jr.rademacher(seed, x.shape, dtype=float) # or normal(k, x.shape)
        _, jvp_out = jvp(vf, (x,), (r,))
        return jnp.dot(r, jvp_out)
    return vmap(one_probe)(seed).mean()



class VelFieldNN(nn.Module):
    out_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, t, x):
        # Encoding time
        freqs = 2 * jnp.pi * jnp.linspace(1,100,10)
        t *= freqs
        t = jnp.concatenate([jnp.sin(t), jnp.cos(t)],axis=-1)
        # Building network
        act_fn = nn.silu
        x = jnp.concatenate([t, x])
        x = nn.Dense(features=self.hidden_dim)(x)
        x = act_fn(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = act_fn(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = act_fn(x)
        x = nn.Dense(features=self.out_dim)(x)
        return x
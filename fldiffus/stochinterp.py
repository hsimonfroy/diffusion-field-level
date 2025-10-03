import matplotlib.pyplot as plt
import numpy as np

from functools import partial
from dataclasses import dataclass
from collections.abc import Callable

from jax import numpy as jnp, random as jr, jit, grad, vmap, tree, value_and_grad, jvp, debug, jacrev, jacfwd
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import flax.linen as nn

from fldiffus.targets import (alpha_OT, beta_OT, alpha_VP, beta_VP, alpha_VE, beta_VE, 
                            drift_OT, diffusion_OT, drift_VP, diffusion_VP, drift_VE, diffusion_VE, 
                            make_gaussian_mixture)
from fldiffus.utils import hutchinson_divergence, integ_sde, integ_ode, ScoreNN



@dataclass
class StochInterp:
    scheduling: str | tuple[Callable, Callable]
    dim: int
    hutch_div: int

    def __post_init__(self):
        # Path params
        if self.scheduling == 'OT':
            self.alpha, self.beta = alpha_OT, beta_OT
            self.drift, self.diffusion = drift_OT, diffusion_OT

        elif self.scheduling == 'VP':
            self.alpha, self.beta = alpha_VP, beta_VP
            self.drift, self.diffusion = drift_VP, diffusion_VP

        elif self.scheduling == 'VE':
            self.alpha, self.beta = alpha_VE, beta_VE
            self.drift, self.diffusion = drift_VE, diffusion_VE

        elif isinstance(self.scheduling, tuple):
            self.alpha, self.beta = self.scheduling
            # General drift and diffusion formula
            self.drift = lambda t, y, args: grad(self.alpha)(t) / self.alpha(t) * y
            self.diffusion = lambda t, y, args: (- self.alpha(t)**2 * grad(lambda t: (self.beta(t) / self.alpha(t))**2)(t))**.5 * jnp.ones_like(y)
        else:
            raise ValueError(f"Unknown scheduling: {self.scheduling}")

        # Integration params
        self.dt0 = 3e-4
        self.eps = 0.
        # self.dt0 = 1e-2
        # self.eps = 1e-6
        self.snapshots = 100 # None, int or array of times
        self.score_target = lambda t, y: grad(self.marg(t).log_prob)(y)
        self.flow_target = lambda t, y: self.drift(t, y, None) + self.diffusion(t, y, None)**2 / 2 * self.score_target(t, y)

        # Neural nets
        class FlowNN(ScoreNN):
            @nn.compact
            def __call__(selfnn, t, x):
                # score = ScoreNN(self.a, self.b, ...)(t, x) # NOTE: Will not have the same params structure
                score = ScoreNN.__call__(selfnn, t, x)
                return self.drift(t, x, None) + self.diffusion(t, x, None)**2 / 2 * score
            
        self.scorenn = ScoreNN(out_dim=self.dim, 
                               hidden_dim=128)
        self.flownn = FlowNN(out_dim=self.dim, 
                             hidden_dim=128)
        self.params = self.scorenn.init(jr.key(0), jnp.zeros(1), jnp.zeros(self.dim))

        
        

        # Plot params
        self.n_discr = 128
        self.xlim = (-3,3)
        if self.dim == 1:
            self.xx = jnp.linspace(*self.xlim, self.n_discr)[:,None]
        elif self.dim == 2:
            xx = jnp.linspace(*self.xlim, self.n_discr)
            self.xx, self.yy = jnp.meshgrid(xx, xx)
            self.xy = jnp.stack([self.xx.flatten(), self.yy.flatten()], axis=1)


    def noise(self, key, t, x1):
        return self.alpha(t) * x1 + self.beta(t) * jr.normal(key, jnp.shape(x1))
    
    def marg(self, t=1., sigma0=.5, probs=[0.4, 0.6]):
        return make_gaussian_mixture(alpha=self.alpha(t), beta=self.beta(t),
                                        d=self.dim, sigma0=sigma0, probs=probs)
    
    @property
    def base(self):
        return self.marg(0.)

    @property
    def target(self):
        return self.marg(1.)
    


    def backward_sde(self, seed, x1):
        t0, t1 = self.eps, 1 - self.eps
        back_drift = lambda t, y, args: -self.drift(1 - t, y, args)
        back_diffusion = lambda t, y, args: self.diffusion(1 - t, y, args)

        ts, ys = integ_sde(seed, t0, t1, self.dt0, x1, back_drift, back_diffusion, snapshots=self.snapshots)
        return ts[::-1], ys

    def forward_sde(self, seed, x0, params=None):
        t0, t1 = self.eps, 1 - self.eps
        if params is None:
            score = self.score_target
        else:
            score = lambda t, y: self.scorenn.apply(params, t, y)
        forw_drift = lambda t, y, args: self.drift(t, y, args) + self.diffusion(t, y, args)**2 * score(t, y)

        ts, ys = integ_sde(seed, t0, t1, self.dt0, x0, forw_drift, self.diffusion, snapshots=self.snapshots)
        return ts, ys



    def logp_drift(self, params, t, y, args):
        x, logp = y
        if params is None:
            flow = self.flow_target
        else:
            flow = lambda t, x: self.flownn.apply(params, t, x)

        dx = flow(t, x)
        if self.hutch_div is None:
            # div = jnp.trace(jacfwd(lambda xx: flow(t, xx))(x)) # Full Jac
            div = jnp.trace(jacrev(lambda xx: flow(t, xx))(x)) # Full Jac
        elif isinstance(self.hutch_div, int):
            seed_hutch = jr.split(jr.key(42), self.hutch_div)
            div = hutchinson_divergence(seed_hutch, lambda xx: self.flow.apply(params, t, xx), x)
        else:
            raise ValueError("hutch_div must be None or int")
        return dx, -div

    def forward_ode(self, x0, params=None):
        # Let us use the (forward) continuous change of variable formula
        # log p_t(x(t)) = log p_0(x(0)) - \int_0^t div(vf(x(s), s)) ds
        t0, t1 = self.eps, 1 - self.eps
        y0 = (x0, self.base.log_prob(x0))

        ts, ys = integ_ode(t0, t1, self.dt0, y0, partial(self.logp_drift, params), snapshots=self.snapshots)
        xs, logps = ys
        return ts, xs, logps[-1]

    def backward_ode(self, x1, params=None):
        # Let us use the (backward) continuous change of variable formula
        # log p_t(x(t)) = log p_0(x(0)) + \int_t^0 div(vf(x(s), s)) ds
        t0, t1 = 1 - self.eps, self.eps
        y0 = (x1, 0.0)

        ts, ys = integ_ode(t0, t1, -self.dt0, y0, partial(self.logp_drift, params), snapshots=self.snapshots)
        xs, logps = ys
        logps = self.base.log_prob(xs[-1]) - logps
        return ts, xs, logps[-1]

















    ############
    # Plotting #
    ############
    def plot_prob(self, log_prob, *args, **kwargs):
        if self.dim == 1:
            prob = jnp.exp(vmap(log_prob)(self.xx))
            out = plt.plot(self.xx, prob, *args, **kwargs)
            plt.xlabel('x')

        elif self.dim == 2:
            prob = jnp.exp(vmap(log_prob)(self.xy)).reshape(self.xx.shape)
            out = plt.contour(self.xx, self.yy, prob, *args, **kwargs)
            plt.xlim(self.xlim), plt.ylim(self.xlim)
            plt.xlabel('x'), plt.ylabel('y')
            plt.gca().set_aspect(1.)
            return out

    def plot_samples(self, samples, *args, **kwargs):
        if self.dim == 1:
            out = plt.hist(samples[...,0], bins=50, range=self.xlim, density=True, 
                     *args,  **{'alpha':0.5} | kwargs)
            plt.xlabel('$x$')

        elif self.dim == 2:
            out = plt.scatter(samples[...,0], samples[...,1], marker='+', 
                        *args, **{'alpha':0.1} | kwargs)
            plt.xlim(self.xlim), plt.ylim(self.xlim)
            plt.xlabel('x'), plt.ylabel('y')
            plt.gca().set_aspect(1.)
            return out

    def plot_margs(self):
        assert self.dim == 1, "Only implemented for dim=1"
        tt, xx = jnp.linspace(0, 1, self.n_discr), jnp.linspace(*self.xlim, self.n_discr)
        tt, xx = jnp.meshgrid(tt, xx)
        tx = jnp.stack([tt.flatten(), xx.flatten()], axis=1)

        def prob_at_tx(tx):
            t, x = tx
            return self.marg(t).prob(x[...,None]).squeeze()

        probas = vmap(prob_at_tx)(tx).reshape(self.n_discr, self.n_discr)
        plt.pcolormesh(tt, xx, probas)
        plt.xlabel('t'), plt.ylabel('x');
        
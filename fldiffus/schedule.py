from jax import numpy as jnp

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


def make_gaussian_mixture(alpha=1., beta=0., d=2, sigma1=1., probs=[0.5, 0.5]):
  """
  Returns a mixture of two d-dimensional Multivariate Gaussians.

  Args:
    alpha: The scaling factor for the mean of the Gaussians.
    beta: The standard deviation of the noise convolved with the Gaussians.
    d: The dimension of the Gaussian distributions.
    sigma0: The initial standard deviation of the Gaussians.
    probs: The probabilities for the categorical distribution.
  """
  return tfd.MixtureSameFamily(
      mixture_distribution=tfd.Categorical(probs=probs),
      components_distribution=tfd.MultivariateNormalDiag(
        #   loc=jnp.stack([-alpha * jnp.ones(d), alpha * jnp.ones(d)]),
          loc=jnp.stack([-alpha * jnp.ones(d), alpha * jnp.ones(d)]),
          scale_diag=jnp.ones((2, d)) * ((alpha * sigma1)**2 + beta**2)**.5)
  )



##### Optimal Transport #####
def alpha_OT(t):
    return t

def beta_OT(t):
    return 1 - t

drift_OT = lambda t, y, args: y / t
diffusion_OT = lambda t, y, args: (2 * (1 / t - 1))**.5 * jnp.ones_like(y)


##### Variance Preserving #####
g2min, g2max = 0.1, 20.
def alpha_VP(t, g2min=g2min, g2max=g2max):
    """https://huggingface.co/docs/diffusers/v0.13.0/en/api/schedulers/score_sde_vp"""
    return jnp.exp((g2min * (t**2 - 1) - g2max * (1 - t)**2) / 4)

def beta_VP(t, g2min=g2min, g2max=g2max):
    alpha = alpha_VP(t, g2min, g2max)
    return (1 - alpha**2)**.5

drift_VP = lambda t, y, args: (t * g2min + (1-t) * g2max) / 2 * y
diffusion_VP = lambda t, y, args: (t * g2min + (1-t) * g2max)**.5 * jnp.ones_like(y)


##### Variance Exploding #####
betamin, betamax = 0.01, 100.
def alpha_VE(t):
    return jnp.ones_like(t)

def beta_VE(t, betamin=betamin, betamax=betamax):
    """https://huggingface.co/docs/diffusers/v0.13.0/en/api/schedulers/score_sde_ve"""
    return betamin**t * betamax**(1 - t)

drift_VE = lambda t, y, args: jnp.zeros_like(y)
diffusion_VE = lambda t, y, args: (2 * beta_VE(t)**2 * jnp.log(betamax / betamin))**.5 * jnp.ones_like(y)


##### Pinned Brownian #####
betamax = 1.
def alpha_PB(t):
    return t

def beta_PB(t, betamax=betamax):
    alpha = alpha_PB(t)
    return 2 * betamax * (alpha * (1 - alpha))**.5

drift_PB = lambda t, y, args: y / t
diffusion_PB = lambda t, y, args: 2 * betamax * jnp.ones_like(y)


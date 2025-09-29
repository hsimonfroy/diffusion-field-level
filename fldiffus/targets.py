from jax import numpy as jnp

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


def make_dist(alpha=1., beta=0., d=2, sigma0=1., probs=[0.5, 0.5]):
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
          scale_diag=jnp.ones((2, d)) * ((alpha * sigma0)**2 + beta**2)**.5)
  )


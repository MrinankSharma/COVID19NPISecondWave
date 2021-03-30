"""
numpyro asymmetric laplace distribution

used for NPI effectiveness prior. See https://en.wikipedia.org/wiki/Asymmetric_Laplace_distribution for more information
We use the same parameterisation as used on wikipedia
"""

import jax.numpy as jnp
import jax.random as random
from jax import lax
from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import promote_shapes, validate_sample


class AsymmetricLaplace(Distribution):
    reparametrized_params = ["scale", "asymmetry"] # see https://en.wikipedia.org/wiki/Asymmetric_Laplace_distribution for parameters
    arg_constraints = {"scale": constraints.positive, "asymmetry": constraints.positive}
    support = constraints.real

    def __init__(self, scale=1.0, asymmetry=1.0, validate_args=None):
        self.scale, self.asymmetry = promote_shapes(scale, asymmetry)
        batch_shape = lax.broadcast_shapes(jnp.shape(scale), jnp.shape(asymmetry))
        super(AsymmetricLaplace, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        u = random.uniform(key, shape=sample_shape + self.batch_shape)
        s = -jnp.log((1 - u) * (1 + self.asymmetry ** 2)) / (
            self.asymmetry * self.scale
        ) * (
            u > ((self.asymmetry ** 2) / (1 + self.asymmetry ** 2))
        ) + self.asymmetry * jnp.log(
            u * (1 + self.asymmetry ** 2) / (self.asymmetry ** 2)
        ) / self.scale * (
            u < ((self.asymmetry ** 2) / (1 + self.asymmetry ** 2))
        )

        return s

    @validate_sample
    def log_prob(self, value):
        return jnp.log(self.scale / (self.asymmetry + (self.asymmetry ** -1))) + (
            -value * self.scale * jnp.sign(value) * (self.asymmetry ** jnp.sign(value))
        )

    @property
    def mean(self):
        return (1 - self.asymmetry ** 2) / (self.scale * self.asymmetry)

    @property
    def variance(self):
        return (1 + self.asymmetry ** 4) / ((self.scale * self.asymmetry) ** 2)

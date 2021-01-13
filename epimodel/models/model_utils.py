import jax
import jax.numpy as jnp
import jax.scipy.signal as jss

import numpyro
import numpyro.distributions as dist


def create_intervention_prior(nCMs, prior_type = 'trunc_normal'):
    if prior_type == 'trunc_normal':
        alpha_i = numpyro.sample("alpha_i", dist.TruncatedNormal(low=-0.1, loc=jnp.zeros(nCMs), scale=0.2))
    else:
        raise ValueError('Intervention effect prior must take a value in [trunc_normal]')
    return alpha_i

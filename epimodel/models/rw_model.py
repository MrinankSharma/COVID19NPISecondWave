import jax
import jax.numpy as jnp
import jax.scipy.signal as jss

import numpyro
import numpyro.distributions as dist

from .model_utils import (
    get_discrete_renewal_transition,
    observe_cases_deaths,
    setup_dr_infection_model,
)


def rw_model(data, ep, r_walk_noise_scale=0.15, noise_scale_period=7, **kwargs):
    # looking at most places in UK, austria, R estimates from the imperial report seem to be at about 1 in local areas
    # setting it to be at about 1 seems pretty reasonable to me.
    basic_R = numpyro.sample(
        "basic_R",
        dist.TruncatedNormal(low=0.1, loc=1.1 * jnp.ones(data.nRs), scale=0.2),
    )

    # number of 'noise points'
    nNP = int(data.nDs / noise_scale_period) + 1
    noisepoint_log_Rt_noise_series = numpyro.sample(
        "noisepoint_log_Rt_noise_series", dist.Normal(loc=jnp.zeros((data.nRs, nNP)))
    )
    log_Rt_noise = jnp.repeat(
        jnp.cumsum(r_walk_noise_scale * noisepoint_log_Rt_noise_series, axis=-1),
        noise_scale_period,
        axis=-1,
    )[: data.nRs, : data.nDs]
    log_Rt = jnp.log(basic_R.reshape((data.nRs, 1))) + log_Rt_noise
    Rt = numpyro.deterministic("Rt", jnp.exp(log_Rt))  # nRs x nDs

    (
        init_infections,
        total_infections,
        infection_noise,
        seeding_padding,
    ) = setup_dr_infection_model(data, ep)
    discrete_renewal_transition = get_discrete_renewal_transition(ep)

    # we need to transpose R because jax.lax.scan scans over the first dimension. We want to scan over time
    # we also will transpose infections at the end
    _, infections = jax.lax.scan(
        discrete_renewal_transition, init_infections, [Rt.T, infection_noise.T]
    )

    total_infections = jax.ops.index_update(
        total_infections,
        jax.ops.index[:, :seeding_padding],
        init_infections[:, -seeding_padding:],
    )
    total_infections = jax.ops.index_update(
        total_infections, jax.ops.index[:, seeding_padding:], infections.T
    )

    total_infections = numpyro.deterministic("total_infections", total_infections)

    cfr = numpyro.sample("cfr", dist.Uniform(low=0.01, high=jnp.ones((data.nRs, 1))))
    future_cases_t = total_infections
    future_deaths_t = jnp.multiply(total_infections, cfr)

    ## at the moment, this is technically neglecting the very earliest infections
    expected_cases = numpyro.deterministic(
        "expected_cases",
        jss.convolve2d(future_cases_t, ep.DPCv, mode="full")[
            :, seeding_padding : data.nDs + seeding_padding
        ],
    )
    expected_deaths = numpyro.deterministic(
        "expected_deaths",
        jss.convolve2d(future_deaths_t, ep.DPDv, mode="full")[
            :, seeding_padding : data.nDs + seeding_padding
        ],
    )

    observe_cases_deaths(data, expected_cases, expected_deaths)

import jax
import jax.numpy as jnp
import jax.scipy.signal as jss

import numpyro
import numpyro.distributions as dist

from epimodel.models.model_utils import (
    create_intervention_prior,
    get_discrete_renewal_transition,
    observe_cases_deaths,
    setup_dr_infection_model,
    get_output_delay_transition,
)


def mixed_csdelay_model(
    data, ep, r_walk_noise_scale=0.15, noise_scale_period=7, **kwargs
):
    alpha_i = create_intervention_prior(data.nCMs)
    cm_reduction = jnp.sum(data.active_cms * alpha_i.reshape((1, data.nCMs, 1)), axis=1)

    # looking at most places in UK, austria, R estimates from the imperial report seem to be at about 1 in local areas
    # setting it to be at about 1 seems pretty reasonable to me.
    basic_R = numpyro.sample(
        "basic_R",
        dist.TruncatedNormal(low=0.1, loc=1.5 * jnp.ones(data.nRs), scale=0.2),
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
    log_Rt = jnp.log(basic_R.reshape((data.nRs, 1))) - cm_reduction + log_Rt_noise
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

    output_delay_transition = get_output_delay_transition(seeding_padding, data)

    _, expected_observations = jax.lax.scan(
        output_delay_transition,
        0.0,
        [future_cases_t, future_deaths_t, ep.DPCv_pa, ep.DPDv_pa],
    )
    expected_cases = numpyro.deterministic("expected_cases", expected_observations[0])
    expected_deaths = numpyro.deterministic("expected_deaths", expected_observations[1])

    observe_cases_deaths(data, expected_cases, expected_deaths)

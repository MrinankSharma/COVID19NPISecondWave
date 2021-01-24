import jax
import jax.numpy as jnp
import jax.scipy.signal as jss

import numpyro
import numpyro.distributions as dist

from .model_utils import (
    create_intervention_prior,
    get_discrete_renewal_transition,
    observe_cases_deaths,
    setup_dr_infection_model,
)


def complex_mixed_intervention_rw_model(
    data,
    ep,
    r_walk_noise_prior_scale=0.15,
    ifr_walk_noise_prior_scale=0.1,
    iar_walk_noise_prior_scale=0.1,
    noise_scale_period=7,
    **kwargs,
):
    alpha_i = create_intervention_prior(data.nCMs)

    # full partial pooling of effects i.e., at region level
    sigma_i = numpyro.sample("sigma_i", dist.HalfNormal(0.2 * jnp.ones((1, data.nCMs))))
    alpha_ic_noise = numpyro.sample(
        "alpha_ic_noise", dist.Normal(loc=jnp.zeros((data.nRs, data.nCMs)))
    )

    alpha_ic = numpyro.deterministic(
        "alpha_ic",
        alpha_i.reshape((1, data.nCMs)).repeat(data.nRs, axis=0)
        + sigma_i * alpha_ic_noise,
    )

    cm_reduction = jnp.sum(
        data.active_cms * alpha_ic.reshape((data.nRs, data.nCMs, 1)), axis=1
    )

    # looking at most places in UK, austria, R estimates from the imperial report seem to be at about 1 in local areas
    # setting it to be at about 1 seems pretty reasonable to me.
    basic_R = numpyro.sample(
        "basic_R",
        dist.TruncatedNormal(low=0.1, loc=1.1 * jnp.ones(data.nRs), scale=0.2),
    )

    # random walk part of transmission
    r_walk_noise_scale = numpyro.sample(
        "r_walk_noise_scale", dist.HalfNormal(r_walk_noise_prior_scale)
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

    nNP = int((data.nDs + seeding_padding) / noise_scale_period) + 1

    iar_0 = 1.0
    ifr_0 = numpyro.sample("cfr", dist.Uniform(low=0.01, high=jnp.ones((data.nRs, 1))))

    # random walk for IFR and IAR
    iar_walk_noise_scale = numpyro.sample(
        "iar_walk_noise_scale", dist.HalfNormal(iar_walk_noise_prior_scale)
    )
    # number of 'noise points'
    noisepoint_log_iar_noise_series = numpyro.sample(
        "noisepoint_log_iar_noise_series", dist.Normal(loc=jnp.zeros((data.nRs, nNP)))
    )
    log_iar_noise = jnp.repeat(
        jnp.cumsum(iar_walk_noise_scale * noisepoint_log_iar_noise_series, axis=-1),
        noise_scale_period,
        axis=-1,
    )[: data.nRs, : data.nDs + seeding_padding]
    iar_t = numpyro.deterministic("iar_t", iar_0 * jnp.exp(log_iar_noise))

    ifr_walk_noise_scale = numpyro.sample(
        "ifr_walk_noise_scale", dist.HalfNormal(ifr_walk_noise_prior_scale)
    )
    # number of 'noise points'
    noisepoint_log_ifr_noise_series = numpyro.sample(
        "noisepoint_log_ifr_noise_series", dist.Normal(loc=jnp.zeros((data.nRs, nNP)))
    )
    log_ifr_noise = jnp.repeat(
        jnp.cumsum(ifr_walk_noise_scale * noisepoint_log_ifr_noise_series, axis=-1),
        noise_scale_period,
        axis=-1,
    )[: data.nRs, : data.nDs + seeding_padding]
    ifr_t = numpyro.deterministic("ifr_t", ifr_0 * jnp.exp(log_ifr_noise))

    future_cases_t = jnp.multiply(total_infections, iar_t)
    future_deaths_t = jnp.multiply(total_infections, ifr_t)

    psi_cases = numpyro.sample(f"psi_cases", dist.HalfNormal(5))
    psi_deaths = numpyro.sample(f"psi_deaths", dist.HalfNormal(5))

    # inefficient loop, but only over about 6 countries. I think numpyro can handle this ok
    # hopefully the compile time ain't too bad
    for la_indices, country_name in zip(data.la_indices, data.countries):
        ## at the moment, this is technically neglecting the very earliest infections

        cd_mean = numpyro.sample(f"cd_mean_{country_name}", dist.Normal(10.0, 2.0))
        cd_disp = numpyro.sample(f"cd_disp_{country_name}", dist.Normal(5.0, 2.0))
        cd_trunc = 32
        cd = dist.GammaPoisson(concentration=cd_disp, rate=cd_disp / cd_mean)

        bins = jnp.arange(0, cd_trunc)
        pmf = jnp.exp(cd.log_prob(bins))
        pmf = pmf / jnp.sum(pmf)
        cases_delay = pmf.reshape((1, cd_trunc))

        dd_mean = numpyro.sample(f"dd_mean_{country_name}", dist.Normal(21.0, 1.0))
        dd_disp = numpyro.sample(f"dd_disp_{country_name}", dist.Normal(14.0, 5.0))
        dd_trunc = 48
        dd = dist.GammaPoisson(concentration=dd_disp, rate=dd_disp / cd_mean)

        bins = jnp.arange(0, dd_trunc)
        pmf = jnp.exp(dd.log_prob(bins))
        pmf = pmf / jnp.sum(pmf)
        deaths_delay = pmf.reshape((1, dd_trunc))

        expected_cases = numpyro.deterministic(
            f"expected_cases_{country_name}",
            jss.convolve2d(future_cases_t[la_indices, :], cases_delay, mode="full")[
                :, seeding_padding : data.nDs + seeding_padding
            ],
        )
        expected_deaths = numpyro.deterministic(
            f"expected_deaths_{country_name}",
            jss.convolve2d(future_deaths_t[la_indices, :], deaths_delay, mode="full")[
                :, seeding_padding : data.nDs + seeding_padding
            ],
        )

        with numpyro.handlers.mask(
            mask=jnp.logical_not(data.new_cases[la_indices, :].mask)
        ):
            observed_cases = numpyro.sample(
                f"observed_cases_{country_name}",
                dist.GammaPoisson(
                    concentration=psi_cases, rate=psi_cases / expected_cases
                ),
                obs=data.new_cases[la_indices, :].data,
            )

        with numpyro.handlers.mask(
            mask=jnp.logical_not(data.new_deaths[la_indices, :].mask)
        ):
            observed_deaths = numpyro.sample(
                f"observed_deaths_{country_name}",
                dist.GammaPoisson(
                    concentration=psi_deaths, rate=psi_deaths / expected_deaths
                ),
                obs=data.new_deaths[la_indices, :].data,
            )

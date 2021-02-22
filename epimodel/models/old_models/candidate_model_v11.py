import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import jax.scipy.signal

from epimodel.models.model_utils import (
    create_intervention_prior,
    create_noisescale_prior,
)

from epimodel.distributions import AsymmetricLaplace


def candidate_model_v11(
    data,
    ep,
    iar_noisescale_prior=None,
    r_walk_noise_scale_period=7,
    ir_walk_noise_scale_period=14,
    **kwargs
):

    alpha_i_tilde = numpyro.sample(
        "alpha_i_tilde", AsymmetricLaplace(asymmetry=0.5 * jnp.ones(data.nCMs), scale=5)
    )
    alpha_i = numpyro.deterministic("alpha_i", alpha_i_tilde / 5.0)

    # no more partial pooling
    cm_reduction = jnp.sum(data.active_cms * alpha_i.reshape((1, data.nCMs, 1)), axis=1)

    basic_R = numpyro.sample(
        "basic_R",
        dist.TruncatedNormal(low=0.1, loc=1.1, scale=0.3 * jnp.ones(data.nRs)),
    )

    # number of 'noise points'
    # -1 since first 2 weeks, no change.
    nNP = int(data.nDs / r_walk_noise_scale_period) - 2

    # r_walk_noise_scale_tilde = numpyro.sample("r_walk_noise_scale_tilde", dist.HalfNormal(30*0.15))
    # r_walk_noise_scale = numpyro.deterministic("r_walk_noise_scale", r_walk_noise_scale_tilde/30)

    r_walk_noise_scale = 0.125

    noisepoint_log_Rt_noise_series = numpyro.sample(
        "noisepoint_log_Rt_noise_series",
        dist.Normal(loc=jnp.zeros((data.nRs, nNP))),
    )

    log_Rt_noise = jnp.repeat(
        r_walk_noise_scale * jnp.cumsum(noisepoint_log_Rt_noise_series, axis=-1),
        r_walk_noise_scale_period,
        axis=-1,
    )[: data.nRs, : (data.nDs - 3 * r_walk_noise_scale_period)]
    full_log_Rt_noise = jnp.zeros_like(cm_reduction)
    full_log_Rt_noise = jax.ops.index_update(
        full_log_Rt_noise,
        jax.ops.index[:, 3 * r_walk_noise_scale_period :],
        log_Rt_noise,
    )

    log_Rt = jnp.log(basic_R.reshape((data.nRs, 1))) - cm_reduction + full_log_Rt_noise
    Rt = numpyro.deterministic("Rt", jnp.exp(log_Rt))  # nRs x nDs
    Rt_walk = numpyro.deterministic("Rt_walk", jnp.exp(full_log_Rt_noise))

    seeding_padding = 7
    total_padding = ep.GIv.size - 1

    log_seeding = numpyro.sample(
        "log_seeding", dist.Normal(loc=jnp.zeros((data.nRs, 1)), scale=1.0)
    )
    seeding = numpyro.deterministic("seeding", jnp.exp(2.0 * log_seeding))

    init_infections = jnp.zeros((data.nRs, total_padding))
    init_infections = jax.ops.index_add(
        init_infections,
        jax.ops.index[:, -seeding_padding:],
        jnp.repeat(seeding, seeding_padding, axis=-1),
    )

    total_infections = jnp.zeros((data.nRs, seeding_padding + data.nDs))

    def discrete_renewal_transition(infections, R):
        mean_new_infections_t = jnp.multiply(R, infections @ ep.GI_flat_rev)
        # enforce that infections remain positive with a softplus
        # and enforce that there is always some noise, even at small noise-scale levels
        new_infections = infections
        new_infections = jax.ops.index_update(
            new_infections, jax.ops.index[:, :-1], infections[:, 1:]
        )
        new_infections = jax.ops.index_update(
            new_infections, jax.ops.index[:, -1], mean_new_infections_t
        )
        return new_infections, mean_new_infections_t

    # we need to transpose R because jax.lax.scan scans over the first dimension. We want to scan over time
    # we also will transpose infections at the end
    _, infections = jax.lax.scan(
        discrete_renewal_transition,
        init_infections,
        Rt.T,
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

    # country level random walks for IFR/IAR changes. Note: country level, **not** area level.
    iar_0 = 1.0
    ifr_0_tilde = numpyro.sample(
        "ifr_0_tilde", dist.Normal(loc=jnp.zeros((data.nCs, 1)))
    )
    ifr_0 = numpyro.deterministic("ifr_0", jax.nn.relu((ifr_0_tilde * 0.001) + 0.15))

    # number of "noisepoints" for these walks
    nNP = int((data.nDs + seeding_padding) / ir_walk_noise_scale_period) - 1

    iar_noise_scale = 0.1

    noisepoint_log_iar_noise_series = numpyro.sample(
        "noisepoint_log_iar_noise_series",
        dist.Normal(loc=jnp.zeros((data.nCs, nNP)), scale=1.0),
    )
    noisepoint_log_ifr_noise_series = numpyro.sample(
        "noisepoint_log_ifr_noise_series",
        dist.Normal(loc=jnp.zeros((data.nCs, nNP)), scale=1.0),
    )

    iar_noise = jnp.repeat(
        iar_noise_scale * jnp.cumsum(noisepoint_log_iar_noise_series, axis=-1),
        ir_walk_noise_scale_period,
        axis=-1,
    )[: data.nCs, : data.nDs + seeding_padding - (2 * ir_walk_noise_scale_period)]
    ifr_noise = jnp.repeat(
        iar_noise_scale * jnp.cumsum(noisepoint_log_ifr_noise_series, axis=-1),
        ir_walk_noise_scale_period,
        axis=-1,
    )[: data.nCs, : data.nDs + seeding_padding - (2 * ir_walk_noise_scale_period)]

    full_iar_noise = jnp.zeros((data.nCs, data.nDs + seeding_padding))
    full_ifr_noise = jnp.zeros((data.nCs, data.nDs + seeding_padding))
    full_iar_noise = jax.ops.index_update(
        full_iar_noise, jax.ops.index[:, 2 * ir_walk_noise_scale_period :], iar_noise
    )
    full_ifr_noise = jax.ops.index_update(
        full_ifr_noise, jax.ops.index[:, 2 * ir_walk_noise_scale_period :], ifr_noise
    )

    iar_t = numpyro.deterministic("iar_t", iar_0 * jnp.exp(full_iar_noise))
    ifr_t = numpyro.deterministic("ifr_t", ifr_0 * jnp.exp(full_ifr_noise))

    # use the `RC_mat` to pull the country level change in the rates for the relevant local area
    future_cases_t = numpyro.deterministic(
        "future_cases_t", jnp.multiply(total_infections, data.RC_mat @ iar_t)
    )
    future_deaths_t = numpyro.deterministic(
        "future_cases_t", jnp.multiply(total_infections, data.RC_mat @ ifr_t)
    )

    def output_delay_transition(loop_carry, scan_slice):
        # this scan function scans over local areas, using their country specific delay, rather than over days
        # we don't need a loop carry, so we just return 0 and ignore the loop carry!
        (
            future_cases,
            future_deaths,
            country_cases_delay,
            country_deaths_delay,
        ) = scan_slice
        expected_cases = jax.scipy.signal.convolve(
            future_cases, country_cases_delay, mode="full"
        )[seeding_padding : data.nDs + seeding_padding]
        expected_deaths = jax.scipy.signal.convolve(
            future_deaths, country_deaths_delay, mode="full"
        )[seeding_padding : data.nDs + seeding_padding]

        return 0.0, (expected_cases, expected_deaths)

    region_cases_delays, region_deaths_delays = ep.get_region_delays()

    _, (expected_cases, expected_deaths) = jax.lax.scan(
        output_delay_transition,
        0,
        [future_cases_t, future_deaths_t, region_cases_delays, region_deaths_delays],
    )

    expected_cases = numpyro.deterministic("expected_cases", expected_cases)
    expected_deaths = numpyro.deterministic("expected_deaths", expected_deaths)

    # make a psi cases and deaths for every country. later, we will use the 'RC' mat
    # to pull the local area value.
    psi_cases = numpyro.sample(
        "psi_cases", dist.HalfNormal(scale=jnp.ones(len(data.unique_Cs)))
    )
    psi_deaths = numpyro.sample(
        "psi_deaths", dist.HalfNormal(scale=jnp.ones(len(data.unique_Cs)))
    )

    cases_conc = 5 * (
        (data.RC_mat @ psi_cases).reshape((data.nRs, 1)).repeat(data.nDs, axis=-1)
    )
    deaths_conc = 5 * (
        (data.RC_mat @ psi_deaths).reshape((data.nRs, 1)).repeat(data.nDs, axis=-1)
    )

    with numpyro.handlers.mask(mask=jnp.logical_not(data.new_cases.mask)):
        observed_cases = numpyro.sample(
            "observed_cases",
            dist.GammaPoisson(
                concentration=cases_conc,
                rate=cases_conc / expected_cases,
            ),
            obs=data.new_cases.data,
        )

    with numpyro.handlers.mask(mask=jnp.logical_not(data.new_deaths.mask)):
        observed_deaths = numpyro.sample(
            "observed_deaths",
            dist.GammaPoisson(
                concentration=deaths_conc,
                rate=deaths_conc / expected_deaths,
            ),
            obs=data.new_deaths.data,
        )


def candidate_model_v10_reparam(
    data,
    ep,
    iar_noisescale_prior=None,
    r_walk_noise_scale_period=7,
    ir_walk_noise_scale_period=14,
    **kwargs
):

    basic_R_noise = numpyro.sample(
        "basic_R_noise", dist.Normal(loc=jnp.zeros(data.nRs), scale=1.0)
    )
    basic_R = numpyro.deterministic("basic_R", jax.nn.relu(1.1 + (basic_R_noise * 0.3)))

    # number of 'noise points'
    # -1 since first 2 weeks, no change.
    nNP = int(data.nDs / r_walk_noise_scale_period) - 2

    r_walk_noise_scale_tilde = numpyro.sample(
        "r_walk_noise_scale_tilde", dist.HalfNormal(1.0)
    )

    r_walk_noise_scale = numpyro.deterministic(
        "r_walk_noise_scale", 0.15 * r_walk_noise_scale_tilde
    )

    noisepoint_log_Rt_noise_series = numpyro.sample(
        "noisepoint_log_Rt_noise_series",
        dist.Normal(loc=jnp.zeros((data.nRs, nNP))),
    )

    log_Rt_noise = jnp.repeat(
        r_walk_noise_scale * jnp.cumsum(noisepoint_log_Rt_noise_series, axis=-1),
        r_walk_noise_scale_period,
        axis=-1,
    )[: data.nRs, : (data.nDs - 3 * r_walk_noise_scale_period)]
    full_log_Rt_noise = jnp.zeros((data.nRs, data.nDs))
    full_log_Rt_noise = jax.ops.index_update(
        full_log_Rt_noise,
        jax.ops.index[:, 3 * r_walk_noise_scale_period :],
        log_Rt_noise,
    )

    log_Rt = jnp.log(basic_R.reshape((data.nRs, 1))) - full_log_Rt_noise
    Rt = numpyro.deterministic("Rt", jnp.exp(log_Rt))  # nRs x nDs
    Rt_walk = numpyro.deterministic("Rt_walk", jnp.exp(full_log_Rt_noise))

    seeding_padding = 7
    total_padding = ep.GIv.size - 1

    # seeding is a lognormal, but reparameterized
    seeding_tilde = numpyro.sample(
        "seeding_tilde", dist.Normal(jnp.zeros((data.nRs, 1)))
    )
    seeding = numpyro.deterministic("seeding", jnp.exp(seeding_tilde * 2.0))
    init_infections = jnp.zeros((data.nRs, total_padding))
    init_infections = jax.ops.index_add(
        init_infections,
        jax.ops.index[:, -seeding_padding:],
        jnp.repeat(seeding, seeding_padding, axis=-1),
    )

    total_infections = jnp.zeros((data.nRs, seeding_padding + data.nDs))

    def discrete_renewal_transition(infections, R):
        mean_new_infections_t = jnp.multiply(R, infections @ ep.GI_flat_rev)
        # enforce that infections remain positive with a softplus
        # and enforce that there is always some noise, even at small noise-scale levels
        new_infections = infections
        new_infections = jax.ops.index_update(
            new_infections, jax.ops.index[:, :-1], infections[:, 1:]
        )
        new_infections = jax.ops.index_update(
            new_infections, jax.ops.index[:, -1], mean_new_infections_t
        )
        return new_infections, mean_new_infections_t

    # we need to transpose R because jax.lax.scan scans over the first dimension. We want to scan over time
    # we also will transpose infections at the end
    _, infections = jax.lax.scan(
        discrete_renewal_transition,
        init_infections,
        Rt.T,
    )

    # extra noise, outside the renewal equation. This noise isn't capturing variability in transmission
    # the random walk does this. It adds some robustness when the case count is very small
    infection_noise_scale = numpyro.sample(
        "infection_noise_scale", dist.HalfNormal(0.5)
    )
    # infection noise is now a normal
    infection_noise = numpyro.sample(
        "infection_noise", dist.Normal(loc=0, scale=jnp.ones((data.nRs, data.nDs)))
    )

    # enforce positivity!
    infections = jax.nn.softplus(
        infections + (infection_noise_scale * infection_noise.T)
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

    # country level random walks for IFR/IAR changes. Note: country level, **not** area level.
    iar_0 = 1.0
    ifr_0 = numpyro.sample(
        "ifr_0", dist.TruncatedNormal(low=1e-3, loc=0.2 * jnp.ones((data.nCs, 1)))
    )

    # number of "noisepoints" for these walks
    nNP = int((data.nDs + seeding_padding) / ir_walk_noise_scale_period) - 1

    # iar_noise_scale_tilde = numpyro.sample(
    #     "iar_noise_scale_tilde", dist.HalfNormal(1)
    # )
    # iar_noise_scale = numpyro.deterministic(
    #     "iar_noise_scale", iar_noise_scale_tilde * 0.01
    # )

    iar_noise_scale = 0.1

    noisepoint_log_iar_noise_series = numpyro.sample(
        "noisepoint_log_iar_noise_series",
        dist.Normal(loc=jnp.zeros((data.nCs, nNP)), scale=1),
    )
    noisepoint_log_ifr_noise_series = numpyro.sample(
        "noisepoint_log_ifr_noise_series",
        dist.Normal(loc=jnp.zeros((data.nCs, nNP)), scale=1),
    )

    iar_noise = jnp.repeat(
        iar_noise_scale * jnp.cumsum(noisepoint_log_iar_noise_series, axis=-1),
        ir_walk_noise_scale_period,
        axis=-1,
    )[: data.nCs, : data.nDs + seeding_padding - (2 * ir_walk_noise_scale_period)]
    ifr_noise = jnp.repeat(
        iar_noise_scale * jnp.cumsum(noisepoint_log_ifr_noise_series, axis=-1),
        ir_walk_noise_scale_period,
        axis=-1,
    )[: data.nCs, : data.nDs + seeding_padding - (2 * ir_walk_noise_scale_period)]

    full_iar_noise = jnp.zeros((data.nCs, data.nDs + seeding_padding))
    full_ifr_noise = jnp.zeros((data.nCs, data.nDs + seeding_padding))
    full_iar_noise = jax.ops.index_update(
        full_iar_noise,
        jax.ops.index[:, 2 * ir_walk_noise_scale_period :],
        iar_noise,
    )
    full_ifr_noise = jax.ops.index_update(
        full_ifr_noise,
        jax.ops.index[:, 2 * ir_walk_noise_scale_period :],
        ifr_noise,
    )

    iar_t = numpyro.deterministic("iar_t", iar_0 * jnp.exp(full_iar_noise))
    ifr_t = numpyro.deterministic("ifr_t", ifr_0 * jnp.exp(full_ifr_noise))

    # use the `RC_mat` to pull the country level change in the rates for the relevant local area
    future_cases_t = numpyro.deterministic(
        "future_cases_t", jnp.multiply(total_infections, data.RC_mat @ iar_t)
    )
    future_deaths_t = numpyro.deterministic(
        "future_cases_t", jnp.multiply(total_infections, data.RC_mat @ ifr_t)
    )

    def output_delay_transition(loop_carry, scan_slice):
        # this scan function scans over local areas, using their country specific delay, rather than over days
        # we don't need a loop carry, so we just return 0 and ignore the loop carry!
        (
            future_cases,
            future_deaths,
            country_cases_delay,
            country_deaths_delay,
        ) = scan_slice
        expected_cases = jax.scipy.signal.convolve(
            future_cases, country_cases_delay, mode="full"
        )[seeding_padding : data.nDs + seeding_padding]
        expected_deaths = jax.scipy.signal.convolve(
            future_deaths, country_deaths_delay, mode="full"
        )[seeding_padding : data.nDs + seeding_padding]

        return 0.0, (expected_cases, expected_deaths)

    region_cases_delays, region_deaths_delays = ep.get_region_delays()

    _, (expected_cases, expected_deaths) = jax.lax.scan(
        output_delay_transition,
        0,
        [
            future_cases_t,
            future_deaths_t,
            region_cases_delays,
            region_deaths_delays,
        ],
    )

    expected_cases = numpyro.deterministic("expected_cases", expected_cases)
    expected_deaths = numpyro.deterministic("expected_deaths", expected_deaths)

    # prior placed over 1/sqrt(psi), as recommended by the Stan prior choice
    # wikipedia page
    cases_scale = numpyro.sample(
        "cases_scale", dist.HalfNormal(scale=jnp.ones(len(data.unique_Cs)))
    )
    deaths_scale = numpyro.sample(
        "deaths_scale", dist.HalfNormal(scale=jnp.ones(len(data.unique_Cs)))
    )
    psi_cases = numpyro.deterministic("psi_cases", (1 / cases_scale) ** 2)
    psi_deaths = numpyro.deterministic("psi_deaths", (1 / deaths_scale) ** 2)

    cases_conc = (
        (data.RC_mat @ psi_cases).reshape((data.nRs, 1)).repeat(data.nDs, axis=-1)
    )
    deaths_conc = (
        (data.RC_mat @ psi_deaths).reshape((data.nRs, 1)).repeat(data.nDs, axis=-1)
    )

    with numpyro.handlers.mask(mask=jnp.logical_not(data.new_cases.mask)):
        observed_cases = numpyro.sample(
            "observed_cases",
            dist.GammaPoisson(
                concentration=cases_conc,
                rate=cases_conc / expected_cases,
            ),
            obs=data.new_cases.data,
        )

    with numpyro.handlers.mask(mask=jnp.logical_not(data.new_deaths.mask)):
        observed_deaths = numpyro.sample(
            "observed_deaths",
            dist.GammaPoisson(
                concentration=deaths_conc,
                rate=deaths_conc / expected_deaths,
            ),
            obs=data.new_deaths.data,
        )

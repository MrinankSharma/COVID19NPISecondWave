import jax.scipy.signal
import jax.numpy as jnp
import jax

from epimodel.models.model_build_utils import *


def default_model(
    data,
    ep,
    intervention_prior=None,
    basic_R_prior=None,
    r_walk_noise_scale_prior=0.15,
    r_walk_period=7,
    n_days_seeding=7,
    seeding_scale=3.0,
    infection_noise_scale=5.0,
    output_noise_scale_prior=5.0,
    **kwargs,
):
    for k in kwargs.keys():
        print(f"{k} is not being used")

    # First, compute R.
    # sample intervention effects from their priors.
    # mean intervention effects
    alpha_i = sample_intervention_effects(data.nCMs, intervention_prior)
    # transmission reduction
    cm_reduction = jnp.sum(data.active_cms * alpha_i.reshape((1, data.nCMs, 1)), axis=1)

    basic_R = sample_basic_R(data.nRs, basic_R_prior)

    # number of 'noise points'
    # -2 since no change for the first 3 weeks.
    nNP = int(data.nDs / r_walk_period) - 1

    r_walk_noise_scale = numpyro.sample(
        "r_walk_noise_scale", dist.HalfNormal(scale=r_walk_noise_scale_prior)
    )

    # rescaling variables by 10 for better NUTS adaptation
    r_walk_noise = numpyro.sample(
        "r_walk_noise",
        dist.Normal(loc=jnp.zeros((data.nRs, nNP)), scale=1.0 / 10),
    )

    # only apply the noise every "r_walk_period" - to get full noise, repeat
    expanded_r_walk_noise = jnp.repeat(
        r_walk_noise_scale * 10.0 * jnp.cumsum(r_walk_noise, axis=-1),
        r_walk_period,
        axis=-1,
    )[: data.nRs, : (data.nDs - 2 * r_walk_period)]
    # except that we assume no noise for the first 3 weeks
    full_log_Rt_noise = jnp.zeros((data.nRs, data.nDs))
    full_log_Rt_noise = jax.ops.index_update(
        full_log_Rt_noise, jax.ops.index[:, 2 * r_walk_period :], expanded_r_walk_noise
    )

    Rt = numpyro.deterministic(
        "Rt",
        jnp.exp(
            jnp.log(basic_R.reshape((data.nRs, 1))) + full_log_Rt_noise - cm_reduction
        ),
    )

    # collect variables in the numpyro trace
    numpyro.deterministic("Rt_walk", jnp.exp(full_log_Rt_noise))
    numpyro.deterministic(
        "Rt_cm", jnp.exp(jnp.log(basic_R.reshape((data.nRs, 1))) - cm_reduction)
    )

    # Infection Model
    seeding_padding = n_days_seeding
    total_padding = ep.GIv.size - 1

    # note; seeding is also rescaled
    init_infections, total_infections_placeholder = seed_infections(
        seeding_scale, data.nRs, data.nDs, seeding_padding, total_padding
    )
    discrete_renewal_transition = get_discrete_renewal_transition(ep)

    # we need to transpose R because jax.lax.scan scans over the first dimension.
    # We want to scan over time instead of regions!
    _, infections = jax.lax.scan(
        discrete_renewal_transition,
        init_infections,
        Rt.T,
    )

    # corrupt infections with additive noise, adding robustness at small case and death
    # counts
    infection_noise = numpyro.sample(
        "infection_noise",
        dist.Normal(loc=0, scale=0.1 * jnp.ones((data.nRs, data.nDs))),
    )
    # enforce positivity!
    infections = jax.nn.softplus(
        infections + (infection_noise_scale * (10.0 * infection_noise.T))
    )

    total_infections = jax.ops.index_update(
        total_infections_placeholder,
        jax.ops.index[:, :seeding_padding],
        init_infections[:, -seeding_padding:],
    )
    total_infections = numpyro.deterministic(
        "total_infections",
        jax.ops.index_update(
            total_infections, jax.ops.index[:, seeding_padding:], infections.T
        ),
    )

    # Time constant case fatality rate (ascertainment rate assumed to be 1
    # throughout the whole period).
    cfr = numpyro.sample("cfr", dist.Uniform(low=1e-3, high=jnp.ones((data.nRs, 1))))

    future_cases_t = numpyro.deterministic("future_cases_t", total_infections)
    future_deaths_t = numpyro.deterministic(
        "future_deaths_t", jnp.multiply(total_infections, cfr)
    )

    # collect expected cases and deaths
    expected_cases = numpyro.deterministic(
        "expected_cases",
        jax.scipy.signal.convolve2d(future_cases_t, ep.DPC, mode="full")[
            :, seeding_padding : seeding_padding + data.nDs
        ],
    )

    expected_deaths = numpyro.deterministic(
        "expected_deaths",
        jax.scipy.signal.convolve2d(future_deaths_t, ep.DPD, mode="full")[
            :, seeding_padding : seeding_padding + data.nDs
        ],
    )

    # country specific psi cases and deaths.
    # We will use the 'RC' matrix to pull the correct local area value.
    psi_cases = numpyro.sample(
        "psi_cases",
        dist.HalfNormal(scale=output_noise_scale_prior * jnp.ones(len(data.unique_Cs))),
    )
    psi_deaths = numpyro.sample(
        "psi_deaths",
        dist.HalfNormal(scale=output_noise_scale_prior * jnp.ones(len(data.unique_Cs))),
    )

    # use the per country psi_cases and psi_deaths and form a nRs x nDs array
    # to use for the output distribution.
    cases_conc = (
        (data.RC_mat @ psi_cases).reshape((data.nRs, 1)).repeat(data.nDs, axis=-1)
    )
    deaths_conc = (
        (data.RC_mat @ psi_deaths).reshape((data.nRs, 1)).repeat(data.nDs, axis=-1)
    )

    with numpyro.handlers.mask(mask=jnp.logical_not(data.new_cases.mask)):
        numpyro.sample(
            "observed_cases",
            dist.GammaPoisson(
                concentration=cases_conc,
                rate=cases_conc / expected_cases,
            ),
            obs=data.new_cases.data,
        )

    with numpyro.handlers.mask(mask=jnp.logical_not(data.new_deaths.mask)):
        numpyro.sample(
            "observed_deaths",
            dist.GammaPoisson(
                concentration=deaths_conc,
                rate=deaths_conc / expected_deaths,
            ),
            obs=data.new_deaths.data,
        )


def default_model_uk_ifriar(
    data,
    ep,
    intervention_prior=None,
    basic_R_prior=None,
    r_walk_noise_scale_prior=0.15,
    r_walk_period=7,
    n_days_seeding=7,
    seeding_scale=3.0,
    infection_noise_scale=5.0,
    output_noise_scale_prior=5.0,
    **kwargs,
):
    for k in kwargs.keys():
        print(f"{k} is not being used")

    # First, compute R.
    # sample intervention effects from their priors.
    # mean intervention effects
    alpha_i = sample_intervention_effects(data.nCMs, intervention_prior)
    # transmission reduction
    cm_reduction = jnp.sum(data.active_cms * alpha_i.reshape((1, data.nCMs, 1)), axis=1)

    basic_R = sample_basic_R(data.nRs, basic_R_prior)

    # number of 'noise points'
    # -2 since no change for the first 3 weeks.
    nNP = int(data.nDs / r_walk_period) - 1

    r_walk_noise_scale = numpyro.sample(
        "r_walk_noise_scale", dist.HalfNormal(scale=r_walk_noise_scale_prior)
    )

    # rescaling variables by 10 for better NUTS adaptation
    r_walk_noise = numpyro.sample(
        "r_walk_noise",
        dist.Normal(loc=jnp.zeros((data.nRs, nNP)), scale=1.0 / 10),
    )

    # only apply the noise every "r_walk_period" - to get full noise, repeat
    expanded_r_walk_noise = jnp.repeat(
        r_walk_noise_scale * 10.0 * jnp.cumsum(r_walk_noise, axis=-1),
        r_walk_period,
        axis=-1,
    )[: data.nRs, : (data.nDs - 2 * r_walk_period)]
    # except that we assume no noise for the first 3 weeks
    full_log_Rt_noise = jnp.zeros((data.nRs, data.nDs))
    full_log_Rt_noise = jax.ops.index_update(
        full_log_Rt_noise, jax.ops.index[:, 2 * r_walk_period :], expanded_r_walk_noise
    )

    Rt = numpyro.deterministic(
        "Rt",
        jnp.exp(
            jnp.log(basic_R.reshape((data.nRs, 1))) + full_log_Rt_noise - cm_reduction
        ),
    )

    # collect variables in the numpyro trace
    numpyro.deterministic("Rt_walk", jnp.exp(full_log_Rt_noise))
    numpyro.deterministic(
        "Rt_cm", jnp.exp(jnp.log(basic_R.reshape((data.nRs, 1))) - cm_reduction)
    )

    # Infection Model
    seeding_padding = n_days_seeding
    total_padding = ep.GIv.size - 1

    # note; seeding is also rescaled
    init_infections, total_infections_placeholder = seed_infections(
        seeding_scale, data.nRs, data.nDs, seeding_padding, total_padding
    )
    discrete_renewal_transition = get_discrete_renewal_transition(ep)

    # we need to transpose R because jax.lax.scan scans over the first dimension.
    # We want to scan over time instead of regions!
    _, infections = jax.lax.scan(
        discrete_renewal_transition,
        init_infections,
        Rt.T,
    )

    # corrupt infections with additive noise, adding robustness at small case and death
    # counts
    infection_noise = numpyro.sample(
        "infection_noise",
        dist.Normal(loc=0, scale=0.1 * jnp.ones((data.nRs, data.nDs))),
    )
    # enforce positivity!
    infections = jax.nn.softplus(
        infections + (infection_noise_scale * (10.0 * infection_noise.T))
    )

    total_infections = jax.ops.index_update(
        total_infections_placeholder,
        jax.ops.index[:, :seeding_padding],
        init_infections[:, -seeding_padding:],
    )
    total_infections = numpyro.deterministic(
        "total_infections",
        jax.ops.index_update(
            total_infections, jax.ops.index[:, seeding_padding:], infections.T
        ),
    )

    # Scale by fixed UK numbers
    iar_t = jnp.array(
        [
            0.32831737,
            0.32831737,
            0.32831737,
            0.32831737,
            0.32831737,
            0.32831737,
            0.32831737,
            0.32831737,
            0.32831737,
            0.32831737,
            0.32831737,
            0.32831737,
            0.32831737,
            0.32831737,
            0.32831737,
            0.32831737,
            0.32831737,
            0.32831737,
            0.32831737,
            0.41706005,
            0.41706005,
            0.41706005,
            0.41706005,
            0.41706005,
            0.41706005,
            0.41706005,
            0.41706005,
            0.41706005,
            0.41706005,
            0.41706005,
            0.41706005,
            0.41706005,
            0.41706005,
            0.49174617,
            0.49174617,
            0.49174617,
            0.49174617,
            0.49174617,
            0.49174617,
            0.49174617,
            0.49174617,
            0.49174617,
            0.49174617,
            0.49174617,
            0.49174617,
            0.49174617,
            0.49174617,
            0.55604659,
            0.55604659,
            0.55604659,
            0.55604659,
            0.55604659,
            0.55604659,
            0.55604659,
            0.55604659,
            0.55604659,
            0.55604659,
            0.55604659,
            0.55604659,
            0.55604659,
            0.55604659,
            0.59710354,
            0.59710354,
            0.59710354,
            0.59710354,
            0.59710354,
            0.59710354,
            0.59710354,
            0.59710354,
            0.59710354,
            0.59710354,
            0.59710354,
            0.59710354,
            0.59710354,
            0.59710354,
            0.61638832,
            0.61638832,
            0.61638832,
            0.61638832,
            0.61638832,
            0.61638832,
            0.61638832,
            0.61638832,
            0.61638832,
            0.61638832,
            0.61638832,
            0.61638832,
            0.61638832,
            0.61638832,
            0.61569382,
            0.61569382,
            0.61569382,
            0.61569382,
            0.61569382,
            0.61569382,
            0.61569382,
            0.61569382,
            0.61569382,
            0.61569382,
            0.61569382,
            0.61569382,
            0.61569382,
            0.61569382,
            0.60523783,
            0.60523783,
            0.60523783,
            0.60523783,
            0.60523783,
            0.60523783,
            0.60523783,
            0.60523783,
            0.60523783,
            0.60523783,
            0.60523783,
            0.60523783,
            0.60523783,
            0.60523783,
            0.58280771,
            0.58280771,
            0.58280771,
            0.58280771,
            0.58280771,
            0.58280771,
            0.58280771,
            0.58280771,
            0.58280771,
            0.58280771,
            0.58280771,
            0.58280771,
            0.58280771,
            0.58280771,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
            0.57764043,
        ]
    )

    ifr_t = jnp.array(
        [
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00858016,
            0.00780975,
            0.00780975,
            0.00780975,
            0.00780975,
            0.00780975,
            0.00780975,
            0.00780975,
            0.00780975,
            0.00780975,
            0.00780975,
            0.00780975,
            0.00780975,
            0.00780975,
            0.00780975,
            0.00707346,
            0.00707346,
            0.00707346,
            0.00707346,
            0.00707346,
            0.00707346,
            0.00707346,
            0.00707346,
            0.00707346,
            0.00707346,
            0.00707346,
            0.00707346,
            0.00707346,
            0.00707346,
            0.00626562,
            0.00626562,
            0.00626562,
            0.00626562,
            0.00626562,
            0.00626562,
            0.00626562,
            0.00626562,
            0.00626562,
            0.00626562,
            0.00626562,
            0.00626562,
            0.00626562,
            0.00626562,
            0.00602792,
            0.00602792,
            0.00602792,
            0.00602792,
            0.00602792,
            0.00602792,
            0.00602792,
            0.00602792,
            0.00602792,
            0.00602792,
            0.00602792,
            0.00602792,
            0.00602792,
            0.00602792,
            0.00616321,
            0.00616321,
            0.00616321,
            0.00616321,
            0.00616321,
            0.00616321,
            0.00616321,
            0.00616321,
            0.00616321,
            0.00616321,
            0.00616321,
            0.00616321,
            0.00616321,
            0.00616321,
            0.00670325,
            0.00670325,
            0.00670325,
            0.00670325,
            0.00670325,
            0.00670325,
            0.00670325,
            0.00670325,
            0.00670325,
            0.00670325,
            0.00670325,
            0.00670325,
            0.00670325,
            0.00670325,
            0.00748796,
            0.00748796,
            0.00748796,
            0.00748796,
            0.00748796,
            0.00748796,
            0.00748796,
            0.00748796,
            0.00748796,
            0.00748796,
            0.00748796,
            0.00748796,
            0.00748796,
            0.00748796,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
            0.00805689,
        ]
    )

    # use the `RC_mat` to pull the country level change in the rates for the relevant local area
    future_cases_t = numpyro.deterministic(
        "future_cases_t", jnp.multiply(total_infections, iar_t)
    )
    future_deaths_t = numpyro.deterministic(
        "future_cases_t", jnp.multiply(total_infections, ifr_t)
    )

    # collect expected cases and deaths
    expected_cases = numpyro.deterministic(
        "expected_cases",
        jax.scipy.signal.convolve2d(future_cases_t, ep.DPC, mode="full")[
            :, seeding_padding : seeding_padding + data.nDs
        ],
    )

    expected_deaths = numpyro.deterministic(
        "expected_deaths",
        jax.scipy.signal.convolve2d(future_deaths_t, ep.DPD, mode="full")[
            :, seeding_padding : seeding_padding + data.nDs
        ],
    )

    # country specific psi cases and deaths.
    # We will use the 'RC' matrix to pull the correct local area value.
    psi_cases = numpyro.sample(
        "psi_cases",
        dist.HalfNormal(scale=output_noise_scale_prior * jnp.ones(len(data.unique_Cs))),
    )
    psi_deaths = numpyro.sample(
        "psi_deaths",
        dist.HalfNormal(scale=output_noise_scale_prior * jnp.ones(len(data.unique_Cs))),
    )

    # use the per country psi_cases and psi_deaths and form a nRs x nDs array
    # to use for the output distribution.
    cases_conc = (
        (data.RC_mat @ psi_cases).reshape((data.nRs, 1)).repeat(data.nDs, axis=-1)
    )
    deaths_conc = (
        (data.RC_mat @ psi_deaths).reshape((data.nRs, 1)).repeat(data.nDs, axis=-1)
    )

    with numpyro.handlers.mask(mask=jnp.logical_not(data.new_cases.mask)):
        numpyro.sample(
            "observed_cases",
            dist.GammaPoisson(
                concentration=cases_conc,
                rate=cases_conc / expected_cases,
            ),
            obs=data.new_cases.data,
        )

    with numpyro.handlers.mask(mask=jnp.logical_not(data.new_deaths.mask)):
        numpyro.sample(
            "observed_deaths",
            dist.GammaPoisson(
                concentration=deaths_conc,
                rate=deaths_conc / expected_deaths,
            ),
            obs=data.new_deaths.data,
        )


def random_walk_model(
    data,
    ep,
    intervention_prior=None,
    basic_R_prior=None,
    r_walk_noise_scale_prior=0.15,
    r_walk_period=7,
    n_days_seeding=7,
    seeding_scale=3.0,
    infection_noise_scale=5.0,
    output_noise_scale_prior=5.0,
    **kwargs,
):
    for k in kwargs.keys():
        print(f"{k} is not being used")

    basic_R = sample_basic_R(data.nRs, basic_R_prior)

    # number of 'noise points'
    # -2 since no change for the first 3 weeks.
    nNP = int(data.nDs / r_walk_period) - 1

    r_walk_noise_scale = numpyro.sample(
        "r_walk_noise_scale", dist.HalfNormal(scale=r_walk_noise_scale_prior)
    )

    # rescaling variables by 10 for better NUTS adaptation
    r_walk_noise = numpyro.sample(
        "r_walk_noise",
        dist.Normal(loc=jnp.zeros((data.nRs, nNP)), scale=1.0 / 10),
    )

    # only apply the noise every "r_walk_period" - to get full noise, repeat
    expanded_r_walk_noise = jnp.repeat(
        r_walk_noise_scale * 10.0 * jnp.cumsum(r_walk_noise, axis=-1),
        r_walk_period,
        axis=-1,
    )[: data.nRs, : (data.nDs - 2 * r_walk_period)]
    # except that we assume no noise for the first 3 weeks
    full_log_Rt_noise = jnp.zeros((data.nRs, data.nDs))
    full_log_Rt_noise = jax.ops.index_update(
        full_log_Rt_noise, jax.ops.index[:, 2 * r_walk_period :], expanded_r_walk_noise
    )

    Rt = numpyro.deterministic(
        "Rt",
        jnp.exp(jnp.log(basic_R.reshape((data.nRs, 1))) + full_log_Rt_noise),
    )

    # collect variables in the numpyro trace
    numpyro.deterministic("Rt_walk", jnp.exp(full_log_Rt_noise))
    numpyro.deterministic("Rt_cm", jnp.exp(jnp.log(basic_R.reshape((data.nRs, 1)))))

    # Infection Model
    seeding_padding = n_days_seeding
    total_padding = ep.GIv.size - 1

    # note; seeding is also rescaled
    init_infections, total_infections_placeholder = seed_infections(
        seeding_scale, data.nRs, data.nDs, seeding_padding, total_padding
    )
    discrete_renewal_transition = get_discrete_renewal_transition(ep)

    # we need to transpose R because jax.lax.scan scans over the first dimension.
    # We want to scan over time instead of regions!
    _, infections = jax.lax.scan(
        discrete_renewal_transition,
        init_infections,
        Rt.T,
    )

    # corrupt infections with additive noise, adding robustness at small case and death
    # counts
    infection_noise = numpyro.sample(
        "infection_noise",
        dist.Normal(loc=0, scale=0.1 * jnp.ones((data.nRs, data.nDs))),
    )
    # enforce positivity!
    infections = jax.nn.softplus(
        infections + (infection_noise_scale * (10.0 * infection_noise.T))
    )

    total_infections = jax.ops.index_update(
        total_infections_placeholder,
        jax.ops.index[:, :seeding_padding],
        init_infections[:, -seeding_padding:],
    )
    total_infections = numpyro.deterministic(
        "total_infections",
        jax.ops.index_update(
            total_infections, jax.ops.index[:, seeding_padding:], infections.T
        ),
    )

    # Time constant case fatality rate (ascertainment rate assumed to be 1
    # throughout the whole period).
    cfr = numpyro.sample("cfr", dist.Uniform(low=1e-3, high=jnp.ones((data.nRs, 1))))

    future_cases_t = numpyro.deterministic("future_cases_t", total_infections)
    future_deaths_t = numpyro.deterministic(
        "future_deaths_t", jnp.multiply(total_infections, cfr)
    )

    # collect expected cases and deaths
    expected_cases = numpyro.deterministic(
        "expected_cases",
        jax.scipy.signal.convolve2d(future_cases_t, ep.DPC, mode="full")[
            :, seeding_padding : seeding_padding + data.nDs
        ],
    )

    expected_deaths = numpyro.deterministic(
        "expected_deaths",
        jax.scipy.signal.convolve2d(future_deaths_t, ep.DPD, mode="full")[
            :, seeding_padding : seeding_padding + data.nDs
        ],
    )

    # country specific psi cases and deaths.
    # We will use the 'RC' matrix to pull the correct local area value.
    psi_cases = numpyro.sample(
        "psi_cases",
        dist.HalfNormal(scale=output_noise_scale_prior * jnp.ones(len(data.unique_Cs))),
    )
    psi_deaths = numpyro.sample(
        "psi_deaths",
        dist.HalfNormal(scale=output_noise_scale_prior * jnp.ones(len(data.unique_Cs))),
    )

    # use the per country psi_cases and psi_deaths and form a nRs x nDs array
    # to use for the output distribution.
    cases_conc = (
        (data.RC_mat @ psi_cases).reshape((data.nRs, 1)).repeat(data.nDs, axis=-1)
    )
    deaths_conc = (
        (data.RC_mat @ psi_deaths).reshape((data.nRs, 1)).repeat(data.nDs, axis=-1)
    )

    with numpyro.handlers.mask(mask=jnp.logical_not(data.new_cases.mask)):
        numpyro.sample(
            "observed_cases",
            dist.GammaPoisson(
                concentration=cases_conc,
                rate=cases_conc / expected_cases,
            ),
            obs=data.new_cases.data,
        )

    with numpyro.handlers.mask(mask=jnp.logical_not(data.new_deaths.mask)):
        numpyro.sample(
            "observed_deaths",
            dist.GammaPoisson(
                concentration=deaths_conc,
                rate=deaths_conc / expected_deaths,
            ),
            obs=data.new_deaths.data,
        )


def default_model_alt_seeding(
    data,
    ep,
    intervention_prior=None,
    basic_R_prior=None,
    r_walk_noise_scale_prior=0.15,
    r_walk_period=7,
    n_days_seeding=7,
    seeding_scale=3.0,
    infection_noise_scale=5.0,
    output_noise_scale_prior=5.0,
    **kwargs,
):
    for k in kwargs.keys():
        print(f"{k} is not being used")

    # First, compute R.
    # sample intervention effects from their priors.
    # mean intervention effects
    alpha_i = sample_intervention_effects(data.nCMs, intervention_prior)
    # transmission reduction
    cm_reduction = jnp.sum(data.active_cms * alpha_i.reshape((1, data.nCMs, 1)), axis=1)

    basic_R = sample_basic_R(data.nRs, basic_R_prior)

    # number of 'noise points'
    # -2 since no change for the first 3 weeks.
    nNP = int(data.nDs / r_walk_period) - 1

    r_walk_noise_scale = numpyro.sample(
        "r_walk_noise_scale", dist.HalfNormal(scale=r_walk_noise_scale_prior)
    )

    # rescaling variables by 10 for better NUTS adaptation
    r_walk_noise = numpyro.sample(
        "r_walk_noise",
        dist.Normal(loc=jnp.zeros((data.nRs, nNP)), scale=1.0 / 10),
    )

    # only apply the noise every "r_walk_period" - to get full noise, repeat
    expanded_r_walk_noise = jnp.repeat(
        r_walk_noise_scale * 10.0 * jnp.cumsum(r_walk_noise, axis=-1),
        r_walk_period,
        axis=-1,
    )[: data.nRs, : (data.nDs - 2 * r_walk_period)]
    # except that we assume no noise for the first 3 weeks
    full_log_Rt_noise = jnp.zeros((data.nRs, data.nDs))
    full_log_Rt_noise = jax.ops.index_update(
        full_log_Rt_noise, jax.ops.index[:, 2 * r_walk_period :], expanded_r_walk_noise
    )

    Rt = numpyro.deterministic(
        "Rt",
        jnp.exp(
            jnp.log(basic_R.reshape((data.nRs, 1))) + full_log_Rt_noise - cm_reduction
        ),
    )

    # collect variables in the numpyro trace
    numpyro.deterministic("Rt_walk", jnp.exp(full_log_Rt_noise))
    numpyro.deterministic(
        "Rt_cm", jnp.exp(jnp.log(basic_R.reshape((data.nRs, 1))) - cm_reduction)
    )

    # Infection Model
    seeding_padding = n_days_seeding
    total_padding = ep.GIv.size - 1

    # note; seeding is also rescaled
    total_infections_placeholder = jnp.zeros((data.nRs, seeding_padding + data.nDs))
    seeding = numpyro.sample("seeding", dist.LogNormal(jnp.zeros((data.nRs, 1)), 1.0))
    seeding = numpyro.sample("seeding", dist.LogNormal(jnp.zeros((data.nRs, 1)), 1.0))
    init_infections = jnp.zeros((data.nRs, total_padding))
    init_infections = jax.ops.index_add(
        init_infections,
        jax.ops.index[:, -seeding_padding:],
        jnp.repeat(seeding ** seeding_scale, seeding_padding, axis=-1),
    )
    discrete_renewal_transition = get_discrete_renewal_transition(ep)

    # we need to transpose R because jax.lax.scan scans over the first dimension.
    # We want to scan over time instead of regions!
    _, infections = jax.lax.scan(
        discrete_renewal_transition,
        init_infections,
        Rt.T,
    )

    # corrupt infections with additive noise, adding robustness at small case and death
    # counts
    infection_noise = numpyro.sample(
        "infection_noise",
        dist.Normal(loc=0, scale=0.1 * jnp.ones((data.nRs, data.nDs))),
    )
    # enforce positivity!
    infections = jax.nn.softplus(
        infections + (infection_noise_scale * (10.0 * infection_noise.T))
    )

    total_infections = jax.ops.index_update(
        total_infections_placeholder,
        jax.ops.index[:, :seeding_padding],
        init_infections[:, -seeding_padding:],
    )
    total_infections = numpyro.deterministic(
        "total_infections",
        jax.ops.index_update(
            total_infections, jax.ops.index[:, seeding_padding:], infections.T
        ),
    )

    # Time constant case fatality rate (ascertainment rate assumed to be 1
    # throughout the whole period).
    cfr = numpyro.sample("cfr", dist.Uniform(low=1e-3, high=jnp.ones((data.nRs, 1))))

    future_cases_t = numpyro.deterministic("future_cases_t", total_infections)
    future_deaths_t = numpyro.deterministic(
        "future_deaths_t", jnp.multiply(total_infections, cfr)
    )

    # collect expected cases and deaths
    expected_cases = numpyro.deterministic(
        "expected_cases",
        jax.scipy.signal.convolve2d(future_cases_t, ep.DPC, mode="full")[
            :, seeding_padding : seeding_padding + data.nDs
        ],
    )

    expected_deaths = numpyro.deterministic(
        "expected_deaths",
        jax.scipy.signal.convolve2d(future_deaths_t, ep.DPD, mode="full")[
            :, seeding_padding : seeding_padding + data.nDs
        ],
    )

    # country specific psi cases and deaths.
    # We will use the 'RC' matrix to pull the correct local area value.
    psi_cases = numpyro.sample(
        "psi_cases",
        dist.HalfNormal(scale=output_noise_scale_prior * jnp.ones(len(data.unique_Cs))),
    )
    psi_deaths = numpyro.sample(
        "psi_deaths",
        dist.HalfNormal(scale=output_noise_scale_prior * jnp.ones(len(data.unique_Cs))),
    )

    # use the per country psi_cases and psi_deaths and form a nRs x nDs array
    # to use for the output distribution.
    cases_conc = (
        (data.RC_mat @ psi_cases).reshape((data.nRs, 1)).repeat(data.nDs, axis=-1)
    )
    deaths_conc = (
        (data.RC_mat @ psi_deaths).reshape((data.nRs, 1)).repeat(data.nDs, axis=-1)
    )

    with numpyro.handlers.mask(mask=jnp.logical_not(data.new_cases.mask)):
        numpyro.sample(
            "observed_cases",
            dist.GammaPoisson(
                concentration=cases_conc,
                rate=cases_conc / expected_cases,
            ),
            obs=data.new_cases.data,
        )

    with numpyro.handlers.mask(mask=jnp.logical_not(data.new_deaths.mask)):
        numpyro.sample(
            "observed_deaths",
            dist.GammaPoisson(
                concentration=deaths_conc,
                rate=deaths_conc / expected_deaths,
            ),
            obs=data.new_deaths.data,
        )

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from epimodel.models.model_utils import (
    create_basic_R_prior,
    create_intervention_prior,
    create_noisescale_prior,
    create_partial_pooling_prior,
    get_output_delay_transition,
)


def candidate_model(
    data,
    ep,
    intervention_prior=None,
    partial_pooling_prior=None,
    r_walk_noisescale_prior=None,
    ifr_noisescale_prior=None,
    iar_noisescale_prior=None,
    basic_r_prior=None,
    r_walk_noise_scale_period=7,
    ir_walk_noise_scale_period=14,
    **kwargs
):
    # partial pool over countries now. i.e., share effects across countries!
    alpha_i = create_intervention_prior(data.nCMs, intervention_prior)
    # full partial pooling of effects i.e., at region level
    sigma_i = create_partial_pooling_prior(data.nCMs, partial_pooling_prior)

    alpha_ic_noise = numpyro.sample(
        "alpha_ic_noise", dist.Normal(loc=jnp.zeros((data.nCs, data.nCMs)))
    )

    alpha_li = numpyro.deterministic(
        "alpha_ic",
        alpha_i.reshape((1, data.nCMs)).repeat(data.nRs, axis=0)
        + (data.RC_mat @ (alpha_ic_noise @ jnp.diag(sigma_i.flatten()))),
    )

    cm_reduction = jnp.sum(
        data.active_cms * alpha_li.reshape((data.nRs, data.nCMs, 1)), axis=1
    )

    basic_R = create_basic_R_prior(data.nRs, basic_r_prior)

    # number of 'noise points'
    nNP = int(data.nDs / r_walk_noise_scale_period) + 1

    noisepoint_log_Rt_noise_series = numpyro.sample(
        "noisepoint_log_Rt_noise_series", dist.Normal(loc=jnp.zeros((data.nRs, nNP)))
    )

    r_walk_noise_scale = create_noisescale_prior(
        "r_walk_noise_scale", r_walk_noisescale_prior, type="r_walk"
    )

    log_Rt_noise = jnp.repeat(
        jnp.cumsum(r_walk_noise_scale * noisepoint_log_Rt_noise_series, axis=-1),
        r_walk_noise_scale_period,
        axis=-1,
    )[: data.nRs, : data.nDs]
    log_Rt = jnp.log(basic_R.reshape((data.nRs, 1))) - cm_reduction + log_Rt_noise
    Rt = numpyro.deterministic("Rt", jnp.exp(log_Rt))  # nRs x nDs

    seeding_padding = 7
    total_padding = ep.GIv.size - 1

    seeding = numpyro.sample("seeding", dist.LogNormal(jnp.zeros((data.nRs, 1)), 5.0))
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
        "ifr_0", dist.Uniform(low=0.01, high=jnp.ones((data.nCs, 1)))
    )

    # number of "noisepoints" for these walks
    nNP = int(data.nDs + seeding_padding / ir_walk_noise_scale_period) + 1

    iar_noise_scale = create_noisescale_prior(
        "iar_noise_scale", iar_noisescale_prior, type="ifr/iar"
    )
    ifr_noise_scale = create_noisescale_prior(
        "ifr_noise_scale", ifr_noisescale_prior, type="ifr/iar"
    )

    noisepoint_log_iar_noise_series = numpyro.sample(
        "noisepoint_log_iar_noise_series", dist.Normal(loc=jnp.zeros((data.nRs, nNP)))
    )
    noisepoint_log_ifr_noise_series = numpyro.sample(
        "noisepoint_log_ifr_noise_series", dist.Normal(loc=jnp.zeros((data.nRs, nNP)))
    )

    iar_noise = jnp.repeat(
        iar_noise_scale * noisepoint_log_iar_noise_series,
        ir_walk_noise_scale_period,
        axis=-1,
    )[: data.nCs, : data.nDs + seeding_padding]
    ifr_noise = jnp.repeat(
        ifr_noise_scale * noisepoint_log_ifr_noise_series,
        ir_walk_noise_scale_period,
        axis=-1,
    )[: data.nCs, : data.nDs + seeding_padding]

    iar_t = numpyro.deterministic("iar_t", iar_0 * jnp.exp(iar_noise))
    ifr_t = numpyro.deterministic("ifr_t", ifr_0 * jnp.exp(ifr_noise))

    # use the `RC_mat` to pull the country level change in the rates for the relevant local area
    future_cases_t = jnp.multiply(total_infections, data.RC_mat @ iar_t)
    future_deaths_t = jnp.multiply(total_infections, data.RC_mat @ ifr_t)

    output_delay_transition = get_output_delay_transition(seeding_padding, data)
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
        "psi_cases", dist.HalfNormal(scale=5 * jnp.ones(len(data.unique_Cs)))
    )
    psi_deaths = numpyro.sample(
        "psi_deaths", dist.HalfNormal(scale=5 * jnp.ones(len(data.unique_Cs)))
    )

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
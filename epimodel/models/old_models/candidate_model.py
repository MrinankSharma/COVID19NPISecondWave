import jax
import jax.numpy as jnp
import jax.scipy.signal as jss
import numpyro
import numpyro.distributions as dist

from epimodel.models.model_utils import (
    create_basic_R_prior,
    create_intervention_prior,
    create_noisescale_prior,
    create_partial_pooling_prior,
    get_discrete_renewal_transition,
    observe_cases_deaths,
    setup_dr_infection_model,
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
    discrete_renewal_transition_type="optim",
    noise_scale_period=7,
    **kwargs
):
    alpha_i = create_intervention_prior(data.nCMs, intervention_prior)
    # full partial pooling of effects i.e., at region level
    sigma_i = create_partial_pooling_prior(data.nCMs, partial_pooling_prior)

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

    basic_R = create_basic_R_prior(data.nRs, basic_r_prior)

    # number of 'noise points'
    nNP = int(data.nDs / noise_scale_period) + 1

    noisepoint_log_Rt_noise_series = numpyro.sample(
        "noisepoint_log_Rt_noise_series", dist.Normal(loc=jnp.zeros((data.nRs, nNP)))
    )

    r_walk_noise_scale = create_noisescale_prior(
        "r_walk_noise_scale", r_walk_noisescale_prior, type="r_walk"
    )

    log_Rt_noise = jnp.repeat(
        jnp.cumsum(r_walk_noise_scale * noisepoint_log_Rt_noise_series, axis=-1),
        noise_scale_period,
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

    infection_noise_scale = numpyro.sample("infection_noise_scale", dist.HalfNormal(3))
    # infection noise is now a normal
    infection_noise = numpyro.sample(
        "infection_noise",
        dist.Normal(loc=0, scale=jnp.ones((data.nRs, data.nDs))),
    )

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

    infections = jax.nn.softplus(
        infections
        + ((infection_noise_scale * (1 + jnp.sqrt(infections))) * infection_noise.T)
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

    iar_0 = 1.0
    ifr_0 = numpyro.sample("cfr", dist.Uniform(low=0.01, high=jnp.ones((data.nRs, 1))))
    nNP = int(data.nDs + seeding_padding / noise_scale_period) + 1

    # # random walk for IFR and IAR
    iar_noise_scale = create_noisescale_prior(
        "iar_noise_scale", iar_noisescale_prior, type="ifr/iar"
    )
    # # number of 'noise points'
    noisepoint_log_iar_noise_series = numpyro.sample(
        "noisepoint_log_iar_noise_series", dist.Normal(loc=jnp.zeros((data.nRs, nNP)))
    )
    iar_noise = jnp.repeat(
        iar_noise_scale * noisepoint_log_iar_noise_series,
        noise_scale_period,
        axis=-1,
    )[: data.nRs, : data.nDs + seeding_padding]
    iar_t = numpyro.deterministic("iar_t", iar_0 * jnp.exp(iar_noise))

    ifr_noise_scale = create_noisescale_prior(
        "ifr_noise_scale", ifr_noisescale_prior, type="ifr/iar"
    )

    # number of 'noise points'
    noisepoint_log_ifr_noise_series = numpyro.sample(
        "noisepoint_log_ifr_noise_series", dist.Normal(loc=jnp.zeros((data.nRs, nNP)))
    )
    ifr_noise = jnp.repeat(
        ifr_noise_scale * noisepoint_log_ifr_noise_series,
        noise_scale_period,
        axis=-1,
    )[: data.nRs, : data.nDs + seeding_padding]
    ifr_t = numpyro.deterministic("ifr_t", ifr_0 * jnp.exp(ifr_noise))

    future_cases_t = jnp.multiply(total_infections, iar_t)
    future_deaths_t = jnp.multiply(total_infections, ifr_t)

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
"""
Contains a bunch of model utility functions, used to construct models while trying to minimise copy and pasteing code.
"""
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from epimodel.distributions import AsymmetricLaplace


def sample_intervention_effects(nCMs, intervention_prior=None):
    if intervention_prior is None:
        intervention_prior = {
            "type": "asymmetric_laplace",
            "scale": 30,
            "asymmetry": 0.5,
        }

    if intervention_prior["type"] == "trunc_normal":
        alpha_i = numpyro.sample(
            "alpha_i",
            dist.TruncatedNormal(
                low=-0.1, loc=jnp.zeros(nCMs), scale=intervention_prior["scale"]
            ),
        )
    elif intervention_prior["type"] == "half_normal":
        alpha_i = numpyro.sample(
            "alpha_i",
            dist.HalfNormal(scale=jnp.ones(nCMs) * intervention_prior["scale"]),
        )
    elif intervention_prior["type"] == "normal":
        alpha_i = numpyro.sample(
            "alpha_i",
            dist.Normal(loc=jnp.zeros(nCMs), scale=intervention_prior["scale"]),
        )
    elif intervention_prior["type"] == "asymmetric_laplace":
        alpha_i = numpyro.sample(
            "alpha_i",
            AsymmetricLaplace(
                asymmetry=intervention_prior["asymmetry"],
                scale=jnp.ones(nCMs) * intervention_prior["scale"],
            ),
        )
    else:
        raise ValueError(
            "Intervention effect prior must take a value in [trunc_normal, normal, asymmetric_laplace, half_normal]"
        )

    return alpha_i


def sample_basic_R(nRs, basic_r_prior=None):
    if basic_r_prior is None:
        basic_r_prior = {"mean": 1.1, "type": "trunc_normal", "variability": 0.3}

    if basic_r_prior["type"] == "trunc_normal":
        basic_R = numpyro.sample(
            "basic_R",
            dist.TruncatedNormal(
                low=0.1,
                loc=basic_r_prior["mean"],
                scale=basic_r_prior["variability"] * jnp.ones(nRs),
            ),
        )
    else:
        raise ValueError("Basic R prior type must be in [trunc_normal]")

    return basic_R


def seed_infections(seeding_scale, nRs, nDs, seeding_padding, total_padding):
    total_infections_placeholder = jnp.zeros((nRs, seeding_padding + nDs))
    seeding = numpyro.sample("seeding", dist.LogNormal(jnp.zeros((nRs, 1)), 1.0))
    init_infections = jnp.zeros((nRs, total_padding))
    init_infections = jax.ops.index_add(
        init_infections,
        jax.ops.index[:, -seeding_padding:],
        jnp.repeat(seeding ** seeding_scale, seeding_padding, axis=-1),
    )
    return init_infections, total_infections_placeholder


def get_discrete_renewal_transition(ep, type="noiseless"):
    """
    Create discrete renewal transition function, used by `jax.lax.scan`

    :param ep: EpidemiologicalParameters() objective

    :return: Discrete Renewal Transition function, with relevant GI parameters
    """

    if type == "optim":

        def discrete_renewal_transition(infections, R_with_noise_tuple):
            # infections is an nR x total_padding size array of infections in the previous
            # total_padding days.
            R, inf_noise = R_with_noise_tuple
            new_infections_t = jax.nn.softplus(
                jnp.multiply(R, infections @ ep.GI_flat_rev) + inf_noise
            )
            new_infections = infections
            new_infections = jax.ops.index_update(
                new_infections, jax.ops.index[:, :-1], infections[:, 1:]
            )
            new_infections = jax.ops.index_update(
                new_infections, jax.ops.index[:, -1], new_infections_t
            )
            return new_infections, new_infections_t

    elif type == "matmul":

        def discrete_renewal_transition(infections, R_with_noise_tuple):
            # infections is an nR x total_padding size array of infections in the previous
            # total_padding days.
            R, inf_noise = R_with_noise_tuple
            new_infections = infections @ ep.GI_projmat
            new_infections = jax.ops.index_update(
                new_infections,
                jax.ops.index[:, -1],
                jax.nn.softplus(jnp.multiply(new_infections[:, -1], R) + inf_noise),
            )
            return new_infections, new_infections[:, -1]

    elif type == "noiseless":

        def discrete_renewal_transition(infections, R):
            new_infections_t = jnp.multiply(R, infections @ ep.GI_flat_rev)
            new_infections = infections
            new_infections = jax.ops.index_update(
                new_infections, jax.ops.index[:, :-1], infections[:, 1:]
            )
            new_infections = jax.ops.index_update(
                new_infections, jax.ops.index[:, -1], new_infections_t
            )
            return new_infections, new_infections_t

    else:
        raise ValueError(
            "Discrete renewal transition type must be in [matmul, optim, noiseless]"
        )

    return discrete_renewal_transition


def get_output_delay_transition(seeding_padding, data):
    def output_delay_transition(loop_carry, scan_slice):
        # this scan function scans over local areas, using their country specific delay, rather than over days
        # therefore the input functions are ** not ** transposed.
        # Also, we don't need a loop carry, so we just return 0 and ignore the loop carry!
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

    return output_delay_transition

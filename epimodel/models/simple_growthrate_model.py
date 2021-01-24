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


def simple_growthrate_model(data, cases_delay=9, deaths_delay=21, **kwargs):
    alpha_i = create_intervention_prior(data.nCMs)
    cm_reduction = jnp.sum(data.active_cms * alpha_i.reshape((1, data.nCMs, 1)), axis=1)

    # g^5 approximately equal 1.5. 1.5^(0.2) = 1.1, so lets go with that for now
    basic_case_growth = numpyro.sample(
        "basic_case_growth",
        dist.TruncatedNormal(low=0.1, loc=1.1 * jnp.ones(data.nRs), scale=0.3),
    )
    basic_death_growth = numpyro.sample(
        "basic_death_growth",
        dist.TruncatedNormal(low=0.1, loc=1.1 * jnp.ones(data.nRs), scale=0.3),
    )

    init_cases = numpyro.sample(
        "init_cases", dist.LogNormal(jnp.zeros((data.nRs, 1)), 5.0)
    )
    init_deaths = numpyro.sample(
        "init_deaths", dist.LogNormal(jnp.zeros((data.nRs, 1)), 5.0)
    )

    cases_delayed_cmred = jnp.zeros_like(cm_reduction)
    deaths_delayed_cmred = jnp.zeros_like(cm_reduction)
    cases_delayed_cmred = jax.ops.index_update(
        cases_delayed_cmred,
        jax.ops.index[:, :cases_delay],
        cm_reduction[:, 0].reshape((data.nRs, 1)).repeat(cases_delay, axis=1),
    )
    deaths_delayed_cmred = jax.ops.index_update(
        deaths_delayed_cmred,
        jax.ops.index[:, :deaths_delay],
        cm_reduction[:, 0].reshape((data.nRs, 1)).repeat(deaths_delay, axis=1),
    )
    cases_delayed_cmred = jax.ops.index_update(
        cases_delayed_cmred,
        jax.ops.index[:, cases_delay:],
        cm_reduction[:, :-cases_delay],
    )
    deaths_delayed_cmred = jax.ops.index_update(
        deaths_delayed_cmred,
        jax.ops.index[:, deaths_delay:],
        cm_reduction[:, :-deaths_delay],
    )

    log_case_growth = jnp.log(basic_case_growth).reshape((80, 1)) - cases_delayed_cmred
    log_death_growth = (
        jnp.log(basic_death_growth).reshape((80, 1)) - deaths_delayed_cmred
    )

    expected_cases = numpyro.deterministic(
        "expected_cases", init_cases * jnp.exp(jnp.cumsum(log_case_growth, axis=-1))
    )
    expected_deaths = numpyro.deterministic(
        "expected_cases", init_deaths * jnp.exp(jnp.cumsum(log_death_growth, axis=-1))
    )

    observe_cases_deaths(data, expected_cases, expected_deaths)

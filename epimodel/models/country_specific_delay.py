import jax
import jax.numpy as jnp
import jax.scipy.signal as jss

import numpyro
import numpyro.distributions as dist

from .model_utils import create_intervention_prior


def get_RVs_to_log():
    return ['alpha_i', 'basic_R', 'Rt', 'expected_cases', 'expected_deaths']


def model_func(data, ep, r_walk_noise_scale=0.15, noise_scale_period=7, **kwargs):
    alpha_i = create_intervention_prior(data.nCMs)
    cm_reduction = jnp.sum(data.active_cms * alpha_i.reshape((1, data.nCMs, 1)), axis=1)

    # looking at most places in UK, austria, R estimates from the imperial report seem to be at about 1 in local areas
    # setting it to be at about 1 seems pretty reasonable to me.
    basic_R = numpyro.sample('basic_R', dist.TruncatedNormal(low=0.1, loc=1.5 * jnp.ones(data.nRs), scale=0.2))

    # number of 'noise points'
    nNP = int(data.nDs / noise_scale_period) + 1
    noisepoint_log_Rt_noise_series = numpyro.sample('noisepoint_log_Rt_noise_series',
                                                    dist.Normal(loc=jnp.zeros((data.nRs, nNP))))
    log_Rt_noise = jnp.repeat(jnp.cumsum(r_walk_noise_scale * noisepoint_log_Rt_noise_series, axis=-1),
                              noise_scale_period, axis=-1)[:data.nRs, :data.nDs]
    log_Rt = jnp.log(basic_R.reshape((data.nRs, 1))) - cm_reduction + log_Rt_noise
    Rt = numpyro.deterministic('Rt', jnp.exp(log_Rt))  # nRs x nDs

    seeding_padding = 7
    total_padding = ep.GIv.size - 1

    seeding = numpyro.sample('seeding', dist.LogNormal(jnp.zeros((data.nRs, 1)), 5.))
    init_infections = jnp.zeros((data.nRs, total_padding))
    init_infections = jax.ops.index_add(init_infections, jax.ops.index[:, -seeding_padding:],
                                        jnp.repeat(seeding, seeding_padding, axis=-1))

    total_infections = jnp.zeros((data.nRs, seeding_padding + data.nDs))

    infection_noise_scale = numpyro.sample("infection_noise_scale", dist.HalfNormal(3))
    infection_noise = numpyro.sample("infection_noise",
                                     dist.HalfNormal(infection_noise_scale * jnp.ones((data.nRs, data.nDs))))

    def discrete_renewal_transition(infections, R_with_noise_tuple):
        # infections is an nR x total_padding size array of infections in the previous
        # total_padding days.
        R, inf_noise = R_with_noise_tuple
        new_infections = infections @ ep.GI_projmat
        new_infections = jax.ops.index_update(new_infections, jax.ops.index[:, -1],
                                              jnp.multiply(new_infections[:, -1], R) + inf_noise)
        return new_infections, new_infections[:, -1]

    # we need to transpose R because jax.lax.scan scans over the first dimension. We want to scan over time
    # we also will transpose infections at the end
    _, infections = jax.lax.scan(discrete_renewal_transition, init_infections, [Rt.T, infection_noise.T])

    total_infections = jax.ops.index_update(total_infections, jax.ops.index[:, :seeding_padding],
                                            init_infections[:, -seeding_padding:])
    total_infections = jax.ops.index_update(total_infections, jax.ops.index[:, seeding_padding:], infections.T)

    total_infections = numpyro.deterministic('total_infections', total_infections)

    cfr = numpyro.sample('cfr', dist.Uniform(low=0.01, high=jnp.ones((data.nRs, 1))))
    future_cases_t = total_infections
    future_deaths_t = jnp.multiply(total_infections, cfr)

    def output_delay_transition(loop_carry, scan_slice):
        # this scan function scans over local areas, using their country specific delay, rather than over days
        # we don't need a loop carry, so we just return 0 and ignore the loop carry!
        future_cases, future_deaths, country_cases_delay, country_deaths_delay = scan_slice
        expected_cases = jnp.convolve(future_cases, country_cases_delay, model='full')[
                         seeding_padding:data.nDs + seeding_padding]
        expected_deaths = jnp.convolve(future_deaths, country_deaths_delay, model='full')[
                          seeding_padding:data.nDs + seeding_padding]

        return 0.0, (expected_cases, expected_deaths)

    _, expected_observations = jax.lax.scan(output_delay_transition, 0.0, [future_cases_t, future_deaths_t, ep.DPCv_pa, ep.DPDv_pa])
    expected_cases = numpyro.deterministic('expected_cases', expected_observations[0])
    expected_deaths = numpyro.deterministic('expected_deaths', expected_observations[1])

    psi_cases = numpyro.sample('psi_cases', dist.HalfNormal(5))
    psi_deaths = numpyro.sample('psi_deaths', dist.HalfNormal(5))

    with numpyro.handlers.mask(mask=jnp.logical_not(data.new_cases.mask)):
        observed_cases = numpyro.sample('observed_cases',
                                        dist.GammaPoisson(concentration=psi_cases,
                                                          rate=psi_cases / expected_cases),
                                        obs=data.new_cases.data)

    with numpyro.handlers.mask(mask=jnp.logical_not(data.new_deaths.mask)):
        observed_deaths = numpyro.sample('observed_deaths',
                                         dist.GammaPoisson(concentration=psi_deaths,
                                                           rate=psi_deaths / expected_deaths),
                                         obs=data.new_deaths.data)

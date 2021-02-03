from datetime import datetime

import time
import numpy as np

from numpyro.infer import MCMC, NUTS, init_to_median
from jax import random

import arviz as az


def run_model(
    model_func,
    data,
    ep,
    num_samples=500,
    num_warmup=500,
    num_chains=4,
    save_results=True,
    output_fname=None,
):
    print(
        f"Running {num_chains} chains, {num_samples} per chain with {num_warmup} warmup steps"
    )
    nuts_kernel = NUTS(model_func, init_strategy=init_to_median, target_accept_prob=0.9)
    mcmc = MCMC(
        nuts_kernel,
        num_samples=num_samples,
        num_warmup=num_warmup,
        num_chains=num_chains,
    )
    rng_key = random.PRNGKey(0)

    start = time.time()
    res = mcmc.run(rng_key, data, ep)
    end = time.time()

    posterior_samples = mcmc.get_samples(np.array([0]))
    time_per_sample = float(end - start) / num_samples
    divergences = int(mcmc.get_extra_fields()["diverging"].sum())

    info_dict = {
        "time_per_sample": time_per_sample,
        "divergences": divergences,
        "model_name": model_func.__name__,
    }

    print(f"Sampling {num_samples} samples per chain took {end - start:.2f}s")
    print(f"There were {divergences} divergences.")

    if save_results:
        try:
            inf_data = az.from_numpyro(mcmc)

            if output_fname is None:
                output_fname = f'{model_func.__name__}-{datetime.now(tz=None).strftime("%d-%m;%H-%M-%S")}.netcdf'

            az.to_netcdf(inf_data, output_fname)

        except Exception as e:
            print(e)

    return posterior_samples, info_dict

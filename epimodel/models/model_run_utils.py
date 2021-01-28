from datetime import datetime

import time
import numpy as np

import numpyro
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
    target_accept=0.8,
    max_tree_depth=10,
    save_results=True,
    output_fname=None,
    model_kwargs=None,
):
    print(
        f"Running {num_chains} chains, {num_samples} per chain with {num_warmup} warmup steps"
    )
    nuts_kernel = NUTS(
        model_func,
        init_strategy=init_to_median,
        target_accept_prob=target_accept,
        max_tree_depth=max_tree_depth,
    )
    mcmc = MCMC(
        nuts_kernel,
        num_samples=num_samples,
        num_warmup=num_warmup,
        num_chains=num_chains,
    )
    rng_key = random.PRNGKey(0)

    start = time.time()
    if model_kwargs is None:
        model_kwargs = {}

    res = mcmc.run(rng_key, data, ep, **model_kwargs)

    posterior_samples = mcmc.get_samples()
    # if you don't block this, the timer won't quite work properly.
    posterior_samples[list(posterior_samples.keys())[0]].block_until_ready()

    end = time.time()
    time_per_sample = float(end - start) / num_samples
    divergences = int(mcmc.get_extra_fields()["diverging"].sum())

    info_dict = {
        "time_per_sample": time_per_sample,
        "divergences": divergences,
        "model_name": model_func.__name__,
    }

    print(f"Sampling {num_samples} samples per chain took {end - start:.2f}s")
    print(f"There were {divergences} divergences.")

    grouped_posterior_samples = mcmc.get_samples(True)

    all_ess = np.array([])
    for k in grouped_posterior_samples.keys():
        ess = numpyro.diagnostics.effective_sample_size(
            np.asarray(grouped_posterior_samples[k])
        )
        all_ess = np.append(all_ess, ess)

    info_dict["ess"] = {
        "med": float(np.percentile(all_ess, 50)),
        "2.5": float(np.percentile(all_ess, 97.5)),
        "97.5": float(np.percentile(all_ess, 2.5)),
        "min": float(np.max(all_ess)),
        "max": float(np.min(all_ess)),
    }
    print(
        f"Mean ESS: {info_dict['ess']['med']:.2f} [{info_dict['ess']['2.5']:.2f} ... {info_dict['ess']['97.5']:.2f}]"
    )

    if num_chains > 1:
        all_rhat = np.array([])
        for k in grouped_posterior_samples.keys():
            rhat = numpyro.diagnostics.gelman_rubin(
                np.asarray(grouped_posterior_samples[k])
            )
            all_rhat = np.append(all_rhat, rhat)

        info_dict["rhat"] = {
            "med": float(np.percentile(all_rhat, 50)),
            "2.5": float(np.percentile(all_rhat, 97.5)),
            "97.5": float(np.percentile(all_rhat, 2.5)),
            "min": float(np.max(all_rhat)),
            "max": float(np.min(all_rhat)),
        }

        print(
            f"Rhat: {info_dict['rhat']['med']:.2f} [{info_dict['rhat']['2.5']:.2f} ... {info_dict['rhat']['97.5']:.2f}]"
        )

    if save_results:
        try:
            inf_data = az.from_numpyro(mcmc)

            if output_fname is None:
                output_fname = f'{model_func.__name__}-{datetime.now(tz=None).strftime("%d-%m;%H-%M-%S")}.netcdf'

            az.to_netcdf(inf_data, output_fname)

        except Exception as e:
            print(e)

    return posterior_samples, info_dict, mcmc

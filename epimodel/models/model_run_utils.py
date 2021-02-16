import time
from datetime import datetime

import arviz as az
import numpy as np
import numpyro
import yaml
from jax import random
from numpyro.infer import MCMC, NUTS, init_to_median


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
    save_yaml=False,
):
    numpyro.set_host_device_count(num_chains)
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

    info_dict = {
        "model_name": model_func.__name__,
    }

    start = time.time()
    if model_kwargs is None:
        model_kwargs = {}

    info_dict["model_kwargs"] = model_kwargs

    # also collect some extra information for better diagonstics!
    print("Warmup")
    mcmc.warmup(
        rng_key,
        data,
        ep,
        **model_kwargs,
        collect_warmup=True,
        extra_fields=["num_steps", "mean_accept_prob"],
    )
    mcmc.get_extra_fields()["num_steps"].block_until_ready()

    info_dict["warmup"] = {}
    info_dict["warmup"]["num_steps"] = np.array(
        mcmc.get_extra_fields()["num_steps"]
    ).tolist()
    info_dict["warmup"]["mean_accept_prob"] = np.array(
        mcmc.get_extra_fields()["mean_accept_prob"]
    ).tolist()

    warmup_samples = mcmc.get_samples()

    print("Sample")
    mcmc.run(
        rng_key,
        data,
        ep,
        **model_kwargs,
        extra_fields=["num_steps", "mean_accept_prob"],
    )

    posterior_samples = mcmc.get_samples()
    # if you don't block this, the timer won't quite work properly.
    posterior_samples[list(posterior_samples.keys())[0]].block_until_ready()

    end = time.time()
    time_per_sample = float(end - start) / num_samples
    divergences = int(mcmc.get_extra_fields()["diverging"].sum())

    info_dict["time_per_sample"] = time_per_sample
    info_dict["divergences"] = divergences

    info_dict["sample"] = {}
    info_dict["sample"]["num_steps"] = np.array(
        mcmc.get_extra_fields()["num_steps"]
    ).tolist()
    info_dict["sample"]["mean_accept_prob"] = np.array(
        mcmc.get_extra_fields()["mean_accept_prob"]
    ).tolist()

    print(f"Sampling {num_samples} samples per chain took {end - start:.2f}s")
    print(f"There were {divergences} divergences.")

    grouped_posterior_samples = mcmc.get_samples(True)

    all_ess = np.array([])
    for k in grouped_posterior_samples.keys():
        ess = numpyro.diagnostics.effective_sample_size(
            np.asarray(grouped_posterior_samples[k])
        )
        all_ess = np.append(all_ess, ess)

    print(f"{np.sum(np.isnan(all_ess))}  ESS were nan")
    all_ess = all_ess[np.logical_not(np.isnan(all_ess))]

    info_dict["ess"] = {
        "med": float(np.percentile(all_ess, 50)),
        "lower": float(np.percentile(all_ess, 2.5)),
        "upper": float(np.percentile(all_ess, 97.5)),
        "min": float(np.min(all_ess)),
        "max": float(np.max(all_ess)),
    }
    print(
        f"Mean ESS: {info_dict['ess']['med']:.2f} [{info_dict['ess']['lower']:.2f} ... {info_dict['ess']['upper']:.2f}]"
    )

    # if num_chains > 1:
    #     all_rhat = np.array([])
    #     for k in grouped_posterior_samples.keys():
    #         rhat = numpyro.diagnostics.gelman_rubin(
    #             np.asarray(grouped_posterior_samples[k])
    #         )
    #         all_rhat = np.append(all_rhat, rhat)
    #
    #     info_dict["rhat"] = {
    #         "med": float(np.percentile(all_rhat, 50)),
    #         "lower": float(np.percentile(all_rhat, 97.5)),
    #         "upper": float(np.percentile(all_rhat, 2.5)),
    #         "min": float(np.max(all_rhat)),
    #         "max": float(np.min(all_rhat)),
    #     }
    #
    #     print(
    #         f"Rhat: {info_dict['rhat']['med']:.2f} [{info_dict['rhat']['lower']:.2f} ... {info_dict['rhat']['upper']:.2f}]"
    #     )

    if save_results:
        print("Saving .netcdf")
        try:
            inf_data = az.from_numpyro(mcmc)

            if output_fname is None:
                output_fname = f'{model_func.__name__}-{datetime.now(tz=None).strftime("%d-%m;%H-%M-%S")}.netcdf'

            az.to_netcdf(inf_data, output_fname)

            yml_fname = output_fname.replace(".netcdf", ".yaml")
            if save_yaml:
                print("Saving Yaml")
                with open(yml_fname, "w") as f:
                    yaml.dump(info_dict, f, sort_keys=True)

        except Exception as e:
            print(e)

    return posterior_samples, warmup_samples, info_dict, mcmc

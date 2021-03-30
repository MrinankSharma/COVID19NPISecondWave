import time
from datetime import datetime

import arviz as az
import numpy as np
import numpyro
import json
from jax import random
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS, init_to_median


def run_model(
    model_func,
    data,
    ep,
    num_samples=500,
    num_warmup=500,
    num_chains=4,
    target_accept=0.75,
    max_tree_depth=15,
    save_results=True,
    output_fname=None,
    model_kwargs=None,
    save_json=False,
    chain_method="parallel",
    heuristic_step_size=True,
):
    """
    Model run utility

    :param model_func: numpyro model
    :param data: PreprocessedData object
    :param ep: EpidemiologicalParameters object
    :param num_samples: number of samples
    :param num_warmup: number of warmup samples
    :param num_chains: number of chains
    :param target_accept: target accept
    :param max_tree_depth: maximum treedepth
    :param save_results: whether to save full results
    :param output_fname: output filename
    :param model_kwargs: model kwargs -- extra arguments for the model function
    :param save_json: whether to save json
    :param chain_method: Numpyro chain method to use
    :param heuristic_step_size: whether to find a heuristic step size
    :return: posterior_samples, warmup_samples, info_dict (dict with assorted diagnostics), Numpyro mcmc object
    """
    print(
        f"Running {num_chains} chains, {num_samples} per chain with {num_warmup} warmup steps"
    )
    nuts_kernel = NUTS(
        model_func,
        init_strategy=init_to_median,
        target_accept_prob=target_accept,
        max_tree_depth=max_tree_depth,
        find_heuristic_step_size=heuristic_step_size,
    )
    mcmc = MCMC(
        nuts_kernel,
        num_samples=num_samples,
        num_warmup=num_warmup,
        num_chains=num_chains,
        chain_method=chain_method,
    )
    rng_key = random.PRNGKey(0)

    # hmcstate = nuts_kernel.init(rng_key, 1, model_args=(data, ep))
    # nRVs = hmcstate.adapt_state.inverse_mass_matrix.size
    # inverse_mass_matrix = init_diag_inv_mass_mat * jnp.ones(nRVs)
    # mass_matrix_sqrt_inv = np.sqrt(inverse_mass_matrix)
    # mass_matrix_sqrt = 1./mass_matrix_sqrt_inv
    # hmcstate = hmcstate._replace(adapt_state=hmcstate.adapt_state._replace(inverse_mass_matrix=inverse_mass_matrix))
    # hmcstate = hmcstate._replace(adapt_state=hmcstate.adapt_state._replace(mass_matrix_sqrt_inv=mass_matrix_sqrt_inv))
    # hmcstate = hmcstate._replace(adapt_state=hmcstate.adapt_state._replace(mass_matrix_sqrt=mass_matrix_sqrt))
    # mcmc.post_warmup_state = hmcstate

    info_dict = {
        "model_name": model_func.__name__,
    }

    start = time.time()
    if model_kwargs is None:
        model_kwargs = {}

    info_dict["model_kwargs"] = model_kwargs

    # also collect some extra information for better diagonstics!
    print(f"Warmup Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    mcmc.warmup(
        rng_key,
        data,
        ep,
        **model_kwargs,
        collect_warmup=True,
        extra_fields=["num_steps", "mean_accept_prob", "adapt_state"],
    )
    mcmc.get_extra_fields()["num_steps"].block_until_ready()

    info_dict["warmup"] = {}
    info_dict["warmup"]["num_steps"] = np.array(
        mcmc.get_extra_fields()["num_steps"]
    ).tolist()
    info_dict["warmup"]["step_size"] = np.array(
        mcmc.get_extra_fields()["adapt_state"].step_size
    ).tolist()
    info_dict["warmup"]["inverse_mass_matrix"] = {}

    all_mass_mats = jnp.array(
        jnp.array_split(
            mcmc.get_extra_fields()["adapt_state"].inverse_mass_matrix,
            num_chains,
            axis=0,
        )
    )

    print(all_mass_mats.shape)

    for i in range(num_chains):
        info_dict["warmup"]["inverse_mass_matrix"][f"chain_{i}"] = all_mass_mats[
            i, -1, :
        ].tolist()

    info_dict["warmup"]["mean_accept_prob"] = np.array(
        mcmc.get_extra_fields()["mean_accept_prob"]
    ).tolist()

    warmup_samples = mcmc.get_samples()

    print(f"Sample Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    mcmc.run(
        rng_key,
        data,
        ep,
        **model_kwargs,
        extra_fields=["num_steps", "mean_accept_prob", "adapt_state"],
    )

    posterior_samples = mcmc.get_samples()
    # if you don't block this, the timer won't quite work properly.
    posterior_samples[list(posterior_samples.keys())[0]].block_until_ready()
    print(f"Sample Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    end = time.time()
    time_per_sample = float(end - start) / num_samples
    divergences = int(mcmc.get_extra_fields()["diverging"].sum())

    info_dict["time_per_sample"] = time_per_sample
    info_dict["total_runtime"] = float(end - start)
    info_dict["divergences"] = divergences

    info_dict["sample"] = {}
    info_dict["sample"]["num_steps"] = np.array(
        mcmc.get_extra_fields()["num_steps"]
    ).tolist()
    info_dict["sample"]["mean_accept_prob"] = np.array(
        mcmc.get_extra_fields()["mean_accept_prob"]
    ).tolist()
    info_dict["sample"]["step_size"] = np.array(
        mcmc.get_extra_fields()["adapt_state"].step_size
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

    if num_chains > 1:
        all_rhat = np.array([])
        for k in grouped_posterior_samples.keys():
            rhat = numpyro.diagnostics.gelman_rubin(
                np.asarray(grouped_posterior_samples[k])
            )
            all_rhat = np.append(all_rhat, rhat)

        print(f"{np.sum(np.isnan(all_rhat))} Rhat were nan")
        all_rhat = all_rhat[np.logical_not(np.isnan(all_rhat))]

        info_dict["rhat"] = {
            "med": float(np.percentile(all_rhat, 50)),
            "upper": float(np.percentile(all_rhat, 97.5)),
            "lower": float(np.percentile(all_rhat, 2.5)),
            "min": float(np.max(all_rhat)),
            "max": float(np.min(all_rhat)),
        }

        print(
            f"Rhat: {info_dict['rhat']['med']:.2f} [{info_dict['rhat']['lower']:.2f} ... {info_dict['rhat']['upper']:.2f}]"
        )

    if save_results:
        print("Saving .netcdf")
        try:
            inf_data = az.from_numpyro(mcmc)

            if output_fname is None:
                output_fname = f'{model_func.__name__}-{datetime.now(tz=None).strftime("%d-%m;%H-%M-%S")}.netcdf'

            az.to_netcdf(inf_data, output_fname)

            json_fname = output_fname.replace(".netcdf", ".json")
            if save_json:
                print("Saving Json")
                with open(json_fname, "w") as f:
                    json.dump(info_dict, f, ensure_ascii=False, indent=4)

        except Exception as e:
            print(e)

    return posterior_samples, warmup_samples, info_dict, mcmc

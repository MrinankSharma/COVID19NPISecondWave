import sys, os

sys.path.append(os.getcwd())  # add current working directory to the path

from epimodel import EpidemiologicalParameters, run_model, preprocess_data
from epimodel.script_utils import *

import argparse
import json
from datetime import datetime

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--seed",
    dest="seed",
    type=int,
    help="seed for bootstrapping",
)

add_argparse_arguments(argparser)
args = argparser.parse_args()

import numpyro

numpyro.set_host_device_count(args.num_chains)

if __name__ == "__main__":
    print(f"Running Sensitivity Analysis {__file__} with config:")
    config = load_model_config(args.model_config)
    pprint_mb_dict(config)

    print("Sampling Params")
    np.random.seed(args.seed)

    gi_shift = np.clip(0.75*np.random.normal(), a_min=-1.5, a_max=1.5)
    cd_shift = np.clip(1.5*np.random.normal(), a_min=-3, a_max=3)
    dd_shift = np.clip(1.5*np.random.normal(), a_min=-3, a_max=3)

    max_frac_voc = np.random.choice([0.1, 0.15, 0.25, 0.5])

    seeding_scale = np.random.choice([2, 2.5, 3, 3.5, 4])
    inf_noise_scale = np.random.choice([1, 3, 5, 7, 9])
    rw_noise_scale_prior = np.random.choice([0.05, 0.1, 0.15, 0.2, 0.25])
    output_noise_scale_prior = np.random.choice([2.5, 5, 10, 15, 20])

    r0_mean = 1.35+np.clip(0.15*np.random.normal(), a_min=-0.35, a_max=0.35)
    r0_scale = np.clip(0.3 + 0.1 * np.random.normal(), a_min=0.1, a_max=0.5)

    rw_period = np.random.choice([5, 7, 9, 11, 14])
    n_days_seeding = np.random.choice([5, 7, 9, 11, 14])

    print("Loading Data")
    data = preprocess_data(get_data_path())
    data.featurize(**config["featurize_kwargs"])
    data.mask_new_variant(
        new_variant_fraction_fname=get_new_variant_path(),
        maximum_fraction_voc=float(max_frac_voc),
    )
    data.mask_from_date("2021-01-09")

    print("Loading EpiParam")
    ep = EpidemiologicalParameters()

    # shift delays
    ep.generation_interval["mean"] = (
        ep.generation_interval["mean"] + gi_shift
    )

    ep.onset_to_death_delay["mean"] = (
        ep.onset_to_death_delay["mean"] + dd_shift
    )

    ep.onset_to_case_delay["mean"] = (
        ep.onset_to_case_delay["mean"] + cd_shift
    )

    ep.generate_delays()

    model_func = get_model_func_from_str(args.model_type)
    ta = get_target_accept_from_model_str(args.model_type)
    td = get_tree_depth_from_model_str(args.model_type)

    base_outpath = generate_base_output_dir(
        args.model_type, args.model_config, args.exp_tag
    )
    ts_str = datetime.now().strftime("%Y-%m-%d;%H:%M:%S")
    summary_output = os.path.join(base_outpath, f"{ts_str}_summary.json")
    full_output = os.path.join(base_outpath, f"{ts_str}_full.netcdf")

    basic_R_prior = {
        "mean": float(r0_mean),
        "type": "trunc_normal",
        "variability": float(r0_scale),
    }

    model_build_dict = config["model_kwargs"]
    model_build_dict["infection_noise_scale"] = int(inf_noise_scale)
    model_build_dict["r_walk_noise_scale_prior"] = float(rw_noise_scale_prior)
    model_build_dict["output_noise_scale_prior"] = float(output_noise_scale_prior)
    model_build_dict["basic_R_prior"] = basic_R_prior
    model_build_dict["seeding_scale"] = float(seeding_scale)
    model_build_dict["r_walk_period"] = int(rw_period)
    model_build_dict["n_days_seeding"] = int(n_days_seeding)

    posterior_samples, _, info_dict, _ = run_model(
        model_func,
        data,
        ep,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        num_warmup=args.num_warmup,
        target_accept=ta,
        max_tree_depth=td,
        model_kwargs=model_build_dict,
        save_results=True,
        output_fname=full_output,
        chain_method="parallel",
    )

    info_dict["model_config_name"] = args.model_config
    info_dict["model_kwargs"] = config["model_kwargs"]
    info_dict["featurize_kwargs"] = config["featurize_kwargs"]
    info_dict["start_dt"] = ts_str
    info_dict["exp_tag"] = args.exp_tag
    info_dict["exp_config"] = {
        "cases_delay_mean_shift": float(cd_shift),
        "deaths_delay_mean_shift": float(dd_shift),
        "gen_int_mean_shift": float(gi_shift),
        "maximum_fraction_voc": float(max_frac_voc),
        "infection_noise_scale": float(inf_noise_scale),
        "r_walk_noise_scale_prior": float(rw_noise_scale_prior),
        "output_noise_scale_prior":  float(output_noise_scale_prior),
        "basic_R_prior": basic_R_prior,
        "seeding_scale": float(seeding_scale),
        "r_walk_period": int(rw_period),
        "n_days_seeding": float(n_days_seeding),
    }
    info_dict["cm_names"] = data.CMs
    info_dict["data_path"] = get_data_path()

    # also need to add sensitivity analysis experiment options to the summary dict!
    summary = load_keys_from_samples(
        get_summary_save_keys(), posterior_samples, info_dict
    )

    with open(summary_output, "w") as f:
        json.dump(info_dict, f, ensure_ascii=False, indent=4)

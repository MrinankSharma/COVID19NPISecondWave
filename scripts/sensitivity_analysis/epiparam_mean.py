import sys, os

sys.path.append(os.getcwd())  # add current working directory to the path

from epimodel import EpidemiologicalParameters, run_model, preprocess_data
from epimodel.script_utils import *

import argparse
import json
from datetime import datetime

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--cases_delay_mean_shift",
    dest="cases_delay_mean_shift",
    type=float,
    help="how much the cases delay mean is shifted for all countries",
)
argparser.add_argument(
    "--death_delay_mean_shift",
    dest="death_delay_mean_shift",
    type=float,
    help="how much the death delay mean is shifted for all countries",
)
argparser.add_argument(
    "--gen_int_mean_shift",
    dest="gen_int_mean_shift",
    type=float,
    help="how much the generation interval mean is shifted",
)

add_argparse_arguments(argparser)
args = argparser.parse_args()

import numpyro

numpyro.set_host_device_count(args.num_chains)

if __name__ == "__main__":
    print(f"Running Sensitivity Analysis {__file__} with config:")
    config = load_model_config(args.model_config)
    pprint_mb_dict(config)

    print("Loading Data")
    data = preprocess_data(get_data_path())
    data.featurize(**config["featurize_kwargs"])
    data.mask_new_variant(
        new_variant_fraction_fname=get_new_variant_path(),
    )
    data.mask_from_date("2021-01-09")

    print("Loading EpiParam")
    ep = EpidemiologicalParameters()

    # shift delays
    ep.generation_interval["mean"] = (
        ep.generation_interval["mean"] + args.gen_int_mean_shift
    )

    ep.onset_to_death_delay["mean"] = (
        ep.onset_to_death_delay["mean"] + args.death_delay_mean_shift
    )

    ep.onset_to_case_delay["mean"] = (
        ep.onset_to_case_delay["mean"] + args.cases_delay_mean_shift
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

    model_build_dict = config["model_kwargs"]

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
        "cases_delay_mean_shift": args.cases_delay_mean_shift,
        "deaths_delay_mean_shift": args.death_delay_mean_shift,
        "gen_int_mean_shift": args.gen_int_mean_shift,
    }
    info_dict["cm_names"] = data.CMs
    info_dict["data_path"] = get_data_path()

    # also need to add sensitivity analysis experiment options to the summary dict!
    summary = load_keys_from_samples(
        get_summary_save_keys(), posterior_samples, info_dict
    )
    with open(summary_output, "w") as f:
        json.dump(info_dict, f, ensure_ascii=False, indent=4)

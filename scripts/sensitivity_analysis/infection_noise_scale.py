import sys, os

sys.path.append(os.getcwd())  # add current working directory to the path

from epimodel import EpidemiologicalParameters, run_model, preprocess_data
from epimodel.script_utils import *

import argparse
from datetime import datetime

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--infection_noise_scale",
    dest="infection_noise_scale",
    type=float,
    help="Infection Noise Scale",
)
add_argparse_arguments(argparser)
args = argparser.parse_args()

if __name__ == "__main__":
    print(f"Running Sensitivity Analysis {__file__} with config:")
    config = load_model_config(args.model_config)
    pprint_mb_dict(config)

    print("Loading Data")
    data = preprocess_data(get_data_path())
    data.featurize(**config["featurize_kwargs"])
    print("Loading EpiParam")
    ep = EpidemiologicalParameters()
    ep.populate_region_delays(data)

    for npi in args.npis:
        data.remove_npi_by_index()

    model_func = get_model_func_from_str(args.model_type)
    ta = get_target_accept_from_model_str(args.model_type)
    td = get_tree_depth_from_model_str(args.model_type)

    base_outpath = generate_base_output_dir(
        args.model_type, args.model_config, args.exp_tag
    )
    ts_str = datetime.now().strftime("%Y-%m-%d;%H:%M:%S")
    summary_output = os.path.join(base_outpath, f"{ts_str}_summary.yaml")
    full_output = os.path.join(base_outpath, f"{ts_str}_full.netcdf")

    model_build_dict = config["model_kwargs"]
    model_build_dict["infection_noise_scale"] = args.infection_noise_scale

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
        save_yaml=False,
        chain_method="parallel",
    )

    info_dict["model_config_name"] = args.model_config
    info_dict["model_kwargs"] = config["model_kwargs"]
    info_dict["featurize_kwargs"] = config["featurize_kwargs"]
    info_dict["start_dt"] = ts_str
    info_dict["exp_tag"] = args.exp_tag
    info_dict["exp_config"] = {"infection_noise_scale": args.infection_noise_scale}
    info_dict["cm_names"] = data.CMs

    # also need to add sensitivity analysis experiment options to the summary dict!
    summary = load_keys_from_samples(
        get_summary_save_keys(), posterior_samples, info_dict
    )
    with open(summary_output, "w") as f:
        yaml.dump(summary, f, sort_keys=True)

import sys, os

sys.path.append(os.getcwd())  # add current working directory to the path

from epimodel import EpidemiologicalParameters, run_model, preprocess_data
from epimodel.script_utils import *

import argparse
import json
import numpyro
from datetime import datetime

argparser = argparse.ArgumentParser()

add_argparse_arguments(argparser)
argparser.add_argument(
    "--n_days_delay",
    dest="n_days_delay",
    type=int,
    help="number of days to delay the school features",
)


args = argparser.parse_args()

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

    school_npis = [
        "Primary Schools Closed",
        "Secondary Schools Closed",
    ]

    nRs, nCMs, nDs = data.active_cms.shape
    new_active_cms = np.copy(data.active_cms)

    for school_npi in school_npis:
        school_npi_index = data.CMs.index(school_npi)
        new_active_cms[:, school_npi_index, args.n_days_delay :] = data.active_cms[
            :, school_npi_index, : -args.n_days_delay
        ]
        new_active_cms[:, school_npi_index, : args.n_days_delay] = (
            data.active_cms[:, school_npi_index, 0]
            .reshape((nRs, 1))
            .repeat(args.n_days_delay, axis=-1)
        )

    data.active_cms = new_active_cms

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
    info_dict["exp_config"] = {"n_days_delay": args.n_days_delay}
    info_dict["cm_names"] = data.CMs
    info_dict["data_path"] = get_data_path()

    # also need to add sensitivity analysis experiment options to the summary dict!
    summary = load_keys_from_samples(
        get_summary_save_keys(), posterior_samples, info_dict
    )
    with open(summary_output, "w") as f:
        json.dump(info_dict, f, ensure_ascii=False, indent=4)

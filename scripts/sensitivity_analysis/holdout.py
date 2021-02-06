import sys, os

sys.path.append(os.getcwd())  # add current working directory to the path

from epimodel import EpidemiologicalParameters, run_model, preprocess_data
from epimodel.script_utils import *

from datetime import datetime
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--rgs", dest="rgs", type=int, help="Region indices to leave out", nargs="+"
)
add_argparse_arguments(argparser)
args = argparser.parse_args()

if __name__ == "__main__":
    print("Loading Data")
    data = preprocess_data(get_data_path())
    data.featurize()
    print("Loading EpiParam")
    ep = EpidemiologicalParameters()
    ep.populate_region_delays(data)

    for rg in args.rgs:
        data.mask_region_by_index(rg)

    model_func = get_model_func_from_str(args.model_type)
    ta = get_target_accept_from_model_str(args.model_type)
    td = get_tree_depth_from_model_str(args.model)

    base_outpath = generate_base_output_dir(
        args.model_type, args.model_config, args.exp_tag
    )
    ts_str = datetime.now().strftime("%Y-%m-%d;%H:%M:%S")
    summary_output = os.path.join(base_outpath, f"{ts_str}_summary.yaml")
    full_output = os.path.join(base_outpath, f"{ts_str}_full.netcdf")

    model_extra_bd = load_model_config(args.model_config)
    pprint_mb_dict(model_extra_bd)

    samples, summary = run_model(
        model_func,
        data,
        ep,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        target_accept=ta,
        max_tree_depth=td,
        model_kwargs=model_extra_bd,
        save_results=True,
        output_fname=full_output,
    )

    summary["model_config"] = args.model_config
    summary["start_dt"] = ts_str
    summary["exp_tag"] = args.exp_tag
    summary["rgs"] = args.rgs

    # also need to add sensitivity analysis experiment options to the summary dict!
    summary = load_keys_from_samples(get_summary_save_keys(), samples, summary)
    with open(summary_output, "w") as f:
        yaml.dump(summary, f, sort_keys=True)

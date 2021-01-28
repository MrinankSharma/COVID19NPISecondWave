"""
:code:`utils.py`

Utilities to support the use of command line sensitivity experiments
"""
import yaml
import epimodel.models
import numpy as np
import os


def get_model_func_from_str(model_type_str):
    if model_type_str == "candidate":
        return epimodel.models.candidate_model


def get_target_accept_from_model_str(model_type_str):
    if model_type_str == "default":
        return 0.8


def get_tree_depth_from_model_str(model_type_str):
    if model_type_str == "default":
        return 12


def add_argparse_arguments(argparse):
    argparse.add_argument(
        "--model_type",
        dest="model_type",
        type=str,
        help="""model""",
        default="candidate",
    )
    argparse.add_argument(
        "--exp_tag", dest="exp_tag", type=str, help="experiment identification tag"
    )
    argparse.add_argument(
        "--n_chains",
        dest="n_chains",
        type=int,
        help="the number of chains to run in parallel",
    )
    argparse.add_argument(
        "--n_samples", dest="n_samples", type=int, help="the number of samples to draw"
    )

    argparse.add_argument(
        "--model_config",
        dest="model_config",
        type=str,
        help="model config file, which is used for overriding default options",
    )


def load_model_config(model_config_str):
    with open("scripts/sensitivity_analysis/model_configs.yaml", "r") as stream:
        try:
            model_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return model_config[model_config_str]


def pprint_mb_dict(d):
    print("Model Build Dict" "----------------")

    for k, v in d.items():
        print(f"    {k}: {v}")


def generate_base_output_dir(model_type, model_config, exp_tag):
    out_path = os.path.join(
        "sensitivity_analysis", f"{model_type}c{model_config}", exp_tag
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    return out_path


def get_summary_save_keys():
    return ["alpha_i", "sigma_i"]


def get_data_path():
    return "data/all_merged_data.csv"


def load_keys_from_samples(keys, posterior_samples, summary_dict):
    for k in keys:
        if k in posterior_samples.keys():
            # save to list
            summary_dict[k] = np.asarray(posterior_samples[k]).tolist()
    return summary_dict

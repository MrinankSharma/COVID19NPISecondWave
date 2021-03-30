"""
:code:`script_utils.py`

Utilities to support the use of command line sensitivity experiments
"""
import os

import numpy as np
import yaml

from epimodel.models import *


def get_model_func_from_str(model_type_str):
    """
    link model function string to actual function

    :param model_type_str:
    :return: model function
    """
    if model_type_str == "default":
        return default_model


def get_target_accept_from_model_str(model_type_str):
    # default
    return 0.75


def get_tree_depth_from_model_str(model_type_str):
    # default
    return 15


def add_argparse_arguments(argparse):
    """
    add argparse arguments to scripts

    :param argparse: argparse object
    """
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
        "--num_chains",
        dest="num_chains",
        type=int,
        help="the number of chains to run in parallel",
    )
    argparse.add_argument(
        "--num_samples",
        dest="num_samples",
        type=int,
        help="the number of samples to draw",
    )

    argparse.add_argument(
        "--num_warmup",
        dest="num_warmup",
        type=int,
        help="the number of warmup samples to draw",
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
    """
    pretty print dictionary

    :param d:
    :return:
    """
    print("Model Build Dict" "----------------")

    for k, v in d.items():
        print(f"    {k}: {v}")


def generate_base_output_dir(model_type, model_config, exp_tag):
    """
    standardise output directory

    :param model_type:
    :param model_config:
    :param exp_tag:
    :return: output directory
    """
    out_path = os.path.join(
        "sensitivity_analysis", f"{model_type}_c{model_config}", exp_tag
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    return out_path


def get_summary_save_keys():
    """
    :return: numpyro variables to save, if available
    """
    return ["alpha_i"]


def get_data_path():
    return "data/all_merged_data_2021-01-22.csv"


def get_new_variant_path():
    return "data/nuts3_new_variant_fraction.csv"


def load_keys_from_samples(keys, posterior_samples, summary_dict):
    for k in keys:
        if k in posterior_samples.keys():
            # save to list
            summary_dict[k] = np.asarray(posterior_samples[k]).tolist()
    return summary_dict

import sys, os

sys.path.append(os.getcwd())  # add current working directory to the path

from epimodel import EpidemiologicalParameters, run_model, preprocess_data
from epimodel.models.candidate_model_v7_all import *

import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--exp", dest="exp", type=int, help="experiment")

argparser.add_argument(
    "--num_samples", dest="num_samples", type=int, help="num_samples"
)

argparser.add_argument("--num_chains", dest="num_chains", type=int, help="num_chains")

argparser.add_argument("--num_warmup", dest="num_warmup", type=int, help="num_chains")

argparser.add_argument(
    "--max_treedepth", dest="max_treedepth", type=int, help="tree depth"
)

args = argparser.parse_args()

if __name__ == "__main__":
    ep = EpidemiologicalParameters()
    data = preprocess_data("data/all_merged_data.csv")
    data.featurize()
    ep.populate_region_delays(data)

    if args.exp == 0:
        # default
        samples, info, mcmc = run_model(
            candidate_model_v7_base,
            data,
            ep,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
            max_tree_depth=args.max_treedepth,
            save_yaml=True,
        )
    elif args.exp == 1:
        # skipping few cases, making more parameters identifiable
        ep = EpidemiologicalParameters()
        data = preprocess_data("data/all_merged_data.csv", skipcases=7, skipdeaths=19)
        data.featurize()
        ep.populate_region_delays(data)
        samples, info, mcmc = run_model(
            candidate_model_v7_base,
            data,
            ep,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
            max_tree_depth=args.max_treedepth,
            save_yaml=True,
        )
    elif args.exp == 2:
        # default
        samples, info, mcmc = run_model(
            candidate_model_v7_fixed_r_var,
            data,
            ep,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
            max_tree_depth=args.max_treedepth,
            save_yaml=True,
        )
    elif args.exp == 3:
        # default
        # samples, info, mcmc = run_model(
        #     candidate_model_v7_alt_conv,
        #     data,
        #     ep,
        #     num_samples=args.num_samples,
        #     num_warmup=args.num_warmup,
        #     num_chains=args.num_chains,
        #     max_tree_depth=args.max_treedepth,
        #     save_yaml=True,
        # )
        pass
    elif args.exp == 4:
        # default
        samples, info, mcmc = run_model(
            candidate_model_v7_centered_ir_noise,
            data,
            ep,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
            max_tree_depth=args.max_treedepth,
            save_yaml=True,
        )
    elif args.exp == 5:
        # default
        samples, info, mcmc = run_model(
            candidate_model_v7_centered_inf_noise,
            data,
            ep,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
            max_tree_depth=args.max_treedepth,
            save_yaml=True,
        )
    elif args.exp == 6:
        # default
        samples, info, mcmc = run_model(
            candidate_model_v7_no_inf_noise,
            data,
            ep,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
            max_tree_depth=args.max_treedepth,
            save_yaml=True,
        )
    elif args.exp == 7:
        # default
        samples, info, mcmc = run_model(
            candidate_model_v7_fixed_inf_noise,
            data,
            ep,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
            max_tree_depth=args.max_treedepth,
            save_yaml=True,
        )
    elif args.exp == 8:
        # default
        samples, info, mcmc = run_model(
            candidate_model_v7_weekly_inf_noise,
            data,
            ep,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
            max_tree_depth=args.max_treedepth,
            save_yaml=True,
        )
    elif args.exp == 9:
        # default
        samples, info, mcmc = run_model(
            candidate_model_v7_relu,
            data,
            ep,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
            max_tree_depth=args.max_treedepth,
            save_yaml=True,
        )

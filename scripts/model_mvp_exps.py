import sys, os

sys.path.append(os.getcwd())  # add current working directory to the path

import numpyro
from epimodel import EpidemiologicalParameters, run_model, preprocess_data
from epimodel.models.candidate_model_v14 import *

import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--exp", dest="exp", type=int, help="experiment")

argparser.add_argument(
    "--num_samples", dest="num_samples", type=int, help="num_samples", default=1000
)

argparser.add_argument(
    "--num_chains", dest="num_chains", type=int, help="num_chains", default=1
)

argparser.add_argument(
    "--num_warmup", dest="num_warmup", type=int, help="num_chains", default=500
)

argparser.add_argument(
    "--max_treedepth", dest="max_treedepth", type=int, help="tree depth", default=20
)

args = argparser.parse_args()
numpyro.set_host_device_count(args.num_chains)

if __name__ == "__main__":
    ep = EpidemiologicalParameters()
    data = preprocess_data("data/all_merged_data.csv", skipcases=8, skipdeaths=20)
    data.featurize()
    ep.populate_region_delays(data)

    if args.exp == 0:
        # default
        samples, warmup_samples, info, mcmc = run_model(
            candidate_model_v14_ppool_variable,
            data,
            ep,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
            max_tree_depth=args.max_treedepth,
            save_yaml=True,
        )

    elif args.exp == 1:
        samples, warmup_samples, info, mcmc = run_model(
            candidate_model_v14_ppool_fixed,
            data,
            ep,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
            max_tree_depth=args.max_treedepth,
            save_yaml=True,
        )

    elif args.exp == 2:
        samples, warmup_samples, info, mcmc = run_model(
            candidate_model_v14_base,
            data,
            ep,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
            max_tree_depth=args.max_treedepth,
            save_yaml=True,
        )

    elif args.exp == 3:
        samples, warmup_samples, info, mcmc = run_model(
            candidate_model_v14_fixed_ir,
            data,
            ep,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
            max_tree_depth=args.max_treedepth,
            save_yaml=True,
        )

    elif args.exp == 4:
        samples, warmup_samples, info, mcmc = run_model(
            candidate_model_v14_fixed_ir_scale,
            data,
            ep,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
            max_tree_depth=args.max_treedepth,
            model_kwargs={"iar_noise_scale": 0.01},
            save_yaml=True,
        )

    elif args.exp == 5:
        samples, warmup_samples, info, mcmc = run_model(
            candidate_model_v14_fixed_ir_scale,
            data,
            ep,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
            max_tree_depth=args.max_treedepth,
            model_kwargs={"iar_noise_scale": 0.025},
            save_yaml=True,
        )

    elif args.exp == 6:
        samples, warmup_samples, info, mcmc = run_model(
            candidate_model_v14_fixed_ir_scale,
            data,
            ep,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
            max_tree_depth=args.max_treedepth,
            model_kwargs={"iar_noise_scale": 0.05},
            save_yaml=True,
        )

    elif args.exp == 7:
        samples, warmup_samples, info, mcmc = run_model(
            candidate_model_v14_fixed_ir_scale,
            data,
            ep,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
            max_tree_depth=args.max_treedepth,
            model_kwargs={"iar_noise_scale": 0.075},
            save_yaml=True,
        )

    elif args.exp == 8:
        samples, warmup_samples, info, mcmc = run_model(
            candidate_model_v14_fixed_ir_scale,
            data,
            ep,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
            max_tree_depth=args.max_treedepth,
            model_kwargs={"iar_noise_scale": 0.1},
            save_yaml=True,
        )

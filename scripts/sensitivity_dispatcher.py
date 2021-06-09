import argparse
import os
import subprocess
import time

import yaml

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--max_parallel_runs",
    dest="max_parallel_runs",
    type=int,
    default=24,
    help="Number of processes to spawn",
)
argparser.add_argument(
    "--categories", nargs="+", dest="categories", type=str, help="Run types to execute"
)
argparser.add_argument(
    "--dry_run",
    default=False,
    action="store_true",
    help="Print run types selected and exit",
)
argparser.add_argument(
    "--model_type",
    default="default",
    dest="model_type",
    type=str,
    help="Model type to use for requested sensitivity analyses",
)

argparser.add_argument(
    "--model_config",
    default="default",
    dest="model_config",
    type=str,
    nargs="+",
    help="Model config used to override default params for **all** requested runs",
)

argparser.add_argument(
    "--exclude_core_indices",
    dest="exclude_core_indices",
    type=int,
    nargs="+",
    help="indices removed",
)

argparser.add_argument(
    "--num_chains",
    default=4,
    dest="num_chains",
    type=int,
    help="Num chains to use",
)

argparser.add_argument(
    "--num_samples",
    default=1000,
    dest="num_samples",
    type=int,
    help="Num samples to use",
)

argparser.add_argument(
    "--num_warmup",
    default=500,
    dest="num_warmup",
    type=int,
    help="Num warmup samples to use",
)

args = argparser.parse_args()


def run_types_to_commands(run_types, exp_options):
    commands = []
    configs = args.model_config

    if len(configs) == 1 and configs[0] == "all":
        with open("scripts/sensitivity_analysis/model_configs.yaml", "r") as stream:
            try:
                config_options = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        print("loading all config options")
        configs = list(config_options.keys())
        configs.remove("default")

    for config in configs:
        model_config = config
        for rt in run_types:
            exp_rt = exp_options[rt]
            experiment_file = exp_rt["experiment_file"]
            num_chains = args.num_chains
            num_samples = args.num_samples
            num_warmup = args.num_warmup
            exp_tag = exp_rt["experiment_tag"]
            model_type = args.model_type

            cmds = [
                f"python scripts/sensitivity_analysis/{experiment_file} --model_type {model_type}"
                f" --num_samples {num_samples} --num_chains {num_chains} --exp_tag {exp_tag}"
                f" --model_config {model_config} --num_warmup {num_warmup} "
            ]

            for key, value in exp_rt["args"].items():
                new_cmds = []
                if isinstance(value, list):
                    for c in cmds:
                        for v in value:
                            if isinstance(v, list):
                                new_cmd = f"{c} --{key}"
                                for v_nest in v:
                                    new_cmd = f"{new_cmd} {v_nest}"
                                new_cmds.append(new_cmd)
                            else:
                                new_cmd = f"{c} --{key} {v}"
                                new_cmds.append(new_cmd)
                else:
                    for c in cmds:
                        new_cmd = f"{c} --{key} {value}"
                        new_cmds.append(new_cmd)

                cmds = new_cmds
            commands.extend(cmds)

    return commands


if __name__ == "__main__":

    with open("scripts/sensitivity_analysis/sensitivity_analysis.yaml", "r") as stream:
        try:
            exp_options = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    commands = run_types_to_commands(args.categories, exp_options)

    print(
        "Running Sensitivity Analysis\n"
        "---------------------------------------\n\n"
        f"Categories: {args.categories}\n"
        f"You have requested {len(commands)} runs"
    )

    if args.dry_run:
        print("Performing Dry Run")
        for c in commands:
            print(c)
    else:
        processes = set()
        available_coresets = set()

        for i in range(args.max_parallel_runs):
            start_core = i * args.num_chains
            end_core = (i + 1) * args.num_chains - 1

            if args.exclude_core_indices:
                for j in range(start_core, end_core + 1):
                    if j in args.exclude_core_indices:
                        continue

            available_coresets.add(f"{start_core}-{end_core}")

        for command in commands:
            # grab set of cpus
            coreset = available_coresets.pop()
            # unfortunately, the best way to parallelise well is to set processor
            # affinities.
            full_cmd = f"taskset -c {coreset} {command}"
            print(f"Running {full_cmd}")
            subproc = subprocess.Popen(full_cmd, shell=True)
            processes.add((coreset, subproc))
            time.sleep(5.0)

            if len(processes) >= args.max_parallel_runs:
                # wait for a child process to complete
                os.wait()
                # if poll returns something, the subprocess has finished
                finished_processes = [p for p in processes if p[1].poll() is not None]

                # remove from running processes
                processes.difference_update(finished_processes)

                # add the free cores to be reused
                for cs, _ in finished_processes:
                    available_coresets.add(cs)

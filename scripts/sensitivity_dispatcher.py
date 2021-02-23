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
    default=1,
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

args = argparser.parse_args()


def run_types_to_commands(run_types, exp_options):
    commands = []
    configs = args.model_config

    for config in configs:
        model_config = config
        for rt in run_types:
            exp_rt = exp_options[rt]
            experiment_file = exp_rt["experiment_file"]
            num_chains = exp_rt["num_chains"]
            num_samples = exp_rt["num_samples"]
            num_warmup = exp_rt["num_warmup"]
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

        for command in commands:
            processes.add(subprocess.Popen(command, shell=True))
            time.sleep(10.0)
            if len(processes) >= args.max_parallel_runs:
                os.wait()
                processes.difference_update(
                    [p for p in processes if p.poll() is not None]
                )
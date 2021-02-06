import os
import subprocess
import argparse
import yaml
import time


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--max_processes",
    dest="max_processes",
    type=int,
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
    help="Model config used to override default params for **all** requested runs",
)

args = argparser.parse_args()


def run_types_to_commands(run_types, exp_options):
    commands = []
    for rt in run_types:
        exp_rt = exp_options[rt]
        experiment_file = exp_rt["experiment_file"]
        n_chains = exp_rt["n_chains"]
        n_samples = exp_rt["n_samples"]
        exp_tag = exp_rt["experiment_tag"]
        model_type = args.model_type
        model_config = args.model_config

        cmds = [
            f"python scripts/sensitivity_analysis/{experiment_file} --model_type {model_type}"
            f" --n_samples {n_samples} --n_chains {n_chains} --exp_tag {exp_tag} --model_config {model_config}"
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
        max_processes = args.max_processes

        for command in commands:
            processes.add(subprocess.Popen(command, shell=True))
            time.sleep(10.0)
            if len(processes) >= max_processes:
                os.wait()
                processes.difference_update(
                    [p for p in processes if p.poll() is not None]
                )

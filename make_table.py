import argparse
import glob
import json
from pathlib import Path

import pandas as pd
from tabulate import tabulate


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment-name", required=True, help="Name of the experiment")
    parser.add_argument("--model", required=True, help="Model name")

    args = parser.parse_args()

    return args


def load_results(experiment_name: str, model: str):
    # Scan for folders of experiments and sort them by name
    folders = sorted(glob.glob(f"logs/schedule/*{experiment_name}/{model}"))

    # Prepare the data structure
    data = {
        "experiment": [],
        "textual_inclusion": [],
        "semantic_similarity": [],
        "concept_semantic_similarity": [],
        "exact_match": [],
        "llama_inclusion": [],
        "file": [],
    }

    # For each experiment, load its results
    for experiment_folder in folders:
        # Get the last file in ascending order, i.e., the most recent one
        # so that we load the most recent experiment for current task
        last_file = sorted(glob.glob(f"{experiment_folder}/*.json"))[-1]
        file_name = Path(last_file).name
        experiment_file_name = file_name.split(".")[0].replace("_results", "")

        # Do the same for the samples file, to ensure they match
        last_file_samples = sorted(glob.glob(f"{experiment_folder}/*.jsonl"))[-1]
        file_samples_name = Path(last_file_samples).name
        samples_file_name = file_samples_name[: file_samples_name.find("_samples")]

        assert experiment_file_name == samples_file_name, (
            f"Experiment file name and samples file name do not match: "
            f"{experiment_file_name} != {samples_file_name}"
        )

        # Load the experiment
        with open(last_file) as f:
            experiment = json.load(f)

        # Load the samples
        with open(last_file_samples) as f:
            samples = [json.loads(x) for x in f.readlines()]

        # Compute LLaMa inclusion score
        llama_inclusion_score = [x.get("textual_inclusion_llama32", 0) for x in samples]
        llama_inclusion_score = sum(llama_inclusion_score) / len(llama_inclusion_score)

        # Store results
        experiment_name = list(experiment["results"].keys())[0]
        data["experiment"].append(experiment_name)
        data["concept_semantic_similarity"].append(
            experiment["results"][experiment_name]["concept_semantic_similarity,none"]
        )
        data["exact_match"].append(experiment["results"][experiment_name]["exact_match,none"])
        data["semantic_similarity"].append(
            experiment["results"][experiment_name]["semantic_similarity,none"]
        )
        data["textual_inclusion"].append(
            experiment["results"][experiment_name]["textual_inclusion,none"]
        )
        data["llama_inclusion"].append(llama_inclusion_score)
        data["file"].append(file_name)

    data = pd.DataFrame(data)

    return data


def make_latex(data: pd.DataFrame, key: str) -> str:
    # Print textual_inclusion values formatted for LaTeX:
    # multiply by 100, round to 1 decimal place, separated by ' & '
    values = []
    for v in data[key]:
        try:
            values.append(f"{float(v) * 100:.1f}")
        except Exception:
            values.append("-")
    return " & ".join(values)


def make_latex_all(data):
    print("\n===== LaTeX Table =====")
    print("Textual inclusion")
    print(make_latex(data, "textual_inclusion"))
    print()
    print("LLaMa inclusion")
    print(make_latex(data, "llama_inclusion"))
    print()
    print("Semantic similarity")
    print(make_latex(data, "semantic_similarity"))
    print()
    print("Concept semantic similarity")
    print(make_latex(data, "concept_semantic_similarity"))


def main(args):
    data = load_results(args.experiment_name, args.model)

    # Print the whole dataframe as a sanity check
    print(tabulate(data, headers="keys", tablefmt="fancy_grid", showindex=False))

    make_latex_all(data)


if __name__ == "__main__":
    args = get_args()
    main(args)

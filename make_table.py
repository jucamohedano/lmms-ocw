import argparse
import glob
import json
from pathlib import Path

import pandas as pd
from tabulate import tabulate


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment-name", required=True, help="Name of the experiment")
    parser.add_argument("--model", required=False, help="Model name", default=None)
    parser.add_argument("--root", required=False, help="Root directory for logs", default="logs/schedule")
    parser.add_argument("--exclude", required=False, help="Exclude certain experiments", default=None)

    args = parser.parse_args()

    return args


def load_results(experiment_name: str, model: str | None, exclude: str | None):
    # Scan for folders of experiments and sort them by name
    _all = False
    if experiment_name == "all":
        experiment_name = ""
        _all = True

    if model is None:
        folders = sorted(glob.glob(f"{args.root}/*{experiment_name}"))
        print(f"{args.root}/*{experiment_name}")
    else:
        folders = sorted(glob.glob(f"{args.root}/*{experiment_name}/{model}"))
        print(f"{args.root}/*{experiment_name}/{model}")

    if exclude is not None:
        folders = [f for f in folders if exclude not in f]

    # Keep only the base experiments if no specific experiment name is given and not loading all
    if not _all and len(experiment_name) == 0 and len(folders) > 0:
        # Folders have this structure: logs/schedule/{experiment}[_other_stuff]/{model}
        # Group them by "{experiment}" part
        experiment_names = {Path(x).parent.name: x for x in folders}
        experiments = []
        for exp in experiment_names:
            # If the previous experiment name is contained in the current one, skip it
            if len(experiments) > 0 and experiments[-1] in exp:
                continue
            experiments.append(exp)

        folders = [experiment_names[x] for x in experiments]

    # Prepare the data structure
    data = {
        "experiment": [],
        "textual_inclusion": [],
        "semantic_similarity": [],
        "concept_semantic_similarity": [],
        "median_concept_semantic_similarity": [],
        "simplified_concept_semantic_similarity": [],
        "textual_iou": [],
        "exact_match": [],
        "llama_inclusion": [],
        "average_context_tokens": [],
        "perplexity": [],
        "file": [],
        "eval_seconds": [],
        "average_eval_seconds": [],
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

        # if not (experiment_file_name == "results" and samples_file_name == "results.json"):
        #     assert experiment_file_name == samples_file_name, (
        #         f"Experiment file name and samples file name do not match: "
        #         f"{experiment_file_name} != {samples_file_name}"
        #     )

        # Load the experiment
        with open(last_file) as f:
            experiment = json.load(f)
        experiment_name = list(experiment["results"].keys())[0]


        # Load the samples
        with open(last_file_samples) as f:
            samples = [json.loads(x) for x in f.readlines()]

        if "textual_inclusion_llama32,none" in experiment["results"][experiment_name]:
            llama_inclusion_score = experiment["results"][experiment_name]["textual_inclusion_llama32,none"]
        else:
            # Compute LLaMa inclusion score
            llama_inclusion_score = [x.get("textual_inclusion_llama32", 0) for x in samples]
            llama_inclusion_score = sum(llama_inclusion_score) / len(llama_inclusion_score)

        if "median_concept_semantic_similarity,none" in experiment["results"][experiment_name]:
            median_concept_semantic_similarity = experiment["results"][experiment_name]["median_concept_semantic_similarity,none"]
        else:
            # Compute median concept semantic similarity
            median_concept_semantic_similarity = [x.get("median_concept_semantic_similarity", 0) for x in samples]
            median_concept_semantic_similarity = sum(median_concept_semantic_similarity) / len(median_concept_semantic_similarity)

        if "simplified_concept_semantic_similarity,none" in experiment["results"][experiment_name]:
            simplified_concept_semantic_similarity = experiment["results"][experiment_name]["simplified_concept_semantic_similarity,none"]
        else:
            # Compute simplified concept semantic similarity
            simplified_concept_semantic_similarity = [x.get("simplified_concept_semantic_similarity", 0) for x in samples]
            simplified_concept_semantic_similarity = sum(simplified_concept_semantic_similarity) / len(simplified_concept_semantic_similarity)

        if "textual_iou,none" in experiment["results"][experiment_name]:
            textual_iou = experiment["results"][experiment_name]["textual_iou,none"]
        else:
            # Compute textual IoU
            textual_iou = [x.get("textual_iou", 0) for x in samples]
            textual_iou = sum(textual_iou) / len(textual_iou)

        if "perplexity,none" in experiment["results"][experiment_name]:
            perplexity = experiment["results"][experiment_name]["perplexity,none"]
        else:
            # Compute average perplexity
            perplexity = [x.get("perplexity", 0) for x in samples]
            perplexity = sum(perplexity) / len(perplexity)

        # Store results
        data["experiment"].append(experiment_name)
        data["concept_semantic_similarity"].append(
            experiment["results"][experiment_name]["concept_semantic_similarity,none"]
        )
        data["median_concept_semantic_similarity"].append(
            median_concept_semantic_similarity
        )
        data["simplified_concept_semantic_similarity"].append(
            simplified_concept_semantic_similarity
        )
        data["textual_iou"].append(textual_iou)
        data["perplexity"].append(perplexity)
        data["exact_match"].append(experiment["results"][experiment_name]["exact_match,none"])
        data["semantic_similarity"].append(
            experiment["results"][experiment_name]["semantic_similarity,none"]
        )
        data["textual_inclusion"].append(
            experiment["results"][experiment_name]["textual_inclusion,none"]
        )
        data["llama_inclusion"].append(llama_inclusion_score)
        data["average_context_tokens"].append(
            round(experiment["results"][experiment_name].get("context_length,none", 0))
        )
        data["file"].append(file_name)
        data["eval_seconds"].append(float(experiment.get("total_evaluation_time_seconds", 0)))
        data["average_eval_seconds"].append(float(experiment.get("total_evaluation_time_seconds", 0)) / len(samples))

    data = pd.DataFrame(data)

    return data


def results_by_group(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    dataset_groups = {
        "prototypical": [
            "caltech101", "sun397"
        ],
        "non-prototypical": [
            "dtd", "ucf101", "eurosat"
        ],
        "fine-grained": [
            "flowers102", "food101", "oxford_pets", "oxfordpets"
        ],
        "very-fine-grained": [
            "stanford_cars", "stanfordcars", "fgvc_aircraft", "fgvcaircraft"
        ]
    }

    grouped_results = {
        "prototypical": [],
        "non-prototypical": [],
        "fine-grained": [],
        "very-fine-grained": []
    }

    for group, datasets in dataset_groups.items():
        for _, result in data.iterrows():
            if any(d in result["experiment"] for d in datasets):
                grouped_results[group].append(result)

    for group in grouped_results.keys():
        grouped_results[group] = pd.DataFrame(grouped_results[group])

    # Average metrics
    for group in grouped_results.keys():
        try:
            grouped_results[group] = grouped_results[group][[
                "textual_inclusion",
                "semantic_similarity",
                "concept_semantic_similarity",
                "median_concept_semantic_similarity",
                "simplified_concept_semantic_similarity",
                "textual_iou",
                "exact_match",
                "llama_inclusion",
                "average_context_tokens",
                "perplexity"
            ]].mean()
        except:
            grouped_results[group] = pd.Series(
                [0.0] * 11,
            )

    # Make a new dataframe with only four rows -- one per group -- and the average of each metric
    # This new dataframe collects the average metrics per group
    grouped_results = pd.concat([
        grouped_results[group].to_frame().T
        for group in grouped_results.keys()
    ])
    grouped_results.insert(0, "experiment", list(dataset_groups.keys()))

    # Reorder columns
    grouped_results = grouped_results[[
        "experiment",
        "textual_inclusion",
        "textual_iou",
        "llama_inclusion",
        "semantic_similarity",
        "concept_semantic_similarity",
        "median_concept_semantic_similarity",
        "simplified_concept_semantic_similarity",
        "exact_match",
    ]]

    return grouped_results


def make_latex(data: pd.DataFrame, key: str) -> str:
    # Print textual_inclusion values formatted for LaTeX:
    # multiply by 100, round to 1 decimal place, separated by ' & '
    values = []
    for v in data[key]:
        try:
            values.append(f"{float(v) * 100:.1f}")
        except Exception:
            values.append("-")

    mean = data[key].mean()
    values.append(f"{mean * 100:.1f}")

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
    print()
    print("Median concept semantic similarity")
    print(make_latex(data, "median_concept_semantic_similarity"))
    print()
    print("Textual IoU")
    print(make_latex(data, "textual_iou"))


def rename_cols(data: pd.DataFrame) -> pd.DataFrame:
    return data.rename(columns={
        "experiment": "Experiment",
        "textual_inclusion": "Txt Incl",
        "semantic_similarity": "Sem Sim",
        "concept_semantic_similarity": "Con Sim",
        "median_concept_semantic_similarity": "Median CS",
        "simplified_concept_semantic_similarity": "!Con Sim",
        "textual_iou": "IoU",
        "exact_match": "Acc",
        "llama_inclusion": "LLaMa",
        "average_context_tokens": "Avg Ctx Tok",
        "perplexity": "PPL",
        "file": "File",
        "eval_seconds": "Time (s)",
        "average_eval_seconds": "Avg. Time (s)",
    })


def main(args):
    data = load_results(args.experiment_name, args.model, args.exclude)
    try:
        data_by_group = results_by_group(data)
    except:
        data_by_group = None
        print("Could not compute grouped results.")


    # Scale numbers to 0-100 and round to one decimal place
    # Columns affected: indices 1 to 6
    data_display_ready = data.copy()
    for col in data.columns[1:-3]:
        data_display_ready[col] = data_display_ready[col].apply(lambda x: round(x * 100, 1))

    if data_by_group is not None:
        for col in data_by_group.columns[1:-1]:
            data_by_group[col] = data_by_group[col].apply(lambda x: round(x * 100, 1))

    # Rename columns for better display
    data_display_ready = rename_cols(data_display_ready)

    if data_by_group is not None:
        data_by_group = rename_cols(data_by_group)

    # Print the whole dataframe as a sanity check
    print(tabulate(data_display_ready, headers="keys", tablefmt="fancy_grid", showindex=False))
    print("\nAverage runtime (s):", data["eval_seconds"].mean())
    print("\nAverage runtime per sample (s):", data["average_eval_seconds"].mean())

    if data_by_group is not None:
        print("\n===== Grouped Results =====")
        print(tabulate(data_by_group, headers="keys", tablefmt="fancy_grid", showindex=False))

    make_latex_all(data)


if __name__ == "__main__":
    args = get_args()
    main(args)

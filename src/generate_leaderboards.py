"""Generate leaderboard CSV files from the ScandEval results."""

from collections import defaultdict
import json
from pathlib import Path
import warnings
import click
from yaml import safe_load
import logging
import pandas as pd
import scipy.stats as stats
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ⋅ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


@click.command()
@click.argument("leaderboard_config")
def main(leaderboard_config: str | Path) -> None:
    """Generate leaderboard CSV files from the ScandEval results.

    Args:
        leaderboard_config:
            The path to the leaderboard configuration file.
    """
    leaderboard_config = Path(leaderboard_config)

    logger.info(f"Generating {leaderboard_config.stem.title()} leaderboard...")

    # Load configs
    with Path("task_config.yaml").open(mode="r") as f:
        task_config: dict[str, dict[str, str]] = safe_load(stream=f)
    with leaderboard_config.open(mode="r") as f:
        config: dict[str, list[str]] = safe_load(stream=f)

    # Load the list of all datasets to be used in the leaderboard
    datasets = [
        dataset for task_datasets in config.values() for dataset in task_datasets
    ]

    # Load results and set them up for the leaderboard
    results = load_results(allowed_datasets=datasets)
    model_results = group_results_by_model(
        results=results, task_config=task_config, required_datasets=datasets
    )
    ranks = compute_ranks(model_results=model_results, config=config)
    metadata_dict = extract_model_metadata(results=results)

    # Generate the leaderboard and store it to disk
    leaderboard_path = Path("leaderboards") / f"{leaderboard_config.stem}.csv"
    df = generate_dataframe(
        model_results=model_results,
        ranks=ranks,
        metadata_dict=metadata_dict,
    )
    df.to_csv(leaderboard_path, index=False)


def load_results(allowed_datasets: list[str]) -> list[dict]:
    """Load processed results.

    Args:
        allowed_datasets:
            The list of datasets to include in the leaderboard.

    Returns:
        The processed results.

    Raises:
        FileNotFoundError:
            If the processed results file is not found.
    """
    results_path = Path("results/results.processed.jsonl")
    if not results_path.exists():
        raise FileNotFoundError("Processed results file not found.")

    results = list()
    with results_path.open() as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            for record in line.replace("}{", "}\n{").split("\n"):
                if not record.strip():
                    continue
                try:
                    results.append(json.loads(record))
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON on line {line_idx:,}: {record}.")

    # Only keep relevant results
    results = [record for record in results if record["dataset"] in allowed_datasets]

    return results


def group_results_by_model(
    results: list[dict],
    task_config: dict[str, dict[str, str]],
    required_datasets: list[str],
) -> dict[str, dict[str, tuple[list[float], float]]]:
    """Group results by model ID.

    Args:
        results:
            The processed results.
        task_config:
            The task configuration.
        required_datasets:
            The list of datasets to include in the leaderboard, which every model must
            have been evaluated on.

    Returns:
        The results grouped by model ID.
    """
    model_scores: dict[str, dict[str, tuple[list[float], float]]] = defaultdict(dict)
    for record in results:
        model_id = extract_model_id_from_record(record=record)
        dataset: str = record["dataset"]
        metric = task_config[record["task"]]["metric"]

        # Get the metrics for the dataset
        total_score: float = record["results"]["total"][f"test_{metric}"]
        if "test" in record["results"]["raw"]:
            raw_scores = [
                result_dict.get(f"test_{metric}", result_dict.get(metric, -1))
                for result_dict in record["results"]["raw"]["test"]
            ]
        else:
            raw_scores = [
                result_dict.get(f"test_{metric}", result_dict.get(metric, -1))
                for result_dict in record["results"]["raw"]
            ]

        # Sometimes the raw scores are normalised to [0, 1], so we need to scale them
        # back to [0, 100]
        if max(raw_scores) <= 1:
            raw_scores = [score * 100 for score in raw_scores]

        model_scores[model_id][dataset] = (raw_scores, total_score)

    # Remove the models that don't have scores for all datasets
    model_scores = {
        model_id: scores
        for model_id, scores in model_scores.items()
        if set(scores.keys()) == set(required_datasets)
    }

    return model_scores


def compute_ranks(
    model_results: dict[str, dict[str, tuple[list[float], float]]],
    config: dict[str, list[str]],
) -> dict[str, float]:
    """Compute the ranks of the models.

    Args:
        model_results:
            The model results.
        config:
            The leaderboard configuration.

    Returns:
        The ranks of the models.
    """
    datasets = [
        dataset for task_datasets in config.values() for dataset in task_datasets
    ]

    model_dataset_ranks: dict[str, dict[str, float]] = defaultdict(dict)
    for dataset in datasets:
        model_dataset_scores = sorted(
            [
                (model_id, scores[dataset][0], scores[dataset][1])
                for model_id, scores in model_results.items()
            ],
            key=lambda x: x[2],
            reverse=True,
        )
        stddev = np.std([score for _, _, score in model_dataset_scores])

        rank_score = 1.0
        previous_scores: list[float] = list()
        for idx, (model_id, raw_scores, _) in enumerate(model_dataset_scores):
            if idx == 0:
                previous_scores = raw_scores
            elif significantly_better(previous_scores, raw_scores):
                difference = np.mean(previous_scores) - np.mean(raw_scores)
                normalised_difference = difference / stddev
                rank_score += normalised_difference.item()
                previous_scores = raw_scores
            model_dataset_ranks[model_id][dataset] = rank_score

    model_task_ranks: dict[str, dict[str, float]] = defaultdict(dict)
    for model_id, dataset_ranks in model_dataset_ranks.items():
        for task, datasets in config.items():
            model_task_ranks[model_id][task] = np.mean(
                [dataset_ranks[dataset] for dataset in datasets]
            ).item()

    return {
        model_id: np.mean(list(task_scores.values())).item()
        for model_id, task_scores in model_task_ranks.items()
    }


def extract_model_metadata(results: list[dict]) -> dict[str, dict]:
    """Extract metadata from the results.

    Args:
        results:
            The processed results.

    Returns:
        The metadata.
    """
    metadata_dict: dict[str, dict] = defaultdict(dict)
    for record in results:
        model_id = extract_model_id_from_record(record=record)
        num_params = (
            round(record["num_model_parameters"] / 1_000_000)
            if record["num_model_parameters"] >= 0
            else "N/A"
        )
        vocab_size = (
            round(record["vocabulary_size"] / 1_000)
            if record["vocabulary_size"] >= 0
            else "N/A"
        )
        context = (
            record["max_sequence_length"]
            if record["max_sequence_length"] >= 0
            else "N/A"
        )
        metadata_dict[model_id] = dict(
            parameters=num_params,
            vocabulary_size=vocab_size,
            context=context,
            commercial=record.get("commercially_licensed", False),
            merge=record.get("merge", False),
        )

    return metadata_dict


def extract_model_id_from_record(record: dict) -> str:
    """Extract the model ID from a record.

    Args:
        record:
            The record.

    Returns:
        The model ID.
    """
    model_id: str = record["model"]
    model_notes: list[str] = list()

    if record.get("generative", True):
        if record.get("few_shot", True):
            model_notes.append("few-shot")
        else:
            model_notes.append("zero-shot")

    if record.get("validation_split", False):
        model_notes.append("val")

    if model_notes:
        model_id += f" ({', '.join(model_notes)})"

    return model_id


def generate_dataframe(
    model_results: dict[str, dict[str, tuple[list[float], float]]],
    ranks: dict[str, float],
    metadata_dict: dict[str, dict],
) -> pd.DataFrame:
    """Generate a DataFrame from the model results.

    Args:
        model_results:
            The model results.
        ranks:
            The ranks of the models.
        metadata_dict:
            The metadata.

    Returns:
        The DataFrame.
    """
    data_dict: dict[str, list] = defaultdict(list)
    for model_id, results in model_results.items():
        total_results = {
            dataset: total_score for dataset, (_, total_score) in results.items()
        }
        metadata = metadata_dict[model_id]

        data_dict["model"].append(model_id)
        data_dict["rank"].append(ranks[model_id])
        for key, value in (metadata | total_results).items():
            data_dict[key].append(value)

    return (
        pd.DataFrame(data_dict)
        .sort_values(by="rank", ascending=True)
        .reset_index(drop=True)
    )


def significantly_better(
    score_values_1: list[float], score_values_2: list[float]
) -> float:
    """Compute one-tailed t-statistic for the difference between two sets of scores.

    Args:
        score_values_1:
            The first set of scores.
        score_values_2:
            The second set of scores.

    Returns:
        The t-statistic of the difference between the two sets of scores, where
        a positive t-statistic indicates that the first set of scores is
        statistically better than the second set of scores.
    """
    assert len(score_values_1) == len(score_values_2)
    if score_values_1 == score_values_2:
        return 0
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=RuntimeWarning)
        test_result = stats.ttest_ind(
            a=score_values_1,
            b=score_values_2,
            alternative="greater",
            equal_var=False,
        )
    return test_result.pvalue < 0.05  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()

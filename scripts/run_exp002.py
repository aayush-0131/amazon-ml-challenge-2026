#!/usr/bin/env python3
"""Run EXP002 multi-pass blocking and compact learned matchers on TRAIN only."""

from __future__ import annotations

import argparse
import csv
import gzip
import heapq
import importlib.metadata
import io
import json
import platform
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from business_entity_resolution.data import iter_source_chunks  # noqa: E402
from business_entity_resolution.evaluation import evaluate_predictions  # noqa: E402
from business_entity_resolution.features import (  # noqa: E402
    FEATURE_NAMES,
    build_feature_row,
)
from business_entity_resolution.modeling import (  # noqa: E402
    NegativeSamplingConfig,
    TrainingPairSampler,
    examples_to_arrays,
    make_hist_gradient_boosting_model,
    make_logistic_model,
    prediction_scores,
    select_global_threshold,
    stable_pair_hash,
)
from business_entity_resolution.multipass import (  # noqa: E402
    PASS_NAMES,
    CandidateBudget,
    MultiPassBlocker,
    MultiPassConfig,
    RetrievedCandidate,
    retrieved_candidate_from_mapping,
)
from business_entity_resolution.sampling import (  # noqa: E402
    SourceRecord,
    load_selected_ground_truth,
    load_selected_source_records,
)
from business_entity_resolution.split import stratified_entity_split  # noqa: E402


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return value / (1024 * 1024)
    return value / 1024


class DeterministicGzipTsvWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.raw = path.open("wb")
        self.compressed = gzip.GzipFile(
            filename="", fileobj=self.raw, mode="wb", mtime=0
        )
        self.text = io.TextIOWrapper(self.compressed, encoding="utf-8", newline="")
        self.writer: csv.DictWriter[str] | None = None

    def write(self, row: dict[str, object]) -> None:
        if self.writer is None:
            self.writer = csv.DictWriter(
                self.text,
                fieldnames=list(row),
                delimiter="\t",
                lineterminator="\n",
            )
            self.writer.writeheader()
        self.writer.writerow(row)

    def close(self) -> None:
        self.text.flush()
        self.text.detach()
        self.compressed.close()
        self.raw.close()

    def __enter__(self) -> DeterministicGzipTsvWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def load_budgets(config: dict[str, object]) -> list[CandidateBudget]:
    return [CandidateBudget(**values) for values in config["blocker_budgets"]]


def prepare_partitions(
    subset_path: Path,
    records: dict[str, SourceRecord],
    truth: dict[str, frozenset[str]],
    *,
    tune_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, str]]:
    exp001 = pd.read_csv(subset_path)
    if set(exp001["source1_entity_id"]) != set(records):
        raise ValueError("EXP001 subset IDs do not match loaded Source-1 records")
    evaluation_ids = frozenset(
        exp001.loc[exp001["partition"].eq("evaluation"), "source1_entity_id"]
    )
    development_ids = frozenset(records) - evaluation_ids
    development_records = [
        {
            "entity_id": entity_id,
            "country": records[entity_id].country,
        }
        for entity_id in development_ids
    ]
    development_truth = {entity_id: truth[entity_id] for entity_id in development_ids}
    split = stratified_entity_split(
        development_records,
        development_truth,
        validation_fraction=tune_fraction,
        seed=seed,
    )
    partition = {
        **{entity_id: "fit" for entity_id in split.train_ids},
        **{entity_id: "tune" for entity_id in split.validation_ids},
        **{entity_id: "evaluation" for entity_id in evaluation_ids},
    }
    table = pd.DataFrame(
        [
            {
                "source1_entity_id": entity_id,
                "country": records[entity_id].country,
                "match_count": len(truth[entity_id]),
                "is_singleton": not truth[entity_id],
                "partition": partition[entity_id],
            }
            for entity_id in sorted(records)
        ]
    )
    return table, partition


def scan_retrieval_pool(
    records: dict[str, SourceRecord],
    data_root: Path,
    artifact_path: Path,
    blocker_config: MultiPassConfig,
    *,
    chunksize: int,
) -> tuple[dict[str, Counter[str]], list[dict[str, object]]]:
    country_counts: dict[str, Counter[str]] = {}
    runtime_rows: list[dict[str, object]] = []
    with DeterministicGzipTsvWriter(artifact_path) as writer:
        for source in ("S2", "S3"):
            started = time.monotonic()
            blocker = MultiPassBlocker(
                records.values(), source=source, config=blocker_config
            )
            source_path = data_root / "train" / f"train_source{source[-1]}.tsv"
            print(f"EXP002 scanning {source_path} ...", flush=True)
            for chunk in iter_source_chunks(source_path, source, chunksize=chunksize):
                blocker.process_chunk(chunk)
            candidates = blocker.finalize()
            for candidate in candidates:
                writer.write(candidate.to_dict())
            country_counts[source] = blocker.target_country_counts
            runtime_rows.append(
                {
                    "stage": f"retrieval_{source.lower()}",
                    "seconds": time.monotonic() - started,
                    "peak_rss_mb": peak_rss_mb(),
                    "row_count": len(candidates),
                }
            )
            print(
                f"{source}: maximum pool {len(candidates):,}; "
                f"peak RSS {peak_rss_mb():.1f} MB.",
                flush=True,
            )
            del candidates, blocker
    return country_counts, runtime_rows


def candidate_selected(row: dict[str, str], budget: CandidateBudget) -> bool:
    return any(
        0 < int(row[f"{pass_name}_rank"]) <= budget.pass_budget(pass_name)
        for pass_name in PASS_NAMES
    )


def evaluate_blocker_budgets(
    artifact_path: Path,
    budgets: list[CandidateBudget],
    truth: dict[str, frozenset[str]],
    records: dict[str, SourceRecord],
    target_country_counts: dict[str, Counter[str]],
) -> tuple[pd.DataFrame, CandidateBudget]:
    selected_counts = {
        budget.name: {"S2": Counter(), "S3": Counter()} for budget in budgets
    }
    retained_counts = {
        budget.name: {"S2": Counter(), "S3": Counter()} for budget in budgets
    }
    started = time.monotonic()
    with gzip.open(artifact_path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            source1_id = row["source1_entity_id"]
            source = row["source"]
            target_id = row["candidate_entity_id"]
            for budget in budgets:
                count = selected_counts[budget.name][source][source1_id]
                if count >= budget.total_per_source:
                    continue
                if candidate_selected(row, budget):
                    selected_counts[budget.name][source][source1_id] += 1
                    if target_id in truth[source1_id]:
                        retained_counts[budget.name][source][source1_id] += 1

    rows: list[dict[str, object]] = []
    countries = sorted({record.country for record in records.values()})
    group_specs: list[tuple[str, str, str | None, list[str]]] = [
        ("overall", "ALL", None, list(records))
    ]
    group_specs.extend(
        (
            "country",
            country,
            None,
            [entity_id for entity_id, record in records.items() if record.country == country],
        )
        for country in countries
    )
    for source in ("S2", "S3"):
        group_specs.append(("source", source, source, list(records)))
        group_specs.extend(
            (
                "source_country",
                f"{source}|{country}",
                source,
                [
                    entity_id
                    for entity_id, record in records.items()
                    if record.country == country
                ],
            )
            for country in countries
        )

    for budget in budgets:
        for dimension, group, source, entity_ids in group_specs:
            counts = []
            true_total = retained_total = all_retained = 0
            matched_count = matched_all = 0
            possible_pairs = 0
            for entity_id in entity_ids:
                if source:
                    candidates = selected_counts[budget.name][source][entity_id]
                    retained = retained_counts[budget.name][source][entity_id]
                    true_count = sum(
                        target.startswith(f"{source}-") for target in truth[entity_id]
                    )
                    eligible = target_country_counts[source][records[entity_id].country]
                else:
                    candidates = sum(
                        selected_counts[budget.name][value][entity_id]
                        for value in ("S2", "S3")
                    )
                    retained = sum(
                        retained_counts[budget.name][value][entity_id]
                        for value in ("S2", "S3")
                    )
                    true_count = len(truth[entity_id])
                    eligible = sum(
                        target_country_counts[value][records[entity_id].country]
                        for value in ("S2", "S3")
                    )
                counts.append(candidates)
                true_total += true_count
                retained_total += retained
                all_retained += int(retained == true_count)
                if true_count:
                    matched_count += 1
                    matched_all += int(retained == true_count)
                possible_pairs += eligible
            values = np.asarray(counts, dtype=np.int64)
            rows.append(
                {
                    "configuration": budget.name,
                    "dimension": dimension,
                    "group": group,
                    "entity_count": len(entity_ids),
                    "true_link_count": true_total,
                    "retained_true_link_count": retained_total,
                    "positive_link_recall": retained_total / true_total,
                    "all_true_links_retained_rate": all_retained / len(entity_ids),
                    "matched_all_true_links_retained_rate": matched_all / matched_count,
                    "average_candidates": float(values.mean()),
                    "median_candidates": float(np.quantile(values, 0.50)),
                    "p95_candidates": float(np.quantile(values, 0.95)),
                    "p99_candidates": float(np.quantile(values, 0.99)),
                    "max_candidates": int(values.max()),
                    "zero_candidate_entities": int((values == 0).sum()),
                    "reduction_ratio": 1.0 - int(values.sum()) / possible_pairs,
                    "evaluation_seconds": time.monotonic() - started,
                    "peak_rss_mb": peak_rss_mb(),
                }
            )
    table = pd.DataFrame(rows)
    overall = table.query('dimension == "overall"')
    selected_name = max(
        overall.itertuples(index=False),
        key=lambda row: (row.positive_link_recall, -row.average_candidates),
    ).configuration
    selected = next(budget for budget in budgets if budget.name == selected_name)
    return table, selected


def build_feature_artifact(
    retrieval_path: Path,
    feature_path: Path,
    selected_budget: CandidateBudget,
    records: dict[str, SourceRecord],
    truth: dict[str, frozenset[str]],
    partition: dict[str, str],
    sampler: TrainingPairSampler,
) -> tuple[dict[str, set[str]], int]:
    selected_per_source: dict[tuple[str, str], int] = Counter()
    retained_evaluation: dict[str, set[str]] = defaultdict(set)
    row_count = 0
    with gzip.open(retrieval_path, "rt", encoding="utf-8", newline="") as handle:
        with DeterministicGzipTsvWriter(feature_path) as writer:
            for raw in csv.DictReader(handle, delimiter="\t"):
                source1_id = raw["source1_entity_id"]
                source = raw["source"]
                key = (source1_id, source)
                if selected_per_source[key] >= selected_budget.total_per_source:
                    continue
                if not candidate_selected(raw, selected_budget):
                    continue
                selected_per_source[key] += 1
                candidate = retrieved_candidate_from_mapping(raw)
                features = build_feature_row(records[source1_id], candidate)
                label = int(candidate.candidate_entity_id in truth[source1_id])
                row = {
                    "source1_entity_id": source1_id,
                    "candidate_entity_id": candidate.candidate_entity_id,
                    "source": source,
                    "country": candidate.country,
                    "partition": partition[source1_id],
                    "label": label,
                    **features,
                }
                writer.write(row)
                row_count += 1
                if partition[source1_id] == "fit":
                    hardness = (
                        0.7 * features["exp001_rule_score"]
                        + 0.3 * min(features["best_retrieval_score"], 100.0)
                    )
                    sampler.add(
                        source1_id,
                        candidate.candidate_entity_id,
                        [features[name] for name in FEATURE_NAMES],
                        label=label,
                        hardness=hardness,
                    )
                elif partition[source1_id] == "evaluation" and label:
                    retained_evaluation[source1_id].add(candidate.candidate_entity_id)
    return dict(retained_evaluation), row_count


def read_feature_chunks(path: Path, *, chunksize: int = 50_000):
    yield from pd.read_csv(path, sep="\t", compression="gzip", chunksize=chunksize)


def score_tuning_pairs(
    feature_path: Path,
    models: dict[str, object],
    truth: dict[str, frozenset[str]],
    tune_ids: frozenset[str],
    *,
    importance_sample_size: int,
    seed: int,
) -> tuple[
    dict[str, dict[str, dict[str, float]]],
    tuple[np.ndarray, np.ndarray],
]:
    score_maps: dict[str, dict[str, dict[str, float]]] = {
        name: defaultdict(dict) for name in models
    }
    sample_heap: list[tuple[int, str, str, tuple[float, ...], int]] = []
    for chunk in read_feature_chunks(feature_path):
        chunk = chunk.loc[chunk["partition"].eq("tune")]
        if chunk.empty:
            continue
        matrix = chunk.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float32)
        scores = {
            "rule": chunk["exp001_rule_score"].to_numpy(dtype=np.float64),
            **{
                name: prediction_scores(model, matrix)
                for name, model in models.items()
                if name != "rule"
            },
        }
        metadata = list(
            chunk.loc[:, ["source1_entity_id", "candidate_entity_id"]].itertuples(
                index=False, name=None
            )
        )
        labels = chunk["label"].to_numpy(dtype=np.int8)
        for index, (source1_id, candidate_id) in enumerate(metadata):
            for name, values in scores.items():
                score_maps[name][source1_id][candidate_id] = float(values[index])
            pair_hash = stable_pair_hash(source1_id, candidate_id, seed)
            item = (
                -pair_hash,
                source1_id,
                candidate_id,
                tuple(float(value) for value in matrix[index]),
                int(labels[index]),
            )
            if len(sample_heap) < importance_sample_size:
                heapq.heappush(sample_heap, item)
            elif item[:3] > sample_heap[0][:3]:
                heapq.heapreplace(sample_heap, item)

    ordered_sample = sorted(sample_heap, key=lambda item: (-item[0], item[1], item[2]))
    sample_x = np.asarray([item[3] for item in ordered_sample], dtype=np.float32)
    sample_y = np.asarray([item[4] for item in ordered_sample], dtype=np.int8)
    return (
        {name: dict(values) for name, values in score_maps.items()},
        (sample_x, sample_y),
    )


def tune_models(
    truth: dict[str, frozenset[str]],
    tune_ids: frozenset[str],
    score_maps: dict[str, dict[str, dict[str, float]]],
    config: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, float], str]:
    tune_truth = {entity_id: truth[entity_id] for entity_id in tune_ids}
    rows: list[dict[str, object]] = []
    thresholds: dict[str, float] = {}
    for model_name, scores in score_maps.items():
        grid = (
            [float(value) for value in config["rule_thresholds"]]
            if model_name == "rule"
            else [float(value) for value in config["probability_thresholds"]]
        )
        threshold, result, grid_rows = select_global_threshold(
            tune_truth, scores, grid
        )
        thresholds[model_name] = threshold
        for row in grid_rows:
            rows.append({"model": model_name, **row})
        print(
            f"{model_name}: tune threshold={threshold}; "
            f"macro F0.5={result.macro_fbeta:.6f}",
            flush=True,
        )
    table = pd.DataFrame(rows)
    best_by_model = (
        table.sort_values(["macro_fbeta", "threshold"], ascending=[False, False])
        .groupby("model", as_index=False)
        .first()
    )
    selected_model = max(
        best_by_model.itertuples(index=False), key=lambda row: row.macro_fbeta
    ).model
    return table, thresholds, selected_model


def evaluate_models(
    feature_path: Path,
    models: dict[str, object],
    thresholds: dict[str, float],
    truth: dict[str, frozenset[str]],
    evaluation_ids: frozenset[str],
    selected_model: str,
) -> tuple[pd.DataFrame, dict[str, set[str]], Counter[str]]:
    predictions: dict[str, dict[str, set[str]]] = {
        name: {entity_id: set() for entity_id in evaluation_ids} for name in models
    }
    categories: Counter[str] = Counter()
    for chunk in read_feature_chunks(feature_path):
        chunk = chunk.loc[chunk["partition"].eq("evaluation")]
        if chunk.empty:
            continue
        matrix = chunk.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float32)
        scores = {
            "rule": chunk["exp001_rule_score"].to_numpy(dtype=np.float64),
            **{
                name: prediction_scores(model, matrix)
                for name, model in models.items()
                if name != "rule"
            },
        }
        metadata = list(
            chunk.loc[:, ["source1_entity_id", "candidate_entity_id"]].itertuples(
                index=False, name=None
            )
        )
        labels = chunk["label"].to_numpy(dtype=np.int8)
        selected_scores = scores[selected_model]
        selected_threshold = thresholds[selected_model]
        for index, (source1_id, candidate_id) in enumerate(metadata):
            for name, values in scores.items():
                if values[index] >= thresholds[name]:
                    predictions[name][source1_id].add(candidate_id)
            accepted = selected_scores[index] >= selected_threshold
            label = bool(labels[index])
            if accepted and not label:
                categories["false_positive_candidate_links"] += 1
                if (
                    chunk.iloc[index]["left_has_numeric"]
                    and chunk.iloc[index]["right_has_numeric"]
                    and chunk.iloc[index]["digit_token_overlap"] == 0
                ):
                    categories["fp_conflicting_numeric_address"] += 1
                if chunk.iloc[index]["right_address_missing"]:
                    categories["fp_missing_target_address"] += 1
                if (
                    chunk.iloc[index]["name_token_set_ratio"] >= 90
                    and chunk.iloc[index]["name_ratio"] < 70
                ):
                    categories["fp_high_token_set_poor_edit"] += 1
                if (
                    chunk.iloc[index]["address_token_set_ratio"] >= 90
                    and chunk.iloc[index]["name_token_set_ratio"] < 60
                ):
                    categories["fp_same_address_weak_name"] += 1
            elif label and not accepted:
                categories["matcher_false_negative_links"] += 1
                if chunk.iloc[index]["name_ratio"] < 50:
                    categories["fn_severe_name_corruption"] += 1
                if chunk.iloc[index]["address_ratio"] < 50:
                    categories["fn_severe_address_corruption"] += 1
                if chunk.iloc[index]["right_address_missing"]:
                    categories["fn_missing_target_address"] += 1
                margin = 2.0 if selected_model == "rule" else 0.05
                if selected_scores[index] >= selected_threshold - margin:
                    categories["fn_just_below_threshold"] += 1

    rows = []
    for name, model_predictions in predictions.items():
        result = evaluate_predictions(
            {entity_id: truth[entity_id] for entity_id in evaluation_ids},
            model_predictions,
        )
        singleton_accuracy = (
            result.correct_singletons / result.singleton_count
            if result.singleton_count
            else 1.0
        )
        rows.append(
            {
                "model": name,
                "threshold": thresholds[name],
                **result.to_dict(),
                "singleton_accuracy": singleton_accuracy,
                "average_predicted_matches": sum(
                    len(values) for values in model_predictions.values()
                )
                / len(model_predictions),
            }
        )
    return pd.DataFrame(rows), predictions[selected_model], categories


def error_slices(
    truth: dict[str, frozenset[str]],
    predictions: dict[str, set[str]],
    retained: dict[str, set[str]],
    records: dict[str, SourceRecord],
    evaluation_ids: frozenset[str],
) -> pd.DataFrame:
    specs: list[tuple[str, str, str | None, list[str]]] = [
        ("overall", "ALL", None, list(evaluation_ids))
    ]
    for country in sorted({records[value].country for value in evaluation_ids}):
        specs.append(
            (
                "country",
                country,
                None,
                [value for value in evaluation_ids if records[value].country == country],
            )
        )
    for source in ("S2", "S3"):
        specs.append(("source", source, source, list(evaluation_ids)))
        for country in sorted({records[value].country for value in evaluation_ids}):
            specs.append(
                (
                    "source_country",
                    f"{source}|{country}",
                    source,
                    [
                        value
                        for value in evaluation_ids
                        if records[value].country == country
                    ],
                )
            )
    specs.extend(
        [
            (
                "entity_type",
                "singleton",
                None,
                [value for value in evaluation_ids if not truth[value]],
            ),
            (
                "entity_type",
                "matched",
                None,
                [value for value in evaluation_ids if truth[value]],
            ),
            (
                "multiplicity",
                "1-2",
                None,
                [value for value in evaluation_ids if 1 <= len(truth[value]) <= 2],
            ),
            (
                "multiplicity",
                "3-4",
                None,
                [value for value in evaluation_ids if 3 <= len(truth[value]) <= 4],
            ),
            (
                "multiplicity",
                "5+",
                None,
                [value for value in evaluation_ids if len(truth[value]) >= 5],
            ),
        ]
    )
    rows = []
    for dimension, group, source, ids in specs:
        tp = fp = fn = blocker_fn = matcher_fn = 0
        for entity_id in ids:
            expected = {
                value
                for value in truth[entity_id]
                if source is None or value.startswith(f"{source}-")
            }
            predicted = {
                value
                for value in predictions[entity_id]
                if source is None or value.startswith(f"{source}-")
            }
            available = {
                value
                for value in retained.get(entity_id, set())
                if source is None or value.startswith(f"{source}-")
            }
            tp += len(expected & predicted)
            fp += len(predicted - expected)
            fn += len(expected - predicted)
            blocker_fn += len(expected - available)
            matcher_fn += len((expected & available) - predicted)
        rows.append(
            {
                "dimension": dimension,
                "group": group,
                "entity_count": len(ids),
                "true_positive_links": tp,
                "false_positive_links": fp,
                "false_negative_links": fn,
                "blocking_false_negative_links": blocker_fn,
                "matcher_false_negative_links": matcher_fn,
            }
        )
    return pd.DataFrame(rows)


def feature_importance_tables(
    models: dict[str, object],
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    *,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    logistic = models["logistic"]
    coefficients = np.abs(logistic.named_steps["model"].coef_[0])
    for name, value in zip(FEATURE_NAMES, coefficients, strict=True):
        rows.append({"model": "logistic", "feature": name, "importance": value})
    permutation = permutation_importance(
        models["hist_gradient_boosting"],
        sample_x,
        sample_y,
        n_repeats=2,
        random_state=seed,
        scoring="average_precision",
        n_jobs=1,
    )
    for name, value in zip(FEATURE_NAMES, permutation.importances_mean, strict=True):
        rows.append(
            {
                "model": "hist_gradient_boosting",
                "feature": name,
                "importance": value,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["model", "importance"], ascending=[True, False]
    )


def write_report(
    path: Path,
    config_path: Path,
    subset: pd.DataFrame,
    blocker_metrics: pd.DataFrame,
    selected_budget: CandidateBudget,
    training_counts: dict[str, int],
    model_metrics: pd.DataFrame,
    selected_model: str,
    error_table: pd.DataFrame,
    error_categories: Counter[str],
    runtime: pd.DataFrame,
    artifact_sizes: dict[str, int],
    importance: pd.DataFrame,
) -> None:
    blocker = blocker_metrics.query(
        'configuration == @selected_budget.name and dimension == "overall"'
    ).iloc[0]
    selected = model_metrics.query("model == @selected_model").iloc[0]
    exp001 = pd.read_csv(
        REPOSITORY_ROOT / "results" / "tables" / "exp001_matcher_metrics.csv"
    ).iloc[0]
    country_audit = pd.read_csv(
        REPOSITORY_ROOT / "results" / "tables" / "exp002_country_link_audit.csv"
    ).query('source == "ALL"').iloc[0]
    top_features = importance.query("model == @selected_model").head(10)
    lines = [
        "# EXP002 — multi-pass blocker and learned matcher",
        "",
        "- Branch: `feat/exp002-multipass-learned-matcher`",
        "- Commit: containing commit; resolve with `git rev-parse HEAD`",
        f"- Config: `{config_path.relative_to(REPOSITORY_ROOT)}`",
        f"- S1 partitions: fit {(subset.partition == 'fit').sum():,}, tune "
        f"{(subset.partition == 'tune').sum():,}, evaluation "
        f"{(subset.partition == 'evaluation').sum():,}",
        "- Evaluation IDs are identical to EXP001.",
        "",
        "## Country constraint",
        "",
        f"All {int(country_audit.true_link_count):,} training links were checked; "
        f"cross-country links: {int(country_audit.cross_country_link_count):,} "
        f"({country_audit.cross_country_link_rate:.6%}). Country remains an exact "
        "string-equality blocking constraint with no closed country enum.",
        "",
        "## Blocker comparison",
        "",
        "| Configuration | Recall | Matched all-links | Mean candidates | p99 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in blocker_metrics.query('dimension == "overall"').itertuples(index=False):
        lines.append(
            f"| {row.configuration} | {row.positive_link_recall:.3%} | "
            f"{row.matched_all_true_links_retained_rate:.3%} | "
            f"{row.average_candidates:.2f} | {row.p99_candidates:.0f} |"
        )
    lines.extend(
        [
            "",
            f"Selected blocker: **{selected_budget.name}**, based first on recall.",
            "",
            "## Training pairs",
            "",
            f"- Positives: {training_counts['positive_count']:,}",
            f"- Hard negatives: {training_counts['hard_negative_count']:,}",
            f"- Easy negatives: {training_counts['easy_negative_count']:,}",
            f"- Total: {training_counts['total_count']:,}",
            "",
            "## Model comparison on held-out evaluation S1",
            "",
            "| Model | Threshold | Macro F0.5 | Macro precision | Macro recall | Singleton accuracy |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in model_metrics.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.threshold:g} | {row.macro_fbeta:.6f} | "
            f"{row.macro_precision:.6f} | {row.macro_recall:.6f} | "
            f"{row.singleton_accuracy:.3%} |"
        )
    overall_errors = error_table.query('dimension == "overall"').iloc[0]
    lines.extend(
        [
            "",
            f"Selected model: **{selected_model}**. Global threshold: "
            f"{selected.threshold:g}.",
            "",
            "## Selected-model error analysis",
            "",
            f"- False-positive links: {int(selected.false_positives):,}",
            f"- False-negative links: {int(selected.false_negatives):,}",
            f"- Blocking FN links: "
            f"{int(overall_errors.blocking_false_negative_links):,}",
            f"- Matcher FN links: "
            f"{int(overall_errors.matcher_false_negative_links):,}",
            f"- FP with conflicting numeric evidence: "
            f"{error_categories['fp_conflicting_numeric_address']:,}",
            f"- FP with missing target address: "
            f"{error_categories['fp_missing_target_address']:,}",
            f"- FN with severe name corruption: "
            f"{error_categories['fn_severe_name_corruption']:,}",
            f"- FN with severe address corruption: "
            f"{error_categories['fn_severe_address_corruption']:,}",
            f"- FN just below threshold: "
            f"{error_categories['fn_just_below_threshold']:,}",
            "",
            "Top learned features:",
            "",
        ]
    )
    lines.extend(
        f"- {row.feature}: {row.importance:.6f}"
        for row in top_features.itertuples(index=False)
    )
    total_runtime = runtime.seconds.sum()
    peak_memory = runtime.peak_rss_mb.max()
    test_scale = 1_732_544 / len(subset)
    estimated_hours = total_runtime * test_scale / 3600
    lines.extend(
        [
            "",
            "## Runtime, memory, and scale",
            "",
            f"- Total measured EXP002 runtime: {total_runtime:.1f}s",
            f"- Peak RSS: {peak_memory:.1f} MB",
            f"- Retrieval artifact: {artifact_sizes['retrieval'] / 1024**2:.1f} MiB",
            f"- Feature artifact: {artifact_sizes['features'] / 1024**2:.1f} MiB",
            f"- Naive linear extrapolation to 1.73M S1: ~{estimated_hours:.1f} hours",
            "- Full test inference is not feasible with this in-memory query-side "
            "implementation on the 8 GB laptop: query/candidate memory scales with "
            "S1 count, while batching would require many full target rescans. A "
            "persistent disk index or stronger/cloud compute is required for the "
            "candidate-generation stage; compact model inference itself is not the "
            "primary bottleneck.",
            "",
            "## EXP001 comparison and verdict",
            "",
            f"- EXP001 macro F0.5: {exp001.macro_fbeta:.6f}",
            f"- EXP002 macro F0.5: {selected.macro_fbeta:.6f}",
            f"- Absolute change: {selected.macro_fbeta - exp001.macro_fbeta:+.6f}",
            "",
            "KEEP if EXP002 materially improves both blocking recall and held-out "
            "macro F0.5; otherwise REVISE. The finalized verdict below is based only "
            "on measured results.",
            "",
        ]
    )
    verdict = (
        "KEEP"
        if blocker.positive_link_recall >= 0.96
        and selected.macro_fbeta > exp001.macro_fbeta
        else "REVISE"
    )
    lines.append(f"**{verdict}**")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "exp002_m2_8gb.json",
    )
    parser.add_argument(
        "--data-root", type=Path, default=REPOSITORY_ROOT / "data" / "raw"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=REPOSITORY_ROOT / "results"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.absolute()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_root = args.data_root.absolute()
    results_dir = args.results_dir.absolute()
    tables_dir = results_dir / "tables"
    artifacts_dir = results_dir / "artifacts"
    tables_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    runtime_rows: list[dict[str, object]] = []

    started = time.monotonic()
    exp001_subset = REPOSITORY_ROOT / str(config["subset_ids_path"])
    subset_ids = frozenset(pd.read_csv(exp001_subset)["source1_entity_id"])
    records = load_selected_source_records(
        data_root / "train" / "train_source1.tsv", "S1", subset_ids
    )
    truth = load_selected_ground_truth(
        data_root / "train" / "train_ground_truth.tsv", subset_ids
    )
    subset, partition = prepare_partitions(
        exp001_subset,
        records,
        truth,
        tune_fraction=float(config["tune_fraction_of_development"]),
        seed=int(config["fit_tune_seed"]),
    )
    subset.to_csv(tables_dir / "exp002_subset_ids.csv", index=False)
    runtime_rows.append(
        {
            "stage": "load_and_partition",
            "seconds": time.monotonic() - started,
            "peak_rss_mb": peak_rss_mb(),
            "row_count": len(subset),
        }
    )
    print(subset["partition"].value_counts().to_dict(), flush=True)

    blocker_config = MultiPassConfig(
        max_pool_per_source=int(config["max_pool_per_source"]),
        min_name_token_length=int(config["min_name_token_length"]),
        min_address_token_length=int(config["min_address_token_length"]),
        max_name_query_df=int(config["max_name_query_df"]),
        max_address_query_df=int(config["max_address_query_df"]),
        max_numeric_query_df=int(config["max_numeric_query_df"]),
        max_char_query_df=int(config["max_char_query_df"]),
        min_char_shared_ngrams=int(config["min_char_shared_ngrams"]),
        min_char_jaccard=float(config["min_char_jaccard"]),
    )
    retrieval_path = artifacts_dir / "exp002_retrieval_pool.tsv.gz"
    country_counts, retrieval_runtime = scan_retrieval_pool(
        records,
        data_root,
        retrieval_path,
        blocker_config,
        chunksize=int(config["chunk_size"]),
    )
    runtime_rows.extend(retrieval_runtime)

    started = time.monotonic()
    budgets = load_budgets(config)
    blocker_metrics, selected_budget = evaluate_blocker_budgets(
        retrieval_path, budgets, truth, records, country_counts
    )
    blocker_metrics.to_csv(
        tables_dir / "exp002_blocker_config_metrics.csv", index=False
    )
    runtime_rows.append(
        {
            "stage": "blocker_evaluation",
            "seconds": time.monotonic() - started,
            "peak_rss_mb": peak_rss_mb(),
            "row_count": len(blocker_metrics),
        }
    )
    print(
        blocker_metrics.query('dimension == "overall"')[
            ["configuration", "positive_link_recall", "average_candidates"]
        ].to_string(index=False),
        flush=True,
    )
    print(f"Selected blocker: {selected_budget.name}", flush=True)

    started = time.monotonic()
    sampling_values = config["negative_sampling"]
    sampler = TrainingPairSampler(NegativeSamplingConfig(**sampling_values))
    feature_path = artifacts_dir / "exp002_candidate_features.tsv.gz"
    retained_evaluation, feature_count = build_feature_artifact(
        retrieval_path,
        feature_path,
        selected_budget,
        records,
        truth,
        partition,
        sampler,
    )
    examples, training_counts = sampler.finalize()
    training_counts["negative_to_positive_ratio"] = (
        (training_counts["hard_negative_count"] + training_counts["easy_negative_count"])
        / training_counts["positive_count"]
    )
    pd.DataFrame([training_counts]).to_csv(
        tables_dir / "exp002_training_pair_counts.csv", index=False
    )
    train_x, train_y = examples_to_arrays(examples)
    runtime_rows.append(
        {
            "stage": "features_and_pair_sampling",
            "seconds": time.monotonic() - started,
            "peak_rss_mb": peak_rss_mb(),
            "row_count": feature_count,
        }
    )
    del examples, sampler

    started = time.monotonic()
    models: dict[str, object] = {
        "rule": None,
        "logistic": make_logistic_model(int(config["model_seed"])),
        "hist_gradient_boosting": make_hist_gradient_boosting_model(
            int(config["model_seed"])
        ),
    }
    for name in ("logistic", "hist_gradient_boosting"):
        models[name].fit(train_x, train_y)
    runtime_rows.append(
        {
            "stage": "model_fit",
            "seconds": time.monotonic() - started,
            "peak_rss_mb": peak_rss_mb(),
            "row_count": len(train_y),
        }
    )

    started = time.monotonic()
    tune_ids = frozenset(
        subset.loc[subset["partition"].eq("tune"), "source1_entity_id"]
    )
    score_maps, importance_sample = score_tuning_pairs(
        feature_path,
        models,
        truth,
        tune_ids,
        importance_sample_size=20_000,
        seed=int(config["model_seed"]),
    )
    threshold_table, thresholds, selected_model = tune_models(
        truth, tune_ids, score_maps, config
    )
    threshold_table.to_csv(
        tables_dir / "exp002_threshold_tuning.csv", index=False
    )
    del score_maps
    evaluation_ids = frozenset(
        subset.loc[subset["partition"].eq("evaluation"), "source1_entity_id"]
    )
    model_metrics, selected_predictions, error_categories = evaluate_models(
        feature_path,
        models,
        thresholds,
        truth,
        evaluation_ids,
        selected_model,
    )
    model_metrics.to_csv(tables_dir / "exp002_model_comparison.csv", index=False)
    errors = error_slices(
        truth,
        selected_predictions,
        retained_evaluation,
        records,
        evaluation_ids,
    )
    errors.to_csv(tables_dir / "exp002_error_slices.csv", index=False)
    pd.DataFrame(
        [
            {"category": category, "count": count}
            for category, count in sorted(error_categories.items())
        ]
    ).to_csv(tables_dir / "exp002_error_categories.csv", index=False)
    runtime_rows.append(
        {
            "stage": "threshold_and_evaluation",
            "seconds": time.monotonic() - started,
            "peak_rss_mb": peak_rss_mb(),
            "row_count": len(evaluation_ids),
        }
    )

    started = time.monotonic()
    importance = feature_importance_tables(
        models,
        importance_sample[0],
        importance_sample[1],
        seed=int(config["model_seed"]),
    )
    importance.to_csv(tables_dir / "exp002_feature_importance.csv", index=False)
    licenses = pd.DataFrame(
        [
            {
                "package": "scikit-learn",
                "version": importlib.metadata.version("scikit-learn"),
                "license": importlib.metadata.metadata("scikit-learn").get(
                    "License-Expression", "BSD-3-Clause"
                ),
            }
        ]
    )
    licenses.to_csv(tables_dir / "exp002_model_licenses.csv", index=False)
    runtime_rows.append(
        {
            "stage": "feature_importance",
            "seconds": time.monotonic() - started,
            "peak_rss_mb": peak_rss_mb(),
            "row_count": len(importance_sample[1]),
        }
    )

    runtime = pd.DataFrame(runtime_rows)
    runtime.to_csv(tables_dir / "exp002_runtime.csv", index=False)
    artifact_sizes = {
        "retrieval": retrieval_path.stat().st_size,
        "features": feature_path.stat().st_size,
    }
    pd.DataFrame(
        [
            {
                "artifact": name,
                "size_bytes": size,
                "size_mib": size / 1024**2,
            }
            for name, size in artifact_sizes.items()
        ]
    ).to_csv(tables_dir / "exp002_artifact_sizes.csv", index=False)
    write_report(
        REPOSITORY_ROOT / "experiments" / "EXP002.md",
        config_path,
        subset,
        blocker_metrics,
        selected_budget,
        training_counts,
        model_metrics,
        selected_model,
        errors,
        error_categories,
        runtime,
        artifact_sizes,
        importance,
    )
    selected_metrics = model_metrics.query("model == @selected_model").iloc[0]
    print(
        f"EXP002 complete: blocker={selected_budget.name}; model={selected_model}; "
        f"macro F0.5={selected_metrics.macro_fbeta:.6f}; "
        f"peak RSS={runtime.peak_rss_mb.max():.1f} MB.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

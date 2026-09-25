#!/usr/bin/env python3
"""Run the EXP001 lexical blocker and rule-based matcher on train data only."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import platform
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from business_entity_resolution.blocking import (  # noqa: E402
    LexicalBlocker,
    LexicalBlockerConfig,
    ScoredCandidate,
)
from business_entity_resolution.blocking_metrics import candidate_metrics  # noqa: E402
from business_entity_resolution.data import iter_source_chunks  # noqa: E402
from business_entity_resolution.evaluation import evaluate_predictions  # noqa: E402
from business_entity_resolution.sampling import (  # noqa: E402
    SourceRecord,
    load_selected_ground_truth,
    load_selected_source_records,
    select_development_entity_ids,
)
from business_entity_resolution.split import stratified_entity_split  # noqa: E402


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return value / (1024 * 1024)
    return value / 1024


class CandidateArtifactWriter:
    """Write deterministic gzip TSV without materializing a giant DataFrame."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.raw = path.open("wb")
        self.gzip_file = gzip.GzipFile(
            filename="", fileobj=self.raw, mode="wb", mtime=0
        )
        self.text = io.TextIOWrapper(self.gzip_file, encoding="utf-8", newline="")
        self.writer: csv.DictWriter[str] | None = None

    def write(self, candidates: list[ScoredCandidate]) -> None:
        for candidate in candidates:
            row = candidate.to_dict()
            if self.writer is None:
                self.writer = csv.DictWriter(
                    self.text, fieldnames=list(row), delimiter="\t", lineterminator="\n"
                )
                self.writer.writeheader()
            self.writer.writerow(row)

    def close(self) -> None:
        self.text.flush()
        self.text.detach()
        self.gzip_file.close()
        self.raw.close()

    def __enter__(self) -> CandidateArtifactWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def subset_table(
    records: dict[str, SourceRecord],
    truth: dict[str, frozenset[str]],
    tuning_ids: frozenset[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source1_entity_id": entity_id,
                "country": records[entity_id].country,
                "match_count": len(truth[entity_id]),
                "is_singleton": not truth[entity_id],
                "partition": "tuning" if entity_id in tuning_ids else "evaluation",
            }
            for entity_id in sorted(records)
        ]
    )


def make_candidate_mapping(
    candidate_scores: dict[str, dict[str, float]],
    entity_ids: set[str] | frozenset[str],
    *,
    source: str | None = None,
) -> dict[str, set[str]]:
    prefix = f"{source}-" if source else None
    return {
        entity_id: {
            candidate_id
            for candidate_id in candidate_scores.get(entity_id, {})
            if prefix is None or candidate_id.startswith(prefix)
        }
        for entity_id in entity_ids
    }


def filter_truth(
    truth: dict[str, frozenset[str]],
    entity_ids: set[str] | frozenset[str],
    *,
    source: str | None = None,
) -> dict[str, frozenset[str]]:
    prefix = f"{source}-" if source else None
    return {
        entity_id: frozenset(
            target_id
            for target_id in truth[entity_id]
            if prefix is None or target_id.startswith(prefix)
        )
        for entity_id in entity_ids
    }


def evaluate_blocking(
    truth: dict[str, frozenset[str]],
    records: dict[str, SourceRecord],
    candidate_scores: dict[str, dict[str, float]],
    target_country_counts: dict[str, Counter[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    entity_ids = frozenset(truth)
    countries = sorted({record.country for record in records.values()})
    country_by_entity = {
        entity_id: record.country for entity_id, record in records.items()
    }
    rows: list[dict[str, object]] = []

    group_specs: list[tuple[str, str, str | None, frozenset[str]]] = [
        ("overall", "ALL", None, entity_ids)
    ]
    group_specs.extend(
        (
            "country",
            country,
            None,
            frozenset(
                entity_id
                for entity_id, record in records.items()
                if record.country == country
            ),
        )
        for country in countries
    )
    for source in ("S2", "S3"):
        group_specs.append(("source", source, source, entity_ids))
        group_specs.extend(
            (
                "source_country",
                f"{source}|{country}",
                source,
                frozenset(
                    entity_id
                    for entity_id, record in records.items()
                    if record.country == country
                ),
            )
            for country in countries
        )

    for dimension, group, source, ids in group_specs:
        group_truth = filter_truth(truth, ids, source=source)
        group_candidates = make_candidate_mapping(
            candidate_scores, ids, source=source
        )
        eligible = (
            dict(target_country_counts[source])
            if source
            else {
                country: sum(counts[country] for counts in target_country_counts.values())
                for country in countries
            }
        )
        metrics = candidate_metrics(
            group_truth,
            group_candidates,
            eligible_target_count_by_country=eligible,
            country_by_entity=country_by_entity,
        )
        rows.append({"dimension": dimension, "group": group, **metrics.to_dict()})

    count_rows = []
    for entity_id in sorted(entity_ids):
        all_ids = candidate_scores.get(entity_id, {})
        count_rows.append(
            {
                "source1_entity_id": entity_id,
                "country": records[entity_id].country,
                "candidate_count": len(all_ids),
                "s2_candidate_count": sum(key.startswith("S2-") for key in all_ids),
                "s3_candidate_count": sum(key.startswith("S3-") for key in all_ids),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(count_rows)


def predictions_at_threshold(
    candidate_scores: dict[str, dict[str, float]],
    entity_ids: frozenset[str],
    threshold: float,
) -> dict[str, set[str]]:
    return {
        entity_id: {
            target_id
            for target_id, score in candidate_scores.get(entity_id, {}).items()
            if score >= threshold
        }
        for entity_id in entity_ids
    }


def tune_threshold(
    truth: dict[str, frozenset[str]],
    candidate_scores: dict[str, dict[str, float]],
    tuning_ids: frozenset[str],
    thresholds: list[float],
) -> tuple[float, pd.DataFrame]:
    tuning_truth = {entity_id: truth[entity_id] for entity_id in tuning_ids}
    rows = []
    for threshold in thresholds:
        predictions = predictions_at_threshold(
            candidate_scores, tuning_ids, threshold
        )
        result = evaluate_predictions(tuning_truth, predictions)
        predicted_count = sum(len(values) for values in predictions.values())
        rows.append(
            {
                "threshold": threshold,
                **result.to_dict(),
                "average_predicted_matches": predicted_count / len(predictions),
            }
        )
    table = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    best = max(rows, key=lambda row: (row["macro_fbeta"], row["threshold"]))
    return float(best["threshold"]), table


def evaluate_matcher(
    truth: dict[str, frozenset[str]],
    candidate_scores: dict[str, dict[str, float]],
    evaluation_ids: frozenset[str],
    threshold: float,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    evaluation_truth = {entity_id: truth[entity_id] for entity_id in evaluation_ids}
    predictions = predictions_at_threshold(
        candidate_scores, evaluation_ids, threshold
    )
    result = evaluate_predictions(evaluation_truth, predictions)
    average_predicted = sum(len(values) for values in predictions.values()) / len(
        predictions
    )
    singleton_total = sum(not values for values in evaluation_truth.values())
    singleton_accuracy = (
        result.correct_singletons / singleton_total if singleton_total else 1.0
    )

    error_counts: Counter[str] = Counter()
    prediction_rows = []
    for entity_id in sorted(evaluation_ids):
        expected = set(evaluation_truth[entity_id])
        predicted = predictions[entity_id]
        false_positives = predicted - expected
        false_negatives = expected - predicted
        candidate_pool = set(candidate_scores.get(entity_id, {}))
        blocking_misses = false_negatives - candidate_pool
        below_threshold = false_negatives & candidate_pool
        error_counts["false_positive_links"] += len(false_positives)
        error_counts["false_negative_links"] += len(false_negatives)
        error_counts["false_negative_blocking_miss_links"] += len(blocking_misses)
        error_counts["false_negative_below_threshold_links"] += len(below_threshold)
        error_counts["false_positive_singleton_links"] += (
            len(false_positives) if not expected else 0
        )
        error_counts["false_positive_matched_entity_links"] += (
            len(false_positives) if expected else 0
        )
        error_counts["false_negative_s2_links"] += sum(
            value.startswith("S2-") for value in false_negatives
        )
        error_counts["false_negative_s3_links"] += sum(
            value.startswith("S3-") for value in false_negatives
        )
        error_counts["entities_with_false_positives"] += int(bool(false_positives))
        error_counts["entities_with_false_negatives"] += int(bool(false_negatives))
        prediction_rows.append(
            {
                "source1_entity_id": entity_id,
                "matched_entity_ids": ",".join(sorted(predicted)),
            }
        )

    summary: dict[str, object] = {
        "threshold": threshold,
        **result.to_dict(),
        "singleton_accuracy": singleton_accuracy,
        "average_predicted_matches": average_predicted,
    }
    errors = pd.DataFrame(
        [
            {"error_category": category, "count": count}
            for category, count in sorted(error_counts.items())
        ]
    )
    return summary, errors, pd.DataFrame(prediction_rows)


def write_experiment_report(
    path: Path,
    config_path: Path,
    config: dict[str, object],
    subset: pd.DataFrame,
    blocking: pd.DataFrame,
    matcher: dict[str, object],
    errors: pd.DataFrame,
    runtime: pd.DataFrame,
) -> None:
    overall = blocking.query('dimension == "overall"').iloc[0]
    india = blocking.query('dimension == "country" and group == "India"').iloc[0]
    us = blocking.query('dimension == "country" and group == "US"').iloc[0]
    s2 = blocking.query('dimension == "source" and group == "S2"').iloc[0]
    s3 = blocking.query('dimension == "source" and group == "S3"').iloc[0]
    tuning_count = int((subset["partition"] == "tuning").sum())
    evaluation_count = int((subset["partition"] == "evaluation").sum())
    error_counts = errors.set_index("error_category")["count"].to_dict()
    profile_tables = path.parents[1] / "results" / "tables"
    profile_rates = pd.read_csv(profile_tables / "positive_pair_profile_rates.csv")
    profile_run = pd.read_csv(profile_tables / "positive_pair_profile_run.csv").iloc[0]
    profile_overall = profile_rates.query('dimension == "overall"').iloc[0]
    profile_india = profile_rates.query(
        'dimension == "country" and group == "India"'
    ).iloc[0]
    profile_us = profile_rates.query(
        'dimension == "country" and group == "US"'
    ).iloc[0]
    lines = [
        "# EXP001 — lexical entity-resolution baseline",
        "",
        "- Branch: `feat/exp001-lexical-baseline`",
        "- Commit: containing commit; resolve with `git rev-parse HEAD`",
        f"- Config: `{config_path.relative_to(REPOSITORY_ROOT)}`",
        f"- Development subset: {len(subset):,} S1 entities, seed "
        f"{config['subset_seed']}, stratified by country × exact match count",
        f"- Threshold tuning entities: {tuning_count:,}",
        f"- Held-out evaluation entities: {evaluation_count:,}",
        "- Candidate generation: country-compatible normalized exact name/address "
        "plus capped rare name/address token indexes, independently for S2 and S3",
        f"- Top-K: {config['top_k_per_source']} per source",
        "- Matcher: deterministic lexical rule score; no fitted model",
        f"- Chosen threshold: {matcher['threshold']}",
        "",
        "## Positive-pair profile",
        "",
        f"- Sample: {int(profile_run.sample_size):,} true links; seed "
        f"{int(profile_run.seed)}",
        f"- Raw → normalized exact name: {profile_overall.raw_name_exact_rate:.3%} "
        f"→ {profile_overall.normalized_name_exact_rate:.3%}",
        f"- Raw → normalized exact address: "
        f"{profile_overall.raw_address_exact_rate:.3%} → "
        f"{profile_overall.normalized_address_exact_rate:.3%}",
        f"- Normalized exact name by country: India "
        f"{profile_india.normalized_name_exact_rate:.3%}, US "
        f"{profile_us.normalized_name_exact_rate:.3%}",
        f"- Target address missing: {profile_overall.right_address_missing_rate:.3%}",
        f"- Profile runtime / peak RSS: {profile_run.runtime_seconds:.1f}s / "
        f"{profile_run.peak_rss_mb:.1f} MB",
        "",
        "## Blocking",
        "",
        f"- Positive-link recall: {overall.positive_link_recall:.3%}",
        f"- Recall by country: India {india.positive_link_recall:.3%}, "
        f"US {us.positive_link_recall:.3%}",
        f"- Recall by source: S2 {s2.positive_link_recall:.3%}, "
        f"S3 {s3.positive_link_recall:.3%}",
        f"- S1 with all true links retained: {overall.all_true_links_retained_rate:.3%}",
        f"- Matched S1 with all true links retained: "
        f"{overall.matched_entity_all_true_links_retained_rate:.3%}",
        f"- Candidates/S1: mean {overall.average_candidates:.3f}, median "
        f"{overall.median_candidates:.1f}, p95 {overall.p95_candidates:.1f}, "
        f"p99 {overall.p99_candidates:.1f}, max {int(overall.max_candidates)}",
        f"- Zero-candidate S1: {int(overall.zero_candidate_entities):,}",
        f"- Reduction ratio: {overall.reduction_ratio:.8%}",
        "",
        "## Held-out matcher evaluation",
        "",
        f"- Macro F0.5: {matcher['macro_fbeta']:.6f}",
        f"- Macro precision: {matcher['macro_precision']:.6f}",
        f"- Macro recall: {matcher['macro_recall']:.6f}",
        f"- Singleton accuracy: {matcher['singleton_accuracy']:.3%}",
        f"- Average predicted matches/S1: {matcher['average_predicted_matches']:.3f}",
        f"- False-positive links: {int(matcher['false_positives']):,}",
        f"- False-negative links: {int(matcher['false_negatives']):,}",
        f"- False negatives absent from candidate set: "
        f"{int(error_counts['false_negative_blocking_miss_links']):,}",
        f"- False negatives retained by blocker but below threshold: "
        f"{int(error_counts['false_negative_below_threshold_links']):,}",
        f"- False-positive links on true singletons: "
        f"{int(error_counts['false_positive_singleton_links']):,}",
        "",
        "## Runtime and resources",
        "",
        f"- Total runtime: {runtime['seconds'].sum():.1f} seconds",
        f"- Peak RSS: {runtime['peak_rss_mb'].max():.1f} MB",
        "",
        "## Decision",
        "",
        f"REVISE — {overall.positive_link_recall:.1%} candidate recall and "
        "near-universal top-K saturation leave too low a ceiling, while the "
        "precision-oriented threshold still misses many retained positives. Improve "
        "candidate ranking/selectivity and the rule score before training a learned "
        "matcher.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / "configs" / "exp001.json"
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
    timings: list[dict[str, object]] = []

    stage_started = time.monotonic()
    ground_truth_path = data_root / "train" / "train_ground_truth.tsv"
    source1_path = data_root / "train" / "train_source1.tsv"
    selected_ids = select_development_entity_ids(
        source1_path,
        ground_truth_path,
        sample_size=int(config["subset_size"]),
        seed=int(config["subset_seed"]),
        chunksize=int(config["chunk_size"]),
    )
    records = load_selected_source_records(
        source1_path,
        "S1",
        selected_ids,
        chunksize=int(config["chunk_size"]),
    )
    truth = load_selected_ground_truth(ground_truth_path, selected_ids)
    split = stratified_entity_split(
        [
            {"entity_id": record.entity_id, "country": record.country}
            for record in records.values()
        ],
        truth,
        validation_fraction=float(config["evaluation_fraction"]),
        seed=int(config["partition_seed"]),
    )
    subset = subset_table(records, truth, split.train_ids)
    subset.to_csv(tables_dir / "exp001_subset_ids.csv", index=False)
    timings.append(
        {
            "stage": "subset",
            "seconds": time.monotonic() - stage_started,
            "peak_rss_mb": peak_rss_mb(),
        }
    )
    print(
        f"Selected {len(selected_ids):,} entities: {len(split.train_ids):,} tuning, "
        f"{len(split.validation_ids):,} evaluation.",
        flush=True,
    )

    blocker_config = LexicalBlockerConfig(
        top_k_per_source=int(config["top_k_per_source"]),
        min_informative_token_length=int(config["min_informative_token_length"]),
        max_query_token_frequency=int(config["max_query_token_frequency"]),
    )
    candidate_scores: dict[str, dict[str, float]] = defaultdict(dict)
    target_country_counts: dict[str, Counter[str]] = {}
    artifact_path = artifacts_dir / "exp001_candidates.tsv.gz"
    with CandidateArtifactWriter(artifact_path) as artifact:
        for source in ("S2", "S3"):
            stage_started = time.monotonic()
            blocker = LexicalBlocker(
                records.values(), source=source, config=blocker_config
            )
            target_path = data_root / "train" / f"train_source{source[-1]}.tsv"
            print(f"Scanning {target_path} for {source} candidates ...", flush=True)
            for chunk in iter_source_chunks(
                target_path, source, chunksize=int(config["chunk_size"])
            ):
                blocker.process_chunk(chunk)
            candidates = blocker.finalize()
            artifact.write(candidates)
            for candidate in candidates:
                evidence = candidate.evidence
                candidate_scores[evidence.source1_entity_id][
                    evidence.candidate_entity_id
                ] = candidate.match_score
            target_country_counts[source] = blocker.target_country_counts
            timings.append(
                {
                    "stage": f"block_{source.lower()}",
                    "seconds": time.monotonic() - stage_started,
                    "peak_rss_mb": peak_rss_mb(),
                }
            )
            print(
                f"{source}: retained {len(candidates):,} candidates; "
                f"peak RSS {peak_rss_mb():.1f} MB.",
                flush=True,
            )
            del candidates, blocker

    stage_started = time.monotonic()
    blocking_table, candidate_counts = evaluate_blocking(
        truth, records, candidate_scores, target_country_counts
    )
    blocking_table.to_csv(tables_dir / "exp001_blocking_metrics.csv", index=False)
    candidate_counts.to_csv(
        tables_dir / "exp001_candidate_counts_by_entity.csv", index=False
    )
    chosen_threshold, threshold_table = tune_threshold(
        truth,
        candidate_scores,
        split.train_ids,
        [float(value) for value in config["thresholds"]],
    )
    threshold_table.to_csv(
        tables_dir / "exp001_threshold_tuning.csv", index=False
    )
    matcher, error_table, predictions = evaluate_matcher(
        truth,
        candidate_scores,
        split.validation_ids,
        chosen_threshold,
    )
    pd.DataFrame([matcher]).to_csv(
        tables_dir / "exp001_matcher_metrics.csv", index=False
    )
    error_table.to_csv(tables_dir / "exp001_error_summary.csv", index=False)
    predictions.to_csv(
        artifacts_dir / "exp001_validation_predictions.tsv.gz",
        sep="\t",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    timings.append(
        {
            "stage": "evaluation",
            "seconds": time.monotonic() - stage_started,
            "peak_rss_mb": peak_rss_mb(),
        }
    )
    runtime = pd.DataFrame(timings)
    runtime.to_csv(tables_dir / "exp001_runtime.csv", index=False)
    write_experiment_report(
        REPOSITORY_ROOT / "experiments" / "EXP001.md",
        config_path,
        config,
        subset,
        blocking_table,
        matcher,
        error_table,
        runtime,
    )
    print(
        f"EXP001 complete. Threshold={chosen_threshold:.1f}; "
        f"macro F0.5={matcher['macro_fbeta']:.6f}; peak RSS={peak_rss_mb():.1f} MB.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build/reuse EXP002 source indexes and benchmark indexed lexical blocking only.

This script deliberately stops before pair-feature construction or model fitting.
It never scans S2/S3 while answering Source-1 queries: target scans are limited
to the explicit, one-time ``--build-index`` step.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from business_entity_resolution.indexed_blocking import (  # noqa: E402
    SourceIndexConfig,
    SourceSideIndex,
    build_or_open_source_index,
)
from business_entity_resolution.multipass import (  # noqa: E402
    CandidateBudget,
    select_candidates_for_budget,
)
from business_entity_resolution.sampling import (  # noqa: E402
    SourceRecord,
    load_selected_ground_truth,
    load_selected_source_records,
)
from business_entity_resolution.split import stable_entity_key  # noqa: E402


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024) if platform.system() == "Darwin" else value / 1024


def load_budget(config: dict[str, object], name: str) -> CandidateBudget:
    for values in config["blocker_budgets"]:
        if values["name"] == name:
            return CandidateBudget(**values)
    available = [values["name"] for values in config["blocker_budgets"]]
    raise ValueError(f"Unknown budget {name!r}; available={available!r}")


def benchmark_ids(records: dict[str, SourceRecord], size: int, seed: int) -> list[str]:
    if not 0 < size <= len(records):
        raise ValueError(f"sample size must be in [1, {len(records):,}]")
    return sorted(
        records,
        key=lambda entity_id: (stable_entity_key(entity_id, seed), entity_id),
    )[:size]


def summarize(
    records: dict[str, SourceRecord],
    truth: dict[str, frozenset[str]],
    selected: dict[str, set[str]],
    sample_ids: list[str],
    sample_name: str,
    query_seconds: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    countries = sorted({records[entity_id].country for entity_id in sample_ids})
    specs: list[tuple[str, str, str | None, list[str]]] = [
        ("overall", "ALL", None, sample_ids)
    ]
    specs.extend(
        ("country", country, None, [entity_id for entity_id in sample_ids if records[entity_id].country == country])
        for country in countries
    )
    for source in ("S2", "S3"):
        specs.append(("source", source, source, sample_ids))
        specs.extend(
            (
                "source_country",
                f"{source}|{country}",
                source,
                [entity_id for entity_id in sample_ids if records[entity_id].country == country],
            )
            for country in countries
        )
    for dimension, group, source, ids in specs:
        candidate_counts: list[int] = []
        true_total = retained_total = matched_total = matched_all = zero = 0
        for entity_id in ids:
            expected = {
                candidate_id
                for candidate_id in truth[entity_id]
                if source is None or candidate_id.startswith(f"{source}-")
            }
            candidates = {
                candidate_id
                for candidate_id in selected[entity_id]
                if source is None or candidate_id.startswith(f"{source}-")
            }
            retained = expected & candidates
            candidate_counts.append(len(candidates))
            true_total += len(expected)
            retained_total += len(retained)
            matched_total += int(bool(expected))
            matched_all += int(bool(expected) and retained == expected)
            zero += int(not candidates)
        values = np.asarray(candidate_counts, dtype=np.int64)
        rows.append(
            {
                "sample": sample_name,
                "dimension": dimension,
                "group": group,
                "entity_count": len(ids),
                "true_link_count": true_total,
                "retained_true_link_count": retained_total,
                "positive_link_recall": retained_total / true_total if true_total else 1.0,
                "matched_all_true_links_retained_rate": matched_all / matched_total if matched_total else 1.0,
                "average_candidates": float(values.mean()),
                "median_candidates": float(np.quantile(values, 0.5)),
                "p95_candidates": float(np.quantile(values, 0.95)),
                "p99_candidates": float(np.quantile(values, 0.99)),
                "max_candidates": int(values.max()),
                "zero_candidate_entities": zero,
                "query_seconds": query_seconds,
                "peak_rss_mb": peak_rss_mb(),
            }
        )
    return rows


def run_benchmark(
    records: dict[str, SourceRecord],
    truth: dict[str, frozenset[str]],
    indexes: dict[str, SourceSideIndex],
    budget: CandidateBudget,
    sample_ids: list[str],
    sample_name: str,
) -> list[dict[str, object]]:
    started = time.monotonic()
    selected: dict[str, set[str]] = defaultdict(set)
    for entity_id in sample_ids:
        record = records[entity_id]
        for source, index in indexes.items():
            candidates = select_candidates_for_budget(index.retrieve(record), budget)
            selected[entity_id].update(candidate.candidate_entity_id for candidate in candidates)
    return summarize(
        records,
        truth,
        selected,
        sample_ids,
        sample_name,
        time.monotonic() - started,
    )


def write_report(path: Path, build: pd.DataFrame, metrics: pd.DataFrame) -> None:
    lines = [
        "# EXP002 indexed-blocker benchmark",
        "",
        "The target-side SQLite FTS5 indexes are built once and then reused for all S1 query batches.",
        "The production path has no character n-gram fallback and no S2/S3 scan during querying.",
        "",
        "## Index build/reuse",
        "",
        "| Source | Reused | Rows | Seconds | Disk MiB | Peak RSS MiB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in build.itertuples(index=False):
        lines.append(
            f"| {row.source} | {bool(row.reused)} | {row.row_count:,} | {row.seconds:.1f} | "
            f"{row.size_mib:.1f} | {row.peak_rss_mb:.1f} |"
        )
    lines.extend(["", "## Candidate retrieval", "", "| Sample | Recall | Matched all-links | Mean candidates | p99 | Query seconds | Peak RSS MiB |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in metrics.query('dimension == "overall"').itertuples(index=False):
        lines.append(
            f"| {row.sample} | {row.positive_link_recall:.3%} | "
            f"{row.matched_all_true_links_retained_rate:.3%} | {row.average_candidates:.2f} | "
            f"{row.p99_candidates:.0f} | {row.query_seconds:.1f} | {row.peak_rss_mb:.1f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPOSITORY_ROOT / "data" / "raw")
    parser.add_argument("--index-dir", type=Path, default=REPOSITORY_ROOT / "results" / "artifacts" / "exp002_source_index")
    parser.add_argument("--results-dir", type=Path, default=REPOSITORY_ROOT / "results")
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs" / "exp002_m2_8gb.json")
    parser.add_argument("--build-index", action="store_true", help="Build missing source indexes once; otherwise require complete indexes.")
    parser.add_argument("--rebuild-index", action="store_true", help="Explicitly replace complete source index files.")
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[1000, 5000])
    parser.add_argument("--budget", default="recall")
    parser.add_argument("--sample-seed", type=int, default=2032)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    budget = load_budget(config, args.budget)
    data_root = args.data_root.absolute()
    index_dir = args.index_dir.absolute()
    results_dir = args.results_dir.absolute()
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    subset_path = REPOSITORY_ROOT / str(config["subset_ids_path"])
    subset_ids = frozenset(pd.read_csv(subset_path)["source1_entity_id"])
    records = load_selected_source_records(data_root / "train" / "train_source1.tsv", "S1", subset_ids)
    truth = load_selected_ground_truth(data_root / "train" / "train_ground_truth.tsv", subset_ids)

    build_rows: list[dict[str, object]] = []
    for source in ("S2", "S3"):
        source_path = data_root / "train" / f"train_source{source[-1]}.tsv"
        index_path = index_dir / f"exp002_{source.lower()}_fts.sqlite"
        if not index_path.exists() and not args.build_index:
            raise FileNotFoundError(
                f"Missing {index_path}. Run once with --build-index; query batches will then reuse it."
            )
        result = build_or_open_source_index(
            source_path,
            source,
            index_dir,
            rebuild=args.rebuild_index,
        )
        build_rows.append(
            {
                "source": result.source,
                "reused": result.reused,
                "row_count": result.row_count,
                "seconds": result.seconds,
                "size_bytes": result.size_bytes,
                "size_mib": result.size_bytes / 1024**2,
                "peak_rss_mb": peak_rss_mb(),
            }
        )
    build = pd.DataFrame(build_rows)
    build.to_csv(tables_dir / "exp002_index_build.csv", index=False)

    with SourceSideIndex(index_dir / "exp002_s2_fts.sqlite", "S2", SourceIndexConfig()) as s2, SourceSideIndex(index_dir / "exp002_s3_fts.sqlite", "S3", SourceIndexConfig()) as s3:
        indexes = {"S2": s2, "S3": s3}
        metric_rows: list[dict[str, object]] = []
        for size in args.sample_sizes:
            ids = benchmark_ids(records, size, args.sample_seed)
            print(f"Querying {size:,} S1 entities against reusable indexes ...", flush=True)
            metric_rows.extend(run_benchmark(records, truth, indexes, budget, ids, f"s1_{size}"))
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(tables_dir / "exp002_indexed_blocker_benchmark.csv", index=False)
    write_report(results_dir / "exp002_indexed_blocker_benchmark.md", build, metrics)
    print(metrics.query('dimension == "overall"')[["sample", "positive_link_recall", "average_candidates", "query_seconds", "peak_rss_mb"]].to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

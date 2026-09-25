#!/usr/bin/env python3
"""Profile a deterministic sample of true S1-to-S2/S3 links."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import platform
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from business_entity_resolution.data import (  # noqa: E402
    DEFAULT_CHUNK_SIZE,
    iter_source_chunks,
)
from business_entity_resolution.ground_truth import iter_ground_truth  # noqa: E402
from business_entity_resolution.sampling import (  # noqa: E402
    SourceRecord,
    load_selected_source_records,
)
from business_entity_resolution.similarity import pair_features  # noqa: E402

QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
SIMILARITY_COLUMNS = (
    "name_ratio",
    "name_token_set_ratio",
    "name_token_sort_ratio",
    "address_ratio",
    "address_token_set_ratio",
    "address_token_sort_ratio",
    "name_token_jaccard",
    "address_token_jaccard",
    "digit_token_overlap",
    "postal_like_agreement",
)
RATE_COLUMNS = (
    "raw_name_exact",
    "normalized_name_exact",
    "raw_address_exact",
    "normalized_address_exact",
    "right_address_missing",
)


def stable_link_hash(source1_id: str, target_id: str, seed: int) -> int:
    payload = f"{seed}\0{source1_id}\0{target_id}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def sample_positive_links(
    ground_truth_path: Path, *, sample_size: int, seed: int
) -> list[tuple[str, str]]:
    """Keep the globally smallest deterministic hashes in O(sample_size) memory."""

    heap: list[tuple[int, str, str]] = []
    link_count = 0
    for source1_id, matches in iter_ground_truth(ground_truth_path):
        for target_id in sorted(matches):
            link_count += 1
            item = (-stable_link_hash(source1_id, target_id, seed), source1_id, target_id)
            if len(heap) < sample_size:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    if sample_size > link_count:
        raise ValueError(
            f"sample_size {sample_size:,} exceeds {link_count:,} positive links"
        )
    return sorted((source1_id, target_id) for _, source1_id, target_id in heap)


def collect_profile_rows(
    links: list[tuple[str, str]],
    source1_records: dict[str, SourceRecord],
    data_root: Path,
    *,
    chunksize: int,
) -> list[dict[str, object]]:
    links_by_target: dict[str, list[str]] = defaultdict(list)
    for source1_id, target_id in links:
        links_by_target[target_id].append(source1_id)

    rows: list[dict[str, object]] = []
    for source in ("S2", "S3"):
        target_ids = {
            target_id for target_id in links_by_target if target_id.startswith(f"{source}-")
        }
        path = data_root / "train" / f"train_source{source[-1]}.tsv"
        print(f"Scanning {path} for {len(target_ids):,} sampled targets ...", flush=True)
        found: set[str] = set()
        for chunk in iter_source_chunks(path, source, chunksize=chunksize):
            subset = chunk.loc[chunk["entity_id"].isin(target_ids)]
            for target in subset.itertuples(index=False):
                found.add(target.entity_id)
                for source1_id in links_by_target[target.entity_id]:
                    source1 = source1_records[source1_id]
                    features = pair_features(
                        source1.business_name,
                        source1.business_address,
                        target.business_name,
                        target.business_address,
                    )
                    rows.append(
                        {
                            "source1_entity_id": source1_id,
                            "target_entity_id": target.entity_id,
                            "source": source,
                            "country": source1.country,
                            **features.to_dict(),
                        }
                    )
        missing = target_ids - found
        if missing:
            raise ValueError(
                f"{len(missing):,} sampled {source} target IDs were not found; "
                f"examples: {sorted(missing)[:5]}"
            )
    if len(rows) != len(links):
        raise AssertionError("profile row count differs from sampled-link count")
    return rows


def group_views(frame: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    groups: list[tuple[str, str, pd.DataFrame]] = [("overall", "ALL", frame)]
    groups.extend(
        ("source", str(key), group) for key, group in frame.groupby("source")
    )
    groups.extend(
        ("country", str(key), group) for key, group in frame.groupby("country")
    )
    groups.extend(
        ("source_country", f"{source}|{country}", group)
        for (source, country), group in frame.groupby(["source", "country"])
    )
    return groups


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rate_rows: list[dict[str, object]] = []
    quantile_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    for dimension, group_name, group in group_views(frame):
        rate_rows.append(
            {
                "dimension": dimension,
                "group": group_name,
                "pair_count": len(group),
                **{f"{column}_rate": float(group[column].mean()) for column in RATE_COLUMNS},
            }
        )
        for metric in SIMILARITY_COLUMNS:
            values = group[metric].astype(float)
            quantiles = values.quantile(QUANTILES)
            quantile_rows.append(
                {
                    "dimension": dimension,
                    "group": group_name,
                    "metric": metric,
                    **{f"p{int(q * 100):02d}": float(quantiles.loc[q]) for q in QUANTILES},
                }
            )

    for (source, country, missing), group in frame.groupby(
        ["source", "country", "right_address_missing"]
    ):
        missing_rows.append(
            {
                "source": source,
                "country": country,
                "target_address_missing": bool(missing),
                "pair_count": len(group),
                "name_ratio_p50": float(group["name_ratio"].median()),
                "name_token_set_ratio_p50": float(
                    group["name_token_set_ratio"].median()
                ),
                "address_ratio_p50": float(group["address_ratio"].median()),
                "normalized_name_exact_rate": float(
                    group["normalized_name_exact"].mean()
                ),
            }
        )
    return (
        pd.DataFrame(rate_rows),
        pd.DataFrame(quantile_rows),
        pd.DataFrame(missing_rows),
    )


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return value / (1024 * 1024)
    return value / 1024


def write_report(
    path: Path,
    rates: pd.DataFrame,
    quantiles: pd.DataFrame,
    *,
    sample_size: int,
    seed: int,
    runtime_seconds: float,
    peak_memory_mb: float,
) -> None:
    overall = rates.query('dimension == "overall"').iloc[0]
    overall_quantiles = quantiles.query('dimension == "overall"').set_index("metric")
    lines = [
        "# EXP001 positive-pair lexical profile",
        "",
        f"Deterministic sample: {sample_size:,} positive links; seed: {seed}.",
        "",
        "## Overall exact-match rates",
        "",
        f"- Raw name exact: {overall.raw_name_exact_rate:.3%}",
        f"- Normalized name exact: {overall.normalized_name_exact_rate:.3%}",
        f"- Raw address exact: {overall.raw_address_exact_rate:.3%}",
        f"- Normalized address exact: {overall.normalized_address_exact_rate:.3%}",
        f"- Target address missing: {overall.right_address_missing_rate:.3%}",
        "",
        "## Overall similarity landmarks",
        "",
        "| Metric | p05 | p50 | p95 |",
        "|---|---:|---:|---:|",
    ]
    for metric in SIMILARITY_COLUMNS:
        row = overall_quantiles.loc[metric]
        lines.append(
            f"| {metric} | {row.p05:.3f} | {row.p50:.3f} | {row.p95:.3f} |"
        )
    lines.extend(
        [
            "",
            "Detailed source, country, and source×country results are in "
            "`results/tables/positive_pair_profile_rates.csv` and "
            "`positive_pair_profile_quantiles.csv`.",
            "",
            f"Runtime: {runtime_seconds:.1f} seconds. Peak RSS: {peak_memory_mb:.1f} MB.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--data-root", type=Path, default=REPOSITORY_ROOT / "data" / "raw"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=REPOSITORY_ROOT / "results"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    data_root = args.data_root.absolute()
    results_dir = args.results_dir.absolute()
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_path = data_root / "train" / "train_ground_truth.tsv"

    print("Selecting deterministic positive-link sample ...", flush=True)
    links = sample_positive_links(
        ground_truth_path, sample_size=args.sample_size, seed=args.seed
    )
    source1_ids = frozenset(source1_id for source1_id, _ in links)
    source1_records = load_selected_source_records(
        data_root / "train" / "train_source1.tsv",
        "S1",
        source1_ids,
        chunksize=args.chunk_size,
    )
    rows = collect_profile_rows(
        links, source1_records, data_root, chunksize=args.chunk_size
    )
    frame = pd.DataFrame(rows)
    rates, quantiles, missing = summarize(frame)
    rates.to_csv(tables_dir / "positive_pair_profile_rates.csv", index=False)
    quantiles.to_csv(tables_dir / "positive_pair_profile_quantiles.csv", index=False)
    missing.to_csv(
        tables_dir / "positive_pair_missing_address_effects.csv", index=False
    )
    runtime_seconds = time.monotonic() - started
    memory_mb = peak_rss_mb()
    pd.DataFrame(
        [
            {
                "sample_size": args.sample_size,
                "seed": args.seed,
                "runtime_seconds": runtime_seconds,
                "peak_rss_mb": memory_mb,
            }
        ]
    ).to_csv(tables_dir / "positive_pair_profile_run.csv", index=False)
    write_report(
        results_dir / "positive_pair_profile_exp001.md",
        rates,
        quantiles,
        sample_size=args.sample_size,
        seed=args.seed,
        runtime_seconds=runtime_seconds,
        peak_memory_mb=memory_mb,
    )
    print(
        f"Profile complete in {runtime_seconds:.1f}s; peak RSS {memory_mb:.1f} MB.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

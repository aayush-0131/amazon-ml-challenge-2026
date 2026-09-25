#!/usr/bin/env python3
"""Reproducible Phase-1 audit of the competition-provided TSV files.

The script is read-only with respect to ``data/raw``. Source files are processed in
chunks, while exact duplicate counts use a temporary on-disk SQLite database so the
largest files do not need to fit in RAM.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from business_entity_resolution.data import (  # noqa: E402
    DEFAULT_CHUNK_SIZE,
    SOURCE_COLUMNS,
    iter_source_chunks,
)
from business_entity_resolution.ground_truth import iter_ground_truth  # noqa: E402

SOURCE_FILES = tuple(
    (split, source, f"{split}_source{source[-1]}.tsv")
    for split in ("train", "test")
    for source in ("S1", "S2", "S3")
)
TEXT_COLUMNS = ("business_name", "business_address")
DUPLICATE_COLUMNS = ("entity_id", *TEXT_COLUMNS)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


@dataclass
class LengthProfile:
    counts: Counter[int] = field(default_factory=Counter)
    total_characters: int = 0
    observations: int = 0

    def update(self, values: pd.Series) -> None:
        non_empty = values.loc[values.str.strip().ne("")]
        lengths = non_empty.str.len().astype("int64")
        chunk_counts = lengths.value_counts()
        self.counts.update(
            {int(length): int(count) for length, count in chunk_counts.items()}
        )
        self.total_characters += int(lengths.sum())
        self.observations += int(lengths.size)

    def quantile(self, probability: float) -> float:
        if not self.observations:
            return math.nan
        rank = max(1, math.ceil(probability * self.observations))
        cumulative = 0
        for length in sorted(self.counts):
            cumulative += self.counts[length]
            if cumulative >= rank:
                return float(length)
        raise AssertionError("length quantile rank was not reached")

    def summary(self) -> dict[str, float | int]:
        if not self.observations:
            return {
                "non_missing_count": 0,
                "mean": math.nan,
                "min": math.nan,
                "p25": math.nan,
                "p50": math.nan,
                "p75": math.nan,
                "p90": math.nan,
                "p95": math.nan,
                "p99": math.nan,
                "max": math.nan,
            }
        lengths = sorted(self.counts)
        return {
            "non_missing_count": self.observations,
            "mean": self.total_characters / self.observations,
            "min": lengths[0],
            "p25": self.quantile(0.25),
            "p50": self.quantile(0.50),
            "p75": self.quantile(0.75),
            "p90": self.quantile(0.90),
            "p95": self.quantile(0.95),
            "p99": self.quantile(0.99),
            "max": lengths[-1],
        }


class ExactDuplicateCounter:
    """Disk-backed exact string counts for high-cardinality columns."""

    def __init__(self, temp_directory: Path | None) -> None:
        temp = tempfile.NamedTemporaryFile(
            prefix="amazon-ber-audit-",
            suffix=".sqlite3",
            dir=temp_directory,
            delete=False,
        )
        temp.close()
        self.path = Path(temp.name)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA temp_store=MEMORY")
        self.connection.execute("PRAGMA locking_mode=EXCLUSIVE")
        self.connection.execute(
            "CREATE TABLE counts ("
            "field TEXT NOT NULL, value TEXT NOT NULL, n INTEGER NOT NULL, "
            "PRIMARY KEY (field, value)) WITHOUT ROWID"
        )

    def update(self, field_name: str, values: pd.Series) -> None:
        if field_name in TEXT_COLUMNS:
            values = values.loc[values.str.strip().ne("")]
        counts = values.value_counts(sort=False)
        rows = (
            (field_name, str(value), int(count))
            for value, count in counts.items()
        )
        self.connection.executemany(
            "INSERT INTO counts(field, value, n) VALUES (?, ?, ?) "
            "ON CONFLICT(field, value) DO UPDATE SET n = n + excluded.n",
            rows,
        )

    def commit(self) -> None:
        self.connection.commit()

    def summarize(self, field_name: str) -> dict[str, int]:
        row = self.connection.execute(
            "SELECT COUNT(*), "
            "COALESCE(SUM(CASE WHEN n > 1 THEN 1 ELSE 0 END), 0), "
            "COALESCE(SUM(CASE WHEN n > 1 THEN n ELSE 0 END), 0), "
            "COALESCE(SUM(CASE WHEN n > 1 THEN n - 1 ELSE 0 END), 0) "
            "FROM counts WHERE field = ?",
            (field_name,),
        ).fetchone()
        assert row is not None
        return {
            "distinct_non_missing_values": int(row[0]),
            "duplicated_value_groups": int(row[1]),
            "rows_in_duplicate_groups": int(row[2]),
            "excess_duplicate_rows": int(row[3]),
        }

    def close(self) -> None:
        self.connection.close()
        self.path.unlink(missing_ok=True)

    def __enter__(self) -> ExactDuplicateCounter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def audit_source_file(
    path: Path,
    split: str,
    source: str,
    *,
    chunksize: int,
    temp_directory: Path | None,
    exact_duplicates: bool,
) -> tuple[list[dict[str, object]], ...]:
    print(f"Auditing {path} ...", flush=True)
    rows = 0
    countries: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    lengths = {column: LengthProfile() for column in TEXT_COLUMNS}
    duplicate_rows: list[dict[str, object]] = []

    duplicate_counter = (
        ExactDuplicateCounter(temp_directory) if exact_duplicates else None
    )
    try:
        for chunk in iter_source_chunks(path, source, chunksize=chunksize):
            rows += len(chunk)
            countries.update(chunk["country"].tolist())
            for column in SOURCE_COLUMNS:
                missing[column] += int(chunk[column].str.strip().eq("").sum())
            for column in TEXT_COLUMNS:
                lengths[column].update(chunk[column])
            if duplicate_counter is not None:
                for column in DUPLICATE_COLUMNS:
                    duplicate_counter.update(column, chunk[column])
                duplicate_counter.commit()

        if duplicate_counter is not None:
            for column in DUPLICATE_COLUMNS:
                result = duplicate_counter.summarize(column)
                duplicate_rows.append(
                    {"split": split, "source": source, "field": column, **result}
                )
    finally:
        if duplicate_counter is not None:
            duplicate_counter.close()

    dataset_summary = [{
        "split": split,
        "source": source,
        "relative_path": display_path(path),
        "file_size_bytes": path.stat().st_size,
        "row_count": rows,
        "schema_valid": True,
        "entity_id_prefix_valid": True,
    }]
    country_rows = [
        {
            "split": split,
            "source": source,
            "country": country if country else "<EMPTY>",
            "row_count": count,
            "row_share": count / rows if rows else math.nan,
        }
        for country, count in sorted(countries.items())
    ]
    missing_rows = [
        {
            "split": split,
            "source": source,
            "field": column,
            "missing_count": missing[column],
            "missing_rate": missing[column] / rows if rows else math.nan,
        }
        for column in SOURCE_COLUMNS
    ]
    length_rows = [
        {
            "split": split,
            "source": source,
            "field": column,
            **lengths[column].summary(),
        }
        for column in TEXT_COLUMNS
    ]
    return dataset_summary, country_rows, missing_rows, duplicate_rows, length_rows


def audit_ground_truth(path: Path) -> tuple[list[dict[str, object]], ...]:
    print(f"Auditing {path} ...", flush=True)
    total_rows = 0
    match_counts: Counter[int] = Counter()
    source_coverage: Counter[str] = Counter()
    positive_links: Counter[str] = Counter()
    seen_source1: set[str] = set()
    duplicate_source1_rows = 0

    for source1_id, matches in iter_ground_truth(path):
        total_rows += 1
        if source1_id in seen_source1:
            duplicate_source1_rows += 1
        seen_source1.add(source1_id)
        match_counts[len(matches)] += 1
        s2_count = sum(match.startswith("S2-") for match in matches)
        s3_count = sum(match.startswith("S3-") for match in matches)
        positive_links["S2"] += s2_count
        positive_links["S3"] += s3_count
        if not matches:
            source_coverage["singleton"] += 1
        elif s2_count and s3_count:
            source_coverage["both_sources"] += 1
        elif s2_count:
            source_coverage["S2_only"] += 1
        else:
            source_coverage["S3_only"] += 1

    count_rows = [
        {
            "match_count": match_count,
            "entity_count": entity_count,
            "entity_share": entity_count / total_rows,
        }
        for match_count, entity_count in sorted(match_counts.items())
    ]
    coverage_order = ("singleton", "S2_only", "S3_only", "both_sources")
    coverage_rows = [
        {
            "match_source_category": category,
            "entity_count": source_coverage[category],
            "entity_share": source_coverage[category] / total_rows,
        }
        for category in coverage_order
    ]
    link_total = sum(positive_links.values())
    link_rows = [
        {
            "matched_source": source,
            "positive_link_count": positive_links[source],
            "positive_link_share": positive_links[source] / link_total,
        }
        for source in ("S2", "S3")
    ]
    integrity_rows = [{
        "relative_path": display_path(path),
        "row_count": total_rows,
        "distinct_source1_entity_ids": len(seen_source1),
        "duplicate_source1_rows": duplicate_source1_rows,
    }]
    return count_rows, coverage_rows, link_rows, integrity_rows


def build_country_shift(country_frame: pd.DataFrame) -> pd.DataFrame:
    summaries = [country_frame]
    all_sources = (
        country_frame.groupby(["split", "country"], as_index=False)["row_count"]
        .sum()
        .assign(source="ALL")
    )
    all_totals = all_sources.groupby("split")["row_count"].transform("sum")
    all_sources["row_share"] = all_sources["row_count"] / all_totals
    summaries.append(all_sources[country_frame.columns])
    combined = pd.concat(summaries, ignore_index=True)

    counts = combined.pivot_table(
        index=["source", "country"],
        columns="split",
        values="row_count",
        fill_value=0,
    ).rename(columns={"train": "train_row_count", "test": "test_row_count"})
    shares = combined.pivot_table(
        index=["source", "country"],
        columns="split",
        values="row_share",
        fill_value=0,
    ).rename(columns={"train": "train_share", "test": "test_share"})
    shift = counts.join(shares).reset_index()
    for column in ("train_row_count", "test_row_count", "train_share", "test_share"):
        if column not in shift:
            shift[column] = 0
    shift["share_change_percentage_points"] = (
        shift["test_share"] - shift["train_share"]
    ) * 100
    shift[["train_row_count", "test_row_count"]] = shift[
        ["train_row_count", "test_row_count"]
    ].astype("int64")
    shift["test_only_country"] = (
        shift["train_row_count"].eq(0) & shift["test_row_count"].gt(0)
    )
    shift = shift.sort_values(["source", "country"]).reset_index(drop=True)
    shift.columns.name = None
    return shift


def write_tables(tables_directory: Path, tables: dict[str, pd.DataFrame]) -> None:
    tables_directory.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(tables_directory / f"{name}.csv", index=False)


def write_report(
    report_path: Path,
    tables: dict[str, pd.DataFrame],
    *,
    chunksize: int,
    exact_duplicates: bool,
) -> None:
    summary = tables["dataset_summary"]
    missingness = tables["missingness"]
    coverage = tables["ground_truth_source_coverage"].set_index(
        "match_source_category"
    )
    links = tables["positive_links_by_source"].set_index("matched_source")
    shift = tables["country_shift"]
    france = shift.loc[(shift["source"] == "ALL") & (shift["country"] == "France")]
    france_test_share = float(france.iloc[0]["test_share"]) if not france.empty else 0.0
    france_s1 = shift.loc[(shift["source"] == "S1") & (shift["country"] == "France")]
    france_s1_share = float(france_s1.iloc[0]["test_share"]) if not france_s1.empty else 0.0
    singleton_count = int(coverage.loc["singleton", "entity_count"])
    total_entities = int(coverage["entity_count"].sum())

    lines = [
        "# Phase-1 dataset audit",
        "",
        "This report profiles only the competition-provided TSVs under `data/raw`. "
        "The audit does not alter raw data or use external information.",
        "",
        "## Dataset inventory",
        "",
        "| Split | Source | Rows | Size (bytes) |",
        "|---|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.split} | {row.source} | {row.row_count:,} | "
            f"{row.file_size_bytes:,} |"
        )

    lines.extend([
        "",
        "## Key findings",
        "",
        f"- Ground truth contains {total_entities:,} Source-1 entities and "
        f"{int(links['positive_link_count'].sum()):,} positive links.",
        f"- Singletons: {singleton_count:,} ({singleton_count / total_entities:.3%}).",
        f"- Matched-source coverage: S2 only "
        f"{int(coverage.loc['S2_only', 'entity_count']):,}, S3 only "
        f"{int(coverage.loc['S3_only', 'entity_count']):,}, both sources "
        f"{int(coverage.loc['both_sources', 'entity_count']):,}.",
        f"- Positive links: S2 {int(links.loc['S2', 'positive_link_count']):,}; "
        f"S3 {int(links.loc['S3', 'positive_link_count']):,}.",
        f"- France is absent from training and accounts for "
        f"{france_s1_share:.3%} of test Source-1 entities and "
        f"{france_test_share:.3%} of all test source rows. It must be treated as "
        "an open-set country value.",
    ])

    address_missing = missingness.loc[missingness["field"] == "business_address"]
    for row in address_missing.itertuples(index=False):
        if row.missing_count:
            lines.append(
                f"- {row.split} {row.source} has {row.missing_count:,} missing "
                f"addresses ({row.missing_rate:.3%})."
            )

    if exact_duplicates:
        duplicates = tables["exact_duplicates"]
        entity_duplicates = duplicates.loc[duplicates["field"] == "entity_id"]
        duplicate_id_rows = int(entity_duplicates["excess_duplicate_rows"].sum())
        lines.append(
            f"- Within-file duplicate entity-ID excess rows: "
            f"{duplicate_id_rows:,}."
        )
        lines.append(
            "- Exact name/address duplicate statistics are in "
            "`results/tables/exact_duplicates.csv`; empty text values are excluded."
        )
    else:
        lines.append("- Exact duplicate profiling was skipped by CLI request.")

    lines.extend([
        "",
        "## Quality interpretation",
        "",
        "- Entity IDs, schemas, country labels, and required fields are validated "
        "during chunked reads; a violation stops the audit.",
        "- Missing Source-2/3 addresses are expected input sparsity but will matter "
        "to later matching work. This audit does not impute or normalize them.",
        "- Exact repeated names or addresses are not automatically data errors in "
        "entity resolution; they indicate ambiguity and should not be treated as "
        "unique identifiers.",
        "- The France test-only shift is a high-confidence distribution shift, not "
        "evidence of bad data. Later pipelines must avoid a closed `{US, India}` "
        "country assumption.",
        "",
        "## Reproducibility and resource behavior",
        "",
        f"- Source chunk size: {chunksize:,} rows.",
        "- Exact duplicate counts use one temporary SQLite database per source file "
        "and remove it after that file is summarized. Peak disk use is therefore "
        "bounded by one source's distinct strings rather than the entire corpus.",
        "- Length quantiles are exact character-count quantiles over non-empty text.",
        "- Machine-readable evidence is under `results/tables/`.",
        "",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "raw",
        help="Competition dataset root (default: %(default)s)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPOSITORY_ROOT / "results",
        help="Audit output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=None,
        help="Directory for temporary exact-count SQLite files",
    )
    parser.add_argument(
        "--skip-exact-duplicates",
        action="store_true",
        help="Skip disk-backed exact name/address/entity-ID duplicate counts",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    # Keep the repository-relative symlink spelling in output tables rather than
    # leaking the machine-specific raw-data target path.
    data_root = args.data_root.absolute()
    results_directory = args.results_dir.resolve()
    exact_duplicates = not args.skip_exact_duplicates

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.temp_dir is not None:
        args.temp_dir.mkdir(parents=True, exist_ok=True)

    table_rows: dict[str, list[dict[str, object]]] = {
        "dataset_summary": [],
        "country_distribution": [],
        "missingness": [],
        "exact_duplicates": [],
        "text_length_summary": [],
    }
    for split, source, filename in SOURCE_FILES:
        path = data_root / split / filename
        outputs = audit_source_file(
            path,
            split,
            source,
            chunksize=args.chunk_size,
            temp_directory=args.temp_dir,
            exact_duplicates=exact_duplicates,
        )
        for name, rows in zip(table_rows, outputs, strict=True):
            table_rows[name].extend(rows)

    ground_truth_outputs = audit_ground_truth(
        data_root / "train" / "train_ground_truth.tsv"
    )
    ground_truth_names = (
        "ground_truth_match_counts",
        "ground_truth_source_coverage",
        "positive_links_by_source",
        "ground_truth_integrity",
    )
    for name, rows in zip(ground_truth_names, ground_truth_outputs, strict=True):
        table_rows[name] = rows

    tables = {name: pd.DataFrame(rows) for name, rows in table_rows.items()}
    tables["country_shift"] = build_country_shift(
        tables["country_distribution"]
    )
    write_tables(results_directory / "tables", tables)
    write_report(
        results_directory / "dataset_audit_phase1.md",
        tables,
        chunksize=args.chunk_size,
        exact_duplicates=exact_duplicates,
    )
    elapsed = time.monotonic() - started
    print(
        f"Audit complete in {elapsed / 60:.1f} minutes. Results: "
        f"{results_directory / 'dataset_audit_phase1.md'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

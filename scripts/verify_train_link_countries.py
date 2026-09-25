#!/usr/bin/env python3
"""Verify whether any training ground-truth link crosses country labels."""

from __future__ import annotations

import argparse
import platform
import resource
import sys
import time
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from business_entity_resolution.data import iter_source_chunks  # noqa: E402
from business_entity_resolution.ground_truth import iter_ground_truth  # noqa: E402


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return value / (1024 * 1024)
    return value / 1024


def verify(data_root: Path) -> pd.DataFrame:
    """Return exact per-source country-agreement counts using dynamic labels."""

    country_codes: dict[str, int] = {}

    def encode(country: str) -> int:
        if country not in country_codes:
            country_codes[country] = len(country_codes)
        return country_codes[country]

    source1_country: dict[str, int] = {}
    for chunk in iter_source_chunks(data_root / "train_source1.tsv", "S1"):
        source1_country.update(
            (entity_id, encode(country))
            for entity_id, country in chunk.loc[:, ["entity_id", "country"]].itertuples(
                index=False, name=None
            )
        )

    rows: list[dict[str, object]] = []
    for source in ("S2", "S3"):
        target_country: dict[str, int] = {}
        for chunk in iter_source_chunks(
            data_root / f"train_source{source[-1]}.tsv", source
        ):
            target_country.update(
                (entity_id, encode(country))
                for entity_id, country in chunk.loc[
                    :, ["entity_id", "country"]
                ].itertuples(index=False, name=None)
            )

        link_count = cross_country = missing_targets = 0
        for source1_id, matches in iter_ground_truth(
            data_root / "train_ground_truth.tsv"
        ):
            left_country = source1_country[source1_id]
            for target_id in matches:
                if not target_id.startswith(f"{source}-"):
                    continue
                link_count += 1
                right_country = target_country.get(target_id)
                if right_country is None:
                    missing_targets += 1
                elif left_country != right_country:
                    cross_country += 1
        rows.append(
            {
                "source": source,
                "true_link_count": link_count,
                "cross_country_link_count": cross_country,
                "cross_country_link_rate": cross_country / link_count,
                "missing_target_count": missing_targets,
            }
        )
        del target_country

    rows.append(
        {
            "source": "ALL",
            "true_link_count": sum(int(row["true_link_count"]) for row in rows),
            "cross_country_link_count": sum(
                int(row["cross_country_link_count"]) for row in rows
            ),
            "cross_country_link_rate": (
                sum(int(row["cross_country_link_count"]) for row in rows)
                / sum(int(row["true_link_count"]) for row in rows)
            ),
            "missing_target_count": sum(
                int(row["missing_target_count"]) for row in rows
            ),
        }
    )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "raw" / "train",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT
        / "results"
        / "tables"
        / "exp002_country_link_audit.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    result = verify(args.data_root.absolute())
    result["runtime_seconds"] = time.monotonic() - started
    result["peak_rss_mb"] = peak_rss_mb()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

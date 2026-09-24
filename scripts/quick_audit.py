from pathlib import Path
from collections import Counter
import csv
import os
import sys

ROOT = Path("data/raw")

SOURCE_FILES = [
    ROOT / "train" / "train_source1.tsv",
    ROOT / "train" / "train_source2.tsv",
    ROOT / "train" / "train_source3.tsv",
    ROOT / "test"  / "test_source1.tsv",
    ROOT / "test"  / "test_source2.tsv",
    ROOT / "test"  / "test_source3.tsv",
]

EXPECTED = {
    "entity_id",
    "business_name",
    "business_address",
    "country",
}

def human_bytes(n):
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024

def audit_source(path):
    if not path.exists():
        print(f"\nMISSING: {path}")
        return

    rows = 0
    missing_name = 0
    missing_address = 0
    missing_country = 0
    countries = Counter()
    prefixes = Counter()

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        fields = set(reader.fieldnames or [])
        schema_ok = fields == EXPECTED

        for row in reader:
            rows += 1

            eid = (row.get("entity_id") or "").strip()
            name = (row.get("business_name") or "").strip()
            addr = (row.get("business_address") or "").strip()
            country = (row.get("country") or "").strip()

            if not name:
                missing_name += 1
            if not addr:
                missing_address += 1
            if not country:
                missing_country += 1

            countries[country or "<EMPTY>"] += 1

            if "-" in eid:
                prefixes[eid.split("-", 1)[0]] += 1
            else:
                prefixes["<INVALID>"] += 1

    print("\n" + "=" * 72)
    print(path)
    print("=" * 72)
    print(f"file size        : {human_bytes(path.stat().st_size)}")
    print(f"rows             : {rows:,}")
    print(f"schema OK        : {schema_ok}")
    print(f"columns          : {reader.fieldnames}")
    print(f"missing name     : {missing_name:,} ({missing_name/max(rows,1):.3%})")
    print(f"missing address  : {missing_address:,} ({missing_address/max(rows,1):.3%})")
    print(f"missing country  : {missing_country:,} ({missing_country/max(rows,1):.3%})")
    print(f"ID prefixes      : {dict(prefixes)}")
    print("countries:")
    for c, n in countries.most_common():
        print(f"  {c!r:<16} {n:>12,}  ({n/max(rows,1):.3%})")

def audit_ground_truth(path):
    print("\n" + "=" * 72)
    print("GROUND TRUTH")
    print("=" * 72)

    if not path.exists():
        print(f"MISSING: {path}")
        return

    rows = 0
    match_count = Counter()
    total_links = 0
    s2_links = 0
    s3_links = 0
    mixed_entities = 0

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            rows += 1
            raw = (row.get("matched_entity_ids") or "").strip()

            ids = [x.strip() for x in raw.split(",") if x.strip()] if raw else []

            k = len(ids)
            match_count[k] += 1
            total_links += k

            has_s2 = any(x.startswith("S2-") for x in ids)
            has_s3 = any(x.startswith("S3-") for x in ids)

            s2_links += sum(x.startswith("S2-") for x in ids)
            s3_links += sum(x.startswith("S3-") for x in ids)

            if has_s2 and has_s3:
                mixed_entities += 1

    print(f"S1 ground-truth rows : {rows:,}")
    print(f"total positive links : {total_links:,}")
    print(f"S2 positive links    : {s2_links:,}")
    print(f"S3 positive links    : {s3_links:,}")

    singletons = match_count.get(0, 0)
    print(
        f"singletons           : {singletons:,} "
        f"({singletons/max(rows,1):.3%})"
    )

    matched_entities = rows - singletons
    print(
        f"matched S1 entities  : {matched_entities:,} "
        f"({matched_entities/max(rows,1):.3%})"
    )

    print(
        f"S1 with both S2 & S3 : {mixed_entities:,} "
        f"({mixed_entities/max(rows,1):.3%})"
    )

    print("\nMatch-count distribution:")
    for k in sorted(match_count):
        print(
            f"  {k:>3} matches : "
            f"{match_count[k]:>12,} "
            f"({match_count[k]/max(rows,1):.3%})"
        )

def main():
    print("Amazon ML Challenge 2026 — Quick Data Audit")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Raw data root: {ROOT.resolve()}")

    for p in SOURCE_FILES:
        audit_source(p)

    audit_ground_truth(ROOT / "train" / "train_ground_truth.tsv")

    print("\n" + "=" * 72)
    print("QUICK AUDIT COMPLETE")
    print("=" * 72)

if __name__ == "__main__":
    main()

"""EXP005: bounded exact-evidence candidate retrieval and deterministic decisions.

No labels or country allowlist enter this inference module. Both the compact
EXP005 index and the existing EXP002 schema-2 index expose the same three tables.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import time
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from .normalize import normalize_address, normalize_name
from .ground_truth import iter_ground_truth
from .evaluation import evaluate_predictions

SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")


@dataclass(frozen=True)
class SourceRecord:
    entity_id: str
    business_name: str
    business_address: str
    country: str


def validate_schema(actual, expected, *, path):
    if tuple(actual or ()) != tuple(expected):
        raise ValueError(f"Unexpected TSV schema in {path}: {actual!r}")

MATCH_HEADER = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_HEADER = ("source1_entity_id", "candidate_entity_ids")
POLICIES = ("both_exact", "name_address_half", "address_name_half", "either_half")
MAX_EXACT_HITS = 200
SUBSET_HASH = "84fa2544c2905fa480ffc0e57a2912143304acb6653a5f6c09a3eefa2a66979c"


def fingerprint(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_records(path: Path, *, limit: int | None = None):
    """Read S1 in file order without retaining the input in memory."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        validate_schema(reader.fieldnames, SOURCE_COLUMNS, path=path)
        for position, row in enumerate(reader):
            if limit is not None and position >= limit:
                break
            eid = row["entity_id"]
            if (not eid.startswith("S1-") or any(c in eid for c in "\t\r\n,")
                    or None in row or any(v is None for v in row.values())):
                raise ValueError(f"Malformed S1 row at position {position}")
            yield SourceRecord(eid, row["business_name"], row["business_address"], row["country"])


def build_index(source_path: Path, source: str, index_dir: Path, *, batch_size: int = 5000) -> Path:
    """Build a compact disk index once. Never reads labels or S1."""
    if source not in {"S2", "S3"} or batch_size < 1:
        raise ValueError("source must be S2/S3 and batch_size positive")
    index_dir.mkdir(parents=True, exist_ok=True)
    path = index_dir / f"exp005_{source.lower()}_exact.sqlite"
    if path.exists():
        with sqlite3.connect(f"file:{path.absolute()}?mode=ro", uri=True) as connection:
            meta = dict(connection.execute("SELECT key, value FROM metadata"))
        if (meta.get("status") != "complete" or meta.get("source") != source
                or json.loads(meta.get("source_fingerprint", "null")) != fingerprint(source_path)):
            raise ValueError(f"Existing EXP005 index is incompatible: {path}")
        return path
    temporary = path.with_suffix(".building")
    temporary.touch(exist_ok=False)
    started = time.monotonic()
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript("""
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=NORMAL;
            PRAGMA cache_size=-32768;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE records (record_id INTEGER PRIMARY KEY, entity_id TEXT NOT NULL UNIQUE,
                                  country TEXT NOT NULL, business_name TEXT NOT NULL,
                                  business_address TEXT NOT NULL);
            CREATE TABLE exact_name (country TEXT NOT NULL, value TEXT NOT NULL, record_id INTEGER NOT NULL);
            CREATE TABLE exact_address (country TEXT NOT NULL, value TEXT NOT NULL, record_id INTEGER NOT NULL);
        """)
        count = 0
        with source_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            validate_schema(reader.fieldnames, SOURCE_COLUMNS, path=source_path)
            records, names, addresses = [], [], []
            for row in reader:
                count += 1
                eid = row["entity_id"]
                if (not eid.startswith(source + "-") or None in row
                        or any(value is None for value in row.values())):
                    raise ValueError(f"Malformed {source} row {count}")
                records.append((count, eid, row["country"], row["business_name"], row["business_address"]))
                name = normalize_name(row["business_name"])
                address = normalize_address(row["business_address"])
                if name:
                    names.append((row["country"], name, count))
                if address:
                    addresses.append((row["country"], address, count))
                if len(records) >= batch_size:
                    _insert_batch(connection, records, names, addresses)
                    records, names, addresses = [], [], []
            if records:
                _insert_batch(connection, records, names, addresses)
        connection.executescript("""
            CREATE INDEX exact_name_lookup ON exact_name(country, value, record_id);
            CREATE INDEX exact_address_lookup ON exact_address(country, value, record_id);
        """)
        connection.executemany("INSERT INTO metadata VALUES (?, ?)", (
            ("status", "complete"), ("source", source), ("schema_version", "exp005-v1"),
            ("row_count", str(count)),
            ("source_fingerprint", json.dumps(fingerprint(source_path), sort_keys=True)),
        ))
        connection.commit()
    except BaseException:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise
    connection.close()
    with sqlite3.connect(f"file:{temporary.absolute()}?mode=ro", uri=True) as check:
        stored = json.loads(check.execute(
            "SELECT value FROM metadata WHERE key='source_fingerprint'").fetchone()[0])
    if fingerprint(source_path) != stored:
        temporary.unlink()
        raise ValueError("Source changed during index build")
    temporary.replace(path)
    print(f"{source}: indexed {count:,} rows in {time.monotonic() - started:.1f}s ({path})", flush=True)
    return path


def _insert_batch(connection, records, names, addresses):
    connection.executemany("INSERT INTO records VALUES (?, ?, ?, ?, ?)", records)
    connection.executemany("INSERT INTO exact_name VALUES (?, ?, ?)", names)
    connection.executemany("INSERT INTO exact_address VALUES (?, ?, ?)", addresses)
    connection.commit()


class ExactIndex:
    def __init__(self, path: Path, source: str, source_path: Path):
        self.connection = sqlite3.connect(f"file:{path.absolute()}?mode=ro", uri=True)
        meta = dict(self.connection.execute("SELECT key, value FROM metadata"))
        if (meta.get("status") != "complete" or meta.get("source") != source
                or json.loads(meta.get("source_fingerprint", "null")) != fingerprint(source_path)):
            self.close()
            raise ValueError(f"Index source fingerprint/status mismatch: {path}")
        self.source = source
        self.overflow = Counter()

    def close(self):
        self.connection.close()

    def exact_ids(self, table: str, country: str, value: str) -> set[int]:
        if not value:
            return set()
        if table not in {"exact_name", "exact_address"}:
            raise ValueError("Invalid posting table")
        rows = self.connection.execute(
            f"SELECT record_id FROM {table} WHERE country=? AND value=? ORDER BY record_id LIMIT ?",
            (country, value, MAX_EXACT_HITS + 1),
        ).fetchall()
        if len(rows) > MAX_EXACT_HITS:
            self.overflow[table] += 1
            return set()  # Skip entire common posting; never take an arbitrary prefix.
        return {int(row[0]) for row in rows}

    def records(self, ids: set[int]):
        if not ids:
            return []
        ordered = sorted(ids)
        placeholders = ",".join("?" for _ in ordered)
        return self.connection.execute(
            f"SELECT entity_id, country, business_name, business_address FROM records "
            f"WHERE record_id IN ({placeholders}) ORDER BY entity_id", ordered,
        ).fetchall()


def open_indexes(index_dir: Path, data_dir: Path, split: str, stack: ExitStack):
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    indexes = {}
    for source in ("S2", "S3"):
        compact = index_dir / f"exp005_{source.lower()}_exact.sqlite"
        legacy = index_dir / f"exp002c_{source.lower()}_postings_v2.sqlite"
        path = compact if compact.is_file() else legacy
        if not path.is_file():
            raise FileNotFoundError(f"No completed exact index for {source} in {index_dir}")
        index = ExactIndex(path, source, data_dir / f"{split}_source{source[-1]}.tsv")
        stack.callback(index.close)
        indexes[source] = index
    return indexes


def _overlap(left: str, right: str) -> float:
    a, b = set(left.split()), set(right.split())
    return len(a & b) / len(a | b) if a and b else 0.0


def candidate_evidence(record: SourceRecord, indexes: dict[str, ExactIndex]):
    """Return exactly the country-scoped candidates whose evidence is evaluated."""
    name, address = normalize_name(record.business_name), normalize_address(record.business_address)
    evidence = []
    for source in ("S2", "S3"):
        index = indexes[source]
        ids = index.exact_ids("exact_name", record.country, name)
        ids |= index.exact_ids("exact_address", record.country, address)
        for eid, country, raw_name, raw_address in index.records(ids):
            if country != record.country:
                raise ValueError("Country-scoped index returned a different country")
            other_name, other_address = normalize_name(raw_name), normalize_address(raw_address)
            same_name = bool(name and name == other_name)
            same_address = bool(address and address == other_address)
            evidence.append((eid, same_name, same_address,
                             _overlap(address, other_address) if same_name else 0.0,
                             _overlap(name, other_name) if same_address else 0.0))
    return sorted(evidence)


def decide(evidence, policy: str) -> list[str]:
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")
    matches = []
    for eid, same_name, same_address, address_overlap, name_overlap in evidence:
        both = same_name and same_address
        name_strong = same_name and address_overlap >= 0.5
        address_strong = same_address and name_overlap >= 0.5
        if (both or (policy in {"name_address_half", "either_half"} and name_strong)
                or (policy in {"address_name_half", "either_half"} and address_strong)):
            matches.append(eid)
    return matches


def id_list(ids) -> str:
    ordered = sorted(set(ids))
    if any(not eid.startswith(("S2-", "S3-")) or any(c in eid for c in "\t\r\n,") for eid in ordered):
        raise ValueError("Invalid target ID")
    return ",".join(ordered)


def run_inference(source1_path: Path, target_dir: Path, index_dir: Path, output_dir: Path,
                  *, split: str, policy: str, smoke_limit: int | None = None,
                  allow_full_test: bool = False) -> dict:
    if policy not in POLICIES:
        raise ValueError("Unknown policy")
    if split == "test":
        if smoke_limit is None and not allow_full_test:
            raise ValueError("Full TEST requires --allow-full-test")
        if smoke_limit is not None and (not 0 < smoke_limit <= 10000 or allow_full_test):
            raise ValueError("Use a 1..10000 smoke limit or --allow-full-test")
    elif split != "train" or smoke_limit is not None or allow_full_test:
        raise ValueError("Invalid TRAIN/TEST inference options")
    if output_dir.exists():
        raise FileExistsError(f"Output directory must be fresh: {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.monotonic()
    initial_source = fingerprint(source1_path)
    candidates_count = predictions_count = rows = 0
    with ExitStack() as stack:
        indexes = open_indexes(index_dir, target_dir, split, stack)
        seen = sqlite3.connect(output_dir / "seen.sqlite")
        stack.callback(seen.close)
        seen.execute("CREATE TABLE seen (entity_id TEXT PRIMARY KEY)")
        match_file = stack.enter_context((output_dir / "matching_results.tsv").open("w", encoding="utf-8", newline=""))
        candidate_file = stack.enter_context((output_dir / "candidate_pairs.tsv").open("w", encoding="utf-8", newline=""))
        match_writer = csv.writer(match_file, delimiter="\t", lineterminator="\n")
        candidate_writer = csv.writer(candidate_file, delimiter="\t", lineterminator="\n")
        match_writer.writerow(MATCH_HEADER)
        candidate_writer.writerow(CANDIDATE_HEADER)
        for record in source_records(source1_path, limit=smoke_limit):
            seen.execute("INSERT INTO seen VALUES (?)", (record.entity_id,))
            evidence = candidate_evidence(record, indexes)
            candidates = [item[0] for item in evidence]
            predictions = decide(evidence, policy)
            if not set(predictions) <= set(candidates):
                raise AssertionError("Prediction outside evaluated candidate set")
            match_writer.writerow((record.entity_id, id_list(predictions)))
            candidate_writer.writerow((record.entity_id, id_list(candidates)))
            rows += 1
            candidates_count += len(candidates)
            predictions_count += len(predictions)
            if rows % 5000 == 0:
                seen.commit()
        seen.commit()
    if fingerprint(source1_path) != initial_source:
        raise ValueError("Source1 changed during inference")
    manifest = {"split": split, "policy": policy, "rows": rows,
                "candidates": candidates_count, "predictions": predictions_count,
                "seconds": time.monotonic() - started,
                "source1_fingerprint": initial_source,
                "overflow": {s: dict(indexes[s].overflow) for s in ("S2", "S3")},
                "full_test_coverage": split == "test" and smoke_limit is None}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def validate_output(source1_path: Path, output_dir: Path, *, limit: int | None = None) -> int:
    """Verify rows, order, deterministic IDs, and prediction subset, streaming."""
    with ExitStack() as stack:
        readers = {}
        for filename, header in (("matching_results.tsv", MATCH_HEADER),
                                 ("candidate_pairs.tsv", CANDIDATE_HEADER)):
            handle = stack.enter_context((output_dir / filename).open(encoding="utf-8", newline=""))
            reader = csv.reader(handle, delimiter="\t")
            if tuple(next(reader, ())) != header:
                raise ValueError(f"Invalid output header: {filename}")
            readers[filename] = reader
        count = 0
        for record in source_records(source1_path, limit=limit):
            values = {}
            for filename, reader in readers.items():
                row = next(reader, None)
                if row is None or len(row) != 2 or row[0] != record.entity_id:
                    raise ValueError(f"Missing, duplicate, or out-of-order row in {filename}")
                ids = row[1].split(",") if row[1] else []
                if id_list(ids) != row[1] or len(ids) != len(set(ids)):
                    raise ValueError(f"Invalid target ordering/IDs in {filename}")
                values[filename] = set(ids)
            if not values["matching_results.tsv"] <= values["candidate_pairs.tsv"]:
                raise ValueError("Prediction is absent from evaluated candidate set")
            count += 1
        if any(next(reader, None) is not None for reader in readers.values()):
            raise ValueError("Extra output rows")
        return count


def _development_inputs(train_dir: Path, subset_path: Path):
    from .split import stratified_entity_split
    if sha256(subset_path) != SUBSET_HASH:
        raise ValueError("Not the authoritative EXP001 20k subset")
    metadata = {}
    with subset_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["source1_entity_id", "country", "match_count", "is_singleton", "partition"]:
            raise ValueError("Unexpected subset schema")
        for row in reader:
            eid = row["source1_entity_id"]
            if eid in metadata or row["partition"] not in {"tuning", "evaluation"}:
                raise ValueError("Duplicate ID or invalid partition in subset")
            metadata[eid] = (row if row["partition"] == "tuning" else
                             {"source1_entity_id": eid, "country": row["country"],
                              "partition": "evaluation"})
    if len(metadata) != 20000:
        raise ValueError("Expected exactly 20,000 development IDs")
    records = {record.entity_id: record for record in source_records(train_dir / "train_source1.tsv")
               if record.entity_id in metadata}
    if set(records) != set(metadata):
        raise ValueError("Subset S1 IDs missing from TRAIN")
    evaluation = {eid for eid, row in metadata.items() if row["partition"] == "evaluation"}
    development = set(metadata) - evaluation
    truth = {eid: matches for eid, matches in iter_ground_truth(train_dir / "train_ground_truth.tsv")
             if eid in development}
    if set(truth) != development:
        raise ValueError("FIT/TUNE ground truth IDs missing from TRAIN")
    for eid, row in metadata.items():
        if records[eid].country != row["country"]:
            raise ValueError(f"Subset metadata mismatch: {eid}")
        if eid in development and (len(truth[eid]) != int(row["match_count"])
                                   or (not truth[eid]) != (row["is_singleton"] == "True")):
            raise ValueError(f"Subset truth metadata mismatch: {eid}")
    split = stratified_entity_split(
        ({"entity_id": eid, "country": records[eid].country} for eid in sorted(development)),
        {eid: truth[eid] for eid in development}, validation_fraction=.25, seed=2029)
    parts = {**dict.fromkeys(split.train_ids, "fit"),
             **dict.fromkeys(split.validation_ids, "tune"),
             **dict.fromkeys(evaluation, "evaluation")}
    if Counter(parts.values()) != {"fit": 5999, "tune": 2001, "evaluation": 12000}:
        raise ValueError("Authoritative FIT/TUNE/EVALUATION counts changed")
    return records, truth, parts, metadata


def run_development(train_dir: Path, subset_path: Path, index_dir: Path, output_dir: Path) -> dict:
    """Select on TUNE, write a frozen policy, then inspect EVALUATION once."""
    if output_dir.exists():
        raise FileExistsError(f"Development output directory must be fresh: {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.monotonic()
    records, truth, parts, metadata = _development_inputs(train_dir, subset_path)
    predictions = {policy: {"fit": {}, "tune": {}} for policy in POLICIES}
    candidate_counts = Counter()
    with ExitStack() as stack:
        indexes = open_indexes(index_dir, train_dir, "train", stack)
        for eid in sorted(eid for eid in records if parts[eid] in {"fit", "tune"}):
            evidence = candidate_evidence(records[eid], indexes)
            part = parts[eid]
            candidate_counts[part] += len(evidence)
            for policy in POLICIES:
                predictions[policy][part][eid] = decide(evidence, policy)
        tune_truth = {eid: truth[eid] for eid in truth if parts[eid] == "tune"}
        tune_results = {policy: evaluate_predictions(tune_truth, predictions[policy]["tune"])
                        for policy in POLICIES}
        selected = max(POLICIES, key=lambda policy: (tune_results[policy].macro_fbeta,
                                                    -POLICIES.index(policy)))
        frozen = {"policy": selected, "subset_sha256": SUBSET_HASH,
                  "tune_macro_fbeta": tune_results[selected].macro_fbeta,
                  "policies": list(POLICIES), "max_exact_hits": MAX_EXACT_HITS}
        (output_dir / "frozen_policy.json").write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
        # EVALUATION labels are not loaded until after the policy is persisted.
        evaluation_truth = {eid: matches for eid, matches in iter_ground_truth(
            train_dir / "train_ground_truth.tsv") if eid in parts and parts[eid] == "evaluation"}
        if set(evaluation_truth) != {eid for eid in parts if parts[eid] == "evaluation"}:
            raise ValueError("EVALUATION ground truth IDs missing from TRAIN")
        with subset_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["partition"] == "evaluation":
                    eid = row["source1_entity_id"]
                    matches = evaluation_truth[eid]
                    if (len(matches) != int(row["match_count"])
                            or (not matches) != (row["is_singleton"] == "True")):
                        raise ValueError(f"EVALUATION truth metadata mismatch: {eid}")
        evaluation_predictions = {}
        for eid in sorted(eid for eid in records if parts[eid] == "evaluation"):
            evidence = candidate_evidence(records[eid], indexes)
            candidate_counts["evaluation"] += len(evidence)
            evaluation_predictions[eid] = decide(evidence, selected)
        overflow = {source: dict(indexes[source].overflow) for source in ("S2", "S3")}
    fit_truth = {eid: truth[eid] for eid in truth if parts[eid] == "fit"}
    report = {
        "selected_policy": selected,
        "partition_counts": dict(Counter(parts.values())),
        "candidate_counts": dict(candidate_counts),
        "fit": evaluate_predictions(fit_truth, predictions[selected]["fit"]).to_dict(),
        "tune": {policy: result.to_dict() for policy, result in tune_results.items()},
        "evaluation": evaluate_predictions(evaluation_truth, evaluation_predictions).to_dict(),
        "overflow": overflow,
        "seconds_total": time.monotonic() - started,
        "train_fingerprints": {name: fingerprint(train_dir / name) for name in
                               ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv",
                                "train_ground_truth.tsv")},
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report

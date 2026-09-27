"""Resumable, deterministic EXP007 TEST inference using the frozen base policy."""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
from threadpoolctl import threadpool_limits

from . import exp007
from .exp003_inference import CANDIDATE_HEADER, MATCH_HEADER, source_records
from .exp007_features import FEATURE_NAMES, feature_row
from .reranker import candidate_pool, fingerprint, index_identity, open_indexes, sha256, write_json_atomic

FORMAT_VERSION = 1
EXPECTED_FULL_S1 = 1_732_544
MAX_SMOKE = 4_000
OUTPUT_FILES = ("matching_results.tsv", "candidate_pairs.tsv")


def validate_mode(smoke_limit, allow_full_test):
    if allow_full_test:
        if smoke_limit is not None:
            raise ValueError("Full TEST cannot also set a smoke limit")
    elif type(smoke_limit) is not int or not 1 <= smoke_limit <= MAX_SMOKE:
        raise ValueError("Full TEST requires --allow-full-test; smoke limit must be 1..4000")


def frozen_bundle(tune_dir):
    # Deliberately omit TRAIN paths: TEST indexes have their own source fingerprints.
    bundle = exp007.load_frozen_bundle(tune_dir)
    policy = bundle["policy"]
    if (policy["variant"] != "base" or policy["guardian"] is not False
            or policy["exclusivity"] is not False or policy["numeric_penalty"] != 0.0
            or policy["calibration"] != "raw_HGB" or policy["feature_count"] != 83
            or (policy["threshold_s2"], policy["threshold_s3"]) != (.965, .956)
            or len(FEATURE_NAMES) != 83 or len(set(FEATURE_NAMES)) != 83
            or bundle["feature_names"] != list(FEATURE_NAMES)):
        raise ValueError("Frozen EXP007 TEST policy differs from the approved base policy")
    return bundle


def target_list(ids):
    ids = list(ids)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate target ID")
    for cid in ids:
        if (not isinstance(cid, str) or not cid.startswith(("S2-", "S3-"))
                or len(cid) <= 3 or cid.strip() != cid or any(c in cid for c in "\t\r\n,")):
            raise ValueError(f"Malformed target ID: {cid!r}")
    return ",".join(sorted(ids))


def score_record(bundle, record, indexes):
    candidates, _ = candidate_pool(record, indexes, bundle["blocker_config"])
    cids = [candidate.candidate_entity_id for candidate in candidates]
    candidate_text = target_list(cids)
    if candidates:
        matrix = np.asarray([list(feature_row(record, candidate).values()) for candidate in candidates],
                            dtype=np.float32).reshape(-1, len(FEATURE_NAMES))
        scores = bundle["models"]["model"].predict_proba(matrix)[:, 1]
        if len(scores) != len(cids) or not np.isfinite(scores).all():
            raise ValueError("Invalid EXP007 base model scores")
    else:
        scores = ()
    policy = bundle["policy"]
    matched = [cid for cid, score in zip(cids, scores) if score >= policy["threshold_s2" if cid.startswith("S2-") else "threshold_s3"]]
    matched_text = target_list(matched)
    if not set(matched) <= set(cids):
        raise AssertionError("Matched IDs escaped the scored candidate pool")
    return matched_text, candidate_text


def _identity(tune_dir, test_dir, index_dir, shards, smoke_limit):
    tune_dir, test_dir, index_dir = map(Path, (tune_dir, test_dir, index_dir))
    return {"format_version": FORMAT_VERSION, "bundle_sha256": sha256(tune_dir / "frozen.joblib"),
            "selected_policy_sha256": sha256(tune_dir / "selected_policy.json"),
            "freeze_sha256": sha256(tune_dir / "freeze.json"),
            "inference_code_sha256": sha256(__file__),
            "test_source1": fingerprint(test_dir / "test_source1.tsv"),
            "indexes": index_identity(index_dir), "shards": shards,
            "smoke_limit": smoke_limit, "full_test": smoke_limit is None}


def _checkpoint_digest(connection):
    digest = hashlib.sha256()
    for position, eid, matched, candidates in connection.execute(
            "SELECT position, entity_id, matched, candidates FROM outputs ORDER BY position"):
        for value in (str(position), eid, matched, candidates):
            data = value.encode("utf-8")
            digest.update(len(data).to_bytes(8, "big"))
            digest.update(data)
    return digest.hexdigest()


def _published_valid(directory, done, identity, shard, connection):
    return (done.get("identity") == identity and done.get("shard") == shard
            and done.get("rows") == connection.execute("SELECT COUNT(*) FROM outputs").fetchone()[0]
            and done.get("checkpoint_sha256") == _checkpoint_digest(connection)
            and set(done.get("files", {})) == set(OUTPUT_FILES)
            and all((directory / name).is_file() and sha256(directory / name) == digest
                    for name, digest in done["files"].items()))


def infer_shard(tune_dir, test_dir, index_dir, shard_dir, *, shard=0, shards=4,
                smoke_limit=100, allow_full_test=False, checkpoint_every=100):
    if type(shards) is not int or not 1 <= shards <= 64 or type(shard) is not int or not 0 <= shard < shards:
        raise ValueError("Use 1..64 shards and 0 <= shard < shards")
    if type(checkpoint_every) is not int or checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    validate_mode(smoke_limit, allow_full_test)
    bundle = frozen_bundle(tune_dir)
    test_dir, shard_dir = Path(test_dir), Path(shard_dir)
    identity = _identity(tune_dir, test_dir, index_dir, shards, smoke_limit)
    directory = shard_dir / f"shard_{shard:02d}"
    if directory.exists() and not (directory / "checkpoint.sqlite").exists():
        raise ValueError("Existing shard directory has no checkpoint identity")
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with ExitStack() as stack:
        lock = stack.enter_context((directory / "worker.lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        indexes = stack.enter_context(open_indexes(index_dir, test_dir, "test", bundle["blocker_config"]))
        connection = sqlite3.connect(directory / "checkpoint.sqlite")
        stack.callback(connection.close)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("CREATE TABLE IF NOT EXISTS outputs (position INTEGER PRIMARY KEY, entity_id TEXT UNIQUE NOT NULL, matched TEXT NOT NULL, candidates TEXT NOT NULL)")
        saved = dict(connection.execute("SELECT key, value FROM meta"))
        encoded = json.dumps({**identity, "shard": shard}, sort_keys=True)
        if saved and saved.get("identity") != encoded:
            raise ValueError("Checkpoint identity mismatch; use a fresh shard directory")
        if not saved:
            connection.execute("INSERT INTO meta VALUES ('identity', ?)", (encoded,))
            connection.commit()
        done_path = directory / "done.json"
        if done_path.exists():
            done = json.loads(done_path.read_text())
            if saved.get("complete") == "true" and _published_valid(directory, done, identity, shard, connection):
                return done
            raise ValueError("Completed shard marker/checksum mismatch; refusing to overwrite published files")
        committed = iter(connection.execute("SELECT position, entity_id FROM outputs ORDER BY position"))
        next_committed = next(committed, None)
        new_rows = 0
        input_count = 0
        scoring_started = time.monotonic()
        previous_seconds = float(saved.get("scoring_seconds", "0"))
        with threadpool_limits(limits=1):
            for position, record in source_records(test_dir / "test_source1.tsv", smoke_limit):
                input_count += 1
                if position % shards != shard:
                    continue
                if next_committed is not None:
                    if next_committed != (position, record.entity_id):
                        raise ValueError("Checkpoint has missing, duplicate, or out-of-order S1")
                    next_committed = next(committed, None)
                    continue
                matched, candidates = score_record(bundle, record, indexes)
                connection.execute("INSERT INTO outputs VALUES (?, ?, ?, ?)",
                                   (position, record.entity_id, matched, candidates))
                new_rows += 1
                if new_rows % checkpoint_every == 0:
                    elapsed = time.monotonic() - scoring_started
                    connection.execute("INSERT OR REPLACE INTO meta VALUES ('scoring_seconds', ?)",
                                       (str(previous_seconds + elapsed),))
                    connection.commit()
                    print(f"shard {shard}: {new_rows} new S1, position {position}, {new_rows/max(elapsed, 1e-9):.2f} new S1/s", flush=True)
        if next_committed is not None or next(committed, None) is not None:
            raise ValueError("Checkpoint contains S1 beyond the selected TEST source")
        expected = EXPECTED_FULL_S1 if smoke_limit is None else smoke_limit
        if input_count != expected:
            raise ValueError(f"Expected {expected} TEST S1, found {input_count}")
        expected_shard_rows = len(range(shard, expected, shards))
        count = connection.execute("SELECT COUNT(*) FROM outputs").fetchone()[0]
        if count != expected_shard_rows:
            raise ValueError("Shard row count mismatch")
        if (fingerprint(test_dir / "test_source1.tsv") != identity["test_source1"]
                or index_identity(index_dir) != identity["indexes"]):
            raise ValueError("TEST source/index changed during inference")
        scoring_seconds = time.monotonic() - scoring_started
        connection.execute("INSERT OR REPLACE INTO meta VALUES ('scoring_seconds', ?)",
                           (str(previous_seconds + scoring_seconds),))
        connection.execute("INSERT OR REPLACE INTO meta VALUES ('complete', 'true')")
        connection.commit()
        for name, header, column in (("matching_results.tsv", MATCH_HEADER, "matched"),
                                     ("candidate_pairs.tsv", CANDIDATE_HEADER, "candidates")):
            temporary = directory / (name + ".building")
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(header)
                writer.writerows(connection.execute(f"SELECT entity_id, {column} FROM outputs ORDER BY position"))
            temporary.replace(directory / name)
        done = {"identity": identity, "shard": shard, "rows": count,
                "checkpoint_sha256": _checkpoint_digest(connection),
                "scoring_seconds": previous_seconds + scoring_seconds,
                "new_rows_this_invocation": new_rows,
                "seconds_this_invocation": time.monotonic() - started,
                "files": {name: sha256(directory / name) for name in OUTPUT_FILES}}
        write_json_atomic(done_path, done)
        return done


def _validated_shards(shard_dir, test_dir, index_dir, shards, allow_full_test):
    if type(shards) is not int or not 1 <= shards <= 64:
        raise ValueError("Use 1..64 shards")
    shard_dir, test_dir = Path(shard_dir), Path(test_dir)
    manifests = [json.loads((shard_dir / f"shard_{i:02d}" / "done.json").read_text()) for i in range(shards)]
    identity = manifests[0]["identity"]
    if identity["format_version"] != FORMAT_VERSION or identity["full_test"] != (identity["smoke_limit"] is None):
        raise ValueError("Invalid EXP007 inference format or mode")
    validate_mode(identity["smoke_limit"], allow_full_test)
    if identity != _identity_from_manifest(identity, test_dir, index_dir, shards):
        raise ValueError("TEST source/index or inference code changed since shards ran")
    expected = EXPECTED_FULL_S1 if identity["smoke_limit"] is None else identity["smoke_limit"]
    for i, done in enumerate(manifests):
        directory = shard_dir / f"shard_{i:02d}"
        if done.get("identity") != identity or done.get("shard") != i or done.get("rows") != len(range(i, expected, shards)):
            raise ValueError("Mixed, duplicate, or incomplete shard identities")
        with sqlite3.connect(f"file:{directory / 'checkpoint.sqlite'}?mode=ro", uri=True) as connection:
            if dict(connection.execute("SELECT key, value FROM meta")).get("identity") != json.dumps({**identity, "shard": i}, sort_keys=True):
                raise ValueError("Shard checkpoint identity mismatch")
            if not _published_valid(directory, done, identity, i, connection):
                raise ValueError("Shard checksum or checkpoint mismatch")
    return identity, manifests, expected


def _identity_from_manifest(identity, test_dir, index_dir, shards):
    current = dict(identity)
    current.update({"test_source1": fingerprint(Path(test_dir) / "test_source1.tsv"),
                    "indexes": index_identity(index_dir), "inference_code_sha256": sha256(__file__),
                    "shards": shards})
    return current


def _read_shard_row(reader, expected_header):
    row = next(reader, None)
    if row is None or len(row) != 2:
        raise ValueError(f"Missing or malformed {expected_header[0]} shard row")
    ids = row[1].split(",") if row[1] else []
    if target_list(ids) != row[1]:
        raise ValueError("Target IDs are unsorted or duplicated")
    return row, set(ids)


def merge_shards(shard_dir, test_dir, index_dir, output_dir, *, shards=4, allow_full_test=False):
    shard_dir, test_dir, output_dir = map(Path, (shard_dir, test_dir, output_dir))
    if output_dir.exists() or output_dir.with_name(output_dir.name + ".building").exists():
        raise FileExistsError("Use a fresh EXP007 merge output directory")
    identity, manifests, expected = _validated_shards(shard_dir, test_dir, index_dir, shards, allow_full_test)
    building = output_dir.with_name(output_dir.name + ".building")
    building.mkdir(parents=True)
    counts = Counter()
    with ExitStack() as stack:
        seen = sqlite3.connect(building / "seen.sqlite")
        stack.callback(seen.close)
        seen.execute("CREATE TABLE seen (entity_id TEXT PRIMARY KEY)")
        readers, writers = {}, {}
        for name, header in (("matching_results.tsv", MATCH_HEADER), ("candidate_pairs.tsv", CANDIDATE_HEADER)):
            handle = stack.enter_context((building / name).open("w", encoding="utf-8", newline=""))
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(header)
            writers[name] = writer
            readers[name] = []
            for i in range(shards):
                handle = stack.enter_context((shard_dir / f"shard_{i:02d}" / name).open(encoding="utf-8", newline=""))
                reader = csv.reader(handle, delimiter="\t")
                if next(reader, None) != header:
                    raise ValueError("Bad EXP007 shard header")
                readers[name].append(reader)
        input_count = 0
        for position, record in source_records(test_dir / "test_source1.tsv", identity["smoke_limit"]):
            input_count += 1
            shard = position % shards
            try:
                seen.execute("INSERT INTO seen VALUES (?)", (record.entity_id,))
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"Duplicate TEST S1 ID: {record.entity_id}") from exc
            match_row, matched = _read_shard_row(readers["matching_results.tsv"][shard], MATCH_HEADER)
            candidate_row, candidates = _read_shard_row(readers["candidate_pairs.tsv"][shard], CANDIDATE_HEADER)
            if match_row[0] != record.entity_id or candidate_row[0] != record.entity_id:
                raise ValueError("Missing, duplicate, or out-of-order shard S1")
            if not matched <= candidates:
                raise ValueError("Matched IDs are not a subset of scored candidates")
            writers["matching_results.tsv"].writerow(match_row)
            writers["candidate_pairs.tsv"].writerow(candidate_row)
            counts[shard] += 1
        if input_count != expected:
            raise ValueError(f"Expected {expected} S1 rows, found {input_count}")
        if any(next(reader, None) is not None for group in readers.values() for reader in group):
            raise ValueError("Unexpected extra shard rows")
        if any(counts[i] != manifests[i]["rows"] for i in range(shards)):
            raise ValueError("Merged shard row count mismatch")
        if fingerprint(test_dir / "test_source1.tsv") != identity["test_source1"]:
            raise ValueError("TEST S1 changed during merge")
        seen.commit()
    write_json_atomic(building / "manifest.json", {"identity": identity, "rows": input_count,
        "files": {name: sha256(building / name) for name in OUTPUT_FILES},
        "submission_ready_full_coverage": identity["smoke_limit"] is None})
    building.replace(output_dir)
    return {"rows": input_count, "files": json.loads((output_dir / "manifest.json").read_text())["files"],
            "output_dir": str(output_dir)}


def audit_output(output_dir, test_dir):
    output_dir, test_dir = Path(output_dir), Path(test_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    identity = manifest["identity"]
    if fingerprint(test_dir / "test_source1.tsv") != identity["test_source1"]:
        raise ValueError("TEST S1 identity changed")
    if set(manifest["files"]) != set(OUTPUT_FILES) or any(
            sha256(output_dir / name) != manifest["files"][name] for name in OUTPUT_FILES):
        raise ValueError("Merged output checksum mismatch")
    with ExitStack() as stack:
        readers = {}
        for name, header in (("matching_results.tsv", MATCH_HEADER), ("candidate_pairs.tsv", CANDIDATE_HEADER)):
            handle = stack.enter_context((output_dir / name).open(encoding="utf-8", newline=""))
            reader = csv.reader(handle, delimiter="\t")
            if next(reader, None) != header:
                raise ValueError("Merged output header mismatch")
            readers[name] = reader
        count, matched_links, candidate_links = 0, 0, 0
        for _, record in source_records(test_dir / "test_source1.tsv", identity["smoke_limit"]):
            match_row, matched = _read_shard_row(readers["matching_results.tsv"], MATCH_HEADER)
            candidate_row, candidates = _read_shard_row(readers["candidate_pairs.tsv"], CANDIDATE_HEADER)
            if match_row[0] != record.entity_id or candidate_row[0] != record.entity_id or not matched <= candidates:
                raise ValueError("Merged output order, coverage, or subset mismatch")
            count += 1
            matched_links += len(matched)
            candidate_links += len(candidates)
        if any(next(reader, None) is not None for reader in readers.values()) or count != manifest["rows"]:
            raise ValueError("Merged output extra/missing S1 rows")
    if identity["smoke_limit"] is None and count != EXPECTED_FULL_S1:
        raise ValueError("Full TEST count mismatch")
    return {"rows": count, "matched_links": matched_links, "candidate_links": candidate_links,
            "full_coverage": identity["smoke_limit"] is None, "files": manifest["files"]}


def benchmark(shard_dir, test_dir, index_dir, *, shards=4):
    _, manifests, _ = _validated_shards(shard_dir, test_dir, index_dir, shards,
                                        allow_full_test=False)
    rates = [done["rows"] / done["scoring_seconds"] if done["scoring_seconds"] > 0 else 0 for done in manifests]
    total_rate = sum(rates)
    if total_rate <= 0:
        raise ValueError("No measured scoring throughput")
    return {"sample_s1": sum(done["rows"] for done in manifests),
        "per_shard_s1_per_second": rates, "aggregate_s1_per_second": total_rate,
        "projected_full_test_hours": EXPECTED_FULL_S1 / total_rate / 3600,
        "projection_assumption": "Independent parallel workers; excludes startup, merge, and resource contention"}

"""Label-free streaming TEST inference and validated deterministic shard merge."""
from __future__ import annotations

from contextlib import ExitStack
from collections import Counter
import csv
import fcntl
import json
from pathlib import Path
import sqlite3
import time

from threadpoolctl import threadpool_limits

from .data import SOURCE_COLUMNS, validate_schema
from .reranker import (accepted_ids, candidate_pool, feature_matrix, fingerprint, index_identity,
                       load_bundle, open_indexes, scores_for, sha256)
from .sampling import SourceRecord

MATCH_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]


def source_records(path, limit=None):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        validate_schema(reader.fieldnames, SOURCE_COLUMNS)
        for position, row in enumerate(reader):
            if limit is not None and position >= limit:
                break
            eid = row["entity_id"]
            if not eid.startswith("S1-") or any(c in eid for c in "\t\r\n,") or None in row or any(v is None for v in row.values()):
                raise ValueError("Malformed S1 row")
            yield position, SourceRecord(eid, row["business_name"], row["business_address"], row["country"])


def id_list(values):
    values = sorted(set(values))
    if any(not x.startswith(("S2-", "S3-")) or any(c in x for c in "\t\r\n,") for x in values):
        raise ValueError("Invalid candidate/matched ID")
    return ",".join(values)


def infer_shard(bundle_path, model_mode, test_dir, index_dir, output_dir, *, shard=0, shards=4,
                smoke_limit=100, allow_full_test=False, checkpoint_every=100):
    if not 0 <= shard < shards or shards != 4 or checkpoint_every < 1:
        raise ValueError("Use four shards, shard IDs 0..3 and a positive checkpoint interval")
    if smoke_limit is None and not allow_full_test:
        raise ValueError("Full TEST requires --allow-full-test")
    if smoke_limit is not None and not 0 < smoke_limit <= 1000:
        raise ValueError("Smoke inference is limited to 1..1000 S1")
    bundle = load_bundle(bundle_path)
    if model_mode not in {"learned", "rule"}:
        raise ValueError("model must be learned or rule")
    if (model_mode == "rule") != (bundle["probes"] == 14 and bundle["model_type"] == "rule"):
        raise ValueError("Rule mode requires the tuned compact_14 bundle; learned mode requires compact_18")
    if model_mode == "learned" and bundle["model_type"] == "rule":
        raise ValueError("Learned mode requires a supervised model bundle")
    test_dir, output_dir = Path(test_dir), Path(output_dir)
    directory = output_dir / f"shard_{shard:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    identity = {"bundle_sha256": sha256(bundle_path), "model": model_mode, "shards": shards,
                "smoke_limit": smoke_limit, "test_source1": fingerprint(test_dir / "test_source1.tsv"),
                "indexes": index_identity(index_dir), "format_version": 1}
    started = time.monotonic()
    with ExitStack() as stack:
        lock = stack.enter_context((directory / "worker.lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        indexes = stack.enter_context(open_indexes(index_dir, test_dir, "test", bundle["blocker_config"]))
        connection = sqlite3.connect(directory / "checkpoint.sqlite")
        stack.callback(connection.close)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        connection.execute("CREATE TABLE IF NOT EXISTS outputs (position INTEGER PRIMARY KEY, entity_id TEXT UNIQUE NOT NULL, matched TEXT NOT NULL, candidates TEXT NOT NULL)")
        saved = dict(connection.execute("SELECT key, value FROM meta"))
        config = json.dumps({**identity, "shard": shard}, sort_keys=True)
        if saved and saved.get("identity") != config:
            raise ValueError("Checkpoint identity mismatch: use a fresh output directory")
        if not saved:
            connection.execute("INSERT INTO meta VALUES ('identity', ?)", (config,))
            connection.commit()
        done_path = directory / "done.json"
        if saved.get("complete") == "true" and done_path.is_file():
            done = json.loads(done_path.read_text())
            if (all(done.get(k) == v for k, v in {**identity, "shard": shard}.items())
                    and set(done.get("files", {})) == {"matching_results.tsv", "candidate_pairs.tsv"}
                    and all((directory / name).is_file() and sha256(directory / name) == digest
                            for name, digest in done["files"].items())):
                return done  # No rescoring or rewriting valid published shard files.
        cursor = connection.execute("SELECT COALESCE(MAX(position), -1) FROM outputs").fetchone()[0]
        processed = 0
        if saved.get("complete") != "true":
            with threadpool_limits(limits=1):
                for position, record in source_records(test_dir / "test_source1.tsv", smoke_limit):
                    if position <= cursor or position % shards != shard:
                        continue
                    candidates, _ = candidate_pool(record, indexes, bundle["blocker_config"], rule_fallback=model_mode == "rule")
                    cids = [c.candidate_entity_id for c in candidates]
                    scores = scores_for(bundle["model_type"], bundle["model"], feature_matrix(record, candidates))
                    matches = accepted_ids(cids, scores, bundle["threshold"])
                    connection.execute("INSERT INTO outputs VALUES (?, ?, ?, ?)",
                                       (position, record.entity_id, id_list(matches), id_list(cids)))
                    processed += 1
                    if processed % checkpoint_every == 0:
                        connection.commit()
                        print(f"shard {shard}: {processed} new rows, input position {position}", flush=True)
            if (fingerprint(test_dir / "test_source1.tsv") != identity["test_source1"]
                    or index_identity(index_dir) != identity["indexes"]):
                raise ValueError("Input/index changed during inference; output not finalized")
            connection.execute("INSERT OR REPLACE INTO meta VALUES ('complete', 'true')")
            connection.commit()
        # Publishing is atomic per file, with done.json as the completion marker.
        for name, header, column in (("matching_results.tsv", MATCH_HEADER, "matched"),
                                     ("candidate_pairs.tsv", CANDIDATE_HEADER, "candidates")):
            temporary = directory / (name + ".building")
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(header)
                writer.writerows(connection.execute(f"SELECT entity_id, {column} FROM outputs ORDER BY position"))
            temporary.replace(directory / name)
        done = {**identity, "shard": shard, "rows": connection.execute("SELECT COUNT(*) FROM outputs").fetchone()[0],
                "seconds_this_invocation": time.monotonic() - started,
                "files": {name: sha256(directory / name) for name in ("matching_results.tsv", "candidate_pairs.tsv")}}
        temporary = directory / "done.building.json"
        temporary.write_text(json.dumps(done, sort_keys=True, indent=2) + "\n")
        temporary.replace(directory / "done.json")
    return done


def merge_shards(shard_dir, test_dir, output_dir, *, allow_full_test=False):
    """O(one S1 list) RAM. Disk UNIQUE constraint catches duplicate S1s globally."""
    shard_dir, test_dir, output_dir = Path(shard_dir), Path(test_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Merged output directory must be fresh")
    manifests = [json.loads((shard_dir / f"shard_{i:02d}/done.json").read_text()) for i in range(4)]
    keys = ("bundle_sha256", "model", "shards", "smoke_limit", "test_source1", "indexes", "format_version")
    identity = {k: manifests[0][k] for k in keys}
    for i, manifest in enumerate(manifests):
        if manifest["shard"] != i or manifest["shards"] != 4 or any(manifest[k] != identity[k] for k in keys):
            raise ValueError("Mixed/duplicate shard identities")
        if set(manifest["files"]) != {"matching_results.tsv", "candidate_pairs.tsv"}:
            raise ValueError("Incomplete shard file inventory")
        for name, digest in manifest["files"].items():
            if sha256(shard_dir / f"shard_{i:02d}" / name) != digest:
                raise ValueError("Shard checksum mismatch")
    if identity["smoke_limit"] is None and not allow_full_test:
        raise ValueError("Merging full TEST requires --allow-full-test")
    if fingerprint(test_dir / "test_source1.tsv") != identity["test_source1"]:
        raise ValueError("S1 source changed since inference")
    building = output_dir.with_name(output_dir.name + ".building")
    building.mkdir(parents=True)  # Never overwrite an interrupted merge.
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
            for i in range(4):
                handle = stack.enter_context((shard_dir / f"shard_{i:02d}" / name).open(encoding="utf-8", newline=""))
                reader = csv.reader(handle, delimiter="\t")
                if next(reader, None) != header:
                    raise ValueError("Bad shard header")
                readers[name].append(reader)
        counts = Counter()
        for position, record in source_records(test_dir / "test_source1.tsv", identity["smoke_limit"]):
            shard = position % 4
            seen.execute("INSERT INTO seen VALUES (?)", (record.entity_id,))
            lists = {}
            for name in writers:
                row = next(readers[name][shard], None)
                if row is None or len(row) != 2 or row[0] != record.entity_id:
                    raise ValueError("Missing/duplicate/out-of-order shard row")
                ids = row[1].split(",") if row[1] else []
                if id_list(ids) != row[1] or len(ids) != len(set(ids)):
                    raise ValueError("Invalid or duplicate target IDs")
                lists[name] = set(ids)
                writers[name].writerow(row)
            if not lists["matching_results.tsv"] <= lists["candidate_pairs.tsv"]:
                raise ValueError("Predictions are not a subset of scored candidates")
            counts[shard] += 1
        if any(next(reader, None) is not None for group in readers.values() for reader in group):
            raise ValueError("Unexpected extra shard rows")
        if any(counts[i] != manifests[i]["rows"] for i in range(4)):
            raise ValueError("Shard row-count mismatch")
        if fingerprint(test_dir / "test_source1.tsv") != identity["test_source1"]:
            raise ValueError("S1 source changed during merge")
        seen.commit()
    # Keep the small disk-backed uniqueness audit as a reproducibility artifact.
    (building / "manifest.json").write_text(json.dumps({**identity, "rows": sum(counts.values()),
                                                        "submission_ready_full_coverage": identity["smoke_limit"] is None}, indent=2) + "\n")
    building.replace(output_dir)
    return sum(counts.values())

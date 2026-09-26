"""EXP006: fixed EXP003 matcher with independently sampled TRAIN S1 entities."""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
import fcntl
import hashlib
import heapq
import json
from pathlib import Path
import sqlite3
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import SOURCE_COLUMNS, validate_schema
from .exp003_training import (FIT_SEED, MODEL_SEED, PROB_THRESHOLDS, PairWriter,
                              evaluate_one, pair_groups, sample_entity)
from .features import FEATURE_NAMES
from .modeling import make_hist_gradient_boosting_model, select_global_threshold
from .reranker import (ROOT, SUBSET_HASH, CONFIG_HASHES, candidate_pool, compatible_code_versions,
                       feature_matrix, fingerprint, frozen_config, index_identity,
                       make_bundle, open_indexes, save_bundle, scores_for, sha256, write_json_atomic)
from .sampling import SourceRecord, load_selected_ground_truth, load_selected_source_records
from .split import stable_entity_key, stratified_entity_split

SEED = 2032
FORMAT = 1
PAIR_HEADER = ["source1_entity_id", "candidate_entity_id", "label"]
DIAGNOSTIC_HEADER = ["source1_entity_id", "country", "true_links", "raw_pairs",
                     "raw_true_links", "eligible_pairs", "eligible_positives",
                     "sampled_pairs", "sampled_positives", "hard_negatives", "easy_negatives"]


def original_partitions(subset=ROOT / "results/tables/exp001_subset_ids.csv", *, authoritative=True):
    subset = Path(subset)
    if authoritative and sha256(subset) != SUBSET_HASH:
        raise ValueError("Original 20k exclusion hash differs from authoritative subset")
    table = pd.read_csv(subset, dtype={"source1_entity_id": str, "country": str})
    if authoritative and len(table) != 20000:
        raise ValueError("Original development universe must contain exactly 20,000 S1")
    if table.source1_entity_id.duplicated().any() or set(table.partition) != {"tuning", "evaluation"}:
        raise ValueError("Invalid original 20k subset")
    evaluation = set(table.loc[table.partition.eq("evaluation"), "source1_entity_id"])
    development = table.loc[table.partition.ne("evaluation")]
    records = [{"entity_id": row.source1_entity_id, "country": row.country} for row in development.itertuples()]
    truth = {row.source1_entity_id: frozenset({"S2-placeholder"}) if int(row.match_count) else frozenset()
             for row in development.itertuples()}
    split = stratified_entity_split(records, truth, validation_fraction=.25, seed=FIT_SEED)
    parts = {**dict.fromkeys(split.train_ids, "fit"), **dict.fromkeys(split.validation_ids, "tune"),
             **dict.fromkeys(evaluation, "evaluation")}
    if authoritative and Counter(parts.values()) != {"fit": 5999, "tune": 2001, "evaluation": 12000}:
        raise ValueError("Original partition counts changed")
    return parts


def _source_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        validate_schema(reader.fieldnames, SOURCE_COLUMNS)
        for row in reader:
            if None in row or any(v is None for v in row.values()) or not row["entity_id"].startswith("S1-"):
                raise ValueError("Malformed TRAIN S1 row")
            yield row


def _allocate(counts, size):
    total = sum(counts.values())
    if size < 1 or size > total:
        raise ValueError(f"Requested {size} EXTRA_FIT but only {total} eligible S1 exist")
    exact = {country: size * count / total for country, count in counts.items()}
    quotas = {c: int(value) for c, value in exact.items()}
    remainder = size - sum(quotas.values())
    for country in sorted(counts, key=lambda c: (-(exact[c] - quotas[c]), c))[:remainder]:
        quotas[country] += 1
    return quotas


def sample(data_dir, output_dir, *, size=50000, seed=SEED, subset=None, authoritative=True):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Use a fresh sample directory")
    parts = original_partitions(subset or ROOT / "results/tables/exp001_subset_ids.csv", authoritative=authoritative)
    excluded = set(parts)
    source = data_dir / "train_source1.tsv"
    truth = data_dir / "train_ground_truth.tsv"
    counts = Counter()
    seen = set()
    for row in _source_rows(source):
        eid = row["entity_id"]
        if eid in seen:
            raise ValueError("Duplicate TRAIN S1 ID")
        seen.add(eid)
        if eid not in excluded:
            counts[row["country"]] += 1
    if not excluded <= seen:
        raise ValueError("Original 20k IDs missing from TRAIN S1")
    quotas = _allocate(counts, size)
    heaps = defaultdict(list)
    for row in _source_rows(source):
        eid, country = row["entity_id"], row["country"]
        if eid in excluded or quotas[country] == 0:
            continue
        key = int.from_bytes(stable_entity_key(eid, seed), "big")
        item = (-key, eid)
        heap = heaps[country]
        if len(heap) < quotas[country]:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    selected = {eid for heap in heaps.values() for _, eid in heap}
    overlap = Counter(parts[eid] for eid in selected & excluded)
    if len(selected) != size or overlap:
        raise AssertionError("EXTRA_FIT size/overlap failure")
    selected_records = load_selected_source_records(source, "S1", selected)
    if Counter(r.country for r in selected_records.values()) != {c: n for c, n in quotas.items() if n}:
        raise ValueError("TRAIN S1 changed during stratified sampling")
    output_dir.mkdir(parents=True)
    sample_path = output_dir / "extra_fit.csv"
    with sample_path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["source1_entity_id", "country"])
        writer.writerows((eid, selected_records[eid].country) for eid in sorted(selected))
    manifest = {"format": FORMAT, "seed": seed, "requested_size": size, "actual_size": len(selected),
                "source_sha256": sha256(source), "ground_truth_sha256": sha256(truth),
                "source_identity": fingerprint(source), "ground_truth_identity": fingerprint(truth),
                "exclusion_sha256": sha256(subset or ROOT / "results/tables/exp001_subset_ids.csv"),
                "country_counts": dict(sorted(Counter(r.country for r in selected_records.values()).items())),
                "eligible_country_counts": dict(sorted(counts.items())), "overlap_original_20k": sum(overlap.values()),
                "overlap_fit": overlap["fit"], "overlap_tune": overlap["tune"],
                "overlap_evaluation": overlap["evaluation"],
                "sample_sha256": sha256(sample_path)}
    write_json_atomic(output_dir / "manifest.json", manifest)
    return manifest


def load_sample(sample_dir, data_dir, *, subset=None, authoritative=True):
    sample_dir, data_dir = Path(sample_dir), Path(data_dir)
    manifest = json.loads((sample_dir / "manifest.json").read_text())
    path = sample_dir / "extra_fit.csv"
    parts = original_partitions(subset or ROOT / "results/tables/exp001_subset_ids.csv", authoritative=authoritative)
    if (manifest["exclusion_sha256"] != sha256(subset or ROOT / "results/tables/exp001_subset_ids.csv")
            or manifest["sample_sha256"] != sha256(path)
            or manifest["source_identity"] != fingerprint(data_dir / "train_source1.tsv")
            or manifest["ground_truth_identity"] != fingerprint(data_dir / "train_ground_truth.tsv")
            or manifest["source_sha256"] != sha256(data_dir / "train_source1.tsv")
            or manifest["ground_truth_sha256"] != sha256(data_dir / "train_ground_truth.tsv")):
        raise ValueError("Sample/source identity mismatch")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [r["source1_entity_id"] for r in rows]
    if (len(ids) != manifest["actual_size"] or len(ids) != len(set(ids)) or ids != sorted(ids)
            or set(ids) & set(parts) or Counter(r["country"] for r in rows) != manifest["country_counts"]):
        raise ValueError("Sample count/order/country/overlap mismatch")
    return manifest, ids


def preflight(data_dir, index_dir, original_dir, *, subset=None, authoritative=True,
              include_evaluation=False):
    data_dir, index_dir, original_dir = Path(data_dir), Path(index_dir), Path(original_dir)
    parts = original_partitions(subset or ROOT / "results/tables/exp001_subset_ids.csv", authoritative=authoritative)
    original = json.loads((original_dir / "pairs/complete.json").read_text())
    if (original["identity"]["subset_sha256"] != sha256(subset or ROOT / "results/tables/exp001_subset_ids.csv")
            or original["identity"]["indexes"] != index_identity(index_dir)
            or not compatible_code_versions(original["identity"]["code"])
            or original["identity"]["inputs"] != {p.name: fingerprint(p) for p in
                 (data_dir / "train_source1.tsv", data_dir / "train_ground_truth.tsv")}):
        raise ValueError("Original EXP003 pair cache identity mismatch")
    for name, digest in original["files"].items():
        if "evaluation" in name and not include_evaluation:
            continue  # Do not read held-out pair contents before the model is frozen.
        if sha256(original_dir / "pairs" / name) != digest:
            raise ValueError(f"Original EXP003 pair cache corrupt: {name}")
    saved_parts = pd.read_csv(original_dir / "partitions.csv", dtype=str)
    if len(saved_parts) != len(parts) or dict(zip(saved_parts.source1_entity_id, saved_parts.partition)) != parts:
        raise ValueError("Original FIT/TUNE/EVALUATION IDs differ")
    with open_indexes(index_dir, data_dir, "train", frozen_config(18)):
        pass
    return {"partition_counts": dict(Counter(parts.values())), "feature_count": len(FEATURE_NAMES),
            "blocker_sha256": CONFIG_HASHES[18], "original_cache_verified": True}


def _identity(sample_dir, data_dir, index_dir, shards):
    manifest, ids = load_sample(sample_dir, data_dir)
    return {"format": FORMAT, "sample_sha256": manifest["sample_sha256"],
            "sample_manifest_sha256": sha256(Path(sample_dir) / "manifest.json"),
            "source_identity": manifest["source_identity"], "ground_truth_identity": manifest["ground_truth_identity"],
            "indexes": index_identity(index_dir), "blocker_sha256": CONFIG_HASHES[18],
            "feature_names": list(FEATURE_NAMES), "sampling_seed": manifest["seed"], "shards": shards}, ids


def _checkpoint_digest(connection):
    digest = hashlib.sha256()
    for position, eid, matrix, pairs, diagnostics in connection.execute(
            "SELECT position, entity_id, matrix, pairs, diagnostics FROM entities ORDER BY position"):
        for value in (str(position).encode(), eid.encode(), matrix, pairs.encode(), diagnostics.encode()):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    return digest.hexdigest()


def generate(sample_dir, data_dir, index_dir, shard_dir, *, shards=4, shard=0, checkpoint_every=100):
    if type(shards) is not int or not 1 <= shards <= 64 or type(shard) is not int or not 0 <= shard < shards or checkpoint_every < 1:
        raise ValueError("Require 1..64 shards, valid shard and positive checkpoint interval")
    sample_dir, data_dir, index_dir, shard_dir = map(Path, (sample_dir, data_dir, index_dir, shard_dir))
    identity, ids = _identity(sample_dir, data_dir, index_dir, shards)
    directory = shard_dir / f"shard_{shard:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with ExitStack() as stack:
        lock = stack.enter_context((directory / "worker.lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        connection = sqlite3.connect(directory / "checkpoint.sqlite")
        stack.callback(connection.close)
        connection.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("CREATE TABLE IF NOT EXISTS entities (position INTEGER PRIMARY KEY, entity_id TEXT UNIQUE NOT NULL, matrix BLOB NOT NULL, pairs TEXT NOT NULL, diagnostics TEXT NOT NULL)")
        config = json.dumps({**identity, "shard": shard}, sort_keys=True)
        saved = dict(connection.execute("SELECT key, value FROM meta"))
        if saved and saved.get("identity") != config:
            raise ValueError("Checkpoint identity mismatch")
        if not saved:
            connection.execute("INSERT INTO meta VALUES ('identity', ?)", (config,))
            connection.commit()
        expected = [(i, eid) for i, eid in enumerate(ids) if i % shards == shard]
        committed = list(connection.execute("SELECT position, entity_id FROM entities ORDER BY position"))
        if committed != expected[:len(committed)]:
            raise ValueError("Checkpoint positions/IDs mismatch")
        done_path = directory / "done.json"
        if done_path.exists():
            done = json.loads(done_path.read_text())
            if (done.get("identity") != {**identity, "shard": shard} or done.get("rows") != len(expected)
                    or committed != expected or done.get("checkpoint_sha256") != _checkpoint_digest(connection)):
                raise ValueError("Completed shard identity mismatch")
            return done
        remaining = expected[len(committed):]
        records = load_selected_source_records(data_dir / "train_source1.tsv", "S1", {eid for _, eid in remaining}) if remaining else {}
        truth = load_selected_ground_truth(data_dir / "train_ground_truth.tsv", records) if remaining else {}
        config18 = frozen_config(18)
        with open_indexes(index_dir, data_dir, "train", config18) as indexes, threadpool_limits(limits=1):
            for j, (position, eid) in enumerate(remaining, 1):
                candidates, raw_ids = candidate_pool(records[eid], indexes, config18)
                cids = [c.candidate_entity_id for c in candidates]
                matrix = feature_matrix(records[eid], candidates)
                labels = [int(cid in truth[eid]) for cid in cids]
                sampled, counts = sample_entity(eid, cids, matrix, labels)
                array = np.asarray([p.features for p in sampled], dtype=np.float32).reshape(-1, len(FEATURE_NAMES))
                pairs = [[p.candidate_entity_id, p.label] for p in sampled]
                diagnostic = [eid, records[eid].country, len(truth[eid]), len(raw_ids), len(raw_ids & truth[eid]),
                              len(cids), sum(labels), len(sampled), counts["positive_count"],
                              counts["hard_negative_count"], counts["easy_negative_count"]]
                connection.execute("INSERT INTO entities VALUES (?, ?, ?, ?, ?)",
                                   (position, eid, array.tobytes(), json.dumps(pairs, separators=(",", ":")),
                                    json.dumps(diagnostic, separators=(",", ":"))))
                if j % checkpoint_every == 0:
                    connection.commit()
                    print(f"shard {shard}: {len(committed)+j}/{len(expected)} S1", flush=True)
        if (_identity(sample_dir, data_dir, index_dir, shards)[0] != identity
                or fingerprint(data_dir / "train_source1.tsv") != identity["source_identity"]):
            raise ValueError("Input identity changed during shard generation")
        connection.commit()
        if list(connection.execute("SELECT position, entity_id FROM entities ORDER BY position")) != expected:
            raise ValueError("Incomplete shard")
        done = {"identity": {**identity, "shard": shard}, "rows": len(expected),
                "pairs": sum(len(json.loads(row[0])) for row in connection.execute("SELECT pairs FROM entities")),
                "checkpoint_sha256": _checkpoint_digest(connection)}
        write_json_atomic(done_path, done)
        print(f"shard {shard}: {len(expected)} total S1, {time.monotonic()-started:.3f} seconds this invocation", flush=True)
        return done


def merge(sample_dir, data_dir, index_dir, shard_dir, output_dir, *, shards=4):
    if type(shards) is not int or not 1 <= shards <= 64:
        raise ValueError("Require 1..64 shards")
    sample_dir, data_dir, index_dir, shard_dir, output_dir = map(Path, (sample_dir, data_dir, index_dir, shard_dir, output_dir))
    if output_dir.exists() or output_dir.with_name(output_dir.name + ".building").exists():
        raise FileExistsError("Use a fresh merged output directory")
    identity, ids = _identity(sample_dir, data_dir, index_dir, shards)
    building = output_dir.with_name(output_dir.name + ".building")
    connections = []
    with ExitStack() as stack:
        for shard in range(shards):
            directory = shard_dir / f"shard_{shard:02d}"
            done = json.loads((directory / "done.json").read_text())
            if done.get("identity") != {**identity, "shard": shard} or done.get("rows") != len(range(shard, len(ids), shards)):
                raise ValueError("Mixed/incomplete shard identity")
            connection = sqlite3.connect(f"file:{directory / 'checkpoint.sqlite'}?mode=ro", uri=True)
            stack.callback(connection.close)
            if dict(connection.execute("SELECT key, value FROM meta")).get("identity") != json.dumps({**identity, "shard": shard}, sort_keys=True):
                raise ValueError("Shard checkpoint identity mismatch")
            if done.get("checkpoint_sha256") != _checkpoint_digest(connection):
                raise ValueError("Shard checkpoint checksum mismatch")
            connections.append(connection)
        building.mkdir(parents=True)
        writer = PairWriter(building / "extra_fit")
        stack.callback(writer.close)
        diagnostics = stack.enter_context((building / "pool_counts.csv").open("w", newline=""))
        diag_writer = csv.writer(diagnostics, lineterminator="\n")
        diag_writer.writerow(DIAGNOSTIC_HEADER)
        cursors = [iter(c.execute("SELECT position, entity_id, matrix, pairs, diagnostics FROM entities ORDER BY position")) for c in connections]
        for position, eid in enumerate(ids):
            row = next(cursors[position % shards], None)
            if row is None or row[:2] != (position, eid):
                raise ValueError("Missing, duplicate or out-of-order shard entity")
            pairs = json.loads(row[3])
            matrix = np.frombuffer(row[2], dtype=np.float32)
            if matrix.size != len(pairs) * len(FEATURE_NAMES):
                raise ValueError("Shard matrix/metadata mismatch")
            writer.add(eid, [p[0] for p in pairs], matrix.reshape(-1, len(FEATURE_NAMES)), [p[1] for p in pairs])
            diag_writer.writerow(json.loads(row[4]))
        if any(next(cursor, None) is not None for cursor in cursors):
            raise ValueError("Unexpected extra shard entities")
        writer.close()
    if _identity(sample_dir, data_dir, index_dir, shards)[0] != identity:
        raise ValueError("Input identity changed during merge")
    frame = pd.read_csv(building / "pool_counts.csv")
    summary = {"entities": len(ids), "raw_candidate_recall": frame.raw_true_links.sum() / frame.true_links.sum() if frame.true_links.sum() else None,
               "eligible_candidate_recall": frame.eligible_positives.sum() / frame.true_links.sum() if frame.true_links.sum() else None,
               "positives_retained": int(frame.sampled_positives.sum()),
               "negatives_retained": int((frame.sampled_pairs - frame.sampled_positives).sum()),
               "average_raw_candidates_per_s1": float(frame.raw_pairs.mean()),
               "average_eligible_candidates_per_s1": float(frame.eligible_pairs.mean())}
    manifest = {"identity": identity, "summary": summary,
                "files": {name: sha256(building / name) for name in ("extra_fit.f32", "extra_fit.tsv", "pool_counts.csv")}}
    write_json_atomic(building / "manifest.json", manifest)
    building.rename(output_dir)
    return manifest


def _validate_merged(merged_dir, sample_dir, data_dir, index_dir):
    merged_dir = Path(merged_dir)
    manifest = json.loads((merged_dir / "manifest.json").read_text())
    identity = manifest["identity"]
    if identity != _identity(sample_dir, data_dir, index_dir, identity["shards"])[0]:
        raise ValueError("Merged pair identity mismatch")
    for name, digest in manifest["files"].items():
        if sha256(merged_dir / name) != digest:
            raise ValueError("Merged pair checksum mismatch")
    return manifest


def fit(data_dir, index_dir, original_dir, sample_dir, merged_dir, output_dir):
    data_dir, index_dir, original_dir, sample_dir, merged_dir, output_dir = map(Path, (data_dir, index_dir, original_dir, sample_dir, merged_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError("Use a fresh fit output directory")
    pre = preflight(data_dir, index_dir, original_dir)
    merged = _validate_merged(merged_dir, sample_dir, data_dir, index_dir)
    if merged["summary"]["entities"] != 50000:
        raise ValueError("Primary EXP006 fit requires 50,000 EXTRA_FIT S1")
    arrays, labels = [], []
    for path in (original_dir / "pairs/fit", merged_dir / "extra_fit"):
        size = path.with_suffix(".f32").stat().st_size
        if size % (4 * len(FEATURE_NAMES)):
            raise ValueError("Invalid pair matrix size")
        n = size // (4 * len(FEATURE_NAMES))
        arrays.append(np.memmap(path.with_suffix(".f32"), mode="r", dtype=np.float32, shape=(n, len(FEATURE_NAMES))))
        labels.append(pd.read_csv(path.with_suffix(".tsv"), sep="\t")["label"].to_numpy(dtype=np.int8))
        if len(labels[-1]) != n:
            raise ValueError("Pair labels/matrix count mismatch")
    x, y = np.concatenate(arrays), np.concatenate(labels)
    if set(y) != {0, 1}:
        raise ValueError("Training labels require both classes")
    model = make_hist_gradient_boosting_model(MODEL_SEED).set_params(early_stopping=False)
    with threadpool_limits(limits=1):
        model.fit(x, y)
    output_dir.mkdir(parents=True)
    joblib.dump(model, output_dir / "model.joblib", compress=3)
    fixed = make_hist_gradient_boosting_model(MODEL_SEED).set_params(early_stopping=False).get_params()
    manifest = {"model_config": fixed, "model_sha256": sha256(output_dir / "model.joblib"),
                "original_fit_pairs_sha256": sha256(original_dir / "pairs/fit.f32"),
                "extra_manifest_sha256": sha256(merged_dir / "manifest.json"),
                "sample_manifest_sha256": sha256(sample_dir / "manifest.json"),
                "partition_counts": pre["partition_counts"], "extra_entities": merged["summary"]["entities"],
                "training_pairs": int(len(y)), "feature_names": list(FEATURE_NAMES),
                "blocker_sha256": CONFIG_HASHES[18]}
    write_json_atomic(output_dir / "fit_manifest.json", manifest)
    return manifest


def select_tune_threshold(path, model, truth):
    """Select one global threshold using only an explicit TUNE truth mapping."""
    scores = {}
    for eid, cids, matrix, _ in pair_groups(path):
        if eid not in truth or eid in scores:
            raise ValueError("TUNE pair artifact contains non-TUNE/duplicate S1")
        scores[eid] = dict(zip(cids, map(float, scores_for("hist_gradient_boosting", model, matrix))))
    thresholds = sorted(set(PROB_THRESHOLDS) | {i / 1000 for i in range(500, 1001)})
    return select_global_threshold(truth, scores, thresholds)


def tune(data_dir, index_dir, original_dir, sample_dir, merged_dir, fit_dir, output_dir):
    data_dir, index_dir, original_dir, sample_dir, merged_dir, fit_dir, output_dir = map(Path, (data_dir, index_dir, original_dir, sample_dir, merged_dir, fit_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError("Use a fresh TUNE output directory")
    preflight(data_dir, index_dir, original_dir)
    _validate_merged(merged_dir, sample_dir, data_dir, index_dir)
    manifest = json.loads((fit_dir / "fit_manifest.json").read_text())
    expected_config = make_hist_gradient_boosting_model(MODEL_SEED).set_params(early_stopping=False).get_params()
    if (manifest["model_config"] != expected_config or manifest["model_sha256"] != sha256(fit_dir / "model.joblib")
            or manifest["extra_manifest_sha256"] != sha256(merged_dir / "manifest.json")
            or manifest["feature_names"] != list(FEATURE_NAMES)):
        raise ValueError("FIT model/config/feature identity mismatch")
    model = joblib.load(fit_dir / "model.joblib")
    parts = original_partitions()
    ids = {e for e, part in parts.items() if part == "tune"}
    truth = load_selected_ground_truth(data_dir / "train_ground_truth.tsv", ids)
    threshold, metric, grid = select_tune_threshold(original_dir / "pairs/tune", model, truth)
    output_dir.mkdir(parents=True)
    pd.DataFrame(grid).to_csv(output_dir / "thresholds.csv", index=False)
    bundle = make_bundle(model, "hist_gradient_boosting", threshold, 18, MODEL_SEED,
                         exp006_fit_manifest_sha256=sha256(fit_dir / "fit_manifest.json"),
                         extra_sample_sha256=json.loads((sample_dir / "manifest.json").read_text())["sample_sha256"],
                         tune_macro_fbeta=metric.macro_fbeta)
    save_bundle(bundle, output_dir / "frozen.joblib")
    frozen = {"threshold": threshold, "tune_macro_fbeta": metric.macro_fbeta,
              "tune_entities": len(ids), "fit_manifest_sha256": sha256(fit_dir / "fit_manifest.json"),
              "bundle_sha256": sha256(output_dir / "frozen.joblib"),
              "threshold_grid_sha256": sha256(output_dir / "thresholds.csv"),
              "evaluation_read": False}
    write_json_atomic(output_dir / "freeze.json", frozen)
    return frozen


def evaluate(data_dir, index_dir, original_dir, fit_dir, tune_dir, output_dir):
    from .reranker import load_bundle
    data_dir, index_dir, original_dir, fit_dir, tune_dir, output_dir = map(Path, (data_dir, index_dir, original_dir, fit_dir, tune_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError("EVALUATION output already exists; run exactly once")
    freeze = json.loads((tune_dir / "freeze.json").read_text())
    if (freeze["fit_manifest_sha256"] != sha256(fit_dir / "fit_manifest.json")
            or freeze["bundle_sha256"] != sha256(tune_dir / "frozen.joblib")
            or freeze["threshold_grid_sha256"] != sha256(tune_dir / "thresholds.csv")
            or freeze["evaluation_read"] is not False):
        raise ValueError("Model/threshold were not frozen before EVALUATION")
    bundle = load_bundle(tune_dir / "frozen.joblib")
    if bundle["threshold"] != freeze["threshold"] or bundle["exp006_fit_manifest_sha256"] != freeze["fit_manifest_sha256"]:
        raise ValueError("Frozen bundle mismatch")
    preflight(data_dir, index_dir, original_dir, include_evaluation=True)
    parts = original_partitions()
    ids = {e for e, part in parts.items() if part == "evaluation"}
    truth = load_selected_ground_truth(data_dir / "train_ground_truth.tsv", ids)
    records = load_selected_source_records(data_dir / "train_source1.tsv", "S1", ids)
    rows = evaluate_one(original_dir / "pairs/evaluation", bundle["model_type"], bundle["model"], bundle["threshold"], truth, records)
    overall = next(r for r in rows if r["dimension"] == "overall")
    output_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(output_dir / "evaluation.csv", index=False)
    summary = {"overall": overall, "baseline_exp003_macro_fbeta": .905874,
               "promotion": bool(overall["macro_fbeta"] > .905874),
               "freeze_sha256": sha256(tune_dir / "freeze.json"),
               "evaluation_csv_sha256": sha256(output_dir / "evaluation.csv")}
    write_json_atomic(output_dir / "summary.json", summary)
    return summary

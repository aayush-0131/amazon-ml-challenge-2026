"""Staged TRAIN-only EXP007 enhanced matcher and frozen entity decision policy."""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
from pathlib import Path
import platform
import sqlite3

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import exp006
from .evaluation import evaluate_predictions
from .exp003_training import MODEL_SEED, SAMPLE_SEED, metric_slices, sample_entity
from .exp007_features import FEATURE_NAMES, ENHANCED_FEATURE_NAMES, NORMALIZATION_VERSION, feature_row, enhanced_row
from .features import build_feature_row
from .ground_truth import iter_ground_truth
from .modeling import make_hist_gradient_boosting_model
from .reranker import (ROOT, CONFIG_HASHES, candidate_pool, code_versions, fingerprint,
                       frozen_config, git_commit, index_identity, open_indexes, sha256,
                       write_json_atomic)
from .sampling import load_selected_ground_truth, load_selected_source_records
from .split import stable_entity_key

FORMAT = 1
PAIR_COLUMNS = ("source1_entity_id", "candidate_entity_id", "label", "candidate_name", "candidate_address")
CODE_FILES = ("exp007.py", "exp007_features.py")
GRID = tuple(i / 100 for i in range(50, 85)) + tuple(i / 1000 for i in range(850, 1001)) + (1.000001,)


def code_hashes():
    return {name: sha256(Path(__file__).parent / name) for name in CODE_FILES}


def identity(data_dir, index_dir, original_dir, sample_dir):
    exp006.preflight(data_dir, index_dir, original_dir)
    sample, ids = exp006.load_sample(sample_dir, data_dir)
    if len(ids) != 50000 or sample["seed"] != exp006.SEED:
        raise ValueError("EXP007 requires the verified EXP006 50k EXTRA_FIT sample")
    return {"format": FORMAT, "blocker_sha256": CONFIG_HASHES[18],
            "indexes": index_identity(index_dir), "subset_sha256": exp006.SUBSET_HASH,
            "sample_manifest_sha256": sha256(Path(sample_dir) / "manifest.json"),
            "original_complete_sha256": sha256(Path(original_dir) / "pairs/complete.json"),
            "source_sha256": sha256(Path(data_dir) / "train_source1.tsv"),
            "truth_sha256": sha256(Path(data_dir) / "train_ground_truth.tsv"),
            "feature_names": list(FEATURE_NAMES), "code_sha256": code_hashes(),
            "legacy_code_sha256": code_versions()}, ids


def partition_ids(partition, sample_ids):
    parts = exp006.original_partitions()
    if partition == "fit":
        return sorted([e for e, p in parts.items() if p == "fit"] + sample_ids)
    if partition in ("tune", "evaluation"):
        return sorted(e for e, p in parts.items() if p == partition)
    raise ValueError("partition must be fit, tune or evaluation")


def preflight(data_dir, index_dir, original_dir, sample_dir):
    manifest, ids = identity(data_dir, index_dir, original_dir, sample_dir)
    return {"identity": manifest, "partition_counts": {p: len(partition_ids(p, ids)) for p in ("fit", "tune", "evaluation")},
            "old_feature_count": len(FEATURE_NAMES)-len(ENHANCED_FEATURE_NAMES),
            "new_feature_count": len(ENHANCED_FEATURE_NAMES), "feature_count": len(FEATURE_NAMES)}


def ownership_audit(ground_truth):
    owners = defaultdict(set)
    links = 0
    seen_s1 = set()
    for eid, matches in ground_truth:
        if eid in seen_s1:
            raise ValueError("Duplicate S1 in complete TRAIN truth")
        seen_s1.add(eid)
        for cid in matches:
            owners[cid].add(eid)
            links += 1
    conflicts = {cid: sorted(v) for cid, v in owners.items() if len(v) > 1}
    return {"total_ground_truth_links": links, "unique_target_ids": len(owners),
            "targets_with_multiple_owners": len(conflicts),
            "max_s1_owners": max(map(len, owners.values()), default=0),
            "s2_conflicts": sum(k.startswith("S2-") for k in conflicts),
            "s3_conflicts": sum(k.startswith("S3-") for k in conflicts),
            "examples": [{"target_id": k, "owners": conflicts[k]} for k in sorted(conflicts)[:20]]}


def audit_ownership(data_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    path = Path(data_dir) / "train_ground_truth.tsv"
    result = ownership_audit(iter_ground_truth(path))
    result["ground_truth_sha256"] = sha256(path)
    write_json_atomic(output_dir / "target_ownership_audit.json", result)
    return result


def oracle(truth, eligible, records):
    if set(truth) != set(eligible):
        raise ValueError("Oracle requires complete eligible candidate pools")
    prediction = {e: set(truth[e]) & set(eligible[e]) for e in truth}
    slices = metric_slices(truth, prediction, eligible, records)
    overall = next(r for r in slices if r["dimension"] == "overall")
    true_links = sum(map(len, truth.values()))
    matched = [e for e in truth if truth[e]]
    return {"overall": overall, "slices": slices,
            "candidate_link_recall": sum(map(len, prediction.values())) / true_links if true_links else 1.0,
            "perfect_candidate_coverage_rate": sum(set(truth[e]) <= set(eligible[e]) for e in truth) / len(truth),
            "matched_with_zero_retained_rate": sum(not prediction[e] for e in matched) / len(matched) if matched else 0.0,
            "blocker_fn": true_links - sum(map(len, prediction.values())),
            "candidate_generation_material_bottleneck": overall["macro_fbeta"] < .985,
            "matcher_decision_priority_if_large_gap": overall["macro_fbeta"] >= .990}


def pair_groups(path):
    path = Path(path)
    size = path.with_suffix(".f32").stat().st_size
    width = len(FEATURE_NAMES)
    if size % (4 * width):
        raise ValueError("Corrupt EXP007 pair matrix")
    n = size // (4 * width)
    matrix = np.memmap(path.with_suffix(".f32"), dtype=np.float32, mode="r", shape=(n, width)) if n else np.empty((0, width), np.float32)
    offset = 0
    with path.with_suffix(".tsv").open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != PAIR_COLUMNS:
            raise ValueError("EXP007 pair metadata schema mismatch")
        for eid, group in itertools.groupby(reader, key=lambda r: r["source1_entity_id"]):
            rows = list(group)
            end = offset + len(rows)
            if end > n or len({r["candidate_entity_id"] for r in rows}) != len(rows):
                raise ValueError("Corrupt EXP007 candidate group")
            yield eid, rows, matrix[offset:end]
            offset = end
    if offset != n:
        raise ValueError("EXP007 matrix/metadata count mismatch")


def _checkpoint_digest(connection):
    digest = hashlib.sha256()
    for position, eid, matrix, rows in connection.execute("SELECT position, entity_id, matrix, rows FROM entities ORDER BY position"):
        for value in (str(position).encode(), eid.encode(), matrix, rows.encode()):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    return digest.hexdigest()


def generate(data_dir, index_dir, original_dir, sample_dir, shard_dir, *, partition, shards=4, shard=0, checkpoint_every=100, tune_dir=None):
    if not 1 <= shards <= 64 or not 0 <= shard < shards or checkpoint_every < 1:
        raise ValueError("Invalid shard configuration")
    if partition == "evaluation":
        if tune_dir is None:
            raise ValueError("Frozen TUNE bundle required before EVALUATION pair generation")
        load_frozen_bundle(tune_dir, data_dir=data_dir, index_dir=index_dir,
                           original_dir=original_dir, sample_dir=sample_dir)
    base, sample_ids = identity(data_dir, index_dir, original_dir, sample_dir)
    ids = partition_ids(partition, sample_ids)
    manifest = {**base, "partition": partition, "shards": shards, "shard": shard}
    directory = Path(shard_dir) / partition / f"shard_{shard:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        import fcntl
        lock = stack.enter_context((directory / "worker.lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        conn = sqlite3.connect(directory / "checkpoint.sqlite")
        stack.callback(conn.close)
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS entities (position INTEGER PRIMARY KEY, entity_id TEXT UNIQUE NOT NULL, matrix BLOB NOT NULL, rows TEXT NOT NULL)")
        saved = dict(conn.execute("SELECT key, value FROM meta"))
        encoded = json.dumps(manifest, sort_keys=True)
        if saved and saved != {"identity": encoded}:
            raise ValueError("EXP007 checkpoint identity mismatch")
        if not saved:
            conn.execute("INSERT INTO meta VALUES ('identity', ?)", (encoded,))
            conn.commit()
        expected = [(i, e) for i, e in enumerate(ids) if i % shards == shard]
        committed = list(conn.execute("SELECT position, entity_id FROM entities ORDER BY position"))
        if committed != expected[:len(committed)]:
            raise ValueError("EXP007 checkpoint IDs/order mismatch")
        done_path = directory / "done.json"
        if done_path.exists():
            done = json.loads(done_path.read_text())
            if done != {"identity": manifest, "rows": len(expected), "checkpoint_sha256": _checkpoint_digest(conn)} or committed != expected:
                raise ValueError("Completed EXP007 shard corrupt")
            return done
        remaining = expected[len(committed):]
        records = load_selected_source_records(Path(data_dir) / "train_source1.tsv", "S1", {e for _, e in remaining}) if remaining else {}
        truth = load_selected_ground_truth(Path(data_dir) / "train_ground_truth.tsv", records) if remaining else {}
        config = frozen_config(18)
        with open_indexes(index_dir, data_dir, "train", config) as indexes, threadpool_limits(limits=1):
            for j, (position, eid) in enumerate(remaining, 1):
                candidates, _ = candidate_pool(records[eid], indexes, config)
                cids = [c.candidate_entity_id for c in candidates]
                labels = [int(cid in truth[eid]) for cid in cids]
                if partition == "fit":
                    old = np.asarray([list(build_feature_row(records[eid], c).values()) for c in candidates], dtype=np.float32).reshape(-1, 51)
                    selected, _ = sample_entity(eid, cids, old, labels)
                    keep = {p.candidate_entity_id for p in selected}
                    if {cid for cid, label in zip(cids, labels) if label} - keep:
                        raise ValueError("EXP007 FIT sampler dropped an eligible positive")
                    candidates = [c for c in candidates if c.candidate_entity_id in keep]
                    # Preserve EXP006 hard/easy negative ordering and selected set.
                    if len(candidates) != len(selected):
                        raise ValueError("FIT sampling mismatch")
                rows = [[eid, c.candidate_entity_id, int(c.candidate_entity_id in truth[eid]), c.candidate_name, c.candidate_address] for c in candidates]
                matrix = np.asarray([list(feature_row(records[eid], c).values()) for c in candidates], dtype=np.float32).reshape(-1, len(FEATURE_NAMES))
                conn.execute("INSERT INTO entities VALUES (?, ?, ?, ?)", (position, eid, matrix.tobytes(), json.dumps(rows, separators=(",", ":"))))
                if j % checkpoint_every == 0:
                    conn.commit()
                    print(f"EXP007 {partition} shard {shard}: {len(committed)+j}/{len(expected)}", flush=True)
        conn.commit()
        if identity(data_dir, index_dir, original_dir, sample_dir)[0] != base:
            raise ValueError("Input identity changed during EXP007 generation")
        done = {"identity": manifest, "rows": len(expected), "checkpoint_sha256": _checkpoint_digest(conn)}
        write_json_atomic(done_path, done)
        return done


def merge(data_dir, index_dir, original_dir, sample_dir, shard_dir, output_dir, *, partition, shards=4):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Use a fresh EXP007 merged partition directory")
    base, sample_ids = identity(data_dir, index_dir, original_dir, sample_dir)
    ids = partition_ids(partition, sample_ids)
    with ExitStack() as stack:
        connections = []
        for shard in range(shards):
            directory = Path(shard_dir) / partition / f"shard_{shard:02d}"
            manifest = {**base, "partition": partition, "shards": shards, "shard": shard}
            done = json.loads((directory / "done.json").read_text())
            conn = sqlite3.connect(f"file:{directory / 'checkpoint.sqlite'}?mode=ro", uri=True)
            stack.callback(conn.close)
            if dict(conn.execute("SELECT key, value FROM meta")) != {"identity": json.dumps(manifest, sort_keys=True)}:
                raise ValueError("EXP007 checkpoint metadata identity mismatch")
            expected = [(i, e) for i, e in enumerate(ids) if i % shards == shard]
            if list(conn.execute("SELECT position, entity_id FROM entities ORDER BY position")) != expected:
                raise ValueError("EXP007 shard IDs/order mismatch")
            if done != {"identity": manifest, "rows": len(range(shard, len(ids), shards)), "checkpoint_sha256": _checkpoint_digest(conn)}:
                raise ValueError("EXP007 shard not complete or corrupt")
            connections.append(conn)
        output_dir.mkdir(parents=True)
        matrix_file = stack.enter_context((output_dir / f"{partition}.f32").open("wb"))
        meta = stack.enter_context((output_dir / f"{partition}.tsv").open("w", newline=""))
        writer = csv.writer(meta, delimiter="\t", lineterminator="\n")
        writer.writerow(PAIR_COLUMNS)
        cursors = [iter(c.execute("SELECT position, entity_id, matrix, rows FROM entities ORDER BY position")) for c in connections]
        pairs = 0
        for position, eid in enumerate(ids):
            row = next(cursors[position % shards], None)
            if row is None or row[:2] != (position, eid):
                raise ValueError("Missing EXP007 merged entity")
            matrix_file.write(row[2])
            records = json.loads(row[3])
            writer.writerows(records)
            pairs += len(records)
        if any(next(cursor, None) is not None for cursor in cursors):
            raise ValueError("EXP007 shard has unexpected extra rows")
    result = {"identity": base, "partition": partition, "entities": len(ids), "pairs": pairs,
              "matrix_sha256": sha256(output_dir / f"{partition}.f32"),
              "metadata_sha256": sha256(output_dir / f"{partition}.tsv")}
    write_json_atomic(output_dir / "manifest.json", result)
    return result


def validate_pairs(path, expected_partition, base):
    path = Path(path)
    saved = json.loads((path / "manifest.json").read_text())
    if saved["partition"] != expected_partition or saved["identity"] != base:
        raise ValueError("EXP007 partition identity mismatch")
    for suffix, key in (("f32", "matrix_sha256"), ("tsv", "metadata_sha256")):
        if saved[key] != sha256(path / f"{expected_partition}.{suffix}"):
            raise ValueError("EXP007 partition checksum mismatch")
    return saved


def blocker_oracle(data_dir, index_dir, original_dir, sample_dir, output_dir, *, partition):
    if partition not in ("tune", "evaluation"):
        raise ValueError("Oracle partition must be tune or evaluation")
    if partition == "evaluation":
        exp006.preflight(data_dir, index_dir, original_dir, include_evaluation=True)
    base, sample_ids = identity(data_dir, index_dir, original_dir, sample_dir)
    ids = partition_ids(partition, sample_ids)
    truth = load_selected_ground_truth(Path(data_dir) / "train_ground_truth.tsv", ids)
    records = load_selected_source_records(Path(data_dir) / "train_source1.tsv", "S1", set(ids))
    # EXP003's complete TUNE/EVALUATION pair TSV is the frozen eligible compact18 pool.
    eligible = {}
    with (Path(original_dir) / "pairs" / f"{partition}.tsv").open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for eid, group in itertools.groupby(reader, key=lambda r: r["source1_entity_id"]):
            if eid in eligible or eid not in truth:
                raise ValueError("Oracle pair entity mismatch")
            rows = list(group)
            cids = [r["candidate_entity_id"] for r in rows]
            if len(cids) != len(set(cids)) or any(int(r["label"]) != int(r["candidate_entity_id"] in truth[eid]) for r in rows):
                raise ValueError("Oracle eligible pool duplicated or labels disagree with truth")
            eligible[eid] = set(cids)
    for eid in ids:
        eligible.setdefault(eid, set())
    result = oracle(truth, eligible, records)
    result["partition"] = partition
    result["input_identity"] = base
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / f"blocker_oracle_{partition}.json", result)
    return {k: v for k, v in result.items() if k != "slices"}


ENTITY_FEATURE_NAMES = ("top", "second", "third", "top_second", "second_third", "candidate_count",
    "above_50", "above_80", "above_95", "best_s2", "best_s3", "s2_margin", "s3_margin",
    "top_retrieval", "top_pass_count", "top_exact_name", "top_core_name", "top_address",
    "top_numeric_conflict", "top_is_s3", "both_sources_strong", "cross_name_agreement")
RELATIVE_FEATURE_NAMES = ("score", "rank", "source_rank", "top", "second", "third", "top_second",
    "second_third", "minus_top", "above_50", "above_80", "above_95", "strong_s2", "strong_s3",
    "source_top_margin", "cross_name", "cross_core", "cross_address", "cross_numeric_agreement", "cross_numeric_conflict")


def _ordered(scores, rows):
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), rows[i][1]))


def cross_evidence(rows, scores, *, top_k=3):
    """Only compare at most top_k squared S2/S3 pairs; return per-candidate evidence."""
    if top_k < 1 or top_k > 5:
        raise ValueError("cross-source top_k must be 1..5")
    by_source = {s: sorted((i for i, r in enumerate(rows) if r[1].startswith(s + "-")),
                            key=lambda i: (-float(scores[i]), rows[i][1]))[:top_k] for s in ("S2", "S3")}
    result = np.zeros((len(rows), 5), np.float32)
    comparisons = 0
    for i in by_source["S2"]:
        for j in by_source["S3"]:
            comparisons += 1
            p = enhanced_row(rows[i][3], rows[j][3], rows[i][4], rows[j][4])
            values = np.asarray([p["trans_name_ratio"], p["core_name_token_set"], p["trans_address_ratio"],
                                 p["primary_number_exact"], p["primary_number_mismatch"]], np.float32)
            result[i] = np.maximum(result[i], values)
            result[j] = np.maximum(result[j], values)
    return result, comparisons


def decision_features(rows, matrix, scores):
    if len(rows) != len(matrix) or len(rows) != len(scores):
        raise ValueError("Entity score shape mismatch")
    order = _ordered(scores, rows)
    ranked = [float(scores[i]) for i in order]
    top, second, third = (ranked + [0.0, 0.0, 0.0])[:3]
    sorder = {s: [i for i in order if rows[i][1].startswith(s + "-")] for s in ("S2", "S3")}
    src_scores = {s: [float(scores[i]) for i in sorder[s]] for s in sorder}
    source_rank = {}
    for s in sorder:
        source_rank.update({i: rank for rank, i in enumerate(sorder[s], 1)})
    rank = {i: n for n, i in enumerate(order, 1)}
    cross, comparisons = cross_evidence(rows, scores)
    idx = {name: FEATURE_NAMES.index(name) for name in ("best_retrieval_score", "pass_count", "normalized_name_exact",
          "core_name_exact", "normalized_address_exact", "primary_number_mismatch")}
    best_s2, best_s3 = (src_scores[s][0] if src_scores[s] else 0 for s in ("S2", "S3"))
    top_matrix = matrix[order[0]] if order else np.zeros(len(FEATURE_NAMES))
    entity = np.asarray([top, second, third, top-second, second-third, len(rows),
        sum(v >= .5 for v in ranked), sum(v >= .8 for v in ranked), sum(v >= .95 for v in ranked),
        best_s2, best_s3, (src_scores["S2"][0]-src_scores["S2"][1]) if len(src_scores["S2"]) > 1 else 0,
        (src_scores["S3"][0]-src_scores["S3"][1]) if len(src_scores["S3"]) > 1 else 0,
        top_matrix[idx["best_retrieval_score"]], top_matrix[idx["pass_count"]],
        top_matrix[idx["normalized_name_exact"]], top_matrix[idx["core_name_exact"]],
        top_matrix[idx["normalized_address_exact"]], top_matrix[idx["primary_number_mismatch"]],
        float(bool(order and rows[order[0]][1].startswith("S3-"))),
        float(best_s2 >= .8 and best_s3 >= .8), float(cross[order[0], 0]) if order else 0], np.float32)
    relative = np.empty((len(rows), len(RELATIVE_FEATURE_NAMES)), np.float32)
    for i, row in enumerate(rows):
        source = "S2" if row[1].startswith("S2-") else "S3"
        own_top = src_scores[source][0] if src_scores[source] else 0
        relative[i] = (scores[i], rank[i], source_rank[i], top, second, third, top-second,
            second-third, scores[i]-top, entity[6], entity[7], entity[8],
            sum(v >= .9 for v in src_scores["S2"]), sum(v >= .9 for v in src_scores["S3"]),
            scores[i]-own_top, *cross[i])
    return entity, relative, comparisons


def score_group(model, rows, matrix, *, relative_model=None, cross=False):
    scores = model.predict_proba(matrix)[:, 1] if len(matrix) else np.empty(0)
    entity, relative, comparisons = decision_features(rows, matrix, scores)
    if relative_model is not None and len(rows):
        width = len(RELATIVE_FEATURE_NAMES) if cross else len(RELATIVE_FEATURE_NAMES)-5
        scores = relative_model.predict_proba(relative[:, :width])[:, 1]
        entity, relative, comparisons = decision_features(rows, matrix, scores)
    return np.asarray(scores, np.float64), entity, comparisons


def fit(data_dir, index_dir, original_dir, sample_dir, fit_pairs_dir, output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Use a fresh EXP007 fit directory")
    base, sample_ids = identity(data_dir, index_dir, original_dir, sample_dir)
    saved = validate_pairs(fit_pairs_dir, "fit", base)
    path = Path(fit_pairs_dir) / "fit"
    n = path.with_suffix(".f32").stat().st_size // (4 * len(FEATURE_NAMES))
    x = np.memmap(path.with_suffix(".f32"), dtype=np.float32, mode="r", shape=(n, len(FEATURE_NAMES)))
    y = np.empty(n, np.int8)
    original_fit = {e for e, p in exp006.original_partitions().items() if p == "fit"}
    original_truth = load_selected_ground_truth(Path(data_dir) / "train_ground_truth.tsv", original_fit)
    fit_groups = []
    offset = 0
    for eid, rows, matrix in pair_groups(path):
        end = offset + len(rows)
        y[offset:end] = [int(r[2]) for r in rows]
        if eid in original_fit:
            fit_groups.append((eid, rows, np.array(matrix), offset, end))
        offset = end
    present = {eid for eid, *_ in fit_groups}
    fit_groups.extend((eid, [], np.empty((0, len(FEATURE_NAMES)), np.float32), 0, 0) for eid in sorted(original_fit-present))
    if offset != n or set(y) != {0, 1}:
        raise ValueError("EXP007 FIT matrix/labels incomplete")
    # Entity-isolated two-fold scores for all learned decision features.
    oof_entity, oof_rel, oof_cross, oof_labels = [], [], [], []
    fold = {eid: int.from_bytes(stable_entity_key(eid, SAMPLE_SEED), "big") % 2 for eid in original_fit}
    for held_out in (0, 1):
        training = [(r, m) for eid, r, m, _, _ in fit_groups if fold[eid] != held_out]
        xx = np.concatenate([m for _, m in training])
        yy = np.fromiter((int(row[2]) for rows, _ in training for row in rows), np.int8)
        small = make_hist_gradient_boosting_model(MODEL_SEED).set_params(early_stopping=False)
        with threadpool_limits(limits=1):
            small.fit(xx, yy)
        for eid, rows, matrix, _, _ in fit_groups:
            if fold[eid] != held_out:
                continue
            scores, entity, _ = score_group(small, rows, matrix)
            _, relative, _ = decision_features(rows, matrix, scores)
            oof_entity.append((eid, entity, int(bool(original_truth[eid]))))
            oof_rel.append(relative)
            oof_cross.extend(int(r[2]) for r in rows)
    guardian = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=MODEL_SEED))
    guardian.fit(np.asarray([v for _, v, _ in oof_entity]), np.asarray([label for _, _, label in oof_entity]))
    rel_x = np.concatenate(oof_rel)
    rel_y = np.asarray(oof_cross)
    relative_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=MODEL_SEED))
    cross_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=MODEL_SEED))
    relative_model.fit(rel_x[:, :-5], rel_y)
    cross_model.fit(rel_x, rel_y)
    model = make_hist_gradient_boosting_model(MODEL_SEED).set_params(early_stopping=False)
    with threadpool_limits(limits=1):
        model.fit(x, y)
    output_dir.mkdir(parents=True)
    models = {"model": model, "guardian": guardian, "relative": relative_model, "relative_cross": cross_model}
    joblib.dump(models, output_dir / "models.joblib", compress=3)
    manifest = {"identity": base, "fit_pairs_manifest_sha256": sha256(Path(fit_pairs_dir) / "manifest.json"),
        "models_sha256": sha256(output_dir / "models.joblib"), "training_pairs": n,
        "fit_entities": len(partition_ids("fit", sample_ids)), "oof_entities": len(oof_entity),
        "oof_protocol": "2 entity-isolated folds of original FIT; EXP006 sampled candidates; seed 2031",
        "feature_names": list(FEATURE_NAMES), "entity_feature_names": ENTITY_FEATURE_NAMES,
        "relative_feature_names": RELATIVE_FEATURE_NAMES, "model_seed": MODEL_SEED,
        "package_versions": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "RapidFuzz", "scikit-learn", "anyascii")},
        "git_commit": git_commit(), "python_version": platform.python_version(),
        "license_notes": "Project-trained models MIT; scikit-learn BSD-3-Clause; AnyAscii ISC"}
    write_json_atomic(output_dir / "fit_manifest.json", manifest)
    return {k: v for k, v in manifest.items() if k != "feature_names"}


def _load_models(fit_dir, base):
    manifest = json.loads((Path(fit_dir) / "fit_manifest.json").read_text())
    if manifest["identity"] != base or manifest["models_sha256"] != sha256(Path(fit_dir) / "models.joblib") or manifest["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("EXP007 FIT model identity/corruption mismatch")
    if manifest["package_versions"] != {p: importlib.metadata.version(p) for p in manifest["package_versions"]}:
        raise ValueError("EXP007 package versions changed")
    return joblib.load(Path(fit_dir) / "models.joblib"), manifest


def _scored_partition(pair_path, ids, models):
    result = {}
    for eid, rows, matrix in pair_groups(pair_path):
        if eid not in ids or eid in result:
            raise ValueError("EXP007 scored partition ID mismatch")
        if any(int(r[2]) not in (0, 1) for r in rows):
            raise ValueError("Invalid EXP007 pair label")
        base_scores, entity, comparisons = score_group(models["model"], rows, matrix)
        if not np.isfinite(base_scores).all():
            raise ValueError("Non-finite EXP007 HGB scores")
        _, relative, _ = decision_features(rows, matrix, base_scores)
        variants = {"base": base_scores}
        if len(rows):
            variants["relative"] = models["relative"].predict_proba(relative[:, :-5])[:, 1]
            variants["cross"] = models["relative_cross"].predict_proba(relative)[:, 1]
        else:
            variants.update({"relative": np.empty(0), "cross": np.empty(0)})
        if any(not np.isfinite(values).all() for values in variants.values()):
            raise ValueError("Non-finite EXP007 decision scores")
        guardian_has_match = float(models["guardian"].predict_proba(entity.reshape(1, -1))[0, 1])
        numeric = matrix[:, FEATURE_NAMES.index("primary_number_mismatch")].astype(np.float64)
        result[eid] = {"rows": rows, "scores": variants, "numeric": numeric,
                       "guardian_has_match": guardian_has_match, "cross_comparisons": comparisons}
    for eid in ids:
        result.setdefault(eid, {"rows": [], "scores": {v: np.empty(0) for v in ("base", "relative", "cross")},
                                "numeric": np.empty(0), "guardian_has_match": 0.0, "cross_comparisons": 0})
    if set(result) != set(ids):
        raise ValueError("Incomplete EXP007 scored partition")
    return result


def _select_thresholds(scored, truth, variant, *, global_only=False, penalty=0.0):
    ids = sorted(truth)
    n = len(ids)
    true_count = np.asarray([len(truth[e]) for e in ids], np.int32)
    counts = {}
    for source in ("S2", "S3"):
        tp, fp = np.zeros((len(GRID), n), np.int32), np.zeros((len(GRID), n), np.int32)
        for j, eid in enumerate(ids):
            item = scored[eid]
            for i, row in enumerate(item["rows"]):
                if not row[1].startswith(source + "-"):
                    continue
                score = float(item["scores"][variant][i] - penalty * item["numeric"][i])
                hit = np.asarray(GRID) <= score
                (tp if int(row[2]) else fp)[:, j] += hit
        counts[source] = (tp, fp)

    best = None
    for a in range(len(GRID)):
        bs = [a] if global_only else range(len(GRID))
        left_tp, left_fp = counts["S2"][0][a], counts["S2"][1][a]
        tp = counts["S3"][0][bs] + left_tp
        fp = counts["S3"][1][bs] + left_fp
        fn = true_count - tp
        denom = 1.25*tp + .25*fn + fp
        values = np.divide(1.25*tp, denom, out=np.zeros_like(tp, dtype=np.float64), where=denom > 0)
        values[:, true_count == 0] = (fp[:, true_count == 0] == 0)
        means = values.mean(axis=1)
        for k, b in enumerate(bs):
            key = (round(float(means[k]), 12), GRID[a]+GRID[b], GRID[a], GRID[b])
            if best is None or key > best[0]:
                best = (key, (GRID[a], GRID[b], float(means[k])))
    return best[1]


def _predict(scored, variant, thresholds, *, guardian=False, guardian_threshold=0.0,
             exclusivity=False, penalty=0.0):
    predictions = {}
    claims = defaultdict(list)
    for eid in sorted(scored):
        item = scored[eid]
        accepted = set()
        if not guardian or item["guardian_has_match"] > guardian_threshold:
            for i, row in enumerate(item["rows"]):
                source = 0 if row[1].startswith("S2-") else 1
                score = float(item["scores"][variant][i] - penalty*item["numeric"][i])
                if score >= thresholds[source]:
                    accepted.add(row[1])
                    claims[row[1]].append((score, eid))
        predictions[eid] = accepted
    before = sum(len(v)-1 for v in claims.values() if len(v) > 1)
    if exclusivity:
        for cid, claimants in claims.items():
            if len(claimants) > 1:
                winner = sorted(claimants, key=lambda item: (-item[0], item[1]))[0][1]
                for _, eid in claimants:
                    if eid != winner:
                        predictions[eid].remove(cid)
    return predictions, {"duplicate_claims_before": before,
        "duplicate_claims_after": 0 if exclusivity else before,
        "claimed_targets": len(claims), "duplicate_claim_rate_before": before/max(1, len(claims)),
        "duplicate_claim_rate_after": (0 if exclusivity else before)/max(1, len(claims))}


def _assess(scored, truth, *, name, variant, thresholds, guardian=False, guardian_threshold=0.0,
            exclusivity=False, penalty=0.0):
    prediction, duplicates = _predict(scored, variant, thresholds, guardian=guardian,
        guardian_threshold=guardian_threshold, exclusivity=exclusivity, penalty=penalty)
    metric = evaluate_predictions(truth, prediction)
    return {"ablation": name, "variant": variant, "threshold_s2": thresholds[0], "threshold_s3": thresholds[1],
        "guardian": guardian, "guardian_threshold": guardian_threshold, "exclusivity": exclusivity,
        "numeric_penalty": penalty, "macro_fbeta": metric.macro_fbeta,
        "macro_precision": metric.macro_precision, "macro_recall": metric.macro_recall,
        "singleton_accuracy": metric.correct_singletons/max(1, metric.singleton_count),
        **duplicates}, prediction


def _guardian_threshold(scored, truth, variant, thresholds, exclusivity):
    best = None
    for cutoff in (i/100 for i in range(0, 91, 2)):
        row, _ = _assess(scored, truth, name="guardian_search", variant=variant, thresholds=thresholds,
                         guardian=True, guardian_threshold=cutoff, exclusivity=exclusivity)
        key = (round(row["macro_fbeta"], 12), -cutoff)
        if best is None or key > best[0]:
            best = (key, cutoff)
    return best[1]


def tune(data_dir, index_dir, original_dir, sample_dir, tune_pairs_dir, fit_dir, ownership_dir, output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Use a fresh EXP007 TUNE directory")
    base, sample_ids = identity(data_dir, index_dir, original_dir, sample_dir)
    validate_pairs(tune_pairs_dir, "tune", base)
    models, fit_manifest = _load_models(fit_dir, base)
    audit_path = Path(ownership_dir) / "target_ownership_audit.json"
    audit = json.loads(audit_path.read_text())
    if audit["ground_truth_sha256"] != base["truth_sha256"]:
        raise ValueError("Ownership audit uses different TRAIN truth")
    exclusive_allowed = audit["targets_with_multiple_owners"] == 0
    ids = partition_ids("tune", sample_ids)
    truth = load_selected_ground_truth(Path(data_dir) / "train_ground_truth.tsv", ids)
    scored = _scored_partition(Path(tune_pairs_dir) / "tune", ids, models)
    for eid in ids:
        if {r[1] for r in scored[eid]["rows"] if int(r[2])} != set(truth[eid]) & {r[1] for r in scored[eid]["rows"]}:
            raise ValueError("EXP007 TUNE pair labels disagree with TRAIN truth")
    global_threshold = _select_thresholds(scored, truth, "base", global_only=True)
    source_thresholds = _select_thresholds(scored, truth, "base")
    guardian_threshold = _guardian_threshold(scored, truth, "base", source_thresholds, False)
    specs = [
        ("global_threshold", "base", global_threshold, False, 0.0, False, 0.0),
        ("source_thresholds", "base", source_thresholds, False, 0.0, False, 0.0),
        ("source_guardian", "base", source_thresholds, True, guardian_threshold, False, 0.0),
        ("source_exclusivity", "base", source_thresholds, False, 0.0, exclusive_allowed, 0.0),
        ("source_guardian_exclusivity", "base", source_thresholds, True, guardian_threshold, exclusive_allowed, 0.0),
    ]
    relative_thresholds = _select_thresholds(scored, truth, "relative")
    cross_thresholds = _select_thresholds(scored, truth, "cross")
    rows = []
    for name, variant, thresholds, guard, cutoff, exclusive, penalty in specs:
        row, _ = _assess(scored, truth, name=name, variant=variant, thresholds=thresholds,
                         guardian=guard, guardian_threshold=cutoff, exclusivity=exclusive, penalty=penalty)
        rows.append(row)
    base_best = max(rows, key=lambda r: (round(r["macro_fbeta"], 12), r["macro_precision"],
                                         r["threshold_s2"]+r["threshold_s3"], -rows.index(r)))
    for name, variant, thresholds in (("relative_decision", "relative", relative_thresholds),
                                      ("cross_source_decision", "cross", cross_thresholds)):
        guard = base_best["guardian"]
        exclusive = base_best["exclusivity"]
        cutoff = _guardian_threshold(scored, truth, variant, thresholds, exclusive) if guard else 0.0
        rows.append(_assess(scored, truth, name=name, variant=variant, thresholds=thresholds,
                    guardian=guard, guardian_threshold=cutoff, exclusivity=exclusive)[0])
    best = max(rows, key=lambda r: (round(r["macro_fbeta"], 12), r["macro_precision"],
                                    r["threshold_s2"]+r["threshold_s3"], -rows.index(r)))
    # Numeric penalty is an optional TUNE ablation on the best non-penalized policy.
    penalty_specs = []
    for penalty in (.01, .03, .05):
        thresholds = _select_thresholds(scored, truth, best["variant"], penalty=penalty)
        penalty_specs.append(_assess(scored, truth, name=f"numeric_penalty_{penalty:.2f}",
            variant=best["variant"], thresholds=thresholds, guardian=best["guardian"],
            guardian_threshold=best["guardian_threshold"], exclusivity=best["exclusivity"], penalty=penalty)[0])
    rows.extend(penalty_specs)
    rows.append({"ablation": "calibration", "status": "not_run_disjoint_calibration_split_unavailable"})
    selected = max((r for r in rows if "macro_fbeta" in r),
                   key=lambda r: (round(r["macro_fbeta"], 12), r["macro_precision"],
                                  r["threshold_s2"]+r["threshold_s3"], -rows.index(r)))
    output_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(output_dir / "tune_ablations.csv", index=False)
    write_json_atomic(output_dir / "feature_names.json", list(FEATURE_NAMES))
    write_json_atomic(output_dir / "feature_count.json", {"old": 51, "enhanced": len(ENHANCED_FEATURE_NAMES), "total": len(FEATURE_NAMES)})
    selected_policy = {k: selected[k] for k in ("ablation", "variant", "threshold_s2", "threshold_s3", "guardian",
        "guardian_threshold", "exclusivity", "numeric_penalty", "macro_fbeta", "macro_precision", "macro_recall")}
    selected_policy.update({"feature_count": len(FEATURE_NAMES), "old_feature_count": 51,
        "enhanced_feature_count": len(ENHANCED_FEATURE_NAMES), "normalization_version": NORMALIZATION_VERSION,
        "blocker_sha256": CONFIG_HASHES[18], "fit_manifest_sha256": sha256(Path(fit_dir)/"fit_manifest.json"),
        "tune_pairs_manifest_sha256": sha256(Path(tune_pairs_dir)/"manifest.json"),
        "ownership_audit_sha256": sha256(audit_path), "exclusivity_disabled_reason": None if exclusive_allowed else "TRAIN target ownership conflicts",
        "calibration": "raw_HGB", "cross_source_top_k": 3 if selected["variant"] == "cross" else 0,
        "numeric_penalty_policy": selected["numeric_penalty"], "evaluation_read": False})
    write_json_atomic(output_dir / "selected_policy.json", selected_policy)
    bundle = {"bundle_version": FORMAT, "models": models, "feature_names": list(FEATURE_NAMES),
        "entity_feature_names": ENTITY_FEATURE_NAMES, "relative_feature_names": RELATIVE_FEATURE_NAMES,
        "blocker_config": frozen_config(18), "identity": base, "policy": selected_policy,
        "fit_manifest": fit_manifest, "git_commit": git_commit(),
        "model_license_notes": fit_manifest["license_notes"], "training_seeds": {"model": MODEL_SEED, "sampling": SAMPLE_SEED, "extra_fit": exp006.SEED}}
    joblib.dump(bundle, output_dir / "frozen.joblib", compress=3)
    freeze = {"selected_policy_sha256": sha256(output_dir / "selected_policy.json"),
        "tune_ablations_sha256": sha256(output_dir / "tune_ablations.csv"),
        "bundle_sha256": sha256(output_dir / "frozen.joblib"), "fit_manifest_sha256": selected_policy["fit_manifest_sha256"],
        "evaluation_read": False}
    write_json_atomic(output_dir / "freeze.json", freeze)
    return {"selected_policy": selected_policy, "tune_ablation_count": len(rows), "freeze": freeze}


def load_frozen_bundle(tune_dir, *, data_dir=None, index_dir=None, original_dir=None, sample_dir=None):
    tune_dir = Path(tune_dir)
    freeze = json.loads((tune_dir / "freeze.json").read_text())
    for file, key in (("selected_policy.json", "selected_policy_sha256"),
                      ("tune_ablations.csv", "tune_ablations_sha256"), ("frozen.joblib", "bundle_sha256")):
        if sha256(tune_dir / file) != freeze[key]:
            raise ValueError("EXP007 frozen policy checksum mismatch")
    bundle = joblib.load(tune_dir / "frozen.joblib")
    policy = json.loads((tune_dir / "selected_policy.json").read_text())
    if (bundle["bundle_version"] != FORMAT or bundle["feature_names"] != list(FEATURE_NAMES)
            or bundle["policy"] != policy or policy["evaluation_read"] is not False
            or bundle["identity"]["code_sha256"] != code_hashes()
            or bundle["identity"]["legacy_code_sha256"] != code_versions()
            or bundle["identity"]["blocker_sha256"] != CONFIG_HASHES[18]
            or bundle["blocker_config"] != frozen_config(18)):
        raise ValueError("EXP007 frozen bundle incompatible")
    if bundle["fit_manifest"]["package_versions"] != {p: importlib.metadata.version(p) for p in bundle["fit_manifest"]["package_versions"]}:
        raise ValueError("EXP007 frozen bundle package versions differ")
    if data_dir is not None:
        current, _ = identity(data_dir, index_dir, original_dir, sample_dir)
        if bundle["identity"] != current:
            raise ValueError("EXP007 frozen bundle input identity mismatch")
    return bundle


def score_source_entity(bundle, source_record, candidates):
    """Inference-available scoring; callers must supply the frozen compact18 pool."""
    matrix = np.asarray([list(feature_row(source_record, c).values()) for c in candidates], np.float32).reshape(-1, len(FEATURE_NAMES))
    rows = [[source_record.entity_id, c.candidate_entity_id, 0, c.candidate_name, c.candidate_address] for c in candidates]
    models = bundle["models"]
    base, entity, comparisons = score_group(models["model"], rows, matrix)
    _, relative, _ = decision_features(rows, matrix, base)
    variants = {"base": base}
    if len(rows):
        variants["relative"] = models["relative"].predict_proba(relative[:, :-5])[:, 1]
        variants["cross"] = models["relative_cross"].predict_proba(relative)[:, 1]
    else:
        variants.update({"relative": np.empty(0), "cross": np.empty(0)})
    if any(not np.isfinite(values).all() for values in variants.values()):
        raise ValueError("Non-finite EXP007 deployment score")
    return {"rows": rows, "scores": variants,
            "numeric": matrix[:, FEATURE_NAMES.index("primary_number_mismatch")].astype(np.float64),
            "guardian_has_match": float(models["guardian"].predict_proba(entity.reshape(1, -1))[0, 1]),
            "cross_comparisons": comparisons}


def score_source_entity_from_index(bundle, source_record, indexes):
    candidates, _ = candidate_pool(source_record, indexes, bundle["blocker_config"])
    return score_source_entity(bundle, source_record, candidates)


def decide_scored(bundle, scored):
    """Apply a frozen policy across S1 entities, including optional global ownership."""
    p = bundle["policy"]
    return _predict(scored, p["variant"], (p["threshold_s2"], p["threshold_s3"]),
                    guardian=p["guardian"], guardian_threshold=p["guardian_threshold"],
                    exclusivity=p["exclusivity"], penalty=p["numeric_penalty"])


def evaluate(data_dir, index_dir, original_dir, sample_dir, evaluation_pairs_dir, fit_dir,
             tune_dir, oracle_dir, ownership_dir, output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("EXP007 EVALUATION already exists; run exactly once")
    base, sample_ids = identity(data_dir, index_dir, original_dir, sample_dir)
    exp006.preflight(data_dir, index_dir, original_dir, include_evaluation=True)
    validate_pairs(evaluation_pairs_dir, "evaluation", base)
    bundle = load_frozen_bundle(tune_dir, data_dir=data_dir, index_dir=index_dir,
                                original_dir=original_dir, sample_dir=sample_dir)
    p = bundle["policy"]
    if p["fit_manifest_sha256"] != sha256(Path(fit_dir) / "fit_manifest.json"):
        raise ValueError("EXP007 FIT manifest changed after policy freeze")
    audit = json.loads((Path(ownership_dir) / "target_ownership_audit.json").read_text())
    if p["ownership_audit_sha256"] != sha256(Path(ownership_dir) / "target_ownership_audit.json"):
        raise ValueError("EXP007 ownership audit changed after policy freeze")
    if p["exclusivity"] and audit["targets_with_multiple_owners"]:
        raise ValueError("Exclusivity forbidden by TRAIN ownership conflicts")
    oracle_result = json.loads((Path(oracle_dir) / "blocker_oracle_evaluation.json").read_text())
    if oracle_result["input_identity"] != base or oracle_result["partition"] != "evaluation":
        raise ValueError("EXP007 blocker oracle identity mismatch")
    ids = partition_ids("evaluation", sample_ids)
    truth = load_selected_ground_truth(Path(data_dir) / "train_ground_truth.tsv", ids)
    records = load_selected_source_records(Path(data_dir) / "train_source1.tsv", "S1", set(ids))
    scored = _scored_partition(Path(evaluation_pairs_dir) / "evaluation", ids, bundle["models"])
    predictions, duplicate_audit = decide_scored(bundle, scored)
    retained = {e: {r[1] for r in scored[e]["rows"] if int(r[2])} for e in ids}
    for eid in ids:
        if retained[eid] != set(truth[eid]) & {r[1] for r in scored[eid]["rows"]}:
            raise ValueError("EXP007 EVALUATION pair labels disagree with TRAIN truth")
    slices = metric_slices(truth, predictions, retained, records)
    overall = next(r for r in slices if r["dimension"] == "overall")
    if p["guardian"]:
        cutoff = p["guardian_threshold"]
        abstained = {e for e in ids if scored[e]["guardian_has_match"] <= cutoff}
    else:
        abstained = set()
    guardian_metrics = {"enabled": p["guardian"], "threshold": p["guardian_threshold"],
        "true_singleton_abstained": sum(not truth[e] for e in abstained),
        "matched_entity_abstained": sum(bool(truth[e]) for e in abstained),
        "true_singleton_not_abstained": sum(not truth[e] and e not in abstained for e in ids),
        "matched_entity_not_abstained": sum(bool(truth[e]) and e not in abstained for e in ids)}
    conflict_pairs = [(e, i, r) for e in ids for i, r in enumerate(scored[e]["rows"]) if scored[e]["numeric"][i] > 0]
    numeric_audit = {"pairs_with_primary_number_conflict": len(conflict_pairs),
        "true_links_with_conflict": sum(int(r[2]) for _, _, r in conflict_pairs),
        "selected_conflict_links": sum(r[1] in predictions[e] for e, _, r in conflict_pairs),
        "selected_penalty": p["numeric_penalty"]}
    cross_audit = {"top_k_per_source": 3, "total_pair_comparisons": sum(scored[e]["cross_comparisons"] for e in ids),
        "max_pair_comparisons_per_s1": max(scored[e]["cross_comparisons"] for e in ids),
        "selected": p["variant"] == "cross", "candidate_expansion": 0}
    f = overall["macro_fbeta"]
    promotion = ("REJECT" if f < .920 else "WEAK" if f < .930 else "MEANINGFUL" if f < .940
                 else "STRONG" if f < .950 else "MAJOR RESULT")
    comparison = {"exp003_macro_fbeta": .905874, "exp006_macro_fbeta": .9136711411208787,
        "delta_vs_exp003": f-.905874, "delta_vs_exp006": f-.9136711411208787,
        "precision_delta_vs_exp006": overall["macro_precision"]-.9511429022366504,
        "recall_delta_vs_exp006": overall["macro_recall"]-.8385030092592611,
        "matcher_fn_delta_vs_exp006": overall["matcher_fn"]-5133,
        "singleton_accuracy_delta_vs_exp006": overall["singleton_accuracy"]-.8746269,
        "blocker_oracle_gap": oracle_result["overall"]["macro_fbeta"]-f}
    output_dir.mkdir(parents=True)
    pd.DataFrame([overall]).to_csv(output_dir / "evaluation.csv", index=False)
    pd.DataFrame(slices).to_csv(output_dir / "evaluation_slices.csv", index=False)
    write_json_atomic(output_dir / "guardian_metrics.json", guardian_metrics)
    write_json_atomic(output_dir / "duplicate_claims_audit.json", duplicate_audit)
    write_json_atomic(output_dir / "numeric_conflict_audit.json", numeric_audit)
    write_json_atomic(output_dir / "cross_source_audit.json", cross_audit)
    summary = {"overall": overall, "promotion": promotion, "comparison": comparison,
        "blocker_oracle": {k: oracle_result[k] for k in ("candidate_link_recall", "perfect_candidate_coverage_rate",
            "matched_with_zero_retained_rate", "blocker_fn")},
        "blocker_oracle_macro_fbeta": oracle_result["overall"]["macro_fbeta"],
        "selected_policy": p, "ownership_audit": audit, "freeze_sha256": sha256(Path(tune_dir)/"freeze.json")}
    write_json_atomic(output_dir / "summary.json", summary)
    files = {q.name: sha256(q) for q in output_dir.iterdir() if q.is_file()}
    write_json_atomic(output_dir / "artifact_manifest.json", {"identity": base, "files": files,
        "evaluation_pairs_manifest_sha256": sha256(Path(evaluation_pairs_dir)/"manifest.json")})
    return summary

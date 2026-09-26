"""Only tiny synthetic TRAIN/TEST fixtures; no Amazon inference or index builds."""
import csv
from dataclasses import fields, replace
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from business_entity_resolution import exp003_inference as inference
from business_entity_resolution import exp003_training as training
from business_entity_resolution.features import FEATURE_NAMES, build_feature_row
from business_entity_resolution.indexed_blocking import build_or_open_source_index
from business_entity_resolution.modeling import make_logistic_model, make_hist_gradient_boosting_model
from business_entity_resolution.multipass import RetrievedCandidate
from business_entity_resolution.reranker import (ROOT, RULE_COLUMN, accepted_ids, candidate_pool,
    feature_matrix, frozen_config, load_bundle, make_bundle, open_indexes, save_bundle, sha256)
from business_entity_resolution.sampling import SourceRecord


def candidate(i, source="S2"):
    values = {f.name: 0 for f in fields(RetrievedCandidate)}
    values.update(source1_entity_id="S1-a", candidate_entity_id=f"{source}-{i:03d}", source=source,
                  country="Pays Ω", candidate_name="Café Alpha", candidate_address="١٢ rue",
                  exact_name_rank=i + 1 if i < 40 else 0, name_token_rank=i - 39 if i >= 40 else 0,
                  best_retrieval_score=100, exact_name_score=100)
    return RetrievedCandidate(**values)


def test_eligible_pool_is_not_capped_at_60():
    raw = [candidate(i) for i in range(90)]  # 80 pass-eligible, 10 rank >40
    indexes = {"S2": SimpleNamespace(retrieve=lambda _: raw), "S3": SimpleNamespace(retrieve=lambda _: [])}
    record = SourceRecord("S1-a", "Café Alpha", "١٢ rue", "Pays Ω")
    eligible, raw_ids = candidate_pool(record, indexes, frozen_config(18))
    assert len(eligible) == 80 and len(raw_ids) == 90
    assert len(candidate_pool(record, indexes, frozen_config(14), rule_fallback=True)[0]) == 60
    matrix = feature_matrix(record, eligible)
    assert matrix.shape == (80, len(FEATURE_NAMES))
    assert matrix[0].tolist() == list(build_feature_row(record, eligible[0]).values())
    assert matrix[0, FEATURE_NAMES.index("country_agreement")] == 1
    assert matrix[0, FEATURE_NAMES.index("normalized_name_exact")] == 1


def test_sampler_keeps_all_positives_and_is_deterministic():
    cids = [f"S2-{i}" for i in range(200)]
    x = np.zeros((200, len(FEATURE_NAMES)), dtype=np.float32)
    x[:, RULE_COLUMN] = np.arange(200) % 101
    labels = np.array([i % 7 == 0 for i in range(200)])
    examples, counts = training.sample_entity("S1-a", cids, x, labels)
    backward, reverse_counts = training.sample_entity("S1-a", cids[::-1], x[::-1], labels[::-1])
    assert examples == backward and counts == reverse_counts
    assert {e.candidate_entity_id for e in examples if e.label} == {cid for cid, label in zip(cids, labels) if label}
    assert counts["hard_negative_count"] <= 20 and counts["easy_negative_count"] <= 3


def test_entity_partitions_preserve_evaluation_and_isolate():
    records = {f"S1-{i}": SourceRecord(f"S1-{i}", "Name", "", "Pays Ω") for i in range(40)}
    truth = {eid: frozenset() if i % 4 == 0 else frozenset({"S2-a"}) for i, eid in enumerate(records)}
    table = pd.DataFrame({"source1_entity_id": list(records), "partition": ["evaluation"] * 12 + ["tuning"] * 28})
    parts = training.partitions(table, records, truth)
    assert parts == training.partitions(table.iloc[::-1], records, truth)
    assert {e for e, p in parts.items() if p == "evaluation"} == set(table.iloc[:12].source1_entity_id)
    groups = [{e for e, p in parts.items() if p == name} for name in ("fit", "tune", "evaluation")]
    assert all(groups) and not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]


@pytest.mark.parametrize("model_type", ["logistic", "hist_gradient_boosting"])
def test_model_bundle_round_trip_and_feature_order(tmp_path, model_type):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(60, len(FEATURE_NAMES))).astype(np.float32)
    y = np.array([0, 1] * 30)
    model = make_logistic_model(2030) if model_type == "logistic" else make_hist_gradient_boosting_model(2030).set_params(max_iter=3, min_samples_leaf=3, early_stopping=False)
    with threadpool_limits(limits=1):
        model.fit(x, y)
        bundle = make_bundle(model, model_type, .7, 18, 2030)
        path = tmp_path / "model.joblib"
        save_bundle(bundle, path)
        restored = load_bundle(path)
        np.testing.assert_array_equal(model.predict_proba(x), restored["model"].predict_proba(x))
    bundle["feature_names"] = list(reversed(FEATURE_NAMES))
    save_bundle(bundle, path)
    with pytest.raises(ValueError, match="feature order"):
        load_bundle(path)


def write_sources(directory, split):
    directory.mkdir()
    for s in ("S1", "S2", "S3"):
        with (directory / f"{split}_source{s[-1]}.tsv").open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["entity_id", "business_name", "business_address", "country"])
            if s == "S1":
                writer.writerows([(f"S1-{i}", "Café Alpha" if i % 2 == 0 else "Absentxyz", "١٢ rue" if i % 2 == 0 else "", "Pays Ω") for i in range(12)])
            else:
                writer.writerows([(f"{s}-{i}", "Café Alpha", "١٢ rue", "Pays Ω") for i in range(2)])


@pytest.fixture
def inference_fixture(tmp_path):
    test = tmp_path / "test"
    write_sources(test, "test")
    index_dir = tmp_path / "indexes"
    for s in ("S2", "S3"):
        build_or_open_source_index(test / f"test_source{s[-1]}.tsv", s, index_dir, chunksize=2)
    bundle = tmp_path / "rule.joblib"
    save_bundle(make_bundle(None, "rule", 96, 14, 2030), bundle)
    return test, index_dir, bundle


def test_rule_shards_resume_merge_and_official_validator(inference_fixture, tmp_path, monkeypatch):
    test, indexes, bundle = inference_fixture
    before = {p.name: sha256(p) for p in indexes.iterdir()}
    shards = tmp_path / "shards"
    for i in range(4):
        done = inference.infer_shard(bundle, "rule", test, indexes, shards, shard=i, smoke_limit=100, checkpoint_every=1)
        assert done["rows"] == 3
    # A completed shard must not score again, even on restart.
    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected retrieval/build")
    monkeypatch.setattr(inference, "candidate_pool", forbidden)
    before_resume = (shards / "shard_00/candidate_pairs.tsv").stat().st_mtime_ns
    inference.infer_shard(bundle, "rule", test, indexes, shards, shard=0)
    assert (shards / "shard_00/candidate_pairs.tsv").stat().st_mtime_ns == before_resume
    output = tmp_path / "output"
    assert inference.merge_shards(shards, test, output) == 12
    assert before == {p.name: sha256(p) for p in indexes.iterdir()}
    rows = list(csv.DictReader((output / "matching_results.tsv").open(), delimiter="\t"))
    assert rows[1]["matched_entity_ids"] == ""
    assert len(rows[0]["matched_entity_ids"].split(",")) == 4
    spec = importlib.util.spec_from_file_location("official_validator", ROOT / "resources/validate_submission.py")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    errors, warnings = validator.validate(str(output / "matching_results.tsv"), str(output / "candidate_pairs.tsv"), str(test), check_ids=True)
    assert errors == [] and warnings == []
    second = tmp_path / "second"
    inference.merge_shards(shards, test, second)
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        assert (second / name).read_bytes() == (output / name).read_bytes()
    # A copied shard cannot stand in for a missing shard.
    path = shards / "shard_01/done.json"
    value = json.loads(path.read_text())
    value["shard"] = 0
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Mixed/duplicate"):
        inference.merge_shards(shards, test, tmp_path / "bad")


def test_interrupted_shard_resumes_committed_rows(inference_fixture, tmp_path, monkeypatch):
    test, indexes, bundle = inference_fixture
    original = inference.candidate_pool
    calls = []
    def interrupt(record, *args, **kwargs):
        calls.append(record.entity_id)
        if len(calls) == 2:
            raise RuntimeError("simulated interruption")
        return original(record, *args, **kwargs)
    monkeypatch.setattr(inference, "candidate_pool", interrupt)
    shards = tmp_path / "resume"
    with pytest.raises(RuntimeError):
        inference.infer_shard(bundle, "rule", test, indexes, shards, shard=0, checkpoint_every=1)
    calls.clear()
    def record_call(record, *args, **kwargs):
        calls.append(record.entity_id)
        return original(record, *args, **kwargs)
    monkeypatch.setattr(inference, "candidate_pool", record_call)
    done = inference.infer_shard(bundle, "rule", test, indexes, shards, shard=0, checkpoint_every=1)
    assert calls == ["S1-4", "S1-8"] and done["rows"] == 3


def test_inference_guards_and_index_identity(inference_fixture, tmp_path):
    test, indexes, bundle = inference_fixture
    with pytest.raises(ValueError, match="Full TEST"):
        inference.infer_shard(bundle, "rule", test, indexes, tmp_path / "out", smoke_limit=None)
    result = subprocess.run([sys.executable, str(ROOT / "scripts/infer_exp003.py"), "run", "--build-index"], capture_output=True, text=True)
    assert result.returncode != 0
    with pytest.raises(FileNotFoundError):
        with open_indexes(tmp_path / "missing", test, "test", frozen_config(18)):
            pass
    # TRAIN-vs-TEST source mismatch cannot silently use a wrong index.
    with (test / "test_source2.tsv").open("a") as handle:
        handle.write("S2-extra\tOther\t\tPays Ω\n")
    with pytest.raises(ValueError, match="fingerprint"):
        with open_indexes(indexes, test, "test", frozen_config(18)):
            pass


def test_merge_rejects_duplicate_s1_across_shards(inference_fixture, tmp_path):
    test, indexes, bundle = inference_fixture
    source1 = test / "test_source1.tsv"
    source1.write_text(source1.read_text().replace("S1-1\t", "S1-0\t"))
    for i in range(4):
        inference.infer_shard(bundle, "rule", test, indexes, tmp_path / "dup", shard=i)
    with pytest.raises(sqlite3.IntegrityError):
        inference.merge_shards(tmp_path / "dup", test, tmp_path / "merged_dup")
    assert not (tmp_path / "merged_dup").exists()


def test_training_refuses_non_authoritative_subset(tmp_path, monkeypatch):
    fake = tmp_path / "results/tables"
    fake.mkdir(parents=True)
    (fake / "exp001_subset_ids.csv").write_text("source1_entity_id,partition\nS1-1,evaluation\n")
    monkeypatch.setattr(training, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="authoritative fixed"):
        training.run_training(tmp_path / "train", tmp_path / "indexes", tmp_path / "output")


def test_cache_completion_and_corruption_checks(tmp_path):
    with pytest.raises(ValueError, match="Incomplete pair cache"):
        training.validate_pair_cache(tmp_path, {"seed": 1})
    for name in training.PAIR_FILES:
        (tmp_path / name).write_bytes(b"fixture")
    training.seal_pair_cache(tmp_path, {"seed": 1})
    training.validate_pair_cache(tmp_path, {"seed": 1})
    with pytest.raises(ValueError, match="identity"):
        training.validate_pair_cache(tmp_path, {"seed": 2})
    (tmp_path / "fit.f32").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="Corrupt pair cache"):
        training.validate_pair_cache(tmp_path, {"seed": 1})


def test_bundle_checksum_and_resume_identity(inference_fixture, tmp_path):
    test, indexes, bundle = inference_fixture
    inference.infer_shard(bundle, "rule", test, indexes, tmp_path / "shards", shard=0)
    with pytest.raises(ValueError, match="Checkpoint identity mismatch"):
        inference.infer_shard(bundle, "rule", test, indexes, tmp_path / "shards", shard=0, smoke_limit=10)
    bundle.write_bytes(bundle.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_bundle(bundle)


def test_threshold_semantics_and_disk_tune_includes_singletons(tmp_path):
    path = tmp_path / "tune"
    writer = training.PairWriter(path)
    x = np.zeros((2, len(FEATURE_NAMES)), dtype=np.float32)
    x[:, RULE_COLUMN] = [96, 95]
    writer.add("S1-a", ["S2-1", "S3-1"], x, [1, 0])
    writer.close()
    truth = {"S1-a": {"S2-1"}, "S1-singleton": set()}
    threshold, result, _ = training.tune_one(path, "rule", None, truth)
    assert threshold == 96 and result.macro_fbeta == 1
    assert accepted_ids(["S2-1", "S3-1"], [.7, .7], .7) == ["S2-1", "S3-1"]
    assert accepted_ids([], [], .7) == []


def test_synthetic_pair_generation_to_learned_inference(tmp_path):
    train = tmp_path / "train"
    write_sources(train, "train")
    indexes = tmp_path / "indexes"
    for s in ("S2", "S3"):
        build_or_open_source_index(train / f"train_source{s[-1]}.tsv", s, indexes, chunksize=2)
    from business_entity_resolution.sampling import load_selected_source_records
    records = load_selected_source_records(train / "train_source1.tsv", "S1", {f"S1-{i}" for i in range(12)})
    truth = {eid: ({"S2-0", "S3-0"} if int(eid[3:]) % 2 == 0 else set()) for eid in records}
    parts = {eid: ("fit" if int(eid[3:]) < 4 else "tune" if int(eid[3:]) < 8 else "evaluation") for eid in records}
    training.generate_pairs(tmp_path / "pairs", records, truth, parts, indexes, train)
    groups = list(training.pair_groups(tmp_path / "pairs/fit"))
    assert all(parts[eid] == "fit" for eid, *_ in groups)
    x = np.concatenate([g[2] for g in groups])
    y = np.concatenate([g[3] for g in groups])
    with threadpool_limits(limits=1):
        model = make_logistic_model(2030).fit(x, y)
        tune_gt = {e: truth[e] for e in truth if parts[e] == "tune"}
        threshold, _, _ = training.tune_one(tmp_path / "pairs/tune", "logistic", model, tune_gt)
        rows = training.evaluate_one(tmp_path / "pairs/evaluation", "logistic", model, threshold,
                                    {e: truth[e] for e in truth if parts[e] == "evaluation"}, records)
    assert rows[0]["blocking_fn"] + rows[0]["matcher_fn"] == rows[0]["false_negatives"]
    bundle = tmp_path / "learned.joblib"
    save_bundle(make_bundle(model, "logistic", threshold, 18, 2030), bundle)
    # TEST has no GT file at all; inference depends only on sources/indexes/bundle.
    test = tmp_path / "test"
    write_sources(test, "test")
    test_indexes = tmp_path / "test_indexes"
    for s in ("S2", "S3"):
        build_or_open_source_index(test / f"test_source{s[-1]}.tsv", s, test_indexes, chunksize=2)
    done = inference.infer_shard(bundle, "learned", test, test_indexes, tmp_path / "learned_out", shard=0)
    assert done["rows"] == 3

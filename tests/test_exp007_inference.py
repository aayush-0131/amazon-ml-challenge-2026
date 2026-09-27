"""Synthetic-only EXP007 TEST inference and artifact safety checks."""
from __future__ import annotations

import csv
from dataclasses import fields
import importlib.util
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from business_entity_resolution import exp007, exp007_inference as inference
from business_entity_resolution.exp007_features import FEATURE_NAMES, feature_row
from business_entity_resolution.indexed_blocking import build_or_open_source_index
from business_entity_resolution.multipass import RetrievedCandidate
from business_entity_resolution.reranker import ROOT, frozen_config, sha256
from business_entity_resolution.sampling import SourceRecord


class SourceModel:
    def __init__(self, s2=.97, s3=.97):
        self.s2, self.s3 = s2, s3

    def predict_proba(self, matrix):
        probability = np.where(matrix[:, FEATURE_NAMES.index("source_is_s3")] > .5, self.s3, self.s2)
        return np.column_stack((1 - probability, probability))


def bundle(model=None):
    return {"models": {"model": model or SourceModel()}, "blocker_config": frozen_config(18),
            "feature_names": list(FEATURE_NAMES),
            "policy": {"variant": "base", "guardian": False, "exclusivity": False,
                       "numeric_penalty": 0.0, "calibration": "raw_HGB", "feature_count": 83,
                       "threshold_s2": .965, "threshold_s3": .956}}


def candidate(source):
    values = {field.name: 0 for field in fields(RetrievedCandidate)}
    values.update(source1_entity_id="S1-0", candidate_entity_id=f"{source}-0", source=source,
                  country="Atlantis", candidate_name="Café Alpha", candidate_address="12 Rue",
                  exact_name_rank=1, exact_name_score=100.0, pass_count=1,
                  best_retrieval_score=100.0)
    return RetrievedCandidate(**values)


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    with (test_dir / "test_source1.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("entity_id", "business_name", "business_address", "country"))
        writer.writerows((f"S1-{i}", "Café Alpha" if i % 2 == 0 else "Absentxyz",
                          "12 Rue" if i % 2 == 0 else "", "Atlantis" if i % 3 else "FR") for i in range(12))
    for source in ("S2", "S3"):
        with (test_dir / f"test_source{source[-1]}.tsv").open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("entity_id", "business_name", "business_address", "country"))
            writer.writerow((f"{source}-0", "Café Alpha", "12 Rue", "Atlantis"))
            writer.writerow((f"{source}-1", "Café Alpha", "12 Rue", "FR"))
    index_dir = tmp_path / "indexes"
    for source in ("S2", "S3"):
        build_or_open_source_index(test_dir / f"test_source{source[-1]}.tsv", source, index_dir, chunksize=2)
    tune_dir = tmp_path / "tune"
    tune_dir.mkdir()
    for name in ("frozen.joblib", "selected_policy.json", "freeze.json"):
        (tune_dir / name).write_bytes(name.encode())
    monkeypatch.setattr(inference, "frozen_bundle", lambda _: bundle())
    return test_dir, index_dir, tune_dir


def run_four(synthetic, shard_dir):
    test_dir, index_dir, tune_dir = synthetic
    return [inference.infer_shard(tune_dir, test_dir, index_dir, shard_dir,
            shard=i, shards=4, smoke_limit=12, checkpoint_every=1) for i in range(4)]


def test_frozen_source_thresholds_subset_zero_candidates_and_unseen_country(monkeypatch):
    s2, s3 = candidate("S2"), candidate("S3")
    record = SourceRecord("S1-0", "Café Alpha", "12 Rue", "Atlantis")
    monkeypatch.setattr(inference, "candidate_pool", lambda *_: ([s2, s3], set()))
    matched, candidates = inference.score_record(bundle(SourceModel(.964, .956)), record, {})
    assert matched == "S3-0"  # S2 is below .965; S3 meets .956.
    assert candidates == "S2-0,S3-0"
    matched, _ = inference.score_record(bundle(SourceModel(.965, .955)), record, {})
    assert matched == "S2-0"
    matched, _ = inference.score_record(bundle(SourceModel(.1, .1)), record, {})
    assert matched == ""
    monkeypatch.setattr(inference, "candidate_pool", lambda *_: ([], set()))
    assert inference.score_record(bundle(), record, {}) == ("", "")
    assert inference.target_list(["S3-0", "S2-0"]) == "S2-0,S3-0"
    with pytest.raises(ValueError, match="Malformed target"):
        inference.target_list(["X-0"])
    with pytest.raises(ValueError, match="Duplicate target"):
        inference.target_list(["S2-0", "S2-0"])


def test_base_scoring_matches_frozen_exp007_semantics(monkeypatch):
    candidates = [candidate("S2"), candidate("S3")]
    record = SourceRecord("S1-0", "Café Alpha", "12 Rue", "Atlantis")
    model = SourceModel(.965, .956)
    selected = bundle(model)
    monkeypatch.setattr(inference, "candidate_pool", lambda *_: (candidates, set()))
    matched, scored_ids = inference.score_record(selected, record, {})
    matrix = np.asarray([list(feature_row(record, c).values()) for c in candidates], np.float32)
    rows = [[record.entity_id, c.candidate_entity_id, 0, c.candidate_name, c.candidate_address] for c in candidates]
    scores, _, _ = exp007.score_group(model, rows, matrix)
    assert list(scores) == pytest.approx([.965, .956])
    expected, _ = exp007._predict({record.entity_id: {"rows": rows, "scores": {"base": scores},
                                  "numeric": np.zeros(len(rows)), "guardian_has_match": 0.0}},
                                 "base", (.965, .956), guardian=False, exclusivity=False, penalty=0.0)
    assert matched == ",".join(sorted(expected[record.entity_id])) == "S2-0,S3-0"
    assert scored_ids == matched


def test_bundle_validation_never_uses_train_identity(monkeypatch, tmp_path):
    approved = bundle()

    def load(path, **kwargs):
        assert path == tmp_path
        assert kwargs == {}
        return approved

    monkeypatch.setattr(exp007, "load_frozen_bundle", load)
    assert inference.frozen_bundle(tmp_path) is approved
    changed = bundle()
    changed["policy"]["threshold_s2"] = .96
    monkeypatch.setattr(exp007, "load_frozen_bundle", lambda *_: changed)
    with pytest.raises(ValueError, match="approved base policy"):
        inference.frozen_bundle(tmp_path)


def test_shards_resume_reuse_merge_audit_and_validator(synthetic, tmp_path, monkeypatch):
    test_dir, index_dir, tune_dir = synthetic
    shards = tmp_path / "shards"
    original = inference.score_record
    calls = []

    def interrupt(*args):
        calls.append(args[1].entity_id)
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return original(*args)

    monkeypatch.setattr(inference, "score_record", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        inference.infer_shard(tune_dir, test_dir, index_dir, shards,
                              shard=0, shards=4, smoke_limit=12, checkpoint_every=1)
    assert calls == ["S1-0", "S1-4"]
    resumed = []

    def record_resume(*args):
        resumed.append(args[1].entity_id)
        return original(*args)

    monkeypatch.setattr(inference, "score_record", record_resume)
    done0 = inference.infer_shard(tune_dir, test_dir, index_dir, shards,
                                  shard=0, shards=4, smoke_limit=12, checkpoint_every=1)
    assert resumed == ["S1-4", "S1-8"]
    assert done0["rows"] == 3
    with sqlite3.connect(shards / "shard_00/checkpoint.sqlite") as connection:
        assert [row[0] for row in connection.execute("SELECT position FROM outputs ORDER BY position")] == [0, 4, 8]
    for i in range(1, 4):
        inference.infer_shard(tune_dir, test_dir, index_dir, shards,
                              shard=i, shards=4, smoke_limit=12, checkpoint_every=1)
    before = (shards / "shard_00/matching_results.tsv").stat().st_mtime_ns
    monkeypatch.setattr(inference, "score_record", lambda *_: pytest.fail("completed shard rescored"))
    assert inference.infer_shard(tune_dir, test_dir, index_dir, shards,
                                  shard=0, shards=4, smoke_limit=12)["rows"] == 3
    assert (shards / "shard_00/matching_results.tsv").stat().st_mtime_ns == before
    output = tmp_path / "merged"
    assert inference.merge_shards(shards, test_dir, index_dir, output)["rows"] == 12
    assert inference.audit_output(output, test_dir)["rows"] == 12
    second = tmp_path / "merged_again"
    inference.merge_shards(shards, test_dir, index_dir, second)
    for name in inference.OUTPUT_FILES:
        assert (output / name).read_bytes() == (second / name).read_bytes()
    assert inference.benchmark(shards, test_dir, index_dir)["projected_full_test_hours"] > 0
    spec = importlib.util.spec_from_file_location("validator", ROOT / "resources/validate_submission.py")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    errors, warnings = validator.validate(str(output / "matching_results.tsv"),
                                          str(output / "candidate_pairs.tsv"), str(test_dir), check_ids=True)
    assert errors == [] and warnings == []


def test_identity_mismatch_and_incomplete_shard_rejected(synthetic, tmp_path):
    test_dir, index_dir, tune_dir = synthetic
    shards = tmp_path / "shards"
    run_four(synthetic, shards)
    with pytest.raises(ValueError, match="identity mismatch"):
        inference.infer_shard(tune_dir, test_dir, index_dir, shards,
                              shard=0, shards=4, smoke_limit=11)
    (shards / "shard_03/done.json").unlink()
    with pytest.raises(FileNotFoundError):
        inference.merge_shards(shards, test_dir, index_dir, tmp_path / "missing")


def test_corrupt_completed_shard_fails_closed(synthetic, tmp_path):
    test_dir, index_dir, tune_dir = synthetic
    shards = tmp_path / "shards"
    run_four(synthetic, shards)
    published = shards / "shard_00/matching_results.tsv"
    before = published.read_bytes()
    done_path = shards / "shard_00/done.json"
    done = json.loads(done_path.read_text())
    done["files"]["matching_results.tsv"] = "0" * 64
    done_path.write_text(json.dumps(done))
    with pytest.raises(ValueError, match="refusing to overwrite"):
        inference.infer_shard(tune_dir, test_dir, index_dir, shards,
                              shard=0, shards=4, smoke_limit=12)
    assert published.read_bytes() == before


@pytest.mark.parametrize("damage", ["missing", "malformed"])
def test_merge_rejects_missing_s1_or_malformed_target(synthetic, tmp_path, damage):
    test_dir, index_dir, _ = synthetic
    shards = tmp_path / "shards"
    run_four(synthetic, shards)
    shard = shards / "shard_00"
    path = shard / "candidate_pairs.tsv"
    lines = path.read_text().splitlines()
    if damage == "missing":
        lines.pop()
    else:
        lines[1] = lines[1].split("\t")[0] + "\tX-0"
    path.write_text("\n".join(lines) + "\n")
    done_path = shard / "done.json"
    done = json.loads(done_path.read_text())
    done["files"]["candidate_pairs.tsv"] = sha256(path)
    done_path.write_text(json.dumps(done))
    with pytest.raises(ValueError):
        inference.merge_shards(shards, test_dir, index_dir, tmp_path / "bad")


def test_duplicate_test_s1_detected_at_merge(synthetic, tmp_path):
    test_dir, index_dir, tune_dir = synthetic
    source = test_dir / "test_source1.tsv"
    lines = source.read_text().splitlines()
    lines[2] = lines[2].replace("S1-1", "S1-0", 1)
    source.write_text("\n".join(lines) + "\n")
    shards = tmp_path / "shards"
    run_four((test_dir, index_dir, tune_dir), shards)
    with pytest.raises(ValueError, match="Duplicate TEST S1"):
        inference.merge_shards(shards, test_dir, index_dir, tmp_path / "duplicate")


def test_smoke_and_full_test_gates_and_protected_hashes(tmp_path):
    with pytest.raises(ValueError, match="allow-full-test"):
        inference.validate_mode(None, False)
    with pytest.raises(ValueError, match="smoke limit"):
        inference.validate_mode(4001, False)
    with pytest.raises(ValueError, match="smoke limit"):
        inference.validate_mode(0, False)
    with pytest.raises(ValueError, match="cannot also"):
        inference.validate_mode(100, True)
    inference.validate_mode(None, True)
    inference.validate_mode(4000, False)
    with pytest.raises(ValueError, match="allow-full-test"):
        inference.infer_shard(tmp_path, tmp_path, tmp_path, tmp_path, smoke_limit=None)
    assert inference.EXPECTED_FULL_S1 == 1_732_544
    assert sha256(ROOT / "src/business_entity_resolution/exp007.py") == "0ff3836f1d53685004876c7f9d6392942894cd81d62440ff6537f03e12eb9774"
    assert sha256(ROOT / "src/business_entity_resolution/exp007_features.py") == "afeb9e66a42aa09209cb7799589630f836b2e8f1f7d294cacd26742a7b571959"

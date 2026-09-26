from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pytest

from business_entity_resolution import exp007
from business_entity_resolution.evaluation import evaluate_predictions
from business_entity_resolution.exp007_features import (FEATURE_NAMES, ENHANCED_FEATURE_NAMES,
    address, core_name, enhanced_row, normalize)
from business_entity_resolution.features import FEATURE_NAMES as OLD_FEATURE_NAMES


@dataclass
class Record:
    country: str = "FR"


def test_secondary_normalization_and_order():
    assert len(OLD_FEATURE_NAMES) == 51
    assert FEATURE_NAMES[:51] == OLD_FEATURE_NAMES
    assert FEATURE_NAMES[51:] == ENHANCED_FEATURE_NAMES
    assert normalize("Société Électricité") == "societe electricite"
    assert normalize("श्री राम") == "sri ram"
    assert normalize("https://www.Acme.com/Widgets") == "acme widgets"
    assert normalize("Bâtiment12 Rue") == "batiment 12 rue"
    assert core_name("Acme Société Anonyme SA") == "acme"
    assert core_name("Acme GmbH") == "acme"
    assert core_name("Bank of India") == "bank of india"
    assert normalize("In Business") == "in business"
    assert address("00012 Baker St").primary == "12"
    assert enhanced_row("Acme", "Acme", "0012 Road", "12 Road")["primary_number_exact"] == 1
    assert enhanced_row("Acme", "Acme", "12 Road", "13 Road")["primary_number_mismatch"] == 1
    assert enhanced_row("Acme", "Acme", "12 Road", "13 Road")["numeric_sets_disjoint"] == 1
    assert enhanced_row("Acme", "Acme", "75001 Paris", "75001 Paris")["postal_like_agreement"] == 1
    assert enhanced_row("Acme", "Acme", "75001 Paris", "75002 Paris")["postal_like_conflict"] == 1
    assert enhanced_row("Acme", "Acme", "SW1A London", "SW1A London")["postal_like_agreement"] == 1
    assert enhanced_row("Acme", "Acme", "", "12 Road")["left_address_missing_x_name"] == 1
    assert normalize("Österreich Handels GmbH") == "osterreich handels gmbh"


def test_exact_blocker_oracle_includes_singletons():
    truth = {"S1-a": {"S2-a", "S3-a"}, "S1-b": set(), "S1-c": {"S2-c"}}
    eligible = {"S1-a": {"S2-a", "S2-wrong"}, "S1-b": {"S3-wrong"}, "S1-c": set()}
    result = exp007.oracle(truth, eligible, {e: Record() for e in truth})
    prediction = {e: truth[e] & eligible[e] for e in truth}
    exact = evaluate_predictions(truth, prediction)
    assert result["overall"]["macro_fbeta"] == exact.macro_fbeta
    assert result["overall"]["false_positives"] == 0
    assert result["candidate_link_recall"] == 1/3
    assert result["perfect_candidate_coverage_rate"] == 1/3
    assert result["matched_with_zero_retained_rate"] == .5
    assert result["blocker_fn"] == 2
    assert exact.macro_fbeta != result["candidate_link_recall"]


def test_ownership_audit_and_exclusivity_tie_break():
    clear = exp007.ownership_audit([("S1-a", {"S2-a"}), ("S1-b", {"S3-a"})])
    assert clear["targets_with_multiple_owners"] == 0
    audit = exp007.ownership_audit([("S1-a", {"S2-a", "S3-a"}),
                                    ("S1-b", {"S2-a", "S3-a"})])
    assert audit["total_ground_truth_links"] == 4
    assert audit["s2_conflicts"] == audit["s3_conflicts"] == 1
    assert audit["max_s1_owners"] == 2
    scored = {e: {"rows": [[e, "S2-x", 0, "", ""]], "scores": {"base": np.asarray([score])},
                  "numeric": np.asarray([0.]), "guardian_has_match": 1.0}
              for e, score in (("S1-b", .9), ("S1-a", .9))}
    pred, dup = exp007._predict(scored, "base", (.8, .8), exclusivity=True)
    assert pred == {"S1-a": {"S2-x"}, "S1-b": set()}
    assert dup["duplicate_claims_before"] == 1
    assert dup["duplicate_claims_after"] == 0
    scored["S1-b"]["scores"]["base"] = np.asarray([.95])
    pred, _ = exp007._predict(scored, "base", (.8, .8), exclusivity=True)
    assert pred["S1-b"] == {"S2-x"}


def test_source_thresholds_guardian_and_conservative_tie():
    scored = {"S1-a": {"rows": [["S1-a", "S2-a", 1, "", ""], ["S1-a", "S3-x", 0, "", ""]],
                        "scores": {"base": np.asarray([.9, .7])}, "numeric": np.zeros(2),
                        "guardian_has_match": .9},
              "S1-b": {"rows": [["S1-b", "S2-x", 0, "", ""]],
                        "scores": {"base": np.asarray([.6])}, "numeric": np.zeros(1),
                        "guardian_has_match": .1}}
    pred, _ = exp007._predict(scored, "base", (.8, .95), guardian=True, guardian_threshold=.5)
    assert pred == {"S1-a": {"S2-a"}, "S1-b": set()}
    assert exp007._select_thresholds(scored, {"S1-a": {"S2-a"}, "S1-b": set()}, "base")[:2] == (.9, 1.000001)


def test_relative_ranks_and_bounded_cross_source():
    rows = [["S1-a", "S2-a", 0, "Acme", "12 Road"], ["S1-a", "S2-b", 0, "Other", "99 Road"],
            ["S1-a", "S3-a", 0, "Acme Ltd", "12 Road"]]
    scores = np.asarray([.9, .5, .8])
    matrix = np.zeros((3, len(FEATURE_NAMES)), np.float32)
    entity, relative, comparisons = exp007.decision_features(rows, matrix, scores)
    assert list(entity[:5]) == pytest.approx([.9, .8, .5, .1, .3])
    assert list(relative[:, 1]) == [1, 3, 2]
    assert list(relative[:, 2]) == [1, 2, 1]
    assert comparisons == 2
    assert relative[0, -5] > 0
    many = rows[:2]*10 + rows[2:]*10
    _, count = exp007.cross_evidence(many, np.linspace(1, 0, len(many)), top_k=3)
    assert count <= 9
    with pytest.raises(ValueError):
        exp007.cross_evidence(rows, scores, top_k=100)


def test_pair_artifact_identity_and_corruption_fail_closed(tmp_path):
    path = tmp_path / "fit"
    path.mkdir()
    (path / "fit.f32").write_bytes(b"\x00" * (4 * len(FEATURE_NAMES)))
    (path / "fit.tsv").write_text("\t".join(exp007.PAIR_COLUMNS) + "\nS1-a\tS2-a\t1\tAcme\t12 Road\n")
    base = {"blocker_sha256": "test", "feature_names": list(FEATURE_NAMES)}
    manifest = {"identity": base, "partition": "fit", "entities": 1, "pairs": 1,
                "matrix_sha256": exp007.sha256(path / "fit.f32"),
                "metadata_sha256": exp007.sha256(path / "fit.tsv")}
    (path / "manifest.json").write_text(json.dumps(manifest))
    assert exp007.validate_pairs(path, "fit", base) == manifest
    with pytest.raises(ValueError, match="identity"):
        exp007.validate_pairs(path, "fit", {"blocker_sha256": "changed"})
    (path / "fit.tsv").write_text((path / "fit.tsv").read_text() + "S1-b\tS2-b\t0\tOther\t99 Road\n")
    with pytest.raises(ValueError, match="checksum"):
        exp007.validate_pairs(path, "fit", base)


def test_evaluation_pair_generation_requires_frozen_policy():
    with pytest.raises(ValueError, match="Frozen TUNE"):
        exp007.generate("unused", "unused", "unused", "unused", "unused",
                        partition="evaluation", shard=0, shards=1)

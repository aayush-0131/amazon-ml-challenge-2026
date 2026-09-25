from __future__ import annotations

import pytest

from business_entity_resolution.evaluation import (
    competition_fbeta,
    entity_metrics,
    evaluate_predictions,
)


def test_correct_singleton() -> None:
    assert entity_metrics(set(), set()).fbeta == 1.0


def test_false_positive_singleton() -> None:
    score = entity_metrics(set(), {"S2-1"})
    assert score.precision == 0.0
    assert score.recall == 0.0
    assert score.fbeta == 0.0


def test_exact_positive_match() -> None:
    score = entity_metrics({"S2-1"}, {"S2-1"})
    assert score.precision == 1.0
    assert score.recall == 1.0
    assert score.fbeta == 1.0


def test_partial_recall() -> None:
    score = entity_metrics({"S2-1", "S3-1"}, {"S2-1"})
    assert score.precision == 1.0
    assert score.recall == 0.5
    assert score.fbeta == pytest.approx(5 / 6)


def test_false_positive() -> None:
    score = entity_metrics({"S2-1"}, {"S2-1", "S3-9"})
    assert score.precision == 0.5
    assert score.recall == 1.0
    assert score.fbeta == pytest.approx(5 / 9)


def test_multiple_matches() -> None:
    score = entity_metrics(
        {"S2-1", "S2-2", "S3-1"}, {"S2-1", "S2-2", "S3-9"}
    )
    assert score.precision == pytest.approx(2 / 3)
    assert score.recall == pytest.approx(2 / 3)
    assert score.fbeta == pytest.approx(2 / 3)


def test_macro_score_and_diagnostics_include_singletons() -> None:
    truth = {"S1-1": set(), "S1-2": {"S2-1", "S3-1"}}
    predictions = {"S1-1": set(), "S1-2": {"S2-1"}}

    result = evaluate_predictions(truth, predictions)

    assert competition_fbeta(truth, predictions) == pytest.approx(11 / 12)
    assert result.macro_fbeta == pytest.approx(11 / 12)
    assert result.true_positives == 1
    assert result.false_positives == 0
    assert result.false_negatives == 1
    assert result.correct_singletons == 1
    assert result.false_positive_singletons == 0


def test_evaluation_requires_exact_source1_coverage() -> None:
    with pytest.raises(ValueError, match="must exactly match"):
        evaluate_predictions({"S1-1": set()}, {})

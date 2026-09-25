"""Exact entity-level competition metric and useful link diagnostics."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EntityMetrics:
    precision: float
    recall: float
    fbeta: float
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass(frozen=True)
class EvaluationResult:
    n_entities: int
    beta: float
    macro_precision: float
    macro_recall: float
    macro_fbeta: float
    micro_precision: float
    micro_recall: float
    micro_fbeta: float
    true_positives: int
    false_positives: int
    false_negatives: int
    singleton_count: int
    correct_singletons: int
    false_positive_singletons: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _fbeta(precision: float, recall: float, beta: float) -> float:
    if beta <= 0:
        raise ValueError("beta must be positive")
    beta_squared = beta * beta
    denominator = beta_squared * precision + recall
    if denominator == 0:
        return 0.0
    return (1.0 + beta_squared) * precision * recall / denominator


def entity_metrics(
    truth: Collection[str],
    prediction: Collection[str],
    *,
    beta: float = 0.5,
) -> EntityMetrics:
    """Score one Source-1 entity using the competition singleton semantics."""

    if beta <= 0:
        raise ValueError("beta must be positive")
    truth_set = set(truth)
    prediction_set = set(prediction)

    if not truth_set:
        correct = not prediction_set
        value = 1.0 if correct else 0.0
        return EntityMetrics(
            precision=value,
            recall=value,
            fbeta=value,
            true_positives=0,
            false_positives=len(prediction_set),
            false_negatives=0,
        )

    true_positives = len(truth_set & prediction_set)
    false_positives = len(prediction_set - truth_set)
    false_negatives = len(truth_set - prediction_set)
    precision = (
        true_positives / len(prediction_set) if prediction_set else 0.0
    )
    recall = true_positives / len(truth_set)
    return EntityMetrics(
        precision=precision,
        recall=recall,
        fbeta=_fbeta(precision, recall, beta),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def evaluate_predictions(
    ground_truth: Mapping[str, Collection[str]],
    predictions: Mapping[str, Collection[str]],
    *,
    beta: float = 0.5,
    require_complete: bool = True,
) -> EvaluationResult:
    """Compute official macro F-beta plus macro/micro diagnostics.

    The official score is ``macro_fbeta``. Every ground-truth Source-1 entity is
    weighted equally, including singletons. With ``require_complete=True`` the
    prediction keys must match the ground-truth keys exactly, mirroring submission
    requirements.
    """

    if not ground_truth:
        raise ValueError("ground_truth must contain at least one entity")
    if beta <= 0:
        raise ValueError("beta must be positive")
    truth_ids = set(ground_truth)
    prediction_ids = set(predictions)
    if require_complete and truth_ids != prediction_ids:
        missing = sorted(truth_ids - prediction_ids)[:5]
        extra = sorted(prediction_ids - truth_ids)[:5]
        raise ValueError(
            "Prediction entity IDs must exactly match ground truth; "
            f"missing={missing!r}, extra={extra!r}"
        )
    if prediction_ids - truth_ids:
        extra = sorted(prediction_ids - truth_ids)[:5]
        raise ValueError(f"Predictions contain unknown Source-1 IDs: {extra!r}")

    n_entities = len(ground_truth)
    macro_precision_sum = 0.0
    macro_recall_sum = 0.0
    macro_fbeta_sum = 0.0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    singleton_count = 0
    correct_singletons = 0
    for source1_id, truth in ground_truth.items():
        prediction = predictions.get(source1_id, ())
        item = entity_metrics(truth, prediction, beta=beta)
        macro_precision_sum += item.precision
        macro_recall_sum += item.recall
        macro_fbeta_sum += item.fbeta
        true_positives += item.true_positives
        false_positives += item.false_positives
        false_negatives += item.false_negatives
        if not truth:
            singleton_count += 1
            correct_singletons += int(not prediction)

    predicted_links = true_positives + false_positives
    true_links = true_positives + false_negatives
    micro_precision = (
        true_positives / predicted_links if predicted_links else float(not true_links)
    )
    micro_recall = (
        true_positives / true_links if true_links else float(not predicted_links)
    )

    return EvaluationResult(
        n_entities=n_entities,
        beta=beta,
        macro_precision=macro_precision_sum / n_entities,
        macro_recall=macro_recall_sum / n_entities,
        macro_fbeta=macro_fbeta_sum / n_entities,
        micro_precision=micro_precision,
        micro_recall=micro_recall,
        micro_fbeta=_fbeta(micro_precision, micro_recall, beta),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        singleton_count=singleton_count,
        correct_singletons=correct_singletons,
        false_positive_singletons=singleton_count - correct_singletons,
    )


def competition_fbeta(
    ground_truth: Mapping[str, Collection[str]],
    predictions: Mapping[str, Collection[str]],
    *,
    beta: float = 0.5,
) -> float:
    """Return the exact official entity-level macro F-beta score."""

    return evaluate_predictions(
        ground_truth, predictions, beta=beta, require_complete=True
    ).macro_fbeta

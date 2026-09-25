"""Deterministic pair sampling, compact models, and threshold selection."""

from __future__ import annotations

import hashlib
import heapq
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import EvaluationResult, evaluate_predictions


@dataclass(frozen=True, slots=True)
class PairExample:
    source1_entity_id: str
    candidate_entity_id: str
    features: tuple[float, ...]
    label: int
    hardness: float


@dataclass(frozen=True)
class NegativeSamplingConfig:
    hard_negatives_per_entity: int = 20
    easy_negatives_per_entity: int = 3
    easy_hardness_ceiling: float = 55.0
    seed: int = 2028


def stable_pair_hash(source1_id: str, candidate_id: str, seed: int) -> int:
    payload = f"{seed}\0{source1_id}\0{candidate_id}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


class TrainingPairSampler:
    """Keep all positives, top hard negatives, and a small easy sample per S1."""

    def __init__(self, config: NegativeSamplingConfig | None = None) -> None:
        self.config = config or NegativeSamplingConfig()
        self.positives: list[PairExample] = []
        self.hard: dict[str, list[tuple[float, str, PairExample]]] = defaultdict(list)
        self.easy: dict[str, list[tuple[int, str, PairExample]]] = defaultdict(list)

    def add(
        self,
        source1_id: str,
        candidate_id: str,
        features: Sequence[float],
        *,
        label: int,
        hardness: float,
    ) -> None:
        example = PairExample(
            source1_id,
            candidate_id,
            tuple(float(value) for value in features),
            int(label),
            float(hardness),
        )
        if label:
            self.positives.append(example)
            return

        hard_heap = self.hard[source1_id]
        hard_item = (example.hardness, candidate_id, example)
        if len(hard_heap) < self.config.hard_negatives_per_entity:
            heapq.heappush(hard_heap, hard_item)
        elif hard_item[:2] > hard_heap[0][:2]:
            heapq.heapreplace(hard_heap, hard_item)

        if hardness <= self.config.easy_hardness_ceiling:
            pair_hash = stable_pair_hash(
                source1_id, candidate_id, self.config.seed
            )
            # Negative hashes make heap[0] the largest original hash.
            easy_item = (-pair_hash, candidate_id, example)
            easy_heap = self.easy[source1_id]
            if len(easy_heap) < self.config.easy_negatives_per_entity:
                heapq.heappush(easy_heap, easy_item)
            elif easy_item[:2] > easy_heap[0][:2]:
                heapq.heapreplace(easy_heap, easy_item)

    def finalize(self) -> tuple[list[PairExample], dict[str, int]]:
        hard_examples = [item[2] for heap in self.hard.values() for item in heap]
        hard_keys = {
            (example.source1_entity_id, example.candidate_entity_id)
            for example in hard_examples
        }
        easy_examples = [
            item[2]
            for heap in self.easy.values()
            for item in heap
            if (item[2].source1_entity_id, item[2].candidate_entity_id)
            not in hard_keys
        ]
        examples = sorted(
            [*self.positives, *hard_examples, *easy_examples],
            key=lambda example: (
                example.source1_entity_id,
                -example.label,
                example.candidate_entity_id,
            ),
        )
        counts = {
            "positive_count": len(self.positives),
            "hard_negative_count": len(hard_examples),
            "easy_negative_count": len(easy_examples),
            "total_count": len(examples),
        }
        return examples, counts


def examples_to_arrays(
    examples: Sequence[PairExample],
) -> tuple[np.ndarray, np.ndarray]:
    if not examples:
        raise ValueError("No pair examples were provided")
    return (
        np.asarray([example.features for example in examples], dtype=np.float32),
        np.asarray([example.label for example in examples], dtype=np.int8),
    )


def make_logistic_model(seed: int = 2028) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=300,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def make_hist_gradient_boosting_model(seed: int = 2028) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=160,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=seed,
        class_weight="balanced",
    )


def prediction_scores(model: object, features: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(features)
    return np.asarray(probabilities[:, 1], dtype=np.float64)


def select_global_threshold(
    ground_truth: Mapping[str, Collection[str]],
    scores: Mapping[str, Mapping[str, float]],
    thresholds: Sequence[float],
) -> tuple[float, EvaluationResult, list[dict[str, float | int]]]:
    rows: list[dict[str, float | int]] = []
    results: dict[float, EvaluationResult] = {}
    for threshold in thresholds:
        predictions = {
            source1_id: {
                candidate_id
                for candidate_id, score in scores.get(source1_id, {}).items()
                if score >= threshold
            }
            for source1_id in ground_truth
        }
        result = evaluate_predictions(ground_truth, predictions)
        results[float(threshold)] = result
        rows.append({"threshold": float(threshold), **result.to_dict()})
    chosen = max(
        results,
        key=lambda threshold: (results[threshold].macro_fbeta, threshold),
    )
    return chosen, results[chosen], rows

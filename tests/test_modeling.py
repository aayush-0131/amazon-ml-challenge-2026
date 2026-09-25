import numpy as np

from business_entity_resolution.modeling import (
    NegativeSamplingConfig,
    TrainingPairSampler,
    make_logistic_model,
    prediction_scores,
    select_global_threshold,
)


def test_hard_negative_sampling_counts_and_determinism() -> None:
    config = NegativeSamplingConfig(
        hard_negatives_per_entity=2,
        easy_negatives_per_entity=1,
        easy_hardness_ceiling=25.0,
        seed=77,
    )

    def sample() -> tuple[list[tuple[str, str, int]], dict[str, int]]:
        sampler = TrainingPairSampler(config)
        sampler.add("S1-1", "S2-p", [1.0], label=1, hardness=100.0)
        for index, hardness in enumerate((10.0, 20.0, 80.0, 90.0)):
            sampler.add(
                "S1-1",
                f"S2-n{index}",
                [hardness],
                label=0,
                hardness=hardness,
            )
        examples, counts = sampler.finalize()
        rows = [
            (item.source1_entity_id, item.candidate_entity_id, item.label)
            for item in examples
        ]
        return rows, counts

    first, counts = sample()
    second, second_counts = sample()
    assert first == second
    assert counts == second_counts
    assert counts == {
        "positive_count": 1,
        "hard_negative_count": 2,
        "easy_negative_count": 1,
        "total_count": 4,
    }
    negative_ids = {candidate for _, candidate, label in first if label == 0}
    assert {"S2-n2", "S2-n3"} <= negative_ids


def test_model_prediction_plumbing() -> None:
    x = np.asarray([[0.0], [0.1], [0.9], [1.0]], dtype=np.float32)
    y = np.asarray([0, 0, 1, 1], dtype=np.int8)
    model = make_logistic_model(seed=5).fit(x, y)
    scores = prediction_scores(model, x)
    assert scores.shape == (4,)
    assert scores[0] < scores[-1]
    assert np.all((scores >= 0) & (scores <= 1))


def test_threshold_selection_handles_singletons_and_multiple_matches() -> None:
    truth = {
        "S1-singleton": frozenset(),
        "S1-multi": frozenset({"S2-1", "S3-1"}),
    }
    scores = {
        "S1-singleton": {"S2-fp": 0.4},
        "S1-multi": {"S2-1": 0.9, "S3-1": 0.7, "S2-fp": 0.2},
    }
    threshold, result, rows = select_global_threshold(
        truth, scores, [0.3, 0.5, 0.8]
    )
    assert threshold == 0.5
    assert result.macro_fbeta == 1.0
    assert result.correct_singletons == 1
    assert len(rows) == 3

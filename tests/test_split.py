from __future__ import annotations

import pandas as pd

from business_entity_resolution.split import stratified_entity_split


def test_entity_split_is_deterministic_disjoint_and_stratified() -> None:
    records = pd.DataFrame(
        [
            {"entity_id": f"S1-{index}", "country": country}
            for country in ("US", "India")
            for index in (range(10) if country == "US" else range(10, 20))
        ]
    )
    truth = {
        entity_id: ({f"S2-{position}"} if position % 2 else set())
        for position, entity_id in enumerate(records["entity_id"])
    }

    first = stratified_entity_split(records, truth, validation_fraction=0.2)
    shuffled = stratified_entity_split(
        records.sample(frac=1.0, random_state=7), truth, validation_fraction=0.2
    )

    assert first == shuffled
    assert first.train_ids.isdisjoint(first.validation_ids)
    assert first.train_ids | first.validation_ids == set(records["entity_id"])
    assert len(first.validation_ids) == 4

    validation_records = records.loc[
        records["entity_id"].isin(first.validation_ids)
    ]
    assert validation_records.groupby("country").size().to_dict() == {
        "India": 2,
        "US": 2,
    }
    assert sum(not truth[entity_id] for entity_id in first.validation_ids) == 2

from business_entity_resolution.multipass import (
    CandidateBudget,
    MultiPassBlocker,
    MultiPassConfig,
    select_candidates_for_budget,
)
from business_entity_resolution.sampling import SourceRecord


def _budget(total: int = 10) -> CandidateBudget:
    return CandidateBudget(
        name="test",
        total_per_source=total,
        exact_name=total,
        exact_address=total,
        name_signature=total,
        name_token=total,
        address_token=total,
        numeric_address=total,
        char_name=total,
    )


def test_multipass_is_deterministic_and_preserves_multiple_matches() -> None:
    queries = [
        SourceRecord("S1-1", "Acme Trading", "12 Market Road 400001", "IN"),
        SourceRecord("S1-2", "Solo Shop", "", "US"),
    ]
    targets = [
        SourceRecord("S2-2", "Trading Acme", "12 Market Rd 400001", "IN"),
        SourceRecord("S2-1", "Acme Trading", "12 Market Road 400001", "IN"),
        SourceRecord("S2-3", "Acme Trading", "12 Market Road 400001", "US"),
    ]
    config = MultiPassConfig(max_pool_per_source=20)

    def retrieve() -> list[tuple[str, str]]:
        blocker = MultiPassBlocker(queries, source="S2", config=config)
        for target in targets:
            blocker.process_record(target)
        selected = select_candidates_for_budget(blocker.finalize(), _budget())
        return [(row.source1_entity_id, row.candidate_entity_id) for row in selected]

    first = retrieve()
    second = retrieve()
    assert first == second
    assert set(first) == {("S1-1", "S2-1"), ("S1-1", "S2-2")}
    assert ("S1-1", "S2-3") not in first
    assert not any(source1_id == "S1-2" for source1_id, _ in first)


def test_candidate_budget_is_bounded() -> None:
    queries = [SourceRecord("S1-1", "Common Name", "10 Main Street", "US")]
    blocker = MultiPassBlocker(
        queries,
        source="S2",
        config=MultiPassConfig(max_pool_per_source=10),
    )
    for index in range(5):
        blocker.process_record(
            SourceRecord(
                f"S2-{index}", "Common Name", f"{index} Main Street", "US"
            )
        )
    selected = select_candidates_for_budget(blocker.finalize(), _budget(total=2))
    assert len(selected) == 2


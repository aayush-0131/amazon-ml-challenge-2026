"""Label-free bounded planning for selective_v1 and selective_v2_compact."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations

# Terms use the existing country-scoped DF and exact schema-2 field/token keys.
Term = tuple[int, str, str]
Pair = tuple[Term, Term]


def term_kind(term: Term) -> str:
    _, field, token = term
    return "name" if field == "name" else "numeric" if token.isdecimal() else "address"


def pair_kind(pair: Pair) -> tuple[str, str]:
    return tuple(sorted(term_kind(term) for term in pair))


@dataclass(frozen=True)
class IntersectionPlan:
    terms: tuple[Term, ...]
    pairs: tuple[Pair, ...]
    considered: int
    skipped_anchor: int
    skipped_selectivity: int
    skipped_term_quota: int


def plan_intersections(
    terms: list[Term], country_count: int, *, term_limit: int, probe_limit: int,
    ordinary_anchor_df: int, selective_anchor_df: int, max_expected_hits: int,
    max_probes_per_term: int, gate_all_pairs: bool = False,
) -> IntersectionPlan:
    """Plan all probes before SQL; changing input order cannot change the plan.

    Reserve the rarest available name/address/numeric term, then fill remaining
    slots by DF. Pairs are oriented by (DF, field, token), so the smaller posting
    list anchors the SQL join. The independence proxy df_left*df_right/N is used
    only for ranking and the larger-anchor gate, never as a true hit estimate.

    Greedy fairness prioritizes less-used endpoints, then less-used pair kinds,
    then selectivity. A hard term-use quota prevents star-shaped probe schedules.
    Enumeration is O(term_limit**2) in Python; issued SQL probes <= probe_limit.
    Compact mode sets gate_all_pairs=True to apply the proxy gate even below
    the ordinary anchor bound. The default preserves selective_v1 exactly.
    """
    if country_count < 0 or any(value <= 0 for value in (
        term_limit, probe_limit, ordinary_anchor_df, selective_anchor_df,
        max_expected_hits, max_probes_per_term,
    )):
        raise ValueError("Invalid intersection scheduling bounds")
    ordered = sorted(set(term for term in terms if term[0] > 0))
    seeds = {}
    for term in ordered:
        seeds.setdefault(term_kind(term), term)
    pool = sorted(seeds.values())[:term_limit]
    reserved = set(pool)
    pool.extend(term for term in ordered if term not in reserved)
    pool = sorted(pool[:term_limit])
    candidates = list(combinations(pool, 2))
    eligible = []
    skipped_anchor = skipped_selectivity = 0
    for left, right in candidates:
        if left[0] > selective_anchor_df:
            skipped_anchor += 1
        elif (gate_all_pairs or left[0] > ordinary_anchor_df) and (
            country_count == 0 or left[0] * right[0] > max_expected_hits * country_count
        ):
            skipped_selectivity += 1
        else:
            eligible.append((left, right))

    usage: Counter[Term] = Counter()
    kind_usage: Counter[tuple[str, str]] = Counter()
    selected = []
    skipped_quota = 0
    while eligible and len(selected) < probe_limit:
        permitted = [pair for pair in eligible if all(usage[term] < max_probes_per_term for term in pair)]
        skipped_quota += len(eligible) - len(permitted)
        eligible = permitted
        if not eligible:
            break
        def priority(pair: Pair) -> tuple:
            left, right = pair
            return (
                max(usage[left], usage[right]), usage[left] + usage[right],
                kind_usage[pair_kind(pair)], left[0] * right[0], left, right,
            )
        pair = min(eligible, key=priority)
        eligible.remove(pair)
        selected.append(pair)
        usage.update(pair)
        kind_usage[pair_kind(pair)] += 1
    return IntersectionPlan(tuple(pool), tuple(selected), len(candidates),
                            skipped_anchor, skipped_selectivity, skipped_quota)

# EXP002d — EXP002c retrieval miss analysis

One retrieval per S1/source; all cap scenarios reuse that pool. These are TRAIN diagnostics.

| Slice | True links | Discovery misses | Cap/ranking misses | Pre-cap recall | Final recall |
|---|---:|---:|---:|---:|---:|
| ALL | 17207 | 1184 | 382 | 93.1191% | 90.8991% |
| India | 6970 | 702 | 24 | 89.9283% | 89.5839% |
| US | 10237 | 482 | 358 | 95.2916% | 91.7945% |
| S2 | 8316 | 573 | 132 | 93.1097% | 91.5224% |
| S2|India | 3361 | 285 | 10 | 91.5204% | 91.2229% |
| S2|US | 4955 | 288 | 122 | 94.1877% | 91.7255% |
| S3 | 8891 | 611 | 250 | 93.1279% | 90.3160% |
| S3|India | 3609 | 417 | 14 | 88.4456% | 88.0576% |
| S3|US | 5282 | 194 | 236 | 96.3271% | 91.8591% |

DF counterfactuals change one field at a time. Eligibility ignores query-term limits; recoverability respects current term ordering, term limits, minimum lengths and raw country equality. Potential recovery is discovery-only and does not guarantee final selection or runtime safety. Overlapping recoveries must not be summed across thresholds or fields.

Cap modes: `total_only` preserves current per-pass rank limits; `widen_pass_limits` also raises each active pass limit to at least the proposed cap. Neither mode adds candidates to the retrieved pool. Blank final-source rank means the candidate failed per-pass eligibility; pool-source rank is provided separately.

Intersection reasons are non-exclusive across shared token combinations. Actual probe order and overflow decisions are captured during retrieval; hit counts stop at max_intersection_hits+1. No new probes are run for counterfactuals. `other_unresolved` indicates an attempted non-overflow combination apparently sharing a missing true link and needs integrity investigation.

## Next-step evidence

- India: prioritize candidate discovery (702 discovery vs 24 cap/ranking misses).
- US: prioritize candidate discovery (482 discovery vs 358 cap/ranking misses).
- name: best single-field theoretical recovery is 209 discovery links at DF 1000; validate cost and final recall before adoption.
- address: best single-field theoretical recovery is 179 discovery links at DF 640; validate cost and final recall before adoption.
- numeric: best single-field theoretical recovery is 241 discovery links at DF 5000; validate cost and final recall before adoption.
- At cap 120/source, `total_only` recovers 358 additional links from the same pool, with final recall 92.9796% and 90.99 candidates/S1. This cannot recover discovery misses.
- At cap 120/source, `widen_pass_limits` recovers 253 additional links from the same pool, with final recall 92.3694% and 124.41 candidates/S1. This cannot recover discovery misses.

No index rebuild, model fitting, TEST inference or recommendation to deploy a counterfactual is implied. Review miss CSVs and country/source slices before designing the next retrieval change.

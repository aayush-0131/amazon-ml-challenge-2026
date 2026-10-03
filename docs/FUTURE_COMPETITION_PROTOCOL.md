# Future competitions — independent-thinking and delivery protocol

**Owner:** Human team leader and teammates. **Role of AI:** technical critic, implementation assistant, source-finder and hypothesis generator, *not* unquestioned decision authority.

## Before committing
- Verify official eligibility, team-size rules, licenses, compute policy, data restrictions, judging metric, deadlines/time zones, public/private split, and all output fields including packaging.
- Each human teammate independently writes a short explanation of: (1) the task and non-obvious constraints; (2) a 10-record worked example; (3) how the scorer handles empty cases; (4) two plausible failure modes; (5) a proposed simple baseline.
- Compare the explanations and resolve disagreements against the official problem statement. Have AI explicitly challenge the accepted interpretation.
- Preserve source links with download dates and artifact hashes. A forum post or competitor's README is not an official rule.

## Architecture review gate (early; time-boxed)
Build a design matrix with **three substantively different** candidates, with input/output semantics, training data, recall ceiling, compute cost, expected failure modes, licensing, test-shift strategy, and package footprint. Have a teammate argue *against* the preferred design. Check at least one published paper/open-source baseline when allowed. AI must include a counterproposal and explicit assumptions. **Do not promise unmeasured gains.**

## Experiment and validation gates
1. Verify a tiny exact evaluator against the organizer's metric, including zero-match records and multiple links; ensure entity-macro is not accidentally replaced by link-micro.
2. Freeze entity-disjoint FIT/TUNE/EVALUATION IDs; do not repeatedly tune on EVALUATION or the public leaderboard without labeling the resulting bias.
3. Measure raw retrieval recall, eligible/capped recall, all-links retention, candidate quantiles, and runtime separately. Report per-source, per-country, singleton and multiplicity slices.
4. Audit model-stage false negatives on *retrieved* positives; sample decoys and numeric/name/address disagreement classes. Use documented source-specific features and cross-encoder alternatives only if feasible and legally permitted.
5. Any unseen language/country is an explicit open-set test plan; check data leakage when learning normalization or adaptation statistics from unlabeled test records.
6. Preserve rejected experiments, versions and seed/hash provenance. Distinguish synthetic tests from challenge-data metrics.
7. Before freezing: produce reproducible outputs from frozen weights/code, rerun the official validator and verify candidate-match subset invariants.

## Delivery gate (start well before final day)
- Generate a measured, representative candidate file and estimate total final *compressed* size; do not assume a matching-results file is the whole submission.
- Rehearse packaging in the exact required directory structure. Verify file names, runtime dependencies, model licenses and reproducibility docs.
- Compare archive size to the portal limit with margin; seek organizer instructions *before* the deadline if the contract is incompatible.
- Record hashes of outputs, archive, code commit, and submission receipt. Never replace a complete package with a partial one silently. A successful portal receipt is **not** review acceptance.

## 72-hour example allocation (adjust to real contest)
- Hours 0–6: independent reading, rule audit, sample metrics, baselines and competing architectures.
- Hours 6–24: data diagnostics, blocker upper bounds, first working submission and packaging smoke.
- Hours 24–48: targeted measured improvements and differentiated approaches, with gates.
- Hours 48–60: decide policy based on held-out evidence; freeze experiments.
- Hours 60–72: full inference, validator, size/reproduction checks, upload evidence and fallback plan.

## Decision record (one paragraph per major change)
`Decision | Alternatives | Evidence on same split | Assumptions | Compute/size cost | Risks | Human owner | Freeze/rollback`.

A recommendation from ChatGPT, Codex, a paper, or a high-scoring competitor is never itself a measured local result. The human lead must be able to explain *why* the decision is supported.

## After results
Write an uncertainty-calibrated retrospective: **observed**, **plausible**, **unknown**. Never assign reasons for selection/rejection without organizer evidence. Publish only what challenge rules and third-party licenses allow. Preserve frozen competition artifacts, and keep post-result learning separate from the submitted solution.

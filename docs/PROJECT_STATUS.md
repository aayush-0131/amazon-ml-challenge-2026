# Amazon ML Challenge 2026 — Project Status

## Current Phase

Business Entity Resolution challenge.

Current development branch:

`feat/exp002c-fast-postings`

Current production decision:

**EXP002b rejected. EXP002c postings implementation ready for AWS 1k/5k verification; results pending.**

---

## Completed Work

### Phase 1 — Dataset / Evaluation

Status: COMPLETE

- exact competition macro-F0.5 evaluator implemented;
- deterministic development splitting implemented;
- complete train/test audit completed;
- TEST contains open-set country France;
- TRAIN country-link audit found zero cross-country true links.

### EXP001 — Lexical Baseline

Status: REVISE

- blocking recall: 86.419%
- held-out macro F0.5: 0.627204
- macro precision: 0.719934
- macro recall: 0.496254
- mean candidates/S1: 49.407

Main weaknesses:

- blocker loses too many true links;
- many retrieved true links are rejected by the rule matcher.

### EXP002a — Query-Side Multi-Pass

Status: REJECTED

Reason:

Runtime architecture did not scale.

### EXP002b — SQLite FTS5 Indexed Blocker

Status: REJECTED

5k benchmark:

- positive-link recall: 81.723%
- all true links retained: 63.923%
- mean candidates: 198.64
- median/p95/p99/max: 200
- query time: 5,030.5 seconds
- peak RSS: ~593 MiB

Persistent index construction itself succeeded:

- S2 index: ~2.56 GiB, 185.5 s
- S3 index: ~2.62 GiB, 193.1 s

Decision:

Do not scale to 20k.
Do not train the learned matcher yet.

---

## Current Best Validated Model

EXP001

Held-out macro F0.5:

`0.627204`

No leaderboard submission yet.

---

## Immediate Work Queue

1. EXP002c implementation: schema-2 country-scoped postings and persisted DF,
   exact passes, bounded rare lists/common-token intersections, batched fetches,
   explicit `configs/exp002c_postings.json`, query diagnostics and synthetic tests.
2. On AWS, build the new indexes once, then benchmark 1k and 5k S1 using the
   same persisted indexes. Commands/artifacts are in `experiments/EXP002.md`.
3. Require high recall and practical runtime.
4. If retrieval passes, benchmark 20k.
5. Train learned pair matcher using hard negatives.
6. Tune using entity-level macro F0.5.
7. Run full TEST inference on AWS.
8. Generate:
   - matching_results.tsv
   - candidate_pairs.tsv
9. Run official Amazon validator.
10. Make leaderboard Submission #1.

Only steps 1–3 are currently authorized. No EXP002c full-data runtime, peak RAM
or recall is known. Schema-1 FTS indexes and prior measured results are retained;
schema-2 builds use new filenames and atomic `.building.sqlite` publication.
The revised benchmark refuses samples over 5k. It does not invoke training.
Required gate: >=96% positive-link recall (stretch 98%), materially improved
matched all-links retention over EXP001's 68.416%, ideally <=80 candidates/S1,
no near-universal cap saturation, and practical projected query runtime.

---

## Infrastructure

### Mac

Development, Git, lightweight testing.

### Codex

Bounded implementation/debugging tasks only.

### AWS

Heavy indexing, large validation and full inference.

Current AWS instance may be stopped between compute stages to preserve credits.

---

## Hard Constraints

- no external entity/business lookup;
- no geocoding API;
- no external augmentation;
- raw challenge data must not be committed;
- France must remain supported as an unseen/open-set country string;
- every serious experiment must be reproducible from a commit/config;
- no long-running compute inside Codex.

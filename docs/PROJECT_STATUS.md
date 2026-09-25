# Amazon ML Challenge 2026 — Project Status

## Current Phase

Business Entity Resolution challenge.

Current development branch:

`feat/exp002d-miss-analysis`

Current production decision:

**EXP002c: REVISE — architecture retained; recall gate not passed.
EXP002d miss-analysis implementation ready for AWS execution; diagnostic results pending.**

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

1. Run `scripts/analyze_exp002c_misses.py` on AWS with the existing schema-2
   indexes and exact EXP002c 5k hash sample (seed 2032).
2. Review discovery versus cap/ranking misses, country/source slices, theoretical
   DF recoverability and same-pool cap scenarios. Commands/artifacts are in
   `experiments/EXP002.md`.
3. Design the next retrieval revision from that evidence, requiring high recall
   and practical runtime before model work.
4. If retrieval passes, benchmark 20k.
5. Train learned pair matcher using hard negatives.
6. Tune using entity-level macro F0.5.
7. Run full TEST inference on AWS.
8. Generate:
   - matching_results.tsv
   - candidate_pairs.tsv
9. Run official Amazon validator.
10. Make leaderboard Submission #1.

Current authorized scope is miss-analysis implementation/tests/documentation
and its AWS handoff. Do not rebuild indexes, run 20k, train or infer on TEST.
EXP002c's AWS 5k final/pre-cap recall is 90.8991%/93.1191%, matched-all retention
75.7512%, mean candidates 63.8378, query time 71.386 s and peak RSS approximately
267 MB. Cap hits remain 60.29% per source query. The approximately 6.87-hour
full-test query projection is not a measured full-test run.
India's 89.5839% final / 89.9283% pre-cap recall primarily indicates discovery
loss; US's 91.7945% / 95.2916% indicates more selection loss. S3 India is weakest
(88.0576% / 88.4456%). EXP002d has no Amazon-data results yet.
All prior evidence is retained. The diagnostic opens schema-2 indexes read-only;
optional probe tracing changes neither index storage nor candidate selection.
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

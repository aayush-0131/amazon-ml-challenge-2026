# Amazon ML Challenge 2026 — Project Status

## Current Phase

Business Entity Resolution challenge.

Current development branch:

`feat/exp002e-selective-intersections`

Current production decision:

**EXP002c: REVISE — architecture retained; recall gate not passed.
EXP002d AWS miss analysis complete. EXP002e selective intersections implemented;
ready for AWS 1k comparison only, with no EXP002e recall/runtime measured yet.**

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

1. Run AWS 1k for EXP002e intersection-only, moderate and balanced variants,
   reusing completed schema-2 indexes. Exact commands: `experiments/EXP002.md`.
2. Review final/pre-cap recall, India/S3 India, all-links retention, candidate
   counts, intersection probes/overflow, CPU time and RAM across all variants.
3. Run 5k only for variants explicitly authorized after that review, using
   `--reviewed-1k` and a fresh output directory. Do not automatically run 20k.
4. Require the recall/runtime gate before separately authorizing larger runs
   or learned-matcher work. EXP001 remains the best validated matcher.

Current authorized Codex scope is code/config/tests/documentation and AWS
handoff only. Do not rebuild indexes, run Amazon benchmarks, train or infer
on TEST. EXP002e keeps schema 2 and the historical EXP002c config unchanged.
The new scheduler is opt-in (`selective_v1`); legacy is the default.
EXP002c's AWS 5k final/pre-cap recall is 90.8991%/93.1191%, matched-all retention
75.7512%, mean candidates 63.8378, query time 71.386 s and peak RSS approximately
267 MB. Cap hits remain 60.29% per source query. The approximately 6.87-hour
full-test query projection is not a measured full-test run.
India's 89.5839% final / 89.9283% pre-cap recall primarily indicates discovery
loss; US's 91.7945% / 95.2916% indicates more selection loss. S3 India is weakest
(88.0576% / 88.4456%).

EXP002d evidence (`c5236f9`, same 5k/seed 2032/17,207 links) identifies 1,184
discovery misses and 382 cap misses. Overlapping intersection reasons include
706 omitted terms, 525 anchors above the DF bound, 609 pair-budget omissions,
8 overflows, 81 without usable combinations and zero unresolved. Even the
aggressive single-token DF union leaves 680 discovery misses: 399 omitted terms,
386 anchor failures, 196 pair-budget omissions, 6 overflows, 53 without a usable
combination, zero unresolved; only 2 have no shared lexical/numeric token.
Moderate/balanced/aggressive DF unions theoretically recover 202/332/504 misses;
they are not measured production retrieval results or runtime guarantees.
Total-only cap 60 gives historical-pool recall 92.6077%, mean 79.6254 candidates
and matched-all 79.8138%. EXP002e therefore keeps active pass limits at 40 and
sets only total_per_source to 60.

EXP002e considers up to 16 terms / 120 Python pairs, issues at most 24 SQL
probes with at most 6 per term, and permits anchors up to DF 50,000 only with
the documented selectivity gate above DF 5,000. Overflow remains whole-pair
rejection at >150 hits. These bounds still need AWS runtime validation.
All prior evidence is retained. Full synthetic suite: **64 passed**. Tests
verify legacy candidate equivalence, bounded/diverse scheduling, Unicode
numeric terms, read-only schema-2 reuse, overflow rejection and no target scan.
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

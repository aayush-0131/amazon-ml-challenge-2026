# Amazon ML Challenge 2026 — Project Status

## Current Phase

Business Entity Resolution challenge.

Current development branch:

`feat/exp002f-compact-intersections`

Current production decision:

**EXP002c: REVISE — architecture retained; recall gate not passed.
EXP002e: REVISE — discovery gate passed, production cost/candidate pressure failed.
EXP002f: implementation only / awaiting AWS 1k frontier.**

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

1. Run AWS 1k for EXP002f compact_6, compact_10, compact_14 and compact_18,
   reusing completed schema-2 indexes. Exact commands: `experiments/EXP002.md`.
2. Review final/pre-cap recall, India/S3 India, all-links retention, candidate
   counts, intersection probes/overflow, CPU time and RAM across all variants.
3. Run 5k only for variants explicitly authorized after that review, using
   `--reviewed-1k` and a fresh output directory. Do not automatically run 20k.
4. Require the recall/runtime gate before separately authorizing larger runs
   or learned-matcher work. EXP001 remains the best validated matcher.

Current authorized Codex scope is code/config/tests/documentation and AWS
handoff only. Do not rebuild indexes, run Amazon benchmarks, train or infer
on TEST. EXP002f keeps schema 2 and historical EXP002c/e configs unchanged.
The new scheduler is opt-in (`selective_v2_compact`); legacy remains the default.
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

EXP002e AWS 1k evidence (`4ac02fc`): intersection-only final/pre-cap recall
93.5847%/96.9330%, matched-all 82.1206%, raw pool 414.509/source, unique
intersection additions 373.2665/source, probes 20.7995/source, overflows
3.274/source, mean selected 119.435/S1 and 98.2% cap hits. India pre-cap is
94.5856%; S3 India is 94.0645%: the overall discovery gate is not a slice-wide
success. Moderate/balanced expand raw pools to 558.625/913.6365 per source
while final recall falls to 93.5003%/93.4159%. Do not expand singleton DFs.

EXP002f retains 16 terms, DF 60/40/250, total cap 60/source and active pass
limits 40. Frontier (probes / per-term quota / unique-intersection budget):
6/3/100, 10/4/150, 14/5/200, 18/6/250. It applies the DF-product/population
gate to all pairs and stops before issuing another probe once a completed
non-overflow probe reaches the output budget. The complete successful result
is retained, so the budget can overshoot by at most 149 IDs. Existing exact/
singleton candidates and duplicate intersection IDs do not consume that budget.
No EXP002f recall/runtime/RAM measurements exist yet; historical results are
preserved and not rerun. Full synthetic suite: **79 passed**. Tests cover
historical v1 planning, compact stopping, complete crossing results, overflow,
selectivity skips, read-only reuse, no target scan/build and AWS guardrails.
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

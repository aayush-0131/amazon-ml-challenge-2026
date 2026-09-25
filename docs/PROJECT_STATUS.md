# Amazon ML Challenge 2026 — Project Status

## Current Phase

Business Entity Resolution challenge.

Current development branch:

`feat/exp002-multipass-learned-matcher`

Current production decision:

**EXP002b rejected. EXP002c fast-postings blocker is next.**

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

1. Implement EXP002c fast postings retrieval.
2. Benchmark 1k and 5k S1.
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

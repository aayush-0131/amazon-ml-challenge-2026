# Amazon ML Challenge 2026 — Project Status

## Round 1 status

**COMPLETE.**

Team: **BlackList**  
Members: Aayush Jha, Anish Singh  
Final Round-1 experiment: **EXP007**  
Public leaderboard macro F0.5: **0.916321**

Production code commit:

`63100d932080e2a783e7dc43eab618abc0d92ad0`

The production commit is preserved on:

`submission/round1-exp007`

The pre-final-main state is preserved on:

`archive/pre-round1-main`

## Final EXP007 policy

- model: scikit-learn HistGradientBoostingClassifier
- ordered pairwise features: 83
- legacy features: 51
- enhanced EXP007 features: 32
- S2 threshold: 0.965
- S3 threshold: 0.956
- guardian: disabled
- exclusivity: disabled
- numeric penalty: 0
- calibration: raw HGB probability
- relative-score and cross-source models: TUNE ablations only; not deployed

Development partitions:

- FIT: 5,999 Source-1 entities
- TUNE: 2,001
- untouched EVALUATION: 12,000
- deterministic EXTRA_FIT: 50,000

Untouched EVALUATION:

- macro F0.5: 0.9384071620652703
- macro precision: 0.9650289021
- macro recall: 0.8835666997
- true positives: 36,723
- false positives: 880
- false negatives: 4,869
- singleton accuracy: 0.9059701
- blocker false negatives: 1,677
- matcher / decision-policy false negatives: 3,192

## Final TEST evidence

- Source-1 rows: 1,732,544
- candidate links: 326,418,278
- selected match links: 5,753,590
- full Source-1 coverage: true
- every selected match is contained in the scored candidate set

Output SHA-256:

- `matching_results.tsv`: `956e80579560c0f8090bec356aae3e849567cf41fe4e0b3079a09ce93a02b20c`
- `candidate_pairs.tsv`: `b39ac0fc31defc311686fc267dd71631537c6f718844a13543d3c046291e0988`

Verified final ZIP SHA-256:

`4174b3d7517628b0b0a58357a28139a005eddf0a77f557f8ae0a67e18725ec7f`

The final ZIP is intentionally not committed to Git.

## Submission progression

- EXP003 public leaderboard: 0.880246
- EXP006 public leaderboard: 0.892649
- EXP007 public leaderboard: 0.916321

## EXP008

Branch:

`exp/exp008-blocker-recall`

Status: **preserved, not deployed in Round 1**.

The TRAIN-only secondary S2 and S3 indexes completed successfully. EXP008 is a
bounded blocker-recall diagnostic with TUNE and EVALUATION commands; it is not a
complete TEST-inference pipeline. No EXP008 policy was used for the final
Round-1 submission.

## Repository / data policy

- competition-provided data only
- no external business lookup or geocoding
- raw challenge data is never committed
- generated TEST TSVs and large SQLite/model artifacts are not committed
- serious experiment claims must map to reproducible committed code and frozen evidence

## Post-submission closeout — 3 October 2026

- Unstop confirmed **Code Submission upload** on 2 October 2026 at approximately 12:48 IST.
- The full ZIP archive exceeded the portal's 1,024 MB upload limit. The smaller successful fallback upload **omitted mandatory `output/candidate_pairs.tsv`**; this was disclosed. Organizer technical acceptance has **not** been confirmed.
- Team BlackList is **not in the published Top 50** results shown on 3 October. No private score or organizer-specific explanation is established; do not infer that the missing file alone determined this outcome.
- EXP007 is frozen and EXP008 remains an unsubmitted diagnostic branch. No post-result modelling changes are authorized.
- See [postmortem](POSTMORTEM_2026.md), [all Top 50 teams / technical evidence](TOP50_RESEARCH_2026.md), [submission/archive register](SUBMISSION_EVIDENCE_2026.md) and [future competition protocol](FUTURE_COMPETITION_PROTOCOL.md).

## Remaining manual actions

1. Capture fallback archive SHA-256; preserve the two original ZIPs, final screenshots, Unstop receipt and organizer correspondence.
2. Inspect AWS across Regions and billing; record actual remaining charges rather than assuming resources were shut down.
3. Follow only official organizer review announcements; do not imply technical acceptance without explicit confirmation.
4. Complete repository licensing decision separately with the team and third-party dependency licenses before promising that the code is an open-source release.


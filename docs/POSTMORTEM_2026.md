# Amazon ML Challenge 2026 — Team BlackList retrospective

**Status:** Round-1 predictions and code submission completed; not listed in the published Top 50.  
**Authors/team:** Team BlackList (Aayush Jha and Anish Singh).  
**Prepared:** 2026-10-03. **Purpose:** Evidence-based postmortem, not a revised solution.  
**Original production commit:** `63100d932080e2a783e7dc43eab618abc0d92ad0`. **Do not modify frozen artifacts.**

## 1. Result and what is not known

- EXP007 public leaderboard macro F0.5: **0.916321**. The prior EXP003/EXP006 public scores were 0.880246 and 0.892649.
- Frozen 12,000-entity EVALUATION macro F0.5: **0.9384071620652703**; macro precision **0.9650289021** and macro recall **0.8835666997**.
- The team's name **does not appear** in the 50-team result spreadsheet shown by the team on 3 October 2026.
- The organizers have **not provided a team-specific rejection explanation or private test labels** in the evidence available for this report. Do **not** claim which particular cause prevented selection.
- The final code-upload receipt confirms upload only; it does not establish technical acceptance, qualification, or approval of the missing candidate file.
- The **Top 50 list**, the historic **public score order**, and the **Top 10 list** are different artifacts and should not be conflated.

Primary references: [competition page](https://unstop.com/hackathons/crp-amazon-ml-challenge-2026-amazon-1743604), [frozen EXP007](../experiments/EXP007.md), [project status](PROJECT_STATUS.md), and [Top 50 research](TOP50_RESEARCH_2026.md).

## 2. The actual EXP007 system

One deduplicated Source-1 reference entity may map to zero, one or many Source-2/Source-3 records. The official metric is macro F0.5 over Source-1 entities with correct singleton handling. The production sequence was:

1. Audit, deterministic entity-level partitions, read-only country indexes.
2. Compact18 candidate generation using lexical/indexed retrieval.
3. 83 ordered tabular pair features: 51 earlier plus 32 additional name/address/transliteration/numeric features.
4. `HistGradientBoostingClassifier` trained using original FIT entities and 50,000 deterministic EXTRA_FIT entities.
5. TUNE-selected thresholds: S2 **0.965**, S3 **0.956**. No deployed guardian, target exclusivity, numeric penalty, score calibration, relative-score model, or cross-source decision model.
6. Frozen, checkpointed, four-shard full TEST inference; merger/audit validated all 1,732,544 S1 rows and that predicted links were drawn from candidates.

The exact source and runbooks remain in the frozen production commit and `experiments/EXP007.md`; retrospective documents do not alter them.

## 3. Where the held-out model lost true links

| Diagnostic, untouched EVALUATION | Value |
|---|---:|
| True positive links | 36,723 |
| False positive links | 880 |
| False negative links | 4,869 |
| False negatives absent from blocker candidates | 1,677 (34.44% of FNs) |
| False negatives retrieved but lost at matcher/decision stage | 3,192 (65.56% of FNs) |
| Singleton accuracy | 0.9059701 |
| Public score minus local EVALUATION score | -0.0220861621 |

**Observed:** Retrieval missed true links, and decision-stage misses were the larger component of remaining false negatives. An excellent downstream classifier cannot recover a missing candidate. A low-threshold classifier cannot necessarily recover decision misses safely, because false positive links and singleton errors are costly under F0.5.

**Not established:** Which countries, scripts, address formats, entity multiplicities, or decoy classes explain the private-test difference. The unlabeled test set includes France; France distribution shift is a hypothesis, **not a proven attribution**. Our TRAIN analysis had country-level differences and cap saturation; these motivated but did not finish EXP008.

**Measurement caution:** The TP/FP/FN figures are pooled link counts used to audit error classes. They cannot be directly substituted into a single micro-F0.5 formula to reproduce the reported *entity-macro* metric.

## 4. Candidate volume, resource engineering and delivery

- Full TEST: **326,418,278** candidate links for 1,732,544 S1 entities (about 188.4 links/S1), versus **5,753,590** predicted matches.
- Complete original ZIP: `BlackList_submission.zip`, **1,971,016,885 bytes**, verified SHA-256 in [submission evidence](SUBMISSION_EVIDENCE_2026.md).
- Unstop portal upload limit reported by the team: **1,024 MB**.
- Submitted fallback: `BlackList_fallback_reviewed.zip`, **41,553,782 bytes**. It contains evaluated predictions, code, frozen model artifacts and documentation but **omits mandatory `output/candidate_pairs.tsv`**. The omission was disclosed.
- Unstop confirmed a **successful Code Submission upload on 2 October 2026, approximately 12:48 IST**. Package compliance/acceptance remains unconfirmed.
- A separate contact-form inquiry requested an authorized way to deliver the complete artifact; no exception approval is established.

**Process failure:** Package-size limits were not continuously measured against the final *auditable* candidate file. A reproduction package that cannot be uploaded as required is a deliverable risk independent of predictive performance.

**Causality limitation:** The published Top 50 results date and upload-deadline sequence alone do not establish whether the omission affected Top 50 selection. Do not write 'the file caused rejection' without an explicit organizer decision.

## 5. What a public finalist did differently (not causal proof)

[Team Androids' publicly indexed solution](https://github.com/PardheevKrishna/amazon-ml-challenge-2026-entity-resolution) discloses an **incoming Source-2/Source-3 → Source-1** cascade with:

- multi-channel inverted-index retrieval (name, address, character and phonetic n-grams);
- a **CatBoost** candidate ranker that keeps up to four ranked Source-1 candidates per incoming record;
- fine-tuned **MiniLM and multilingual-e5-based cross-encoders** on raw names and addresses;
- **per-country LightGBM** stacking and score/margin-based link decisions;
- synthetic hard decoy clusters and separate India/US rescue procedures;
- France-specific text normalization, pseudo-label adaptation and explicit rules;
- approximately **12.293 million** reported final candidate links (about 7.1 per S1).

Their public repository **self-reports** a public leaderboard score of 0.991300; do not use it as a private final score or assert it is independently verified. They disclose public-leaderboard-informed choices and no fully untouched end-to-end evaluation of their final solution. Their per-incoming-record ownership convention is also a modelling assumption to audit against ground truth, not something to import blindly.

The differences suggest useful *future experiments*, not retrospective proof that cross-encoders, CatBoost, a smaller candidate list, or a country stacker *caused* a particular score increase. Other Top 50 methodologies remain **unverified** unless tied to a team with an attributable code/report source; see [Top 50 research](TOP50_RESEARCH_2026.md).

## 6. Our decision-making failure: AI assistance without enough independent challenge

This is a **process diagnosis**, not a claim that any person or assistant alone caused the outcome.

- The team adopted an increasingly elaborate version of one broad architecture while its blocker recall and candidate caps remained material.
- We did not reserve enough early time for genuinely **independent solution families**, e.g. incoming-record competition, cross-encoders, decoy-stress tests and country-specialized routes.
- We designed ablations, but the deployed system remained a single 83-feature HGB with two source thresholds.
- The student/team must understand the problem, metric, constraints and outputs themselves. AI advice should generate falsifiable hypotheses and counterarguments, **not serve as the sole decision authority**.
- The assistant should have insisted on an architecture-diversity review, demanded proof of resource/package compliance, clearly marked architectural assumptions, and surfaced the gap between local EVALUATION and public score without offering causal certainties.

The goal is *stronger human ownership*, not refusing useful AI collaboration.

## 7. Five changes for the next challenge

1. **Read the official statement independently:** each teammate writes task semantics, metric, edge cases, restrictions, unseen domains and exact submission contract before prompting a model for approaches.
2. **Compare at least three candidate solution families early:** cheap lexical baseline; pairwise tabular matcher; ranked-retrieval + semantic reranking (or another genuinely distinct family). Record compute and data requirements; implement only after gates.
3. **Separate retrieval ceilings from decision quality:** blocker recall, matched-all recall, per-country recall, hard-negative errors, singleton performance, calibration and entity-macro F0.5; hold out an evaluation partition untouched by architecture selection.
4. **Audit distribution shifts and assumptions:** unseen countries, scripts, legal forms, address shifts, record ownership, multiple matches, decoy clusters and sampling effects.
5. **Run a delivery rehearsal by the midpoint:** write a realistic full output sample, extrapolate file and ZIP sizes with a safety margin, check validator and packaging, ask organizers about hard limits immediately.

These actions are specified as executable gates in [Future Competition Protocol](FUTURE_COMPETITION_PROTOCOL.md).

## 8. Open questions — do not manufacture answers

- Official private leaderboard result / cutoff and why BlackList was not selected.
- Whether the submitted fallback passed technical review despite the missing mandatory candidate file.
- Whether the separate methodology email required anything beyond the submitted documentation.
- AWS resources and actual final charges (not verified here).
- Whether other shortlisted teams publish attributable implementations after the finale.
- Whether any top-50 selection criteria differed materially from public standings.

The valid outcome today is: **Round 1 closed, evidence preserved, Top 50 absent, causality uncertain, EXP007 frozen.** No model retraining or retroactive result changes are authorized by this document.

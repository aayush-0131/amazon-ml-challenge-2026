# TEAMMATE_HANDOFF — Amazon ML Challenge 2026

> Snapshot written during the competition on **2026-09-27 (IST)**.  
> This file is an orientation document, not a live process monitor. Ask the repo owner for the latest AWS/submission status before taking any operational action.

## 1. What this repository is doing

This repository solves the Amazon ML Challenge 2026 **Business Entity Resolution** task.

For each Source-1 (S1) business entity, the system retrieves candidate entities from Source-2/Source-3 (S2/S3), computes a fixed feature vector for each candidate pair, scores those pairs, and outputs zero, one, or many matched entity IDs.

The current learned pipeline is:

```text
S1 record
  ↓
compact_18 candidate retrieval / blocking
  ↓
pass-eligible candidate pool
  ↓
51 fixed pair features
  ↓
HistGradientBoosting classifier
  ↓
global probability threshold
  ↓
predicted S2/S3 matches
```

The competition metric is **macro F0.5**, so precision matters more than recall.

## 2. Current experiment lineage

### EXP003 — established learned baseline

EXP003 is the current baseline system.

- Original development universe: **20,000 S1**
- FIT: **5,999 S1**
- TUNE: **2,001 S1**
- untouched EVALUATION: **12,000 S1**
- candidate pool: frozen `compact_18`
- features: **51**
- model: HistGradientBoosting
- selected threshold: **0.95**
- EVALUATION macro F0.5: **0.905874**
- EVALUATION macro precision: **0.936684**
- EVALUATION macro recall: **0.848791**

A full TEST inference for EXP003 is currently running as the first submission candidate.

### EXP006 — scaled-training experiment

EXP006 keeps the EXP003 candidate system, 51 features, HGB family, evaluation protocol, and TEST inference implementation fixed. The main question is whether more supervised TRAIN entities improve the matcher.

It adds **50,000 independently sampled TRAIN S1 entities** outside the original 20k development universe.

Important controls:

- EXTRA_FIT excludes all original 20,000 S1 IDs.
- TUNE remains the original 2,001-S1 TUNE partition.
- EVALUATION remains the original untouched 12,000-S1 EVALUATION partition.
- No TEST labels or external data are used.
- Threshold selection happens on TUNE before EVALUATION is read.
- The frozen HGB configuration is unchanged from EXP003.

Verified EXP006 50k result:

- TUNE threshold: **0.97**
- TUNE macro F0.5: **0.916077**
- EVALUATION macro F0.5: **0.913671**
- EVALUATION macro precision: **0.951143**
- EVALUATION macro recall: **0.838503**
- EVALUATION micro F0.5: **0.937682**
- singleton accuracy: **0.874627**
- promotion flag: **true**

Relative to EXP003, EXP006 raises held-out macro F0.5 from **0.905874 → 0.913671**. The gain comes mainly from substantially better precision, with some recall traded away.

A 1,000-S1 TEST deployment smoke passed on all four shards. Full TEST inference for EXP006 is currently running independently as a possible later submission candidate.

## 3. Current code reference

For EXP006, use:

- branch: `exp/exp006-scaled-training`
- correctness-patch commit: `004c3fb168fcea7657040edd8189c7d98af0782c`

The patch fixes two important correctness issues:

1. EXP006 `merge` no longer incorrectly requires/passes `--merged-dir`.
2. shard/merge identity now records the **actual validated sample manifest seed**, rather than a hard-coded seed.

The branch passed **120 tests** after this patch.

Do not assume `main` contains the current competition state.

## 4. Read the repository in this order

Start here:

1. `experiments/EXP006.md`  
   Design, leakage controls, sampling, fixed model protocol, and intended AWS workflow.

2. `scripts/run_exp006.py`  
   CLI orchestration for:
   `preflight → sample → generate → merge → fit → tune → evaluate`.

3. `src/business_entity_resolution/exp006.py`  
   EXP006 implementation.

4. `src/business_entity_resolution/reranker.py`  
   Candidate retrieval, bundle validation, feature scoring, index handling, and model-bundle compatibility.

5. `src/business_entity_resolution/features.py`  
   The fixed feature definition/order.

6. `src/business_entity_resolution/exp003_training.py`  
   EXP003 training/evaluation utilities reused by EXP006.

7. `src/business_entity_resolution/exp003_inference.py`  
   Resumable sharded TEST inference and deterministic validated merge.

8. `scripts/infer_exp003.py`  
   CLI used for both EXP003 and EXP006 full TEST inference.

9. `tests/test_exp006.py`  
   Read this early: it documents many invariants more clearly than prose.

## 5. Key data / artifact concepts

The raw Amazon challenge data are intentionally not committed to Git.

Important local/AWS artifacts include:

```text
results/exp003_20k/
    original FIT/TUNE/EVALUATION caches and learned baseline artifacts

results/artifacts/exp002c_source_index/
    TRAIN-side source indexes

results/artifacts/test_source_index/
    TEST-side source indexes

results/exp006_50k_sample/
    deterministic EXTRA_FIT sample manifest

results/exp006_50k_shards/
    checkpointed 50k TRAIN pair-generation shards

results/exp006_50k_merged/
    merged EXTRA_FIT training pairs/features

results/exp006_50k_fit/
    fitted HGB

results/exp006_50k_tune/
    frozen model bundle + threshold

results/exp006_50k_evaluation/
    untouched held-out evaluation result
```

Most of `results/` is ignored by git and exists only on the machines where the experiments were executed.

Do not assume a fresh clone can reproduce an AWS run without the challenge dataset and prebuilt indexes/artifacts.

## 6. Reproducibility / leakage rules

These rules are non-negotiable:

- Never use TEST labels; there are none available for legitimate training.
- Never tune on the 12k EVALUATION partition.
- Never mix original TUNE/EVALUATION IDs into EXTRA_FIT.
- Do not change blocker/features/model and still call the result the same controlled EXP006 experiment.
- Preserve deterministic seeds and manifest/hash checks.
- Never replace or edit completed experiment artifacts in place.
- New runs should use fresh output directories.
- Candidate predictions must remain a subset of scored candidate pairs.
- Do not add external lookup/web data to the matching system.

## 7. How to work safely

Do **not** push experimental edits directly to `main` or to a branch backing a running AWS job.

Use a new branch or worktree:

```bash
git fetch --all
git switch exp/exp006-scaled-training
git status

git switch -c <your-name>/<experiment-name>
```

Before coding, write down:

1. hypothesis,
2. what changes,
3. what stays frozen,
4. evaluation gate,
5. expected runtime/cost,
6. rollback condition.

Run the full unit suite before handing work back:

```bash
python -m pytest -q
git diff --check
```

Do not delete, rename, clean, or regenerate AWS `results/` directories unless the repo owner explicitly asks you to.

## 8. Good ways to help right now

Useful independent work includes:

- inspect EXP003/EXP006 error slices and propose a clearly isolated hypothesis;
- review feature definitions for bugs/leakage/redundancy;
- review candidate/blocking diagnostics to identify where remaining false negatives originate;
- inspect threshold/rank/margin post-processing ideas without touching TEST labels;
- review the submission validator / packaging path;
- improve documentation and reproducibility;
- add tests around any proposed change;
- profile a proposed method on TRAIN/TUNE/EVALUATION before suggesting a full TEST run.

Avoid launching expensive full TEST runs without first showing a convincing held-out gain.

## 9. Current operational snapshot

At the time this handoff was written:

- **EXP003** full TEST inference is still running on one 4-vCPU EC2 instance and is intended to become Submission #1 once merge + validation pass.
- **EXP006** has already passed TRAIN/TUNE/EVALUATION and a 1k TEST smoke; its full TEST inference is running on a separate 4-vCPU EC2 instance.
- EXP006 held-out macro F0.5 is **0.913671**, versus EXP003 **0.905874**.
- An AWS vCPU quota increase has been appealed, but current experiments do not depend on it.

Do not interrupt either AWS run.

## 10. Fast mental model

If you remember only one thing:

```text
EXP003 = proven baseline

EXP006 = same retrieval/features/HGB protocol
         + 50k extra independent TRAIN S1
         + stricter tuned threshold
         → better untouched held-out macro F0.5
```

The current question is no longer “does EXP006 train?” It does. The remaining question is how its full TEST submission performs relative to EXP003 on the competition leaderboard.

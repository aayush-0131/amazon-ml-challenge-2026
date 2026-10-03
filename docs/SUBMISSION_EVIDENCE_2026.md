# Round-1 submission and reproducibility evidence — BlackList

**Reference date:** 2026-10-03. Facts below are recorded from the team's frozen reports and submission confirmation, not a fresh download or SHA computation. This document does not alter any ZIP.

## Frozen identifiers
- Team: **BlackList**; members: **Aayush Jha and Anish Singh**.
- Model: **EXP007**.
- Production code commit: [`63100d932080e2a783e7dc43eab618abc0d92ad0`](https://github.com/aayush-0131/amazon-ml-challenge-2026/commit/63100d932080e2a783e7dc43eab618abc0d92ad0).
- Frozen branch: `submission/round1-exp007`; archived original base: `archive/pre-round1-main`.
- Frozen local EVALUATION macro F0.5: **0.9384071620652703**; public score **0.916321**.
- Full TEST inference: 1,732,544 S1 entities; 326,418,278 candidate links; 5,753,590 selected links.

## Archive registry

| File | Size | State |
|---|---:|---|
| `BlackList_submission.zip` | 1,971,016,885 bytes | Original complete archive; **not submitted due to portal size limit** |
| `BlackList_fallback_reviewed.zip` | 41,553,782 bytes | Uploaded on Unstop, 2 Oct 2026 ~12:48 IST; candidate TSV omitted |

Original complete archive SHA-256 (team-reported and recorded before submission):
```text
4174b3d7517628b0b0a58357a28139a005eddf0a77f557f8ae0a67e18725ec7f
```

Full TEST outputs SHA-256 (from project audit, **not** claims about fallback ZIP):
```text
matching_results.tsv: 956e80579560c0f8090bec356aae3e849567cf41fe4e0b3079a09ce93a02b20c
candidate_pairs.tsv: b39ac0fc31defc311686fc267dd71631537c6f718844a13543d3c046291e0988
```

The fallback archive SHA-256 has **not yet been independently recorded in this public register**. Maintain original and fallback checksums in the team-controlled private evidence register, and preserve both source archives without modification. Reproducibility or technical acceptance must not be inferred from a checksum alone.

## Submission details, distinguishing knowns from unknowns

**Known from the team's Unstop confirmation email**: the Code Submission upload completed on 2 Oct 2026 at approximately 12:48 IST.

**Known from the submitted fallback description**: it includes evaluated `matching_results.tsv`, complete EXP007 implementation, frozen model artifacts, dependency information and corrected methodology; it does **not** contain the mandatory `output/candidate_pairs.tsv`. This was disclosed explicitly.

**Known from organizer requirements**: the final package should contain *both* TSVs, runnable code and method documentation. See [Unstop opportunity](https://unstop.com/hackathons/crp-amazon-ml-challenge-2026-amazon-1743604) and [problem-statement reproduction in a public project (secondary source)](https://github.com/anmol-228/amazon-ml-challenge-2026/blob/main/docs/OFFICIAL_REQUIREMENTS.md).

**Unknown**: whether Amazon technically accepted the incomplete ZIP, independently reproduced EXP007, or disqualified this package; no reviewer response or private evaluation result is established here. **Do not conflate upload success with package compliance**.

**Evidence to retain privately outside Git**: confirmation email, screenshots of Unstop upload/leaderboard, 1 October support inquiry, archive hash logs, AWS billing export and instance-stop evidence. Do not publish private competition data, credentials, account identifiers or any unrestricted raw TSVs.

## Reproducibility boundary

The source code and experiment commands are in `experiments/EXP007.md`, `experiments/EXP007_TEST_INFERENCE.md`, and the frozen production branch. The GitHub repository **does not contain** raw competition data, SQLite indexes, trained binary artifacts or large output TSVs. Therefore a reader cannot reproduce the numerical results from this repository alone; they need authorized competition datasets, frozen artifacts or independently rebuilt ones, pinned dependencies and adequate compute. The archive remains separate from Git by design.

## Closeout checklist (manual account access required)

- [ ] Record fallback ZIP SHA-256.
- [ ] Preserve screenshots and both originals in at least two safe locations, with private access.
- [ ] Verify AWS EC2 instances in every Region; EBS snapshots/volumes, Elastic/public IP resources, SageMaker and associated storage; review actual Billing/Cost Explorer charges.
- [ ] Preserve any organizer response about the omitted candidate TSV; update status only after reading it.
- [ ] Confirm whether organizer requests or result updates change the team-specific review status.

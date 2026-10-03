# Lessons for future ML projects and competitions

This public document collects high-level lessons from Team BlackList's Amazon ML Challenge 2026 experience. It is **not** a complete competition operations manual.

## Problem understanding
Understanding the task definition, evaluation metric, domain assumptions and deliverable format matters as much as implementing a model. Teams should make important modelling decisions deliberately and be able to explain their rationale.

## Scientific evaluation
Candidate generation and matching accuracy are separate failure points. Report both, evaluate on entity-disjoint held-out data, examine errors across data slices, and distinguish measured improvements from hypotheses. Strong local results do not guarantee equivalent performance on unseen test distributions.

## Engineering and reproducibility
A valid solution must meet the organizer's complete output and packaging contract, not merely obtain a good prediction score. Preserve immutable experiment references, honest documentation and output integrity evidence where publication is permitted.

## Learning from outcomes
A competition result cannot be assigned one cause without reviewer evidence. We publish documented technical outcomes and acknowledged limitations, while retaining detailed internal operational plans in team-controlled private storage.

For the factual BlackList results, see the [retrospective](POSTMORTEM_2026.md) and [project status](PROJECT_STATUS.md).
# Lab 9 — Deep Research Agent

**Skill:** Deep Research · **Worklet mapping:** Multi-hop research with evidence synthesis — a cited draft built only from literature cards, never from the model's own memory.

## What this demonstrates

Search → relevance gate → full-text discussion mining → gap analysis → a cited draft. Every claim traces to a row: paper → point → section → DOI → status, in a machine-checkable evidence table.

## Real code exercised

- [`Course-Project/code/services/research-discovery/`](../../Course-Project/code/services/research-discovery) and [`research-writing/`](../../Course-Project/code/services/research-writing)
- [`Course-Project/code/tests/services/test_discovery_pipeline_integration.py`](../../Course-Project/code/tests/services/test_discovery_pipeline_integration.py) — a real, unmocked, end-to-end integration test against the live local model (deliberately **not** re-run for this lab today — it takes 7–20 minutes per run on this hardware; its real pass is already on record, see below).

## The experiment (already run live, real output kept as evidence)

Four full, fresh, live pipeline runs, each end to end on the real local model:

| Run | Status | Words | QA overall | Verified / invalid / unverifiable |
|---|---|---|---|---|
| Intermittent fasting → cognition | complete | 3,357 | 3.5 | 7 / 0 / 0 |
| Remote work → productivity | complete | 2,709 | 4.25 | 6 / 0 / 2 |
| Urban green space → mental health | partial | 2,238 | 4.25 | 6 / 0 / 0 |
| Related: fasting → memory (library on) | complete | 3,648 | 4.25 | 7 / 0 / 0 |

Real artifacts for each run: [`Course-Project/docs/evidence/live-pass/runs/`](../../Course-Project/docs/evidence/live-pass/runs) — a real `paper.pdf`, `evidence_table.md`, and `decision_brief.md` per run. One DOI from this evidence table was independently spot-checked against the live Crossref API by hand and matched exactly.

## Why this is the real evidence, not a toy

0 invalid references across all 4 runs, and one partial draft was reported as partial, not quietly upgraded to "complete."

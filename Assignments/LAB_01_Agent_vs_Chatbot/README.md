# Lab 1 — Agent vs Chatbot

**Skill:** Agents · **Worklet mapping:** Discovery runs a plan → act → observe → revise loop, not a one-shot prompt/response.

## What this demonstrates

ResearchGenie's Discovery stage does not just answer a research question — it asks one clarifying question when the request is vague, and re-plans with new search queries when too few papers turn out to be on topic. Both triggers are deterministic code, not a model's guess, so the behaviour is explainable and testable.

## Real code exercised

- [`Course-Project/code/services/research-discovery/discovery/pipeline.py`](../../Course-Project/code/services/research-discovery/discovery/pipeline.py) — the clarify step and the re-plan-on-thin-corpus step.
- [`Course-Project/code/tests/orchestrator/test_clarify.py`](../../Course-Project/code/tests/orchestrator/test_clarify.py)
- [`Course-Project/code/tests/services/test_discovery_replan.py`](../../Course-Project/code/tests/services/test_discovery_replan.py)

## The experiment

Run the real test suite for this lab directly against the real pipeline code (no mocked logic — these tests exercise the actual `discovery/pipeline.py` control flow):

```bash
cd Course-Project/code
pip install -e .
pytest -v tests/orchestrator/test_clarify.py tests/services/test_discovery_replan.py
```

**Real captured result** (see [`sample_run.txt`](./sample_run.txt) for the full console output from this exact run):

- Every clarify/re-plan test passes against the live pipeline code.
- Independently, across 4 fresh full live runs (`Course-Project/docs/evidence/live-pass/LIVE_PASS.md`), Discovery's loop ran with no scripted path — it adapted only where the evidence (a vague question, a thin corpus) actually required it.

## Why this is the real evidence, not a toy

This is not a simplified stand-in written for the assignment — it is the exact code path a real ResearchGenie run takes in production, exercised by the same tests that gate every pull request to the project.

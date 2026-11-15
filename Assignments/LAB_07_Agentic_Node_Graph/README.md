# Lab 7 — Agentic Node Graph

**Skill:** Swarms · **Worklet mapping:** Graph orchestration with conditional edges and checkpoints — two cooperating graphs, not a linear script.

## What this demonstrates

Writing is a 9-node LangGraph with three conditional edges and SQLite checkpoints. The Dev Agent is a second, separate graph: plan → write a failing test → patch → test, at most 3 rounds, working in a throwaway git worktree.

## Real code exercised

- [`Course-Project/code/services/research-writing/writing/graph.py`](../../Course-Project/code/services/research-writing/writing/graph.py) — the 9-node writing graph.
- [`Course-Project/code/dev_agent/agent.py`](../../Course-Project/code/dev_agent/agent.py) — the Dev Agent's own graph.
- [`Course-Project/code/tests/dev_agent/test_dev_agent.py`](../../Course-Project/code/tests/dev_agent/test_dev_agent.py)

## The experiment

```bash
cd Course-Project/code
pytest -v tests/dev_agent/test_dev_agent.py
```

**Real captured result** ([`sample_run.txt`](./sample_run.txt)): the Dev Agent graph's plan/test/patch structure, and its refusal rules (never overwrite an existing test, require red before green), pass against the real graph implementation.

**Live evidence:** every one of today's live research runs passed fully through the real 9-node writing graph — not a subset, every run in `Course-Project/docs/evidence/live-pass/`.

## Why this is the real evidence, not a toy

The Dev Agent graph is the same one used for real, in Lab 11 below, on a real product ticket — not a demo graph built only for this assignment.

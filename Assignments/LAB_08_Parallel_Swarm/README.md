# Lab 8 — Parallel Swarm

**Skill:** Swarms · **Worklet mapping:** Parallel workers with a fault-tolerant merge, and a recurrence threshold so one paper's wish list is never mistaken for a real research gap.

## What this demonstrates

One literature-card worker runs per paper, with retry on failure. Source search and full-text fetch run in parallel. The Trend Advisor's gap-mining swarm only keeps a "gap" that several papers raise independently.

## Real code exercised

- [`Course-Project/code/services/trend-advisor/`](../../Course-Project/code/services/trend-advisor) — the gap-mining swarm.
- [`Course-Project/code/tests/services/test_trend_gap_mining.py`](../../Course-Project/code/tests/services/test_trend_gap_mining.py)

## The experiment

```bash
cd Course-Project/code
pytest -v tests/services/test_trend_gap_mining.py
```

**Real captured result** ([`sample_run.txt`](./sample_run.txt)): the recurrence-threshold logic passes — a single paper's future-work line never becomes a reported gap on its own.

**Live evidence:** a real live advisor run produced **34 real gap statements across 3 academic fields** (CS/AI/ML, ECE, Mechanical) — see `Course-Project/docs/evidence/live-pass/`.

## Why this is the real evidence, not a toy

Every gap statement traces back to real OpenAlex/arXiv publication data, not a simulated corpus.

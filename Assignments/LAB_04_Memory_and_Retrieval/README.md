# Lab 4 — Memory & Retrieval

**Skill:** Skills · **Worklet mapping:** Short-term state (LangGraph + SQLite checkpoints) vs. long-term retrieval (a research library + response cache).

## What this demonstrates

ResearchGenie remembers papers and search results across runs — a SQLite full-text research library stores every run and paper, and a response cache keeps searches 7 days and DOIs 30 days. Critically, a seeded past paper still has to pass the exact same relevance gate as a brand-new one: memory can make a run richer, never less checked.

## Real code exercised

- [`Course-Project/code/shared/utilities/library.py`](../../Course-Project/code/shared/utilities/library.py)
- [`Course-Project/code/shared/utilities/response_cache.py`](../../Course-Project/code/shared/utilities/response_cache.py)
- [`Course-Project/code/tests/shared/test_library.py`](../../Course-Project/code/tests/shared/test_library.py)
- [`Course-Project/code/tests/shared/test_response_cache.py`](../../Course-Project/code/tests/shared/test_response_cache.py)

## The experiment

```bash
cd Course-Project/code
pytest -v tests/shared/test_library.py tests/shared/test_response_cache.py
```

**Real captured result** ([`sample_run.txt`](./sample_run.txt)): every library/cache unit test passes, including the explicit test that an off-topic seeded paper is rejected by the relevance gate.

**Live evidence** (not re-run today, already real and captured): on a related live run in `Course-Project/docs/evidence/live-pass/LIVE_PASS.md`, memory surfaced 10 past papers (1 reused excerpt, 9 seeded) — Discovery time was *higher*, not lower, because seeding widens the corpus to screen rather than skipping work.

## Why this is the real evidence, not a toy

The honest, measured finding — memory widens the corpus but didn't speed up that run — is kept in, not edited out.

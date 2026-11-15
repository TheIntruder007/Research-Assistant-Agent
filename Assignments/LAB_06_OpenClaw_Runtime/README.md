# Lab 6 — OpenCLA / OpenClaw Runtime

**Skill:** OpenCLA/OpenClaw · **Worklet mapping:** An agent runtime with real logging and background execution, reachable from chat.

## What this demonstrates

Every pipeline stage emits one standard event to a durable per-run ledger (`events.jsonl`), so a run's history survives a restart. A Telegram bot was wired end-to-end through OpenClaw: created, paired, authenticated, and the skill confirmed visible to the model.

## Real code exercised

- [`Course-Project/code/orchestrator/pipeline.py`](../../Course-Project/code/orchestrator/pipeline.py) — the event ledger.
- [`Course-Project/code/orchestrator/api.py`](../../Course-Project/code/orchestrator/api.py) — FastAPI + SSE web surface reading the same ledger.
- [`Course-Project/code/tests/orchestrator/test_pipeline.py`](../../Course-Project/code/tests/orchestrator/test_pipeline.py), [`test_api.py`](../../Course-Project/code/tests/orchestrator/test_api.py)
- [`Course-Project/code/dashboards/logs_dashboard.py`](../../Course-Project/code/dashboards/logs_dashboard.py) and [`telegram_sync_dashboard.py`](../../Course-Project/code/dashboards/telegram_sync_dashboard.py) — two live dashboards built specifically to show this runtime working, reading the real event ledger and the real OpenClaw gateway log, live.

## The experiment

```bash
cd Course-Project/code
pytest -v tests/orchestrator/test_pipeline.py tests/orchestrator/test_api.py
```

**Real captured result** ([`sample_run.txt`](./sample_run.txt)): the event ledger and API tests pass against the real orchestrator.

## An honest, not-yet-solved gap (reported, not hidden)

The Telegram **connection** is proven solid end to end. But when asked a research question, the chat agent currently answers from its own knowledge instead of invoking the real pipeline — tried and ruled out across 5 independent fixes (PATH resolution, exec permission modes, a completely fresh session, a full permission bypass, and stronger skill wording). This is a real integration boundary between OpenClaw's `claude-cli` model backend and skill-triggered tool use, documented precisely so the next attempt doesn't repeat the same five dead ends.

## Why this is the real evidence, not a toy

We built a live dashboard specifically so this exact gap can be watched happening in real time, rather than just described in a report.

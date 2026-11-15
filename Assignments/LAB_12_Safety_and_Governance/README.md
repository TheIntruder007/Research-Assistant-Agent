# Lab 12 — Safety & Governance

**Skill:** Cloud SDLC · **Worklet mapping:** Approval gates, audit logs, independent evaluation, and rollback — not an afterthought bolted onto the pipeline.

## What this demonstrates

A human approves the screened corpus before Writing starts. A human confirms every Doctor fix before it applies. Finished runs are versioned (`final_v1/`, `final_v2/` — never overwritten). A deterministic QA checker, independent of the model that wrote the draft, grades the result.

## Real code exercised

- [`Course-Project/code/orchestrator/decision_brief.py`](../../Course-Project/code/orchestrator/decision_brief.py) — rule-built, no LLM call.
- [`Course-Project/code/tests/orchestrator/test_resume.py`](../../Course-Project/code/tests/orchestrator/test_resume.py)
- [`Course-Project/code/tests/services/test_discovery_approval.py`](../../Course-Project/code/tests/services/test_discovery_approval.py)
- [`Course-Project/code/tests/orchestrator/test_decision_brief.py`](../../Course-Project/code/tests/orchestrator/test_decision_brief.py)
- [`Course-Project/docs/GOVERNANCE.md`](../../Course-Project/docs/GOVERNANCE.md) — 6 named risks, each mapped to a control, a test, and a rollback.

## The experiment

```bash
cd Course-Project/code
pytest -v tests/orchestrator/test_resume.py tests/services/test_discovery_approval.py tests/orchestrator/test_decision_brief.py
```

**Real captured result** ([`sample_run.txt`](./sample_run.txt)): the approval gate, the versioned-resume logic, and the rule-built decision brief all pass against the real implementation.

## Why this is the real evidence, not a toy

The decision brief is deterministic by design — no LLM in that loop at all — specifically so grading it can never drift with a model update.

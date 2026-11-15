# Lab 10 — Cloud Operations Agent

**Skill:** Cloud SDLC · **Worklet mapping:** A 12-rule Doctor that diagnoses a failure, quotes its evidence, proposes one fix, waits for human confirmation, then resumes — never restarts.

## What this demonstrates

CI runs on every push. A container packages both a local-Ollama and a cloud-Gemini profile. The Doctor reads the real event ledger and `/system` state, matches against a 12-rule table, and only ever proposes a fix after naming its cause from real evidence.

## Real code exercised

- [`Course-Project/code/orchestrator/doctor.py`](../../Course-Project/code/orchestrator/doctor.py)
- [`Course-Project/code/tests/orchestrator/test_doctor.py`](../../Course-Project/code/tests/orchestrator/test_doctor.py) — one real test per rule, 12 rules.

## The experiment

```bash
cd Course-Project/code
pytest -v tests/orchestrator/test_doctor.py
```

**Real captured result** ([`sample_run.txt`](./sample_run.txt)): all 12 rules pass against the real Doctor implementation.

**Live evidence — proven end to end, not just in unit tests:** the first live attempt surfaced a genuine gap (a broken draft wasn't being diagnosed because Ollama was down but the run had still "completed"). We fixed it live (the `ollama_stopped_mid_writing` rule), then proved full recovery for real: killed Ollama deliberately mid-run, the Doctor caught it, fixed it, and the run resumed to a completed draft. Full record in `Course-Project/docs/evidence/live-pass/LIVE_PASS.md` and GitHub issue #13 (closed).

## Why this is the real evidence, not a toy

The bug the Doctor fixed live was a real bug we found by testing, not a scripted failure written in advance for the assignment.

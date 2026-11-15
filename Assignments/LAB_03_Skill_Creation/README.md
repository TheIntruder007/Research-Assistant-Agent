# Lab 3 — Skill Creation

**Skill:** Skills · **Worklet mapping:** Reusable skills — instructions + tools + a typed contract, not a one-off script.

## What this demonstrates

Every ResearchGenie capability sits behind a typed Pydantic contract, and one Agent Skill folder (`skills/researchgenie/`) lets any agent runtime — Claude Code or OpenClaw — start a run from chat using only the real headless CLI. A dedicated test fails the build the moment the skill's documented commands drift from the real CLI's own parser.

## Real code exercised

- [`Course-Project/code/shared/contracts/`](../../Course-Project/code/shared/contracts) — the typed request/result contracts per service.
- [`Course-Project/code/researchgenie/headless.py`](../../Course-Project/code/researchgenie/headless.py) — the headless JSON CLI (`run` / `status` / `result` / `resume` / `doctor`), fixed exit codes.
- [`Course-Project/code/skills/researchgenie/SKILL.md`](../../Course-Project/code/skills/researchgenie/SKILL.md) — the Agent Skill itself.
- [`Course-Project/code/tests/researchgenie/test_headless.py`](../../Course-Project/code/tests/researchgenie/test_headless.py)
- [`Course-Project/code/tests/test_agent_skill.py`](../../Course-Project/code/tests/test_agent_skill.py)

## The experiment

```bash
cd Course-Project/code
pytest -v tests/researchgenie/test_headless.py tests/test_agent_skill.py
```

**Real captured result** ([`sample_run.txt`](./sample_run.txt)): every command line documented in `SKILL.md` is parsed by the real headless CLI's own argument parser in the test — if a future change to the CLI breaks a documented command, this test fails immediately, not at a demo.

## Why this is the real evidence, not a toy

The skill is also installed and checked live in OpenClaw (`openclaw skills check`) — see Lab 6 — using this exact file, not a simplified copy.

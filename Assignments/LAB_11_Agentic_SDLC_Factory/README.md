# Lab 11 — Agentic SDLC Factory

**Skill:** Cloud SDLC · **Worklet mapping:** Requirements → tested code, written by an agent: plan → failing test → patch → green, with a human applying the final diff.

## What this demonstrates

The Dev Agent plans a fix from a ticket, writes a failing test (red is required first), then patches with search/replace edits until the test passes — in a throwaway git worktree, never committing on its own.

## Real code exercised

- [`Course-Project/code/dev_agent/agent.py`](../../Course-Project/code/dev_agent/agent.py)
- [`Course-Project/code/tests/dev_agent/test_dev_agent.py`](../../Course-Project/code/tests/dev_agent/test_dev_agent.py)
- [`Course-Project/docs/tickets/26-apa-7.md`](../../Course-Project/docs/tickets/26-apa-7.md) — the real ticket and the full attempt history.

## The experiment (already run live, real output kept as evidence — 8 real attempts)

This is the one lab we report as an honest negative result, by design:

- **5 attempts on the local model (Ollama):** every single time, it correctly refused to overwrite the real test file rather than fake a pass. Never reached a patch round.
- **3 attempts on Gemini:** reached 3 patch rounds each, close but never fully green — one missed a shared helper, one had a whitespace mismatch in its edit, one regressed into a double-suffix bug.
- **The fix was then written by hand** and merged through a normal pull request (commit `90e44ec`, PR #37) — recorded plainly as hand-written, not claimed as the agent's own work.

## Why this is the real evidence, not a toy

We report the negative result as it is. The Dev Agent's safety rules — refuse to overwrite real tests, require red before green, never commit — are the actual achievement here, proven under genuine repeated failure, not a lucky one-shot demo.

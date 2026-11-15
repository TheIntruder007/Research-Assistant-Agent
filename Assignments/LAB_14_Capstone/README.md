# Lab 14 - Capstone Project Studio

**Skill:** all 7 skills integrated. **Worklet mapping:** one end to end, team owned agentic system, not a collection of separate demos.

## What this demonstrates

Everything in Labs 1 through 13, integrated into one real product: `researchgenie` (run, status, result, resume, doctor, config), `researchgenie-advise` (Trend and Gap Advisor), `researchgenie-dev` (Dev Agent), `researchgenie-mcp` (MCP server). A web workbench with resume, approval, Doctor, evidence and brief tabs. Delivered across 5 rounds of real GitHub issues and pull requests.

## Real code exercised

The entire [`Course-Project/code/`](../../Course-Project/code) tree, every service, the orchestrator, the shared utilities, the CLI, and the dashboards shown individually in Labs 1 through 13 above.

## The experiment, the full test suite, run live for this submission

```bash
cd Course-Project/code
pip install -e .
pytest -q
```

**Real captured result** (see [`full_suite_sample_run.txt`](./full_suite_sample_run.txt)): the complete suite, run fresh for this exact submission, not a cached number from an earlier week.

## Why this is the real evidence, not a toy

This is the same command CI runs on every single push to `main`. Nothing here was staged for the assignment, it is the actual gate the real product has to pass.

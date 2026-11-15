# Lab 2 — Tool-Using Agent

**Skill:** Agents · **Worklet mapping:** Native tool calling — the model picks which scholarly databases to search and what query to send each; code double-checks the choice against the field.

## What this demonstrates

Discovery doesn't hard-code which database to search. The local 9B model returns a tool plan (database + query per source) via native function calling; code validates each choice against the research field (e.g. refuses PubMed for a mechanical-engineering question), with a rule-based fallback if the plan is invalid.

## Real code exercised

- [`Course-Project/code/services/research-discovery/discovery/source_plan.py`](../../Course-Project/code/services/research-discovery/discovery/source_plan.py) — the tool definitions and field-validation logic.
- [`Course-Project/code/shared/utilities/scholarly_gateway.py`](../../Course-Project/code/shared/utilities/scholarly_gateway.py) — pacing, retry, circuit breaker and credit budget shared by every source.
- [`Course-Project/code/scripts/spike_tool_calling.py`](../../Course-Project/code/scripts/spike_tool_calling.py) — the live spike script itself.
- [`Course-Project/code/tests/shared/test_scholarly_gateway.py`](../../Course-Project/code/tests/shared/test_scholarly_gateway.py)

## The experiment (already run live, real output captured)

The spike script asks the real local model to plan sources for 12 research questions spanning disciplines (mechanical, electrical, computer science, biomedical, public health, physics…) and checks whether each plan is valid and sensible. This is the actual script, run against the actual local model — not a notebook stand-in:

```bash
cd Course-Project/code
python scripts/spike_tool_calling.py --model qwen3.5:9b --json out.json
```

**Real captured result** (full raw output in [`spike_output.json`](./spike_output.json), copied from the project's live run):

> `qwen3.5:9b: 12/12 valid tool plans, 11/12 sensible (PubMed only where it belongs).`

The one "not sensible" case was a borderline field, not a wrong database choice. The gateway unit tests additionally show 250 DOI lookups batch into 3 Crossref requests + 1 OpenAlex request — the model never has to think about request batching, the gateway owns it.

## Why this is the real evidence, not a toy

No part of this is a rule-based stand-in for the model — the actual `qwen3.5:9b` model, running locally, produced every one of the 12 tool plans graded above.

# Lab 5 — MCP-Style Connector

**Skill:** MCP · **Worklet mapping:** One Model Context Protocol server, usable by any MCP client.

## What this demonstrates

`researchgenie-mcp` exposes six read-only tools — list runs, read an artifact, search the library, summarize a run, search the literature, verify a DOI — over the official MCP Python SDK. It is read-only by design: no tool writes a run folder or starts a run, and it refuses path traversal.

## Real code exercised

- [`Course-Project/code/researchgenie/mcp_server.py`](../../Course-Project/code/researchgenie/mcp_server.py)
- [`Course-Project/code/tests/researchgenie/test_mcp_server.py`](../../Course-Project/code/tests/researchgenie/test_mcp_server.py) — a real MCP client session, not a mocked transport.

## The experiment

```bash
cd Course-Project/code
pytest -v tests/researchgenie/test_mcp_server.py
```

**Real captured result** ([`sample_run.txt`](./sample_run.txt)): a genuine MCP client connects over stdio and calls every tool, including a path-traversal attempt on `read_run_artifact`, which is refused.

**Separately, live-verified in Claude Desktop** during this review cycle: asking "list my ResearchGenie runs" returned all of the real runs on disk, correctly — not a scripted answer, a real MCP tool call from a real MCP client.

## Why this is the real evidence, not a toy

The same server is wired into Claude Desktop, Claude Code, and OpenClaw's `mcp.servers` config — one real implementation, three real clients.

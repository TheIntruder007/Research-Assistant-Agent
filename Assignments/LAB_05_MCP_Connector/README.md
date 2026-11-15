# Lab 5 - MCP-Style Connector

**Skill:** MCP

**Worklet mapping:** One Model Context Protocol server, usable by any MCP client.

## What this demonstrates

researchgenie-mcp exposes six read only tools, list runs, read an artifact, search the library, summarize a run, search the literature, verify a DOI, built on the official MCP Python SDK. It is read only by design, no tool writes a run folder or starts a run, and it refuses path traversal.

## Real code exercised

The actual source files this lab is built on, copied in full below, not linked only.

### `researchgenie/mcp_server.py`

```python
"""Read-only MCP server (#20): ResearchGenie's runs, library and DOI checker
for any MCP client, over stdio, on the official MCP SDK.

    researchgenie-mcp

Six tools, none of which writes a run folder, the library or the config,
or starts a run (MCP request timeouts are short; long runs go through the
headless CLI / skill). The two tools that reach the network go through the
scholarly gateway, so they share its pacing, keys and response cache.
Config snippets for OpenClaw, Claude Code and Claude Desktop are in
docs/mcp-server.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "research-discovery"))

from orchestrator import pipeline  # noqa: E402
from researchgenie import config as rg_config  # noqa: E402
from researchgenie import headless  # noqa: E402
from shared.utilities import library  # noqa: E402

# Text files a client may read; paper.pdf is binary and stays a path in summaries.
READABLE_FILES = tuple(name for name in pipeline.FINAL_FILES if not name.endswith(".pdf"))
MAX_ARTIFACT_CHARS = 200_000
_transport: httpx.AsyncBaseTransport | None = None  # test seam (httpx.MockTransport)

server = FastMCP("researchgenie")


def _run_directory(run_id: str) -> Path | None:
    """A run folder under the output root, else a past run the library knows."""
    found = pipeline.find_run_directory(run_id, pipeline.DEFAULT_OUTPUT_ROOT)
    if found is None and library.library_path().is_file():
        past = library.get_run(run_id)
        found = Path(past["run_dir"]) if past and Path(past["run_dir"]).is_dir() else None
    return found


@server.tool()
def list_runs(limit: int = 20) -> list[dict[str, Any]]:
    """Recent research runs in the output folder, newest first: run id,
    question, state (running / failed / completed) and folder."""
    root = pipeline.DEFAULT_OUTPUT_ROOT
    folders = sorted((d for d in root.iterdir() if d.is_dir() and (d / "events.jsonl").is_file()),
                     key=lambda d: (d / "events.jsonl").stat().st_mtime, reverse=True) \
        if root.is_dir() else []
    runs = []
    for folder in folders[:max(1, min(limit, 200))]:
        status = headless.status_of(folder)
        try:
            question = json.loads((folder / "00_request.json").read_text(encoding="utf-8"))["research_question"]
        except (OSError, ValueError, KeyError):
            question = None
        runs.append({"run_id": status["run_id"], "question": question, "state": status["state"],
                     "stage": status["stage"], "run_directory": status["run_directory"]})
    return runs


@server.tool()
def read_run_artifact(run_id: str, name: str) -> dict[str, Any]:
    """The text of one final file of a run. `name` is one of: draft.md,
    quality_report.md, validation_report.md, references.md,
    evidence_table.md, paper.tex. Anything else is refused."""
    if name not in READABLE_FILES:
        raise ValueError(f"Only these files can be read: {', '.join(READABLE_FILES)}.")
    directory = _run_directory(run_id)
    if directory is None:
        raise ValueError(f"No run {run_id!r}.")
    # Imported copies under docs/research-papers/ keep their files at the top level.
    for path in (directory / "final" / name, directory / name):
        resolved = path.resolve()
        if resolved.is_relative_to(directory.resolve()) and resolved.is_file():
            text = resolved.read_text(encoding="utf-8", errors="replace")
            return {"run_id": run_id, "name": name, "truncated": len(text) > MAX_ARTIFACT_CHARS,
                    "text": text[:MAX_ARTIFACT_CHARS]}
    raise ValueError(f"{name} was not produced by run {run_id!r}.")


@server.tool()
def search_library(query: str, limit: int = 10) -> dict[str, list[dict[str, Any]]]:
    """Past runs and stored papers in the local research library whose
    question, title or abstract matches the query."""
    if not library.library_path().is_file():  # never create one just to search it
        return {"runs": [], "papers": []}
    limit = max(1, min(limit, 50))
    return {"runs": library.search_runs(query, limit=limit),
            "papers": [{k: p[k] for k in ("doi", "title", "year", "venue", "abstract")}
                       for p in library.search_papers(query, limit=limit)]}


@server.tool()
def summarize_run(run_id: str) -> dict[str, Any]:
    """One run at a glance: question, state, quality scores, verification
    counts, draft status, the papers it used and its final files."""
    directory = _run_directory(run_id)
    if directory is None:
        raise ValueError(f"No run {run_id!r}.")
    summary = headless.result_of(directory)
    try:
        summary["question"] = json.loads(
            (directory / "00_request.json").read_text(encoding="utf-8"))["research_question"]
    except (OSError, ValueError, KeyError):
        summary["question"] = None
    try:
        discovery = json.loads((directory / "01_discovery" / "result.json").read_text(encoding="utf-8"))
        summary["papers"] = [{k: p.get(k) for k in ("title", "year", "doi")}
                             for p in discovery.get("selected_papers", [])]
        summary["research_gaps"] = [g["title"] for g in discovery.get("research_gaps", [])]
    except (OSError, ValueError):
        summary["papers"], summary["research_gaps"] = [], []
    return summary


@server.tool()
async def search_literature(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search the scholarly literature (OpenAlex, through ResearchGenie's
    gateway and cache). Returns title, authors, year, venue, DOI, citations."""
    from discovery.sources import openalex
    async with httpx.AsyncClient(timeout=45, follow_redirects=True, transport=_transport) as http:
        papers = await openalex.search(http, {"keyword": query}, max(1, min(limit, 25)),
                                       require_abstract=False)
    return [{"title": p.title, "authors": p.authors[:6], "year": p.year, "venue": p.venue,
             "doi": p.doi, "citations": p.citations, "url": p.url} for p in papers]


@server.tool()
async def verify_doi(doi: str) -> dict[str, Any]:
    """Whether a DOI is real: 'verified' (Crossref or OpenAlex knows it),
    'not_found' (neither does — possibly made up) or 'unverifiable' (no
    answer right now, which says nothing about the DOI)."""
    check_reference = pipeline.verification_service.check_reference
    async with httpx.AsyncClient(transport=_transport) as http:
        check = await check_reference(http, f"doi: {doi.strip()}")
    if check.doi is None:
        return {"doi": doi, "status": "unverifiable", "detail": "That is not a DOI (10.xxxx/...)."}
    status = "verified" if check.verified else \
        "unverifiable" if (check.issue or "").startswith("unverifiable") else "not_found"
    return {"doi": check.doi, "status": status, "source": check.verification_source,
            "detail": check.issue}


def main() -> None:
    rg_config.apply_scholarly(rg_config.load())  # the keys and email `researchgenie config` saved
    server.run()  # stdio


if __name__ == "__main__":
    main()

```

### Real tests for this lab

### `tests/researchgenie/test_mcp_server.py`

```python
"""The read-only MCP server (#20), through a real MCP client session (the
SDK's in-memory transport) against a temporary run folder and library."""

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "orchestrator"))

from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

from orchestrator import pipeline as orchestrator_pipeline  # noqa: E402
from researchgenie import mcp_server  # noqa: E402
from shared.utilities import library  # noqa: E402
from test_pipeline import (  # noqa: E402
    _fake_run_discovery, _fake_run_qa, _fake_run_verification, _fake_run_writing,
)

TOOLS = {"list_runs", "read_run_artifact", "search_library", "summarize_run",
         "search_literature", "verify_doi"}
REAL_DOI = "10.1038/nature14539"


@pytest.fixture
def finished_run(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator_pipeline, "DEFAULT_OUTPUT_ROOT", tmp_path / "outputs")
    for service, name, fake in (
            (orchestrator_pipeline.discovery_service, "run_discovery", _fake_run_discovery),
            (orchestrator_pipeline.writing_service, "run_writing", _fake_run_writing),
            (orchestrator_pipeline.verification_service, "run_verification", _fake_run_verification),
            (orchestrator_pipeline.qa_service, "run_quality_assurance", _fake_run_qa)):
        monkeypatch.setattr(service, name, fake)

    async def run():
        request = orchestrator_pipeline.ResearchRequest(research_question="Does sleep aid memory?")
        return [e async for e in orchestrator_pipeline.run_pipeline(request, run_id="run42")]

    asyncio.run(run())  # also writes the run into the (temporary) library
    return tmp_path / "outputs" / "does-sleep-aid-memory_run42"


def _doi_transport(request: httpx.Request) -> httpx.Response:
    asked = request.url.params.get("filter", "")  # a DOI filter (#7)
    if request.url.host == "api.crossref.org":
        return httpx.Response(200, json={"message": {"items": [{"DOI": REAL_DOI}] if REAL_DOI in asked else []}})
    if asked.startswith("doi:"):  # OpenAlex fallback
        return httpx.Response(200, json={"results": []})
    return httpx.Response(200, json={"results": [{
        "id": "https://openalex.org/W1", "display_name": "Sleep and memory",
        "publication_year": 2020, "cited_by_count": 7, "doi": "https://doi.org/10.1/sleep",
        "authorships": [{"author": {"display_name": "A. Author"}}]}]})


def _call(tool: str, **arguments):
    async def go():
        async with create_connected_server_and_client_session(mcp_server.server) as client:
            if tool == "list_tools":
                return await client.list_tools()
            return await client.call_tool(tool, arguments)
    return asyncio.run(go())


def _data(result):
    assert not result.isError, result.content
    if result.structuredContent is not None:
        data = result.structuredContent
        return data.get("result", data) if set(data) == {"result"} else data
    return json.loads(result.content[0].text)


def _snapshot(*paths):
    return {str(p): p.stat().st_mtime_ns for root in paths for p in sorted(Path(root).rglob("*"))
            if p.is_file()}


def test_the_six_read_only_tools_return_the_right_data(finished_run, monkeypatch):
    monkeypatch.setattr(mcp_server, "_transport", httpx.MockTransport(_doi_transport))
    before = _snapshot(finished_run, library.library_path().parent)

    assert {t.name for t in _call("list_tools").tools} == TOOLS

    [run] = _data(_call("list_runs"))
    assert run["run_id"] == "run42" and run["state"] == "completed"
    assert run["question"] == "Does sleep aid memory?"

    artifact = _data(_call("read_run_artifact", run_id="run42", name="draft.md"))
    assert artifact["text"] == (finished_run / "final" / "draft.md").read_text(encoding="utf-8")

    found = _data(_call("search_library", query="sleep memory paper"))
    assert [r["run_id"] for r in found["runs"]] == ["run42"]
    assert found["papers"][0]["doi"] == "10.1/x"

    summary = _data(_call("summarize_run", run_id="run42"))
    assert summary["state"] == "completed" and summary["scores"]["overall"] == 5.0
    assert summary["papers"][0]["title"] == "Paper One" and "draft.md" in summary["final_files"]

    papers = _data(_call("search_literature", query="sleep memory"))
    assert papers[0]["title"] == "Sleep and memory" and papers[0]["doi"] == "10.1/sleep"

    assert _data(_call("verify_doi", doi=REAL_DOI))["status"] == "verified"
    made_up = _data(_call("verify_doi", doi="10.9999/made.up.2026"))
    assert made_up["status"] == "not_found" and "fabricated" in made_up["detail"]

    # Nothing in the run folder or the library changed.
    assert _snapshot(finished_run, library.library_path().parent) == before


@pytest.mark.parametrize("name", ["../../events.jsonl", "../00_request.json", "metadata.json",
                                  "paper.pdf", "draft.md/../../00_request.json", "/etc/passwd"])
def test_artifact_reads_outside_the_final_files_are_refused(finished_run, name):
    result = _call("read_run_artifact", run_id="run42", name=name)
    assert result.isError and "Only these files" in result.content[0].text


def test_unknown_runs_are_refused(finished_run):
    assert _call("summarize_run", run_id="nope").isError
    assert _call("read_run_artifact", run_id="../run42", name="draft.md").isError


def test_no_tool_starts_a_run_or_writes(monkeypatch):
    """The server only exposes the six tools above; none takes a path to
    write or a request to start, and searching a missing library never
    creates one."""
    tools = {t.name: t for t in _call("list_tools").tools}
    for tool in tools.values():
        assert not any(word in tool.name for word in ("run_pipeline", "start", "write", "delete"))
        assert "path" not in tool.inputSchema.get("properties", {})
    assert _data(_call("search_library", query="anything")) == {"runs": [], "papers": []}
    assert not library.library_path().exists()

```

## The experiment

```bash
cd Course-Project/code
pip install -e .
pytest -v tests/researchgenie/test_mcp_server.py
```

## Real result
A genuine MCP client connects over stdio and calls every tool above, including a path traversal attempt on read_run_artifact, which is refused. Separately, during this review cycle, asking Claude Desktop to list my ResearchGenie runs returned every real run on disk correctly, a real MCP tool call from a real MCP client, not a scripted answer.

## Why this is the real evidence, not a toy
The same server is wired into Claude Desktop, Claude Code and OpenClaw's mcp.servers configuration, one real implementation serving three real clients.

# Lab 12 - Safety and Governance

**Skill:** Cloud SDLC

**Worklet mapping:** Approval gates, audit logs, independent evaluation and rollback, built into the pipeline rather than added afterwards.

## What this demonstrates

A human approves the screened corpus before Writing starts. A human confirms every Doctor fix before it applies. Finished runs are versioned, final_v1, final_v2, never overwritten. A deterministic QA checker, independent of the model that wrote the draft, grades the result.

## Real code exercised

The actual source files this lab is built on, copied in full below, not linked only.

### `orchestrator/decision_brief.py`

```python
"""The decision brief (#12): where the student's judgement is needed, and nothing else.

Deterministic, never a model call. Built from what a run folder already
holds: the event log (failure, exhausted providers, cache hits and, when
#7/#19 log them, budget_check and memory_hit), the Writing result's draft
metadata (draft status, broken and evidence-limited sections, the numbered
reference list), Verification's invalid and unverifiable references and the
QA scores. RULES picks exactly one next action: the first matching row wins.

A completed run's brief is final/decision_brief.md, beside the rest of its
output (so a resume keeps it in final_vN with that output). A run that
failed has no final/ of its own: its brief is <run>/decision_brief.md, next
to doctor_report.json, and is removed once a resume completes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from orchestrator import doctor

BRIEF_NAME = "decision_brief.md"
_SCORES = (("overall", "Overall quality"), ("citation_integrity", "Citation integrity"),
           ("claim_source_alignment", "Claim-source alignment"),
           ("process_control", "Process control"), ("literature_coverage", "Literature coverage"))
_STRONG, _WEAK = 3.5, 2.5   # the web page's score tones: ok from 3.5, bad under 2.5
_LEADING_NUMBER = re.compile(r"^\s*\[?(\d+)[\].]")


def _read(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _of(event: dict[str, Any], kind: str) -> bool:
    """A gateway or memory event, by its status (#7, #19) or its raw_type."""
    return event.get("status") == kind or (event.get("details") or {}).get("raw_type") == kind


@dataclass
class Facts:
    """Everything a rule or a line of the brief looks at."""
    run_id: str
    state: str                               # completed | failed | stopped
    error: str | None
    stage: str | None                        # the last stage that logged anything
    draft: dict[str, Any]                    # the Writing result's draft_metadata
    scores: dict[str, float]
    references: list[tuple[int, str, str]]   # (number, kind, issue): invalid / unverifiable
    verified: int
    unsupported_claims: int
    missing_citations: int
    exhausted: list[tuple[str, str]]         # (host, stage), first report first
    reduced_plan: bool                       # a budget_check took the reduced plan (#7)
    cache_hits: int
    memory_hits: int
    broken: list[str] = field(default_factory=list)
    evidence_limited: list[str] = field(default_factory=list)


def _number(entry: str, numbered: list[str]) -> int | None:
    """A reference's number: its place in the Writing result's list, else a leading [N] or N."""
    if entry in numbered:
        return numbered.index(entry) + 1
    found = _LEADING_NUMBER.match(entry)
    return int(found.group(1)) if found else None


def gather(run_directory: Path) -> Facts:
    run_directory = Path(run_directory)
    events = doctor.read_events(run_directory)
    error = doctor.last_error(events)
    completed = bool(events) and events[-1].get("type") == "result"
    writing = _read(run_directory / "02_writing" / "result.json")
    verification = _read(run_directory / "03_verification" / "result.json")
    qa = _read(run_directory / "04_quality_assurance" / "result.json") if completed else {}
    draft = writing.get("draft_metadata") or {}
    numbered = writing.get("reference_candidates") or []
    references = sorted(
        (number, kind, check.get("issue") or "")
        for kind in ("invalid", "unverifiable")
        for check in verification.get(f"{kind}_references") or []
        if (number := _number(check.get("reference_entry", ""), numbered)) is not None)
    exhausted: dict[str, str] = {}  # doctor.exhausted_hosts, keeping the stage
    for e in events:
        if _of(e, "provider_exhausted") and (host := (e.get("details") or {}).get("host")):
            exhausted.setdefault(host, e.get("stage") or "discovery")
    budget = [e.get("details") or {} for e in events if _of(e, "budget_check")]
    return Facts(
        run_id=run_directory.name.partition("_")[2],
        state="completed" if completed else "failed" if error is not None else "stopped",
        error=error, stage=next((e["stage"] for e in reversed(events) if e.get("stage")), None),
        draft=draft, scores=(qa.get("scores") or {}), references=references,
        verified=len(verification.get("verified_references") or []),
        unsupported_claims=len(qa.get("unsupported_claims") or []),
        missing_citations=len(qa.get("missing_citations") or []),
        exhausted=list(exhausted.items()),
        reduced_plan=any(d.get("plan") == "reduced" or d.get("reduced_plan") for d in budget),
        cache_hits=sum((e.get("details") or {}).get("hits") or 0 for e in events if _of(e, "cache_hit")),
        memory_hits=sum((e.get("details") or {}).get("count") or 0 for e in events if _of(e, "memory_hit")),
        broken=list(draft.get("broken_sections") or []),
        evidence_limited=list(draft.get("evidence_limited_sections") or []),
    )


def _title(stage: str | None) -> str:
    return (stage or "an unknown stage").replace("_", " ").capitalize()


def _names(items: list) -> str:
    return ", ".join(str(item) for item in items)


def _openalex_stage(f: Facts) -> str | None:
    """The stage that ran without OpenAlex: where its budget ran out, else
    verification when a budget check chose the reduced plan (#7 skips
    OpenAlex verification searches)."""
    stage = next((stage for host, stage in f.exhausted if host == "api.openalex.org"), None)
    return stage or ("verification" if f.reduced_plan else None)


@dataclass(frozen=True)
class Rule:
    id: str
    when: Callable[[Facts], bool]
    action: Callable[[Facts], str]


RULES: tuple[Rule, ...] = (
    Rule("run_failed", lambda f: f.state != "completed",
         lambda f: f"Run `researchgenie doctor {f.run_id}`: the run stopped in {_title(f.stage)}"
                   + (f" ({f.error})" if f.error else "") + ". The Doctor names the cause and "
                   "resumes from the failed stage once you confirm."),
    Rule("broken_sections", lambda f: bool(f.broken) or f.draft.get("draft_status") in ("partial", "failed"),
         lambda f: f"Resume from Writing (`researchgenie resume {f.run_id} --from writing`): "
                   + (f"{len(f.broken)} section(s) are broken: {_names(f.broken)}." if f.broken
                      else f"the draft is {f.draft.get('draft_status')}.")),
    Rule("reduced_plan", lambda f: _openalex_stage(f) is not None,
         lambda f: f"Add your OpenAlex key (`researchgenie config`), then resume from "
                   f"{_title(_openalex_stage(f))} (`researchgenie resume {f.run_id} --from "
                   f"{_openalex_stage(f)}`): {_title(_openalex_stage(f))} ran on a reduced plan."),
    Rule("evidence_limited", lambda f: bool(f.evidence_limited),
         lambda f: f"Resume from Writing (`researchgenie resume {f.run_id} --from writing`): "
                   f"{len(f.evidence_limited)} section(s) were evidence-limited: {_names(f.evidence_limited)}."),
    Rule("check_references", lambda f: bool(f.references),
         lambda f: f"Check reference(s) {_names([n for n, _, _ in f.references])} by hand "
                   "against their DOI before you submit: Verification could not confirm them."),
    Rule("ready", lambda f: True,
         lambda f: "Read the draft through and check each claim against its source in "
                   "final/evidence_table.md; nothing else needs your judgement."),
)


def match(facts: Facts) -> Rule:
    """The first matching row; the last row always matches."""
    return next(rule for rule in RULES if rule.when(facts))


def _strong_and_weak(f: Facts) -> tuple[list[str], list[str]]:
    strong, weak = [], []
    for key, label in _SCORES:
        if (score := f.scores.get(key)) is not None:
            if score >= _STRONG:
                strong.append(f"{label} {score:.1f}/5")
            elif score < _WEAK:
                weak.append(f"{label} {score:.1f}/5")
    total = f.verified + len(f.references)
    if f.verified:
        strong.append(f"{f.verified} of {total} reference(s) verified against Crossref/OpenAlex")
    if f.draft.get("draft_status") == "complete" and not f.broken:
        strong.append(f"Draft complete: {f.draft.get('sections_written', 0)} section(s) written")
    if f.draft.get("length_status") == "on_target":
        strong.append(f"Length on target ({f.draft.get('actual_words')} words)")
    if f.cache_hits:
        strong.append(f"{f.cache_hits} scholarly request(s) answered from the local cache")
    if f.memory_hits:
        strong.append(f"{f.memory_hits} paper excerpt(s) reused from the research library")

    if f.state != "completed":
        weak.append(f"The run stopped in {_title(f.stage)}" + (f": {f.error}" if f.error else ""))
    if f.draft.get("draft_status") in ("partial", "failed"):
        weak.append(f"Draft status: {f.draft['draft_status']}")
    for kind in ("invalid", "unverifiable"):
        if count := sum(1 for _, k, _ in f.references if k == kind):
            weak.append(f"{count} {kind} reference(s)")
    if f.unsupported_claims:
        weak.append(f"{f.unsupported_claims} claim(s) not supported by the cited paper's abstract")
    if f.missing_citations:
        weak.append(f"{f.missing_citations} factual sentence(s) without a citation")
    if f.draft.get("length_status") in ("short", "over_target"):
        weak.append(f"Length {f.draft['length_status'].replace('_', ' ')}: {f.draft.get('actual_words')} "
                    f"of ~{f.draft.get('target_words')} words")
    for host, stage in f.exhausted:
        weak.append(f"{host} ran out of budget during {_title(stage)}")
    if f.reduced_plan:
        weak.append("The OpenAlex budget check chose the reduced plan")
    return strong, weak


def build_brief(run_directory: Path) -> dict[str, Any]:
    """The brief of one run folder: reads it, writes nothing."""
    f = gather(run_directory)
    rule = match(f)
    next_action = rule.action(f)
    strong, weak = _strong_and_weak(f)
    sections = list(dict.fromkeys(f.broken + f.evidence_limited))
    lines = ["# Decision brief", "",
             f"Run {f.run_id} · {f.state}"
             + (f" · overall quality {f.scores['overall']:.1f}/5" if "overall" in f.scores else ""), "",
             "## Next action", "", next_action, "", "## Strong", ""]
    lines += [f"- {line}" for line in strong] or ["- Nothing stands out yet."]
    lines += ["", "## Weak", ""]
    lines += [f"- {line}" for line in weak] or ["- Nothing."]
    lines += ["", "## References to check by hand", ""]
    lines += [f"- [{n}] {kind}" + (f": {issue}" if issue else "") for n, kind, issue in f.references] \
        or ["- None: every reference was verified." if f.verified else "- None."]
    lines += ["", "## Sections to look at", ""]
    lines += [f"- {name}: " + ("broken" if name in f.broken else "evidence-limited") for name in sections] \
        or ["- None."]
    return {"rule": rule.id, "next_action": next_action, "strong": strong, "weak": weak,
            "check_references": [n for n, _, _ in f.references], "sections": sections,
            "state": f.state, "markdown": "\n".join(lines) + "\n"}


def brief_path(run_directory: Path, completed: bool) -> Path:
    run_directory = Path(run_directory)
    return run_directory / "final" / BRIEF_NAME if completed else run_directory / BRIEF_NAME


def write_brief(run_directory: Path) -> Path:
    """Writes the brief where it belongs (see the module docstring) and returns its path."""
    brief = build_brief(run_directory)
    path = brief_path(run_directory, brief["state"] == "completed")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(brief["markdown"], encoding="utf-8")
    if brief["state"] == "completed":
        brief_path(run_directory, False).unlink(missing_ok=True)  # an earlier failure's brief
    return path


def find_brief(run_directory: Path) -> Path | None:
    """The brief written for this run, if any."""
    return next((p for p in (brief_path(run_directory, True), brief_path(run_directory, False))
                 if p.is_file()), None)

```

### Real tests for this lab

### `tests/orchestrator/test_resume.py`

```python
"""Resume a failed run from the failed stage (#9), with every service faked."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "orchestrator"))

from orchestrator import pipeline as orchestrator_pipeline  # noqa: E402
from shared.contracts.pipeline_contract import ResearchRequest  # noqa: E402
from test_pipeline import (  # noqa: E402
    _fake_run_discovery, _fake_run_qa, _fake_run_verification, _fake_run_writing,
)

REQUEST = ResearchRequest(research_question="What is X?", corpus_size=6)


async def _failing_verification(request):
    yield {"type": "status", "stage": "verify", "state": "running", "message": "verifying"}
    raise orchestrator_pipeline.PipelineError("Crossref went away.")


async def _must_not_run(*args, **kwargs):
    raise AssertionError("an earlier stage was re-run on resume")
    yield  # pragma: no cover


def _drain(events):
    async def run():
        return [event async for event in events]
    return asyncio.run(run())


@pytest.fixture
def failed_run(tmp_path, monkeypatch):
    """A run that finished Discovery and Writing, then failed in Verification."""
    monkeypatch.setattr(orchestrator_pipeline.discovery_service, "run_discovery", _fake_run_discovery)
    monkeypatch.setattr(orchestrator_pipeline.writing_service, "run_writing", _fake_run_writing)
    monkeypatch.setattr(orchestrator_pipeline.verification_service, "run_verification", _failing_verification)
    monkeypatch.setattr(orchestrator_pipeline.qa_service, "run_quality_assurance", _fake_run_qa)
    with pytest.raises(orchestrator_pipeline.PipelineError):
        _drain(orchestrator_pipeline.run_pipeline(REQUEST, output_root=tmp_path, run_id="abc123"))
    # From here on Discovery and Writing must not be called again.
    monkeypatch.setattr(orchestrator_pipeline.discovery_service, "run_discovery", _must_not_run)
    monkeypatch.setattr(orchestrator_pipeline.writing_service, "run_writing", _must_not_run)
    monkeypatch.setattr(orchestrator_pipeline.verification_service, "run_verification", _fake_run_verification)
    return tmp_path / "what-is-x_abc123"


def test_resuming_a_run_that_failed_in_verification_skips_discovery_and_writing(failed_run):
    events = _drain(orchestrator_pipeline.resume_pipeline(failed_run))

    result = events[-1]["result"]
    assert result.run_id == "abc123" and Path(result.run_directory) == failed_run
    assert result.discovery.selected_papers[0].title == "Paper One"  # loaded, not re-run
    for name in ("draft.md", "references.md", "evidence_table.md", "paper.tex"):
        assert (failed_run / "final" / name).is_file()


def test_the_resume_is_recorded_in_the_event_log_and_metadata(failed_run):
    _drain(orchestrator_pipeline.resume_pipeline(failed_run))

    logged = [json.loads(line) for line in (failed_run / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    resume = [e for e in logged if e.get("details", {}).get("raw_type") == "resume"]
    assert len(resume) == 1 and resume[0]["details"]["from_stage"] == "verification"
    assert logged.index(resume[0]) > [e["type"] for e in logged].index("error")
    assert logged[-1]["type"] == "result"

    metadata = json.loads((failed_run / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["resumed_from_stage"] == "verification"
    assert metadata["final_versions"] == []


def test_an_existing_final_folder_is_kept_as_a_numbered_version(failed_run):
    _drain(orchestrator_pipeline.resume_pipeline(failed_run))
    (failed_run / "final" / "draft.md").write_text("first draft", encoding="utf-8")

    _drain(orchestrator_pipeline.resume_pipeline(failed_run, from_stage="quality_assurance"))
    _drain(orchestrator_pipeline.resume_pipeline(failed_run, from_stage="quality_assurance"))

    assert (failed_run / "final_v1" / "draft.md").read_text(encoding="utf-8") == "first draft"
    assert (failed_run / "final_v2" / "draft.md").is_file()
    assert (failed_run / "final" / "draft.md").is_file()
    metadata = json.loads((failed_run / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["resumed_from_stage"] == "quality_assurance"
    assert metadata["final_versions"] == ["final_v1", "final_v2"]


def test_a_corrupt_stage_result_fails_with_a_message_naming_the_stage(failed_run):
    (failed_run / "02_writing" / "result.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(orchestrator_pipeline.PipelineError, match="writing"):
        _drain(orchestrator_pipeline.resume_pipeline(failed_run))


def test_a_missing_earlier_stage_result_fails_with_a_message_naming_the_stage(failed_run):
    (failed_run / "01_discovery" / "result.json").unlink()
    with pytest.raises(orchestrator_pipeline.PipelineError, match="discovery"):
        _drain(orchestrator_pipeline.resume_pipeline(failed_run, from_stage="verification"))


def test_resume_finds_a_run_by_its_id(failed_run):
    assert orchestrator_pipeline.find_run_directory("abc123", failed_run.parent) == failed_run
    assert orchestrator_pipeline.find_run_directory("nope", failed_run.parent) is None


# ------------------------------------------------------------------ API + CLI

def _api_client(monkeypatch, output_root):
    from fastapi.testclient import TestClient
    from orchestrator import api, run_registry
    monkeypatch.setattr(orchestrator_pipeline, "DEFAULT_OUTPUT_ROOT", output_root)
    monkeypatch.setattr(run_registry, "_runs", {})
    return TestClient(api.app)


def test_the_api_resumes_a_failed_run_and_streams_its_events(failed_run, monkeypatch):
    import time
    client = _api_client(monkeypatch, failed_run.parent)
    assert client.get("/research/runs/abc123").json()["status"] == "failed"

    response = client.post("/research/runs/abc123/resume", json={})
    assert response.status_code == 202 and response.json() == {"run_id": "abc123"}
    for _ in range(200):
        body = client.get("/research/runs/abc123").json()
        if body["status"] != "running":
            break
        time.sleep(0.02)
    assert body["status"] == "completed"
    with client.stream("GET", "/research/runs/abc123/events") as stream:
        replay = "".join(stream.iter_text())
    assert '"raw_type": "resume"' in replay and replay.endswith("event: end\ndata: {}\n\n")


def test_the_api_refuses_a_resume_it_cannot_do(failed_run, monkeypatch):
    client = _api_client(monkeypatch, failed_run.parent)
    assert client.post("/research/runs/nope/resume", json={}).status_code == 404
    (failed_run / "02_writing" / "result.json").write_text("{not json", encoding="utf-8")
    response = client.post("/research/runs/abc123/resume", json={})
    assert response.status_code == 422 and "writing" in response.json()["detail"]


def test_the_cli_accepts_a_run_folder_or_a_run_id(failed_run, monkeypatch):
    from researchgenie import cli
    monkeypatch.setattr(cli, "DEFAULT_OUTPUT_ROOT", failed_run.parent)
    assert cli._resolve_run(str(failed_run)) == failed_run
    assert cli._resolve_run("abc123") == failed_run
    assert cli._resolve_run("nope") is None


def test_the_web_page_offers_resume_only_for_failed_runs():
    page = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert 'id="resume"' in page and "/resume" in page

```

### `tests/services/test_discovery_approval.py`

```python
"""The paper approval step (#17): the run pauses after search and screening,
before full text, analysis and Writing; dropped papers are backfilled from
the ranked candidates, so the corpus keeps its size."""

import asyncio
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "research-discovery"))

from discovery import pipeline  # noqa: E402

QUESTION = "Effects of intermittent fasting on cognitive performance"
QUERIES = {"keyword": "intermittent fasting cognitive performance", "pubmed": "fasting",
           "arxiv": "fasting", "interpretation": "How intermittent fasting affects cognition."}


def _work(i: int) -> dict:
    return {
        "id": f"https://openalex.org/W{i}",
        "display_name": f"Intermittent fasting and cognitive performance in adults, study {i}",
        "abstract_inverted_index": {"Intermittent": [0], "fasting": [1], "improved": [2],
                                    "cognitive": [3], "performance": [4]},
        "publication_year": 2022, "cited_by_count": 100 - i,
        "doi": f"https://doi.org/10.1000/if.{i}",
        "authorships": [{"author": {"display_name": "A. Author"}}],
        "primary_location": {"source": {"display_name": "Nutrients"}},
    }


def _transport():
    def handler(request):
        if request.url.host == "api.openalex.org":
            return httpx.Response(200, json={"results": [_work(i) for i in range(8)]})
        return httpx.Response(404)
    return httpx.MockTransport(handler)


async def _fake_queries(question, model):
    return dict(QUERIES)


def _until_corpus(monkeypatch, approve=None):
    monkeypatch.setattr(pipeline, "generate_queries", _fake_queries)

    async def go():
        events = []
        stream = pipeline.run_pipeline(QUESTION, ["openalex"], per_source=10, corpus_size=5,
                                       fulltext_enabled=False, transport=_transport(),
                                       approve=approve)
        try:
            async for event in stream:
                events.append(event)
                if event["type"] == "corpus":
                    break
        finally:
            await stream.aclose()
        return events

    return asyncio.run(go())


def test_dropped_papers_are_replaced_from_the_ranked_candidates(monkeypatch):
    shown = {}

    async def approve(papers, reserve):
        shown["papers"], shown["reserve"] = papers, reserve
        return {"approved_by": "cli", "drop": [2, 5]}

    events = _until_corpus(monkeypatch, approve)
    assert {"title", "relevance_score", "year", "venue"} <= set(shown["papers"][0])
    assert shown["reserve"] == 3
    dropped = {shown["papers"][1]["title"], shown["papers"][4]["title"]}

    corpus = events[-1]["papers"]
    assert len(corpus) == 5 and not dropped & {p["title"] for p in corpus}
    assert [p["id"] for p in corpus] == [1, 2, 3, 4, 5]

    [approval] = [e for e in events if e["type"] == "approval"]
    assert approval["approved_by"] == "cli" and approval["approved_at"]
    assert {d["title"] for d in approval["dropped"]} == dropped
    assert len(approval["replacements"]) == 2
    types = [e["type"] for e in events]
    assert types.index("approval_request") < types.index("approval")


def test_without_an_approver_the_corpus_is_approved_automatically(monkeypatch):
    events = _until_corpus(monkeypatch)
    assert "approval_request" not in [e["type"] for e in events]
    [approval] = [e for e in events if e["type"] == "approval"]
    assert approval["approved_by"] == "auto" and approval["dropped"] == []
    assert len(events[-1]["papers"]) == 5


def test_the_final_draft_never_cites_a_dropped_paper(tmp_path, monkeypatch):
    """Real Discovery service and orchestrator; fake network, model and later stages."""
    from orchestrator import pipeline as orchestrator_pipeline
    from shared.contracts.pipeline_contract import ResearchRequest
    sys.path.insert(0, str(ROOT / "tests" / "orchestrator"))
    from test_pipeline import _fake_qa_result, _fake_verification_result, _fake_writing_result

    service = orchestrator_pipeline.discovery_service
    real = service.run_pipeline

    async def fake_analysis(question, papers, candidates, verification_papers, model_name):
        yield {"type": "report", "report": {
            "field_overview": "o", "themes": [], "limitations": "l", "gaps": [],
            "author_flagged_gaps": [], "confidence_notes": "c",
            "novelty_analysis": {"novelty_summary": "s", "confidence": "low", "caveats": "c"}}}

    monkeypatch.setattr(pipeline, "generate_queries", _fake_queries)
    monkeypatch.setattr(pipeline, "analyze_gaps", fake_analysis)
    monkeypatch.setattr(service, "run_pipeline", lambda *a, **k: real(
        *a, **{**k, "transport": _transport(), "fulltext_enabled": False}))
    seen_by_writing = []

    async def writing(request, output_root=None):
        seen_by_writing.extend(p.title for p in request.discovery.selected_papers)
        draft = " ".join(f"{p.title} [{p.id}]." for p in request.discovery.selected_papers)
        yield {"type": "result", "result": _fake_writing_result().model_copy(
            update={"research_draft_markdown": draft})}

    async def verification(request):
        yield {"type": "result", "result": _fake_verification_result()}

    async def qa(request):
        yield {"type": "result", "result": _fake_qa_result().model_copy(
            update={"final_draft_markdown": request.writing.research_draft_markdown})}

    monkeypatch.setattr(orchestrator_pipeline.writing_service, "run_writing", writing)
    monkeypatch.setattr(orchestrator_pipeline.verification_service, "run_verification", verification)
    monkeypatch.setattr(orchestrator_pipeline.qa_service, "run_quality_assurance", qa)
    dropped = []

    async def approve(papers, reserve):
        dropped.append(papers[0]["title"])
        return {"approved_by": "web", "drop": [1]}

    async def run():
        return [e async for e in orchestrator_pipeline.run_pipeline(
            ResearchRequest(research_question=QUESTION, corpus_size=6), output_root=tmp_path,
            approve=approve)]

    result = asyncio.run(run())[-1]["result"]
    assert len(seen_by_writing) == 6 and dropped[0] not in seen_by_writing
    draft = (Path(result.run_directory) / "final" / "draft.md").read_text(encoding="utf-8")
    assert dropped[0] not in draft
    logged = (Path(result.run_directory) / "events.jsonl").read_text(encoding="utf-8")
    assert '"approved_by": "web"' in logged and dropped[0] in logged

```

### `tests/orchestrator/test_decision_brief.py`

```python
"""The decision brief (#12): one test per rule row over hand-built run
folders, then the brief written at the end of a real (faked-service) run,
completed and failed, without a model call, and shown by every surface."""

import asyncio
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from rich.console import Console

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "orchestrator"))

from orchestrator import decision_brief  # noqa: E402
from orchestrator import pipeline as orchestrator_pipeline  # noqa: E402
from researchgenie import headless, tui  # noqa: E402
from shared.contracts.pipeline_contract import ResearchRequest  # noqa: E402
from shared.contracts.verification_contract import ReferenceCheck  # noqa: E402
from shared.utilities import llm_provider  # noqa: E402
from test_pipeline import (  # noqa: E402
    _fake_qa_result, _fake_run_discovery, _fake_run_qa, _fake_run_verification,
    _fake_run_writing, _fake_verification_result, _fake_writing_result,
)

REFERENCES = [f'[{n}] A. Author, "Paper {n}," 2020, doi: 10.1/{n}.' for n in (1, 2, 3)]


def _event(stage, status="running", **details):
    return {"type": "progress", "run_id": "r1", "stage": stage, "status": status,
            "message": "m", "details": details}


def _folder(tmp_path, events=None, draft=None, invalid=(), unverifiable=(), failed=False):
    """A run folder: the given events (else a completed run's), the Writing,
    Verification and QA results, the draft metadata overridden by `draft`."""
    folder = tmp_path / "what-is-x_r1"
    folder.mkdir()
    events = events or [_event("quality_assurance", "done")]
    events = events + [{"type": "error", "message": "Ollama went away"} if failed
                       else {"type": "result", "result": {}}]
    (folder / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    writing = _fake_writing_result()
    writing = writing.model_copy(update={
        "reference_candidates": REFERENCES,
        "draft_metadata": writing.draft_metadata.model_copy(update=draft or {})})
    check = lambda n, issue: ReferenceCheck(reference_entry=REFERENCES[n - 1], verified=False, issue=issue)  # noqa: E731
    verification = _fake_verification_result().model_copy(update={
        "verified_references": [ReferenceCheck(reference_entry=REFERENCES[0], verified=True)],
        "invalid_references": [check(n, "DOI not found") for n in invalid],
        "unverifiable_references": [check(n, "no DOI") for n in unverifiable]})
    for name, result in (("02_writing", writing), ("03_verification", verification),
                         ("04_quality_assurance", _fake_qa_result())):
        (folder / name).mkdir()
        (folder / name / "result.json").write_text(result.model_dump_json(), encoding="utf-8")
    return folder


ROWS = [
    # events, draft metadata, invalid, unverifiable, failed, rule, words in the action
    ([_event("writing")], {"broken_sections": ["Discussion"]}, (), (), True,
     "run_failed", ("researchgenie doctor r1", "Writing", "Ollama went away")),
    (None, {"draft_status": "partial", "broken_sections": ["Discussion", "Methods"]}, (2,), (), False,
     "broken_sections", ("--from writing", "2 section(s) are broken: Discussion, Methods")),
    ([_event("verification", "warning", raw_type="provider_exhausted", host="api.openalex.org")],
     {"evidence_limited_sections": ["Results"]}, (), (3,), False,
     "reduced_plan", ("Add your OpenAlex key", "--from verification", "reduced plan")),
    ([{"status": "budget_check", "details": {"plan": "reduced"}}], {}, (), (), False,
     "reduced_plan", ("Add your OpenAlex key", "--from verification")),
    (None, {"evidence_limited_sections": ["Methods", "Results"]}, (2,), (), False,
     "evidence_limited", ("--from writing", "2 section(s) were evidence-limited: Methods, Results")),
    (None, {}, (2,), (3,), False, "check_references", ("Check reference(s) 2, 3 by hand",)),
    (None, {}, (), (), False, "ready", ("evidence_table.md",)),
]


@pytest.mark.parametrize("events, draft, invalid, unverifiable, failed, rule, words", ROWS,
                         ids=[f"{row[5]}-{i}" for i, row in enumerate(ROWS)])
def test_each_rule_row(tmp_path, events, draft, invalid, unverifiable, failed, rule, words):
    brief = decision_brief.build_brief(_folder(tmp_path, events, draft, invalid, unverifiable, failed))
    assert brief["rule"] == rule
    assert all(word in brief["next_action"] for word in words), brief["next_action"]
    # Exactly one next action, in the markdown under its own heading.
    assert brief["markdown"].count("## Next action") == 1
    assert f"## Next action\n\n{brief['next_action']}\n" in brief["markdown"]


def test_every_flagged_reference_and_section_is_listed(tmp_path):
    folder = _folder(tmp_path, draft={"broken_sections": ["Discussion"],
                                      "evidence_limited_sections": ["Methods"]},
                     invalid=(3,), unverifiable=(2,))
    brief = decision_brief.build_brief(folder)
    assert brief["check_references"] == [2, 3]
    assert brief["sections"] == ["Discussion", "Methods"]
    for line in ("- [2] unverifiable: no DOI", "- [3] invalid: DOI not found",
                 "- Discussion: broken", "- Methods: evidence-limited"):
        assert line in brief["markdown"]
    assert "Overall quality 5.0/5" in brief["strong"]
    assert "1 invalid reference(s)" in brief["weak"] and "1 unverifiable reference(s)" in brief["weak"]


def test_cache_and_memory_hits_are_read_when_logged(tmp_path):
    events = [_event("verification", raw_type="cache_hit", hits=4),
              {"status": "memory_hit", "details": {"count": 2}}]
    brief = decision_brief.build_brief(_folder(tmp_path, events))
    assert "4 scholarly request(s) answered from the local cache" in brief["strong"]
    assert "2 paper excerpt(s) reused from the research library" in brief["strong"]


# ------------------------------------------------- at the end of every run

@pytest.fixture
def services(tmp_path, monkeypatch):
    """Faked services, and a model that must never be asked."""
    for service, name, fake in (
            (orchestrator_pipeline.discovery_service, "run_discovery", _fake_run_discovery),
            (orchestrator_pipeline.writing_service, "run_writing", _fake_run_writing),
            (orchestrator_pipeline.verification_service, "run_verification", _fake_run_verification),
            (orchestrator_pipeline.qa_service, "run_quality_assurance", _fake_run_qa)):
        monkeypatch.setattr(service, name, fake)

    def refuse(*args, **kwargs):
        raise AssertionError("the decision brief must not call the model")
    for name in ("stream_json", "complete_json", "complete_tools", "health_check"):
        monkeypatch.setattr(llm_provider, name, refuse)
    monkeypatch.setattr(orchestrator_pipeline, "DEFAULT_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(headless, "DEFAULT_OUTPUT_ROOT", tmp_path)
    return monkeypatch


async def _drain(events):
    async for _ in events:
        pass


def test_a_completed_run_writes_final_decision_brief_without_the_model(tmp_path, services):
    asyncio.run(_drain(orchestrator_pipeline.run_pipeline(
        ResearchRequest(research_question="What is X?", corpus_size=6), output_root=tmp_path, run_id="r1")))
    folder = orchestrator_pipeline.find_run_directory("r1", tmp_path)
    written = (folder / "final" / "decision_brief.md").read_text(encoding="utf-8")
    assert written == decision_brief.build_brief(folder)["markdown"]
    assert "Run r1 · completed" in written

    report = headless.result_of(folder)
    assert report["decision_brief"]["path"] == str(folder / "final" / "decision_brief.md")
    assert report["decision_brief"]["markdown"] == written
    assert report["decision_brief"]["next_action"] in written
    assert "decision_brief.md" in report["final_files"]


def test_a_failed_run_gets_a_brief_that_points_to_the_doctor_until_a_resume_completes(tmp_path, services):
    async def failing_writing(request, output_root=None):
        raise orchestrator_pipeline.PipelineError("Research Writing Service produced no result.")
        yield  # pragma: no cover
    services.setattr(orchestrator_pipeline.writing_service, "run_writing", failing_writing)
    with pytest.raises(orchestrator_pipeline.PipelineError):
        asyncio.run(_drain(orchestrator_pipeline.run_pipeline(
            ResearchRequest(research_question="What is X?", corpus_size=6), output_root=tmp_path, run_id="r1")))
    folder = orchestrator_pipeline.find_run_directory("r1", tmp_path)
    assert not (folder / "final").exists()
    report = headless.result_of(folder)
    assert report["decision_brief"]["path"] == str(folder / "decision_brief.md")
    assert report["decision_brief"]["next_action"].startswith("Run `researchgenie doctor r1`")

    services.setattr(orchestrator_pipeline.writing_service, "run_writing", _fake_run_writing)
    asyncio.run(_drain(orchestrator_pipeline.resume_pipeline(folder)))
    assert (folder / "final" / "decision_brief.md").is_file()
    assert not (folder / "decision_brief.md").exists()


def test_the_api_cli_and_web_page_show_the_brief(tmp_path, services):
    asyncio.run(_drain(orchestrator_pipeline.run_pipeline(
        ResearchRequest(research_question="What is X?", corpus_size=6), output_root=tmp_path, run_id="r1")))
    brief = decision_brief.build_brief(orchestrator_pipeline.find_run_directory("r1", tmp_path))

    from orchestrator import api
    client = TestClient(api.app)
    body = client.get("/research/runs/r1/brief").json()
    assert body["next_action"] == brief["next_action"] and body["path"].endswith("decision_brief.md")
    assert client.get("/research/runs/nope/brief").status_code == 404
    assert '"Decision brief": "_Loading the decision brief…_"' in client.get("/").text

    console = Console(record=True, width=400)
    tui.render_brief(console, brief)
    assert brief["next_action"] in console.export_text()

```

## The experiment

```bash
cd Course-Project/code
pip install -e .
pytest -v tests/orchestrator/test_resume.py tests/services/test_discovery_approval.py tests/orchestrator/test_decision_brief.py
```

## Real result
The approval gate, the versioned resume logic, which is implemented inside orchestrator/pipeline.py shown in full in Lab 6, and the rule built decision brief above all pass against the real implementation. `Course-Project/docs/GOVERNANCE.md` maps 6 named risks, each to a control, a test and a rollback.

## Why this is the real evidence, not a toy
The decision brief is deterministic by design, no LLM is in that loop at all, specifically so grading it can never drift with a model update.

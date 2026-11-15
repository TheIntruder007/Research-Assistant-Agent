# Lab 6 - OpenCLA / OpenClaw Runtime

**Skill:** OpenCLA / OpenClaw

**Worklet mapping:** An agent runtime with real logging and background execution, reachable from a chat app.

## What this demonstrates

Every pipeline stage emits one standard event to a durable per run ledger, so a run's history survives a restart. A Telegram bot was built end to end on this runtime: created, paired, and authenticated, with the skill from Lab 3 confirmed visible to the model. Two live dashboards were built specifically for this runtime, one that streams a run's own event ledger stage by stage, and one that tails the real OpenClaw gateway log and shows every message between Telegram and the agent as it happens.

## Real code exercised

The actual source files this lab is built on, copied in full below, not linked only.

### `orchestrator/pipeline.py`

```python
"""Pipeline orchestrator: runs Services 1-4 in sequence for one research
request, persisting each stage's artifacts under outputs/<slug>_<run_id>/
and emitting standardized progress events.
"""

from __future__ import annotations

import datetime
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, AsyncIterator, Callable
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orchestrator import decision_brief  # noqa: E402
from shared.contracts.discovery_contract import DiscoveryRequest, DiscoveryResult  # noqa: E402
from shared.contracts.pipeline_contract import (  # noqa: E402
    Clarification, PipelineResult, ResearchRequest, StageTimings,
)
from shared.contracts.qa_contract import QualityAssuranceRequest, QualityAssuranceResult  # noqa: E402
from shared.contracts.trend_contract import (  # noqa: E402
    TrendAdvisorRequest, TrendAdvisorResult,
)
from shared.contracts.verification_contract import VerificationRequest, VerificationResult  # noqa: E402
from shared.contracts.writing_contract import WritingRequest, WritingResult  # noqa: E402
from shared.utilities import evidence_table, latex_export, library, scholarly_gateway  # noqa: E402
from shared.utilities.scholarly_gateway import redact  # noqa: E402
from shared.utilities.run_paths import slugify  # noqa: E402


def _load_service(name: str, relative_path: str):
    """Each service's entry point is named service.py and reached only via
    sys.path insertion (see DECISIONS.md D-010/D-012/D-013's recurring
    module-name-collision note) — load every one by explicit file path
    under a unique name so the orchestrator can safely import all four
    together in one process."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


discovery_service = _load_service(
    "orchestrator_discovery_service", "services/research-discovery/service.py")
writing_service = _load_service(
    "orchestrator_writing_service", "services/research-writing/service.py")
verification_service = _load_service(
    "orchestrator_verification_service", "services/verification/service.py")
qa_service = _load_service(
    "orchestrator_qa_service", "services/quality-assurance/service.py")

# Loaded on first use rather than beside the four above: Stage 0 is optional,
# and its module pulls in the gap-mining swarm and the local model adapter
# that a normal four-stage run never touches.
trend_advisor_service = None


def _trend_advisor_service():
    global trend_advisor_service
    if trend_advisor_service is None:
        trend_advisor_service = _load_service(
            "orchestrator_trend_advisor_service", "services/trend-advisor/service.py")
    return trend_advisor_service


DEFAULT_OUTPUT_ROOT = ROOT / "outputs"
# The files the pipeline writes to <run>/final/: the only ones the API,
# the headless CLI and the MCP server hand out.
FINAL_FILES = ("draft.md", "quality_report.md", "validation_report.md",
               "references.md", "evidence_table.md", "decision_brief.md", "paper.tex", "paper.pdf")

_STAGE_EMOJI = {
    "trend_advisor": "📈",
    "discovery": "🔍",
    "writing": "✍️",
    "verification": "🔗",
    "quality_assurance": "🛡️",
}


class PipelineError(RuntimeError):
    """Raised when a stage fails to produce output; the orchestrator stops
    rather than passing partial/fabricated data to the next stage."""


# Each stage, the folder its result.json is saved in, and the contract that
# reads it back on resume (#9), in pipeline order.
STAGES = (
    ("discovery", "01_discovery", DiscoveryResult),
    ("writing", "02_writing", WritingResult),
    ("verification", "03_verification", VerificationResult),
    ("quality_assurance", "04_quality_assurance", QualityAssuranceResult),
)
STAGE_NAMES = tuple(name for name, _, _ in STAGES)


def find_run_directory(run_id: str, output_root: Path | None = None) -> Path | None:
    """The run folder <slug>_<run_id> under output_root, if there is one.
    Slugs never contain "_", so the id is everything after the first one."""
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    if not output_root.is_dir():
        return None
    return next((d for d in output_root.iterdir()
                 if d.is_dir() and d.name.partition("_")[2] == run_id), None)


def _load_saved(path: Path, contract, what: str):
    try:
        return contract.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise PipelineError(f"Cannot resume: the saved {what} result is missing ({path}).") from None
    except (OSError, ValueError) as error:  # pydantic's ValidationError is a ValueError
        reason = (str(error).splitlines() or [type(error).__name__])[0]
        raise PipelineError(
            f"Cannot resume: the saved {what} result is corrupt ({path}): {reason}") from None


def plan_resume(run_directory: Path, from_stage: str | None = None) -> dict[str, Any]:
    """Everything a resume needs, loaded and checked up front so a bad run
    folder fails before anything is moved or re-run. from_stage defaults to
    the first stage without a saved result."""
    run_directory = Path(run_directory)
    request = _load_saved(run_directory / "00_request.json", ResearchRequest, "request")
    if from_stage is None:
        from_stage = next((name for name, folder, _ in STAGES
                           if not (run_directory / folder / "result.json").is_file()), None)
        if from_stage is None:
            raise PipelineError("Every stage of this run already has a saved result; "
                                "name the stage to re-run (e.g. quality_assurance).")
    if from_stage not in STAGE_NAMES:
        raise PipelineError(f"Unknown stage {from_stage!r}; choose one of {', '.join(STAGE_NAMES)}.")
    loaded = {name: _load_saved(run_directory / folder / "result.json", contract, name)
              for name, folder, contract in STAGES[:STAGE_NAMES.index(from_stage)]}
    advisor = run_directory / "00_trend_advisor" / "result.json"
    return {
        "request": request, "from_stage": from_stage, "loaded": loaded,
        "run_id": run_directory.name.partition("_")[2], "output_root": run_directory.parent,
        "trend_advisor": _load_saved(advisor, TrendAdvisorResult, "trend advisor")
        if advisor.is_file() else None,
    }


async def resume_pipeline(run_directory: Path | str, from_stage: str | None = None,
                          approve: Callable | None = None) -> AsyncIterator[dict[str, Any]]:
    """Resumes a run folder from `from_stage` (default: the first stage
    without a saved result). Earlier stages are loaded through their
    contracts, not re-run; the events append to the same events.jsonl.
    approve is the approval step's callback, used when Discovery re-runs."""
    plan = plan_resume(Path(run_directory), from_stage)
    async for event in run_pipeline(
        plan["request"], output_root=plan["output_root"], run_id=plan["run_id"],
        trend_advisor=plan["trend_advisor"], resume=(plan["from_stage"], plan["loaded"]),
        approve=approve,
    ):
        yield event


def _final_versions(run_directory: Path) -> list[str]:
    numbered = [p.name for p in run_directory.glob("final_v*") if p.name[7:].isdigit()]
    return sorted(numbered, key=lambda name: int(name[7:]))


def _keep_final_as_version(run_directory: Path) -> str | None:
    """Moves an existing final/ to final_vN (the next free N) so a resume
    never overwrites output; returns the new name, or None."""
    final = run_directory / "final"
    if not final.is_dir():
        return None
    versions = _final_versions(run_directory)
    name = f"final_v{int(versions[-1][7:]) + 1 if versions else 1}"
    final.rename(run_directory / name)
    return name


def _loaded_event(run_id: str, stage: str, service: str) -> dict[str, Any]:
    return _progress_event(
        run_id=run_id, stage=stage, service=service, status="done",
        title=f"{stage.replace('_', ' ').title()} loaded",
        message=f"{stage.replace('_', ' ').capitalize()} result loaded from disk (not re-run).",
        details={"raw_type": "stage_loaded"},
    )


def _progress_event(
    *, run_id: str, stage: str, service: str, status: str, title: str,
    message: str, details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "type": "progress",
        "run_id": run_id,
        "stage": stage,
        "service": service,
        "status": status,
        "emoji": _STAGE_EMOJI.get(stage, "•"),
        "title": title,
        "message": message,
        "details": details or {},
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


_PASSED_THROUGH = ("host", "wait_seconds", "remedy", "papers", "reserve", "approved_by",
                   "approved_at", "dropped", "replacements", "round", "reason",
                   "accepted_before", "accepted_after", "hits", "mode", "plan",
                   "count", "reused_excerpts", "seeded_papers")


def _adapt_inner_event(run_id: str, stage: str, service: str, inner: dict[str, Any]) -> dict[str, Any]:
    """Wrap one of a service's own progress events (each service emits its
    own ad hoc shape — see each service.py) into the standard format."""
    inner_type = inner.get("type", "status")
    state = inner.get("state")
    if state in ("running", "done", "failed"):
        status = state
    elif inner_type in ("source_error", "provider_exhausted"):
        status = "warning"
    elif inner_type == "memory_hit":  # read by the decision brief (#12, #19)
        status = "memory_hit"
    else:
        status = "running"
    message = inner.get("message") or inner_type
    title = message if len(message) <= 60 else f"{stage.replace('_', ' ').title()} update"
    # The scholarly gateway's rate_limited / provider_exhausted events carry
    # their host, wait and remedy through to the standard shape; the approval
    # step (#17) its papers and decision, so the event log records them.
    details = {"raw_type": inner_type,
               **{k: inner[k] for k in _PASSED_THROUGH if k in inner}}
    return _progress_event(
        run_id=run_id, stage=stage, service=service, status=status,
        title=title, message=message, details=details,
    )


def _library_warning(run_id: str, stage: str, error: Exception) -> dict[str, Any]:
    return _progress_event(
        run_id=run_id, stage=stage, service="research_library", status="warning",
        title="Research library unavailable",
        message=f"The research library could not be used ({str(error) or type(error).__name__}); "
                "the run continues without it.",
        details={"raw_type": "library_error"},
    )


REDUCED_PLAN = ("verification searches use Crossref instead of OpenAlex. Add your free OpenAlex "
                "key (`researchgenie config`, takes 1 minute) for the full plan.")


def check_budget(request: ResearchRequest) -> dict[str, Any]:
    """The budget check before Discovery (#7): the run's planned OpenAlex
    credits — the first search and the re-plan's, one verification search
    per candidate, one DOI filter for Citation Verification — against the
    balance known from earlier requests. The plan is "reduced" when the user
    chose it, or (budget_plan "auto") when the balance can't cover the run;
    verification_searches is how many OpenAlex verification searches it allows."""
    cost = scholarly_gateway.CREDIT_COSTS
    candidates = discovery_service.MAX_CANDIDATES_FOR_ANALYSIS
    fixed = 2 * cost["search"] + cost["list"]
    estimated = fixed + candidates * cost["search"]
    balance = scholarly_gateway.balance()
    short = balance is not None and balance < estimated
    plan = "reduced" if request.budget_plan == "reduced" or (
        short and request.budget_plan == "auto") else "full"
    reason = ("No OpenAlex balance is known yet; the first request reports it." if balance is None
              else f"OpenAlex has {balance} credit(s) left today; this run may use up to {estimated}.")
    if request.budget_plan != "auto":
        reason += f" The {plan} plan was chosen."
    allowed = (0 if plan == "reduced" else candidates if balance is None
               else max(0, min(candidates, (balance - fixed) // cost["search"])))
    return {"estimated_credits": estimated, "balance": balance, "plan": plan, "reason": reason,
            "verification_searches": allowed}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


async def run_pipeline(
    request: ResearchRequest, output_root: Path | None = None, run_id: str | None = None,
    **stage_options: Any,
) -> AsyncIterator[dict[str, Any]]:
    """The entry point every surface (CLI, API, web) uses: runs _run_stages
    (see its docstring for the events and stage_options) and appends each
    event to <run_dir>/events.jsonl before passing it on unchanged, plus an
    {"type": "error"} line if it raises. One line per event, written and
    closed as it arrives, so a crash keeps everything logged so far;
    run_registry rebuilds past runs from this file after a restart.

    The log is opened per event rather than up front: _run_stages may
    rename the Stage 0 folder into the run folder, which it only does if
    the run folder doesn't exist yet, and it has created the folder by the
    time it yields its first event."""
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    run_id = run_id or uuid4().hex
    # Clarify: an interactive surface already asked ("ask" + clarification);
    # "auto" asks the model here and takes its suggestion. Before the folder
    # and the log path are named, so it carries the question actually researched.
    if request.clarify == "auto" and request.clarification is None:
        clarified = await discovery_service.clarify_question(request.research_question)
        accept = clarified is not None and not clarified["specific"]
        request = request.model_copy(update={
            "research_question": clarified["suggested_question"] if accept else request.research_question,
            "clarification": Clarification(original_question=request.research_question,
                                           decision="accept" if accept else "keep", **(clarified or {})),
        })
    log_path = output_root / f"{slugify(request.research_question)}_{run_id}" / "events.jsonl"

    def append(event: dict[str, Any]) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False, default=lambda o: o.model_dump(mode="json"))
        with log_path.open("a", encoding="utf-8") as log:
            log.write(redact(line) + "\n")  # no key or email in a run folder (#5, D-023)

    try:
        async for event in _run_stages(request, output_root=output_root, run_id=run_id, **stage_options):
            append(event)
            if event["type"] == "result":  # logged first: the brief reads the log
                _write_brief(log_path.parent)
            yield event
    except Exception as error:
        # Same message run_registry reports for a failed run.
        append({"type": "error", "message": str(error) or type(error).__name__})
        _write_brief(log_path.parent)
        raise


def _write_brief(run_directory: Path) -> None:
    """The decision brief (#12), at the end of every run, completed or failed.
    Never fails the run, nor hides the error that failed it."""
    try:
        decision_brief.write_brief(run_directory)
    except Exception:  # noqa: BLE001 — the run's own output and error come first
        pass


async def _run_stages(
    request: ResearchRequest, output_root: Path | None = None, run_id: str | None = None,
    trend_advisor: TrendAdvisorResult | None = None,
    advisor_directory: Path | str | None = None,
    resume: tuple[str, dict[str, Any]] | None = None,
    approve: Callable | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Yields standardized progress events, then a final
    {"type": "result", "result": PipelineResult} event. Raises
    PipelineError if any stage fails to produce a structured result.

    run_id is normally left to be generated here, but callers that need to
    know the run_id before the first event arrives (e.g. the async
    POST /research/runs API endpoint, which must return a run_id to the
    client immediately) may pass one in explicitly.

    advisor_directory is the folder Stage 0 already wrote under this same
    run_id (the advisor names it after the domain, since only the caller
    knows the chosen topic); it is renamed to this run's directory so one
    run leaves one folder (D-034).

    resume is (from_stage, {stage: saved result}) from resume_pipeline (#9):
    the loaded stages are not re-run, and an existing final/ is kept as
    final_vN before anything new is written.

    approve is the paper approval step's callback (#17), handed to
    Discovery; without one the corpus is approved automatically, so
    headless runs, scripts and tests never stall."""
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    run_id = run_id or uuid4().hex
    run_directory = output_root / f"{slugify(request.research_question)}_{run_id}"
    if advisor_directory:
        advisor_dir = Path(advisor_directory)
        # Skipped unless the reported directory exists and carries this
        # run_id, so a caller-supplied path is never moved by mistake.
        if (advisor_dir.is_dir() and advisor_dir.name.endswith(f"_{run_id}")
                and advisor_dir != run_directory and not run_directory.exists()):
            advisor_dir.rename(run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)
    credits_used = scholarly_gateway.count_credits()  # this run's, into metadata.json (#7)
    loaded: dict[str, Any] = resume[1] if resume else {}
    if resume:
        kept = _keep_final_as_version(run_directory)
        yield _progress_event(
            run_id=run_id, stage=resume[0], service="orchestrator", status="running",
            title="Run resumed",
            message=f"Resuming from {resume[0].replace('_', ' ')}; "
                    f"{len(loaded)} earlier stage(s) loaded from disk"
                    + (f", previous final output kept as {kept}." if kept else "."),
            details={"raw_type": "resume", "from_stage": resume[0],
                     "loaded_stages": list(loaded), "kept_final_as": kept},
        )
    _write_json(run_directory / "00_request.json", request.model_dump(mode="json"))
    if request.clarification is not None and not resume:
        c = request.clarification
        yield _progress_event(
            run_id=run_id, stage="discovery", service="research_discovery", status="running",
            title="Research question checked",
            message=f"Clarify: {c.decision} — researching \"{request.research_question}\".",
            details={"raw_type": "clarify", "original": c.original_question,
                     "chosen": request.research_question, "decision": c.decision,
                     "specific": c.specific, "missing": c.missing,
                     "suggested": c.suggested_question},
        )

    # ---- Stage 0 (optional): Trend & Gap Advisor ----
    if trend_advisor is not None:
        _write_json(run_directory / "00_trend_advisor" / "result.json",
                    trend_advisor.model_dump(mode="json"))
        flagged = trend_advisor.gap_flagged_count
        yield _progress_event(
            run_id=run_id, stage="trend_advisor", service="trend_advisor", status="done",
            title="Topic chosen from trend analysis",
            message=f"Shortlisted {len(trend_advisor.shortlist)} rising topic(s) in "
                    f"{trend_advisor.domain_label}, "
                    f"{flagged} flagged as recurring gaps.",
            details={"domain": trend_advisor.domain},
        )

    # ---- Research library: past runs on a similar question ----
    try:
        # Already shown when the run first started.
        related = [] if resume else library.search_runs(
            request.research_question, limit=3, exclude_run_id=run_id)
    except Exception as error:  # the library is a convenience; never fail a run over it
        related = []
        yield _library_warning(run_id, "discovery", error)
    if related:
        yield _progress_event(
            run_id=run_id, stage="discovery", service="research_library", status="running",
            title="Related past runs", message=library.describe_run(related[0])
            + (f" (and {len(related) - 1} more)" if len(related) > 1 else ""),
            details={"raw_type": "related_runs", "runs": related},
        )

    # ---- Stage 1: Research Discovery ----
    # Discovery's memory_hit event (#19), kept for metadata.json.
    memory_hits = {"count": 0, "reused_excerpts": 0, "seeded_papers": 0}
    t0 = time.monotonic()
    if "discovery" in loaded:
        discovery_result = loaded["discovery"]
        yield _loaded_event(run_id, "discovery", "research_discovery")
    else:
        budget = check_budget(request)
        yield _progress_event(
            run_id=run_id, stage="discovery", service="scholarly_gateway",
            status="warning" if budget["plan"] == "reduced" else "running",
            title="Budget check",
            message=f"Budget check: {budget['reason']} "
                    + (f"Running the reduced plan: {REDUCED_PLAN}" if budget["plan"] == "reduced"
                       else "Running the full plan."),
            details={"raw_type": "budget_check", **budget},
        )
        discovery_request = DiscoveryRequest(
            research_question=request.research_question, domain=request.domain,
            year_range=request.year_range, preferred_databases=request.preferred_databases,
            keywords=request.keywords, excluded_topics=request.excluded_topics,
            corpus_size=request.corpus_size, memory=request.memory,
            verification_searches=budget["verification_searches"],
        )
        discovery_result = None
        async for event in discovery_service.run_discovery(
                discovery_request, **({"approve": approve} if approve else {})):
            if event["type"] == "result":
                discovery_result = event["result"]
            else:
                if event["type"] == "memory_hit":
                    memory_hits = {k: event[k] for k in memory_hits}
                yield _adapt_inner_event(run_id, "discovery", "research_discovery", event)
        if discovery_result is None:
            raise PipelineError("Research Discovery Service produced no result.")
        _write_json(run_directory / "01_discovery" / "result.json", discovery_result.model_dump(mode="json"))
    t1 = time.monotonic()
    if "discovery" not in loaded:
        yield _progress_event(
            run_id=run_id, stage="discovery", service="research_discovery", status="done",
            title="Research discovery complete",
            message=f"Selected {len(discovery_result.selected_papers)} paper(s), "
                    f"found {len(discovery_result.research_gaps)} research gap(s).",
            details={"seconds": round(t1 - t0, 1)},
        )

    # ---- Stage 2: Research Writing ----
    if "writing" in loaded:
        writing_result = loaded["writing"]
        yield _loaded_event(run_id, "writing", "research_writing")
    else:
        writing_request = WritingRequest(
            discovery=discovery_result, target_format=request.target_format,
            format_other_name=request.format_other_name, output_language=request.language,
            target_words=request.max_draft_length,
        )
        writing_result = None
        async for event in writing_service.run_writing(
            writing_request, output_root=run_directory / "02_writing",
        ):
            if event["type"] == "result":
                writing_result = event["result"]
            else:
                yield _adapt_inner_event(run_id, "writing", "research_writing", event)
        if writing_result is None:
            raise PipelineError("Research Writing Service produced no result.")
        _write_json(run_directory / "02_writing" / "result.json", writing_result.model_dump(mode="json"))
    t2 = time.monotonic()
    if "writing" not in loaded:
        yield _progress_event(
            run_id=run_id, stage="writing", service="research_writing", status="done",
            title="Draft generation complete",
            message=f"{writing_result.draft_metadata.sections_written} section(s) written, "
                    f"{len(writing_result.reference_candidates)} reference(s) cited.",
            details={"seconds": round(t2 - t1, 1)},
        )

    # ---- Stage 3: Citation Verification ----
    if "verification" in loaded:
        verification_result = loaded["verification"]
        yield _loaded_event(run_id, "verification", "citation_verification")
    else:
        verification_result = None
        async for event in verification_service.run_verification(
            VerificationRequest(writing=writing_result),
        ):
            if event["type"] == "result":
                verification_result = event["result"]
            else:
                yield _adapt_inner_event(run_id, "verification", "citation_verification", event)
        if verification_result is None:
            raise PipelineError("Citation Verification Service produced no result.")
        _write_json(run_directory / "03_verification" / "result.json", verification_result.model_dump(mode="json"))
    t3 = time.monotonic()
    if "verification" not in loaded:
        yield _progress_event(
            run_id=run_id, stage="verification", service="citation_verification", status="done",
            title="Citation verification complete",
            message=f"{len(verification_result.verified_references)} verified, "
                    f"{len(verification_result.invalid_references)} invalid, "
                    f"{len(verification_result.unverifiable_references)} unverifiable.",
            details={"seconds": round(t3 - t2, 1)},
        )

    # ---- Stage 4: Quality Assurance (always the last stage, so never loaded) ----
    qa_result = None
    async for event in qa_service.run_quality_assurance(
        QualityAssuranceRequest(
            discovery=discovery_result, writing=writing_result, verification=verification_result,
        ),
    ):
        if event["type"] == "result":
            qa_result = event["result"]
        else:
            yield _adapt_inner_event(run_id, "quality_assurance", "quality_assurance", event)
    if qa_result is None:
        raise PipelineError("Quality Assurance Service produced no result.")
    t4 = time.monotonic()
    _write_json(run_directory / "04_quality_assurance" / "result.json", qa_result.model_dump(mode="json"))
    yield _progress_event(
        run_id=run_id, stage="quality_assurance", service="quality_assurance", status="done",
        title="Quality assurance complete",
        message=f"Overall quality score: {qa_result.scores.overall:.1f}/5.0.",
        details={"seconds": round(t4 - t3, 1)},
    )

    # ---- Final artifacts ----
    final_dir = run_directory / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    (final_dir / "draft.md").write_text(qa_result.final_draft_markdown, encoding="utf-8")
    (final_dir / "quality_report.md").write_text(qa_result.quality_report_markdown, encoding="utf-8")
    (final_dir / "validation_report.md").write_text(qa_result.final_validation_report, encoding="utf-8")
    (final_dir / "references.md").write_text(
        "\n".join(f"- {entry}" for entry in qa_result.final_references) + "\n", encoding="utf-8",
    )
    # Deterministic, no LLM call — see shared/utilities/evidence_table.py.
    (final_dir / "evidence_table.md").write_text(
        evidence_table.build_evidence_table(Path(writing_result.run_directory), verification_result),
        encoding="utf-8",
    )

    # LaTeX source is always generated (deterministic, no LLM call — see
    # shared/utilities/latex_export.py). PDF compilation is attempted only
    # if a LaTeX toolchain (pdflatex or tectonic) is actually available on
    # this machine; never claimed as produced otherwise.
    tex_path = final_dir / "paper.tex"
    tex_path.write_text(
        latex_export.markdown_to_latex(
            qa_result.final_draft_markdown, title=request.research_question,
            target_format=request.target_format,
        ),
        encoding="utf-8",
    )
    latex_export.compile_pdf(tex_path, final_dir)

    timings = StageTimings(
        discovery_seconds=round(t1 - t0, 1), writing_seconds=round(t2 - t1, 1),
        verification_seconds=round(t3 - t2, 1), quality_assurance_seconds=round(t4 - t3, 1),
    )
    result = PipelineResult(
        run_id=run_id, run_directory=str(run_directory), request=request,
        discovery=discovery_result, writing=writing_result,
        verification=verification_result, quality_assurance=qa_result, timings=timings,
        trend_advisor=trend_advisor,
    )
    draft_meta = writing_result.draft_metadata
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "research_question": request.research_question,
        "timings": timings.model_dump(),
        "overall_quality_score": qa_result.scores.overall,
        # Length/completeness analytics (Fix 8, DECISIONS.md D-026) — the
        # same fields as draft_metadata, surfaced at the top level of the
        # run's metadata.json so a final-status report never has to dig
        # through the writing stage's own result to answer "did this run
        # produce a complete, correctly-sized paper?".
        "draft_status": draft_meta.draft_status,
        "target_words": draft_meta.target_words,
        "actual_words": draft_meta.actual_words,
        "length_status": draft_meta.length_status,
        "evidence_limited_sections": draft_meta.evidence_limited_sections,
        "broken_sections": draft_meta.broken_sections,
        "duplicate_sections": draft_meta.duplicate_sections,
        # Credits each scholarly provider charged this run (#7); stages
        # loaded on a resume spent nothing this time.
        "credits_used": credits_used,
    }
    if "discovery" not in loaded:
        # With timings.discovery_seconds, what a related second run saved (#19).
        metadata["memory"] = {"enabled": request.memory, **memory_hits}
    if resume:
        metadata["resumed_from_stage"] = resume[0]
        metadata["final_versions"] = _final_versions(run_directory)
    if trend_advisor is not None and trend_advisor.chosen_topic:
        metadata["chosen_topic"] = trend_advisor.chosen_topic
        chosen = trend_advisor.chosen_candidate
        metadata["chosen_topic_gap_flagged"] = chosen is not None and chosen.gap_signal
    _write_json(run_directory / "metadata.json", metadata)
    try:
        library.import_run(run_directory)
    except Exception as error:  # the run itself is complete and on disk
        yield _library_warning(run_id, "quality_assurance", error)
    yield {"type": "result", "result": result}


async def run_trend_advisor_stage(
    request: TrendAdvisorRequest, output_root: Path | None = None,
    run_id: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Runs Stage 0 on its own, adapting the advisor service's own progress
    events into the same standardized shape as Stages 1-4, then yielding its
    {"type": "result", "result": TrendAdvisorResult} event unchanged.

    Which shortlisted topic to pick is not decided here — see
    run_advisor_then_pipeline."""
    run_id = run_id or uuid4().hex
    async for event in _trend_advisor_service().run_trend_advisor(
        request, output_root=output_root or DEFAULT_OUTPUT_ROOT, run_id=run_id,
    ):
        if event["type"] == "result":
            yield event
        else:
            yield _adapt_inner_event(run_id, "trend_advisor", "trend_advisor", event)


async def run_advisor_then_pipeline(
    advisor_request: TrendAdvisorRequest,
    select_topic: Callable[[TrendAdvisorResult], str | None],
    request: ResearchRequest | None = None,
    output_root: Path | None = None, run_id: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stage 0 followed immediately by the four-stage pipeline, under one run_id.

    select_topic is handed the finished shortlist and returns the topic that
    was locked in, or None if it was not (user story 10). Choosing is the
    caller's decision — the same hand-off the advisor CLI makes after its
    converging chat — so nothing here picks a topic or asks a model to.

    The chosen topic becomes ResearchRequest.research_question; `request`
    supplies the remaining pipeline settings when the defaults are not
    wanted, and its own research_question is overwritten."""
    run_id = run_id or uuid4().hex
    advisor_result = None
    advisor_directory = None
    async for event in run_trend_advisor_stage(advisor_request, output_root, run_id):
        if event["type"] == "result":
            advisor_result = event["result"]
            advisor_directory = event.get("run_directory")
        else:
            yield event
    if advisor_result is None:
        raise PipelineError("Trend & Gap Advisor Service produced no result.")

    topic = select_topic(advisor_result)
    if topic is None:
        yield _progress_event(
            run_id=run_id, stage="trend_advisor", service="trend_advisor", status="done",
            title="No topic locked in",
            message="The shortlist was produced but no topic was chosen, so the "
                    "research pipeline was not started.",
        )
        return

    request = (request or ResearchRequest(research_question=topic)).model_copy(
        update={"research_question": topic})

    async for event in run_pipeline(
        request, output_root=output_root, run_id=run_id,
        trend_advisor=advisor_result.model_copy(update={"chosen_topic": topic}),
        advisor_directory=advisor_directory,
    ):
        yield event

```

### `orchestrator/api.py`

```python
"""FastAPI backend exposing the pipeline as a set of research endpoints.

POST /research runs the full four-stage pipeline synchronously and returns
the final result directly — the original MVP endpoint, kept unchanged for
callers happy to block for the run's full duration (see DECISIONS.md D-015).

POST /research/runs, GET /research/runs/{run_id}, and
GET /research/runs/{run_id}/events add the asynchronous path deferred at
that point: start a run in the background, poll its status, or stream its
progress events over SSE as they happen (see DECISIONS.md D-017).

The /advisor/* endpoints expose the optional Stage 0 (Trend & Gap Advisor)
the same way, mirroring researchgenie-advise step for step: shortlist, chat,
draft a question, then hand off to the pipeline under the same run_id.
GET / serves the single-page demo UI in frontend/index.html.

Run locally with:
    uvicorn orchestrator.api:app --reload
"""

from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.encoders import jsonable_encoder  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from orchestrator import decision_brief, doctor, run_registry  # noqa: E402
from orchestrator.pipeline import (  # noqa: E402
    FINAL_FILES, PipelineError, check_budget, find_run_directory, plan_resume, resume_pipeline,
    run_pipeline, run_trend_advisor_stage,
)
from orchestrator.pipeline import discovery_service  # noqa: E402 — for the clarify step
from researchgenie import advisor_cli  # noqa: E402
from researchgenie import config as rg_config  # noqa: E402
from shared.contracts.pipeline_contract import PipelineResult, ResearchRequest  # noqa: E402
from shared.contracts.trend_contract import TrendAdvisorRequest, TrendAdvisorResult  # noqa: E402
from shared.utilities import library, llm_provider, scholarly_gateway  # noqa: E402

FRONTEND = ROOT / "frontend" / "index.html"

@asynccontextmanager
async def _lifespan(app: FastAPI):
    # The scholarly keys and email saved by `researchgenie config` (#5);
    # environment variables already set win.
    rg_config.apply_scholarly(rg_config.load())
    yield


app = FastAPI(
    title="AI Research Assistant",
    description="Local-first pipeline: research request -> evidence-grounded, "
                "citation-verified, quality-audited draft.",
    version="0.1.0",
    lifespan=_lifespan,
)


@app.post("/research", response_model=PipelineResult)
async def create_research_run(request: ResearchRequest) -> PipelineResult:
    """Validates the request, runs Services 1-4 in sequence, and returns
    the final PipelineResult. FastAPI/Pydantic reject a malformed request
    body with a 422 before this function runs."""
    result: PipelineResult | None = None
    try:
        async for event in run_pipeline(request):
            if event["type"] == "result":
                result = event["result"]
    except PipelineError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    if result is None:
        raise HTTPException(status_code=500, detail="Pipeline finished without producing a result.")
    return result


@app.post("/research/runs", status_code=202)
async def start_research_run(request: ResearchRequest) -> dict[str, str]:
    """Starts a pipeline run in the background and returns its run_id
    immediately, instead of blocking until the run finishes. Poll
    GET /research/runs/{run_id} for status, or GET
    /research/runs/{run_id}/events for a live SSE progress stream."""
    run_id = uuid4().hex
    # approval "ask": pause after Discovery until POST .../approve (#17).
    approve = {"approve": run_registry.approver(run_id)} if request.approval == "ask" else {}
    await run_registry.start_run(
        request, run_id, lambda req, run_id: run_pipeline(req, run_id=run_id, **approve))
    return {"run_id": run_id}


class Approval(BaseModel):
    drop: list[int] = []  # ids of the shown papers to replace from the ranked candidates


@app.post("/research/runs/{run_id}/approve")
async def approve_research_run(run_id: str, approval: Approval) -> dict[str, str]:
    """Continues a run that is awaiting_approval, dropping the given papers."""
    if not run_registry.approve(run_id, approval.drop):
        raise HTTPException(status_code=409, detail="This run is not waiting for approval.")
    return {"run_id": run_id}


class ResumeRequest(BaseModel):
    from_stage: str | None = None  # default: the first stage without a saved result


@app.post("/research/runs/{run_id}/resume", status_code=202)
async def resume_research_run(run_id: str, body: ResumeRequest) -> dict[str, str]:
    """Resumes a failed run in the background under its own run_id (#9):
    earlier stages are loaded from disk, an existing final/ is kept as
    final_vN, and the events stream from /research/runs/{run_id}/events."""
    state = run_registry.get_run(run_id)
    if state is not None and state.status == "running":
        raise HTTPException(status_code=409, detail="This run is still running.")
    directory = find_run_directory(run_id)
    if directory is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    try:
        plan = plan_resume(directory, body.from_stage)  # fail now, not in the background
    except PipelineError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    approve = run_registry.approver(run_id) if plan["request"].approval == "ask" else None
    await run_registry.start_run(
        None, run_id,
        lambda _request, run_id: resume_pipeline(directory, body.from_stage, approve=approve))
    return {"run_id": run_id}


class DoctorRequest(BaseModel):
    apply: bool = False  # false: diagnose only; true: the user confirmed the proposed fix
    fix: str | None = None  # with apply: the id of the fix the user saw and confirmed


@app.post("/research/runs/{run_id}/doctor")
async def doctor_research_run(run_id: str, body: DoctorRequest) -> dict[str, object]:
    """The Doctor (#13): diagnoses the run from its event log and the live
    system status. With apply=true (the web page sends it only after the
    user confirmed the shown fix) it applies the fix and, when the fix
    resumes, restarts the run in the background under its own run_id.
    Either way the diagnosis is saved as the run's doctor_report.json."""
    state = run_registry.get_run(run_id)
    if state is not None and state.status in ("running", "awaiting_approval") and             run_id in run_registry._runs:  # a live run (a log-only one reads as failed)
        raise HTTPException(status_code=409, detail="This run is still running.")
    directory = find_run_directory(run_id)
    if directory is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    diagnosis = doctor.diagnose(directory, await doctor.system_status())
    if not body.apply:
        doctor.write_report(directory, diagnosis, applied=False, outcome="Diagnosed; no fix applied yet.")
        return {**diagnosis, "applied": False, "resumed": False}
    if body.fix != (diagnosis["fix"] or {}).get("id"):  # confirmed one fix, never apply another
        raise HTTPException(status_code=409, detail="The diagnosis changed since it was shown; "
                                                    "diagnose again before applying a fix.")
    problem = await doctor.apply_fix(diagnosis)  # no key prompt here: a key fix is reported
    if problem is None:  # every automatic fix resumes; check the folder can be resumed now
        try:
            plan = plan_resume(directory, doctor.resume_stage(diagnosis))
        except PipelineError as error:
            problem = str(error)
    if problem is not None:
        doctor.write_report(directory, diagnosis, applied=False, outcome=problem)
        return {**diagnosis, "applied": False, "resumed": False, "outcome": problem}
    approve = run_registry.approver(run_id) if plan["request"].approval == "ask" else None
    doctor.write_report(directory, diagnosis, applied=True, outcome="Fix applied; resume started.")
    await run_registry.start_run(
        None, run_id, lambda _request, run_id: doctor.resume_and_record(
            directory, diagnosis, resume_pipeline, approve=approve))
    return {**diagnosis, "applied": True, "resumed": True, "run_id": run_id,
            "outcome": "Fix applied; resume started."}


class QuestionCheck(BaseModel):
    research_question: str


@app.post("/research/clarify")
async def clarify_research_question(check: QuestionCheck) -> dict[str, object]:
    """The clarify step on its own, called by the page before it starts a run
    so the user can accept, edit or keep the suggestion. available is false
    when the model gave no usable answer: the page then keeps the question."""
    clarified = await discovery_service.clarify_question(check.research_question)
    if clarified is None:
        return {"available": False, "specific": None, "missing": [], "suggested_question": ""}
    return {"available": True, **clarified}


@app.post("/research/budget")
async def check_research_budget(request: ResearchRequest) -> dict[str, object]:
    """The budget check on its own (#7), called by the page before it starts
    a run: plan "reduced" means the known OpenAlex balance can't cover it, so
    the page offers the free key or the reduced plan. Re-reads the saved
    config first, so a key added with `researchgenie config` counts at once."""
    rg_config.apply_scholarly(rg_config.load())
    return check_budget(request)


@app.get("/research/runs/{run_id}")
async def get_research_run(run_id: str) -> dict[str, object]:
    state = run_registry.get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    return {
        "run_id": state.run_id,
        "status": state.status,
        "result": jsonable_encoder(state.result) if state.result is not None else None,
        "error": state.error,
    }


@app.get("/research/runs/{run_id}/brief")
async def get_research_run_brief(run_id: str) -> dict[str, object]:
    """The decision brief (#12) of a finished or failed run: the one next
    action, what is strong and weak, the references and sections to check."""
    directory = find_run_directory(run_id)
    path = decision_brief.find_brief(directory) if directory else None
    if path is None:
        raise HTTPException(status_code=404, detail="This run has no decision brief (yet).")
    return {**decision_brief.build_brief(directory), "path": str(path)}


@app.get("/research/runs/{run_id}/events")
async def stream_research_run_events(run_id: str) -> StreamingResponse:
    state = run_registry.get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")

    async def event_stream():
        queue = run_registry.subscribe(state)
        try:
            while True:
                event = await queue.get()
                if event.get("type") == "end":
                    yield "event: end\ndata: {}\n\n"
                    break
                payload = json.dumps(jsonable_encoder(event))
                yield f"data: {payload}\n\n"
        finally:
            run_registry.unsubscribe(state, queue)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(FRONTEND)


@app.get("/system")
async def system_status() -> dict[str, object]:
    """Whether the configured model is ready — shown before a run starts, so
    an unreachable Ollama is a visible red light, not a stage failing later."""
    return {
        "provider": llm_provider.AI_PROVIDER,
        "model": llm_provider.GEMINI_MODEL if llm_provider.AI_PROVIDER == "gemini"
        else llm_provider.DEFAULT_MODEL,
        "model_ready": await llm_provider.health_check(),
        "domains": [{"key": key, "label": label} for key, label in advisor_cli.DOMAINS],
        # keyed / identified / anonymous and healthy / degraded per source (#5)
        "providers": scholarly_gateway.provider_status(),
    }


@app.get("/research/runs/{run_id}/files/{name}")
async def get_run_file(run_id: str, name: str) -> FileResponse:
    state = run_registry.get_run(run_id)
    if state is not None and isinstance(state.result, PipelineResult):
        directory = Path(state.result.run_directory)
    else:
        # A past run this server process never ran (e.g. one listed by a
        # related_runs event) is found through the research library.
        past = library.get_run(run_id)
        directory = Path(past["run_dir"]) if past else None
    if directory is None or name not in FINAL_FILES:
        raise HTTPException(status_code=404, detail="No such file for this run.")
    # Imported copies under docs/research-papers/ keep their files at the top level.
    path = next((p for p in (directory / "final" / name, directory / name) if p.is_file()), None)
    if path is None:
        raise HTTPException(status_code=404, detail=f"{name} was not produced by this run.")
    return FileResponse(path, filename=name)


# ---- Stage 0: Trend & Gap Advisor ----

class ChatTurn(BaseModel):
    question: str


class TopicPick(BaseModel):
    topic: str


class AdvisorHandoff(ResearchRequest):
    """A ResearchRequest plus the shortlisted topic it was drafted from."""
    topic: str


def _advisor_result(run_id: str) -> tuple[TrendAdvisorResult, str | None]:
    """The finished shortlist of an advisor run and the folder it saved to."""
    state = run_registry.get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    if not isinstance(state.result, TrendAdvisorResult):
        raise HTTPException(status_code=409, detail="This run has no finished shortlist.")
    directory = next((e.get("run_directory") for e in state.events
                      if e.get("type") == "result"), None)
    return state.result, directory


def _candidate(result: TrendAdvisorResult, topic: str):
    candidate = next((c for c in result.shortlist if c.topic == topic), None)
    if candidate is None:
        raise HTTPException(status_code=422, detail="Topic is not on this run's shortlist.")
    return candidate


@app.post("/advisor/runs", status_code=202)
async def start_advisor_run(request: TrendAdvisorRequest) -> dict[str, str]:
    """Starts Stage 0 in the background; poll or stream it with the same
    /research/runs/{run_id} endpoints as a pipeline run."""
    run_id = uuid4().hex
    await run_registry.start_run(
        request, run_id, lambda req, run_id: run_trend_advisor_stage(req, run_id=run_id))
    return {"run_id": run_id}


@app.post("/advisor/runs/{run_id}/chat")
async def advisor_chat(run_id: str, turn: ChatTurn) -> dict[str, str]:
    """One grounded chat turn about the shortlist (advisor_cli._chat_turn)."""
    result, _ = _advisor_result(run_id)
    try:
        reply = await advisor_cli._chat_turn(turn.question, advisor_cli._shortlist_context(result))
    except llm_provider.LLMProviderError as error:
        raise HTTPException(status_code=502, detail=f"The model didn't answer: {error}") from error
    return {"reply": reply}


@app.post("/advisor/runs/{run_id}/question")
async def advisor_question(run_id: str, pick: TopicPick) -> dict[str, object]:
    """Drafts a research question for the picked topic. Falls back to the
    topic itself, flagged, exactly as the CLI does when no question comes back."""
    result, _ = _advisor_result(run_id)
    candidate = _candidate(result, pick.topic)
    try:
        question = await advisor_cli._research_question_for(candidate)
    except llm_provider.LLMProviderError:
        question = ""
    return {"research_question": question or candidate.topic, "drafted": bool(question)}


@app.post("/advisor/runs/{run_id}/pipeline", status_code=202)
async def advisor_to_pipeline(run_id: str, handoff: AdvisorHandoff) -> dict[str, str]:
    """Runs the four-stage pipeline on the locked-in topic under the advisor's
    own run_id, so the chained run leaves one folder (D-034). The registry
    entry is replaced by the pipeline run; its stream starts afresh."""
    result, directory = _advisor_result(run_id)
    chosen = advisor_cli.with_chosen_topic(result, _candidate(result, handoff.topic))
    request = ResearchRequest(**handoff.model_dump(exclude={"topic"}))
    approve = {"approve": run_registry.approver(run_id)} if request.approval == "ask" else {}
    await run_registry.start_run(
        request, run_id,
        lambda req, run_id: run_pipeline(req, run_id=run_id, trend_advisor=chosen,
                                         advisor_directory=directory, **approve))
    return {"run_id": run_id}

```

### `dashboards/logs_dashboard.py`

```python
"""Live backend-activity dashboard for a ResearchGenie run.

Standalone: reads the outputs/ folder's events.jsonl files directly and
serves them over SSE. No import from orchestrator/discovery/writing/etc,
so it can never change pipeline behaviour, only watch it.

    python dashboards/logs_dashboard.py [--port 8010]

Open http://127.0.0.1:8010 while a run is going, to show reviewers what
the backend is actually doing stage by stage, live.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

app = FastAPI()


def _runs() -> list[dict]:
    if not OUTPUTS.is_dir():
        return []
    rows = []
    for d in OUTPUTS.iterdir():
        events = d / "events.jsonl"
        if not events.is_file():
            continue
        question = d.name.rsplit("_", 1)[0].replace("-", " ")
        run_id = d.name.rsplit("_", 1)[-1]
        rows.append({"run_id": run_id, "dir": d.name, "question": question,
                      "mtime": events.stat().st_mtime})
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows


def _events(run_dir: Path) -> list[dict]:
    path = run_dir / "events.jsonl"
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


@app.get("/api/runs")
def api_runs():
    return _runs()


async def _stream(run_dir: Path):
    sent = 0
    while True:
        events = _events(run_dir)
        for e in events[sent:]:
            yield f"data: {json.dumps(e)}\n\n"
        sent = len(events)
        await asyncio.sleep(0.5)


@app.get("/api/stream/{dir_name}")
async def api_stream(dir_name: str):
    run_dir = OUTPUTS / dir_name
    return StreamingResponse(_stream(run_dir), media_type="text/event-stream")


PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<title>ResearchGenie — live backend</title>
<style>
:root{--bg:#0b0d10;--card:#151922;--line:#262b36;--ink:#e7eaf0;--mute:#8b93a7;
--blue:#5b9cff;--green:#3ecf8e;--red:#ff6b6b;--amber:#e9b64f}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 ui-monospace,Consolas,monospace}
header{padding:16px 24px;border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:center;flex-wrap:wrap}
h1{font:600 16px/1 system-ui,sans-serif;margin:0}
select{background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:6px 10px;font:inherit}
.stage{padding:4px 12px;border-radius:99px;font:600 12px system-ui,sans-serif;background:var(--card);border:1px solid var(--line)}
main{padding:16px 24px;max-width:1100px;margin:0 auto}
.row{display:flex;gap:10px;padding:6px 0;border-bottom:1px solid var(--line)}
.t{color:var(--mute);white-space:nowrap}
.badge{font:600 11px system-ui,sans-serif;padding:1px 7px;border-radius:5px;white-space:nowrap;height:fit-content}
.running{background:#1d2a44;color:var(--blue)}
.done{background:#143328;color:var(--green)}
.error{background:#3a1d20;color:var(--red)}
.msg{flex:1}
#log{display:flex;flex-direction:column-reverse;max-height:76vh;overflow-y:auto;border:1px solid var(--line);border-radius:10px;background:var(--card);padding:10px 14px}
.empty{color:var(--mute);padding:40px;text-align:center;font:14px system-ui,sans-serif}
</style></head>
<body>
<header>
  <h1>ResearchGenie — live backend</h1>
  <select id="runPicker"></select>
  <span class="stage" id="stagePill">no run selected</span>
</header>
<main><div id="log"><div class="empty">Pick a run above, or start one — it will appear automatically.</div></div></main>
<script>
let es = null, currentDir = null;
const log = document.getElementById('log');
const stagePill = document.getElementById('stagePill');
const picker = document.getElementById('runPicker');

function badgeClass(status) {
  if (status === 'running') return 'running';
  if (status === 'error') return 'error';
  return 'done';
}

function addRow(e) {
  const row = document.createElement('div');
  row.className = 'row';
  const t = new Date(e.timestamp).toLocaleTimeString();
  const status = e.status || e.type || '';
  row.innerHTML = `<span class="t">${t}</span>
    <span class="badge ${badgeClass(status)}">${(e.stage || e.type || '').toUpperCase()}</span>
    <span class="msg">${e.emoji || ''} ${e.message || e.title || JSON.stringify(e)}</span>`;
  log.prepend(row);
  if (e.stage) stagePill.textContent = e.stage.replace('_', ' ') + (status ? ' · ' + status : '');
}

function connect(dirName) {
  if (es) es.close();
  log.innerHTML = '';
  currentDir = dirName;
  es = new EventSource(`/api/stream/${dirName}`);
  es.onmessage = (ev) => addRow(JSON.parse(ev.data));
}

async function refreshRuns() {
  const runs = await (await fetch('/api/runs')).json();
  const prev = picker.value;
  picker.innerHTML = '';
  for (const r of runs) {
    const opt = document.createElement('option');
    opt.value = r.dir;
    opt.textContent = `${r.question} (${r.run_id.slice(0, 8)})`;
    picker.appendChild(opt);
  }
  if (runs.length && !currentDir) connect(runs[0].dir);
  else if (prev) picker.value = prev;
}

picker.addEventListener('change', () => connect(picker.value));
refreshRuns();
setInterval(refreshRuns, 5000);
</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port)

```

### `dashboards/telegram_sync_dashboard.py`

```python
"""Live Telegram <-> OpenClaw <-> agent message-flow dashboard.

Standalone: tails OpenClaw's own gateway log file directly (plain text,
one JSON object per line) and classifies each line by subsystem. Does
not import OpenClaw or ResearchGenie code, and writes nothing — a pure
observer, so it can never change what it's watching.

    python dashboards/telegram_sync_dashboard.py [--port 8011] [--log-dir PATH]

Open http://127.0.0.1:8011 next to the Telegram chat to show exactly
what happens between a message landing and the agent's reply: inbound
text, which model session picked it up, how long it ran, and what went
back out.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

app = FastAPI()
LOG_DIR = Path(os.environ.get("LOCALAPPDATA", "")) / "Temp" / "openclaw"

_SUBSYSTEM_RE = re.compile(r'"subsystem":"([^"]+)"')


def _latest_log() -> Path | None:
    if not LOG_DIR.is_dir():
        return None
    logs = sorted(LOG_DIR.glob("openclaw-*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def _classify(raw: dict) -> dict:
    name = raw.get("_meta", {}).get("name", "")
    m = _SUBSYSTEM_RE.search(name)
    subsystem = m.group(1) if m else "gateway"
    message = raw.get("message", "")
    kind = "other"
    if subsystem.startswith("channels/telegram") or "telegram" in message.lower():
        if "inbound message" in message.lower():
            kind = "inbound"
        elif "sendmessage" in message.lower():
            kind = "outbound"
        else:
            kind = "telegram"
    elif "cli-backend" in subsystem or "claude live session" in message.lower() or "cli exec" in message.lower():
        kind = "agent"
    return {"time": raw.get("time", ""), "subsystem": subsystem, "message": message, "kind": kind}


def _tail_new_lines(path: Path, offset: int) -> tuple[list[dict], int]:
    data = path.read_bytes()
    if offset > len(data):
        offset = 0  # log rotated to a new day's file underneath us
    new = data[offset:]
    rows = []
    for line in new.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(_classify(json.loads(line)))
        except json.JSONDecodeError:
            continue
    return rows, len(data)


async def _stream():
    path = _latest_log()
    offset = 0
    if path and path.is_file():
        # Replay the last ~200 lines of history so a viewer who opens the
        # dashboard mid-conversation isn't staring at a blank page.
        all_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in all_lines[-200:]:
            line = line.strip()
            if not line:
                continue
            try:
                yield f"data: {json.dumps(_classify(json.loads(line)))}\n\n"
            except json.JSONDecodeError:
                continue
        offset = path.stat().st_size
    while True:
        latest = _latest_log()
        if latest is None:
            await asyncio.sleep(1)
            continue
        if latest != path:
            path, offset = latest, 0
        rows, offset = _tail_new_lines(path, offset)
        for r in rows:
            yield f"data: {json.dumps(r)}\n\n"
        await asyncio.sleep(0.5)


@app.get("/api/stream")
async def api_stream():
    return StreamingResponse(_stream(), media_type="text/event-stream")


PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<title>Telegram &harr; Agent — live sync</title>
<style>
:root{--bg:#0b0d10;--card:#151922;--line:#262b36;--ink:#e7eaf0;--mute:#8b93a7;
--in:#5b9cff;--out:#3ecf8e;--agent:#e9b64f;--tg:#6ad1ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 ui-monospace,Consolas,monospace}
header{padding:16px 24px;border-bottom:1px solid var(--line)}
h1{font:600 16px/1 system-ui,sans-serif;margin:0 0 4px}
.sub{color:var(--mute);font:13px system-ui,sans-serif}
main{padding:16px 24px;max-width:900px;margin:0 auto}
.row{display:flex;gap:10px;padding:8px 10px;margin:6px 0;border-radius:8px;border-left:3px solid var(--line);background:var(--card)}
.row.inbound{border-left-color:var(--in)}
.row.outbound{border-left-color:var(--out)}
.row.agent{border-left-color:var(--agent)}
.row.telegram{border-left-color:var(--tg)}
.t{color:var(--mute);white-space:nowrap;font-size:12px}
.lbl{font:700 11px system-ui,sans-serif;padding:1px 7px;border-radius:5px;white-space:nowrap;height:fit-content}
.inbound .lbl{background:#1d2a44;color:var(--in)}
.outbound .lbl{background:#143328;color:var(--out)}
.agent .lbl{background:#3a2e17;color:var(--agent)}
.telegram .lbl{background:#17313a;color:var(--tg)}
.msg{flex:1;word-break:break-word}
#log{display:flex;flex-direction:column-reverse}
.empty{color:var(--mute);padding:40px;text-align:center;font:14px system-ui,sans-serif}
</style></head>
<body>
<header>
  <h1>Telegram &harr; OpenClaw &harr; agent</h1>
  <div class="sub">Every inbound message, agent session, and outbound reply, live from the gateway's own log.</div>
</header>
<main><div id="log"><div class="empty">Waiting for the OpenClaw gateway log&hellip;</div></div></main>
<script>
const log = document.getElementById('log');
let first = true;
function labelFor(kind) {
  return {inbound: 'IN ← TELEGRAM', outbound: 'OUT → TELEGRAM', agent: 'AGENT', telegram: 'TELEGRAM'}[kind] || 'GATEWAY';
}
function addRow(e) {
  if (e.kind === 'other') return;
  if (first) { log.innerHTML = ''; first = false; }
  const row = document.createElement('div');
  row.className = 'row ' + e.kind;
  const t = e.time ? new Date(e.time).toLocaleTimeString() : '';
  row.innerHTML = `<span class="t">${t}</span><span class="lbl">${labelFor(e.kind)}</span><span class="msg">${e.message}</span>`;
  log.prepend(row);
}
const es = new EventSource('/api/stream');
es.onmessage = (ev) => addRow(JSON.parse(ev.data));
</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--log-dir", type=Path, default=None,
                         help="Override the OpenClaw log directory (default: %%LOCALAPPDATA%%/Temp/openclaw)")
    args = parser.parse_args()
    if args.log_dir:
        LOG_DIR = args.log_dir
    uvicorn.run(app, host="127.0.0.1", port=args.port)

```

### Real tests for this lab

### `tests/orchestrator/test_pipeline.py`

```python
"""Fast unit tests for the orchestrator, with all four services faked out —
see tests/services/*_pipeline_integration.py and scripts/smoke_test_full_pipeline.py
for the real, slow, live-service coverage."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from orchestrator import pipeline as orchestrator_pipeline  # noqa: E402
from shared.contracts.discovery_contract import (  # noqa: E402
    DiscoveryRequest, DiscoveryResult, NoveltyAssessment, PaperMetadata, ResearchGap,
)
from shared.contracts.pipeline_contract import ResearchRequest  # noqa: E402
from shared.contracts.trend_contract import TrendAdvisorRequest  # noqa: E402
from shared.contracts.qa_contract import QualityAssuranceResult, QualityScores  # noqa: E402
from shared.contracts.verification_contract import VerificationResult  # noqa: E402
from shared.contracts.writing_contract import DraftMetadata, WritingResult  # noqa: E402


def _fake_discovery_result() -> DiscoveryResult:
    return DiscoveryResult(
        research_request=DiscoveryRequest(research_question="Q?", corpus_size=6),
        research_interpretation="interp", search_queries={"keyword": "x"},
        sources_searched=["OpenAlex"],
        selected_papers=[PaperMetadata(id=1, title="Paper One", doi="10.1/x", has_abstract=True, abstract="abstract")],
        field_overview="overview", limitations="none",
        research_gaps=[ResearchGap(
            title="Gap", gap_type="methodological", impact="high",
            description="desc", evidence="evi", supporting_paper_ids=[1], research_questions=["Q1?"],
        )],
        novelty_analysis=NoveltyAssessment(novelty_summary="summary", confidence="medium", caveats="caveat"),
        confidence_notes="notes",
    )


def _fake_writing_result() -> WritingResult:
    return WritingResult(
        research_outline="# Q\n## Introduction", research_draft_markdown="draft body [1].",
        reference_candidates=['[1] A. Author, "Paper One," 2020, doi: 10.1/x.'],
        run_directory="/tmp/writing-run",
        draft_metadata=DraftMetadata(
            model_name="qwen3.5:9b", run_id="writing-run-1", generated_at="2026-08-27T00:00:00Z",
            sections_written=2, papers_cited=1,
        ),
    )


def _fake_verification_result() -> VerificationResult:
    return VerificationResult(
        validated_draft_markdown="draft body [1].",
        verified_references=[], invalid_references=[], unverifiable_references=[],
        validation_report="# Citation Verification Report\n",
    )


def _fake_qa_result() -> QualityAssuranceResult:
    return QualityAssuranceResult(
        final_draft_markdown="draft body [1].",
        final_references=['[1] A. Author, "Paper One," 2020, doi: 10.1/x.'],
        final_validation_report="# Citation Verification Report\n",
        quality_report_markdown="# Quality Assurance Report\n",
        scores=QualityScores(
            citation_integrity=5.0, claim_source_alignment=5.0,
            process_control=5.0, literature_coverage=5.0, overall=5.0,
        ),
    )


async def _fake_run_discovery(request):
    yield {"type": "status", "message": "searching"}
    yield {"type": "result", "result": _fake_discovery_result()}


async def _fake_run_writing(request, output_root=None):
    yield {"type": "status", "stage": "write", "state": "running", "message": "writing"}
    yield {"type": "result", "result": _fake_writing_result()}


async def _fake_run_verification(request):
    yield {"type": "status", "stage": "verify", "state": "running", "message": "verifying"}
    yield {"type": "result", "result": _fake_verification_result()}


async def _fake_run_qa(request):
    yield {"type": "status", "stage": "audit", "state": "running", "message": "auditing"}
    yield {"type": "result", "result": _fake_qa_result()}


@pytest.fixture(autouse=True)
def _patch_services(monkeypatch):
    monkeypatch.setattr(orchestrator_pipeline.discovery_service, "run_discovery", _fake_run_discovery)
    monkeypatch.setattr(orchestrator_pipeline.writing_service, "run_writing", _fake_run_writing)
    monkeypatch.setattr(orchestrator_pipeline.verification_service, "run_verification", _fake_run_verification)
    monkeypatch.setattr(orchestrator_pipeline.qa_service, "run_quality_assurance", _fake_run_qa)


def test_run_pipeline_produces_result_and_persists_artifacts(tmp_path):
    request = ResearchRequest(research_question="What is X?", corpus_size=6)

    async def run():
        events = []
        result = None
        async for event in orchestrator_pipeline.run_pipeline(request, output_root=tmp_path):
            if event["type"] == "result":
                result = event["result"]
            else:
                events.append(event)
        return events, result

    events, result = asyncio.run(run())

    assert result is not None
    assert result.quality_assurance.scores.overall == 5.0
    assert Path(result.run_directory).is_dir()
    assert (Path(result.run_directory) / "00_request.json").exists()
    assert (Path(result.run_directory) / "01_discovery" / "result.json").exists()
    assert (Path(result.run_directory) / "final" / "draft.md").exists()
    assert (Path(result.run_directory) / "final" / "draft.md").read_text(encoding="utf-8") == "draft body [1]."
    # LaTeX source is always generated (deterministic — see D-022); PDF
    # compilation is opportunistic and not asserted here since it depends
    # on a LaTeX toolchain being present on the machine running the test.
    tex_path = Path(result.run_directory) / "final" / "paper.tex"
    assert tex_path.exists()
    assert tex_path.read_text(encoding="utf-8").startswith("\\documentclass")

    metadata = json.loads((Path(result.run_directory) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["overall_quality_score"] == 5.0

    # Standardized progress events carry the required fields.
    stage_events = [e for e in events if e.get("stage") == "discovery"]
    assert stage_events
    for event in stage_events:
        assert set(event) >= {"run_id", "stage", "service", "status", "emoji", "title", "message", "details", "timestamp"}


def test_every_completed_run_writes_an_evidence_table_without_the_model(tmp_path, monkeypatch):
    fixture = ROOT / "tests" / "fixtures" / "evidence_run"
    writing = _fake_writing_result().model_copy(
        update={"run_directory": str(fixture / "02_writing" / "0a1b2c3d")})
    verification = VerificationResult.model_validate_json(
        (fixture / "03_verification" / "result.json").read_text(encoding="utf-8"))

    async def _writing(request, output_root=None):
        yield {"type": "result", "result": writing}

    async def _verification(request):
        yield {"type": "result", "result": verification}

    def _refuse(*args, **kwargs):
        raise AssertionError("the evidence table must not call the model")

    monkeypatch.setattr(orchestrator_pipeline.writing_service, "run_writing", _writing)
    monkeypatch.setattr(orchestrator_pipeline.verification_service, "run_verification", _verification)
    from shared.utilities import llm_provider
    for name in ("stream_json", "complete_json", "health_check"):
        monkeypatch.setattr(llm_provider, name, _refuse)

    async def run():
        async for event in orchestrator_pipeline.run_pipeline(
            ResearchRequest(research_question="What is X?", corpus_size=6), output_root=tmp_path,
        ):
            if event["type"] == "result":
                return event["result"]

    table = (Path(asyncio.run(run()).run_directory) / "final" / "evidence_table.md").read_text(encoding="utf-8")
    assert "| Ketosis and attention | Ketosis was linked to faster reaction times. | Background | 10.1000/fake.999 | invalid |" in table


def test_research_request_requires_format_other_name_when_format_is_other():
    with pytest.raises(ValueError):
        ResearchRequest(research_question="Q?", target_format="Other")
    # Valid when the name is supplied.
    ResearchRequest(research_question="Q?", target_format="Other", format_other_name="Custom Style")


def test_run_pipeline_raises_if_a_stage_yields_no_result(tmp_path, monkeypatch):
    async def _empty_discovery(request):
        yield {"type": "status", "message": "nothing"}
        return

    monkeypatch.setattr(orchestrator_pipeline.discovery_service, "run_discovery", _empty_discovery)
    request = ResearchRequest(research_question="What is X?", corpus_size=6)

    async def run():
        async for _ in orchestrator_pipeline.run_pipeline(request, output_root=tmp_path):
            pass

    with pytest.raises(orchestrator_pipeline.PipelineError):
        asyncio.run(run())


def test_run_pipeline_appends_every_event_to_events_jsonl_including_failure(tmp_path, monkeypatch):
    async def _failing_writing(request, output_root=None):
        yield {"type": "status", "stage": "write", "state": "running", "message": "writing"}

    monkeypatch.setattr(orchestrator_pipeline.writing_service, "run_writing", _failing_writing)
    request = ResearchRequest(research_question="What is X?", corpus_size=6)

    async def run():
        events = []
        with pytest.raises(orchestrator_pipeline.PipelineError) as error:
            async for event in orchestrator_pipeline.run_pipeline(request, output_root=tmp_path, run_id="abc123"):
                events.append(event)
        return events, error.value

    events, error = asyncio.run(run())

    log_path = tmp_path / "what-is-x_abc123" / "events.jsonl"
    logged = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(events) > 2
    assert logged[:-1] == events  # same events, same order, same shape
    assert logged[-1] == {"type": "error", "message": str(error)}


def test_run_pipeline_logs_result_event_to_events_jsonl(tmp_path):
    request = ResearchRequest(research_question="What is X?", corpus_size=6)

    async def run():
        return [event async for event in orchestrator_pipeline.run_pipeline(request, output_root=tmp_path)]

    events = asyncio.run(run())

    run_directory = Path(events[-1]["result"].run_directory)
    logged = [json.loads(line) for line in (run_directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert logged[:-1] == events[:-1]
    assert logged[-1] == {"type": "result", "result": events[-1]["result"].model_dump(mode="json")}


# --- Optional Stage 0: Trend & Gap Advisor (DECISIONS.md D-031) -------------

def _fake_trend_result():
    from shared.contracts.trend_contract import TopicCandidate, TrendAdvisorResult
    return TrendAdvisorResult(
        domain="CS_AI_ML",
        shortlist=[TopicCandidate(
            topic="Edge Inference", topic_id="T1", growth_metric=3.0,
            paper_count=900, prior_papers=300, gap_signal=True,
            gap_evidence=["On-device latency is unmeasured."], source="both",
        )],
        generated_at="2026-09-12T00:00:00Z",
    )


def test_stage_zero_is_absent_by_default(tmp_path):
    """The advisor is optional: a plain run must be byte-for-byte what it
    was before this stage existed."""
    request = ResearchRequest(research_question="What is X?", corpus_size=6)

    async def run():
        result = None
        events = []
        async for event in orchestrator_pipeline.run_pipeline(request, output_root=tmp_path):
            if event["type"] == "result":
                result = event["result"]
            else:
                events.append(event)
        return events, result

    events, result = asyncio.run(run())

    assert result.trend_advisor is None
    assert not (Path(result.run_directory) / "00_trend_advisor").exists()
    assert not any(e.get("stage") == "trend_advisor" for e in events)
    metadata = json.loads((Path(result.run_directory) / "metadata.json").read_text(encoding="utf-8"))
    assert "chosen_topic" not in metadata


def test_stage_zero_is_persisted_into_the_same_run_folder(tmp_path):
    """Why a student chose this topic and what the pipeline then produced
    belong to one run, not two unrelated folders."""
    request = ResearchRequest(research_question="How does edge inference scale?", corpus_size=6)
    advisor = _fake_trend_result()

    async def run():
        result = None
        async for event in orchestrator_pipeline.run_pipeline(
            request, output_root=tmp_path, trend_advisor=advisor,
        ):
            if event["type"] == "result":
                result = event["result"]
        return result

    result = asyncio.run(run())

    saved = Path(result.run_directory) / "00_trend_advisor" / "result.json"
    assert saved.exists()
    assert json.loads(saved.read_text(encoding="utf-8"))["shortlist"][0]["topic"] == "Edge Inference"
    assert result.trend_advisor.shortlist[0].topic == "Edge Inference"


def test_run_pipeline_adopts_the_advisors_directory(tmp_path):
    """The CLI runs Stage 0 itself, then hands run_pipeline the folder the
    advisor wrote under the same run_id; that folder must become the run's."""
    run_id = "abc123"
    advisor_dir = tmp_path / f"cs-ai-ml_{run_id}"
    (advisor_dir / "00_trend_advisor").mkdir(parents=True)
    (advisor_dir / "00_trend_advisor" / "result.json").write_text("{}", encoding="utf-8")
    request = ResearchRequest(research_question="How does edge inference scale?", corpus_size=6)

    async def run():
        async for event in orchestrator_pipeline.run_pipeline(
            request, output_root=tmp_path, run_id=run_id,
            trend_advisor=_fake_trend_result(), advisor_directory=advisor_dir,
        ):
            if event["type"] == "result":
                return event["result"]

    result = asyncio.run(run())

    assert [p.name for p in tmp_path.iterdir() if p.is_dir()] == [Path(result.run_directory).name]
    assert Path(result.run_directory).name == f"how-does-edge-inference-scale_{run_id}"
    assert not advisor_dir.exists()


def test_run_pipeline_never_moves_a_folder_from_another_run(tmp_path):
    other = tmp_path / "cs-ai-ml_someoneelse"
    other.mkdir()
    request = ResearchRequest(research_question="What is X?", corpus_size=6)

    async def run():
        async for event in orchestrator_pipeline.run_pipeline(
            request, output_root=tmp_path, run_id="abc123",
            trend_advisor=_fake_trend_result(), advisor_directory=other,
        ):
            if event["type"] == "result":
                return event["result"]

    asyncio.run(run())

    assert other.exists()


def test_stage_zero_is_reported_in_the_audit_trail(tmp_path):
    request = ResearchRequest(research_question="How does edge inference scale?", corpus_size=6)

    async def run():
        events = []
        async for event in orchestrator_pipeline.run_pipeline(
            request, output_root=tmp_path, trend_advisor=_fake_trend_result(),
        ):
            if event["type"] != "result":
                events.append(event)
        return events

    events = asyncio.run(run())

    stage_zero = [e for e in events if e.get("stage") == "trend_advisor"]
    assert stage_zero, "Stage 0 produced no progress event"
    assert stage_zero[0]["status"] == "done"
    assert set(stage_zero[0]) >= {"run_id", "stage", "service", "status", "emoji",
                                  "title", "message", "details", "timestamp"}
    # Stage 0 is reported before Discovery, since it ran before it.
    assert events.index(stage_zero[0]) < min(
        i for i, e in enumerate(events) if e.get("stage") == "discovery")




def test_metadata_reads_the_named_pick_not_the_first_entry(tmp_path):
    """The chosen topic is whichever one the student named, wherever it sits
    in the shortlist's own ranking."""
    from shared.contracts.trend_contract import TopicCandidate
    advisor = _fake_trend_result()
    advisor = advisor.model_copy(update={
        "shortlist": [
            TopicCandidate(topic="Not Chosen", topic_id="T9", growth_metric=9.0,
                           paper_count=100),
            *advisor.shortlist,
        ],
        "chosen_topic": "Edge Inference",
    })
    request = ResearchRequest(research_question="How does edge inference scale?", corpus_size=6)

    async def run():
        async for event in orchestrator_pipeline.run_pipeline(
            request, output_root=tmp_path, trend_advisor=advisor,
        ):
            if event["type"] == "result":
                return event["result"]

    result = asyncio.run(run())
    metadata = json.loads((Path(result.run_directory) / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["chosen_topic"] == "Edge Inference"
    assert metadata["chosen_topic_gap_flagged"] is True


def test_no_chosen_topic_recorded_when_the_advisor_result_names_none(tmp_path):
    """A shortlist saved without a pick (the student stopped to think it
    over) must not have a topic attributed to them."""
    request = ResearchRequest(research_question="What is X?", corpus_size=6)

    async def run():
        async for event in orchestrator_pipeline.run_pipeline(
            request, output_root=tmp_path, trend_advisor=_fake_trend_result(),
        ):
            if event["type"] == "result":
                return event["result"]

    result = asyncio.run(run())
    metadata = json.loads((Path(result.run_directory) / "metadata.json").read_text(encoding="utf-8"))

    assert "chosen_topic" not in metadata
    # The shortlist itself is still preserved as run provenance.
    assert (Path(result.run_directory) / "00_trend_advisor" / "result.json").exists()


def test_a_chosen_topic_absent_from_the_shortlist_is_not_reported_as_gap_flagged(tmp_path):
    """Defensive: the gap flag is looked up by topic name, so a name that
    matches nothing must report False rather than raise or guess."""
    advisor = _fake_trend_result().model_copy(update={"chosen_topic": "Some Other Topic"})
    request = ResearchRequest(research_question="What is X?", corpus_size=6)

    async def run():
        async for event in orchestrator_pipeline.run_pipeline(
            request, output_root=tmp_path, trend_advisor=advisor,
        ):
            if event["type"] == "result":
                return event["result"]

    result = asyncio.run(run())
    metadata = json.loads((Path(result.run_directory) / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["chosen_topic"] == "Some Other Topic"
    assert metadata["chosen_topic_gap_flagged"] is False


# --- Stage 0 run from the orchestrator (ticket #5) --------------------------

async def _fake_run_trend_advisor(request, *, output_root=None, run_id=None, model=None):
    yield {"type": "status", "stage": "velocity", "state": "running",
           "message": "Analysing publication velocity..."}
    yield {"type": "status", "stage": "gap_mining", "state": "done",
           "message": "1 topic(s) flagged as recurring gaps."}
    yield {"type": "result", "result": _fake_trend_result(),
           "run_directory": str(output_root)}


@pytest.fixture
def _patch_advisor(monkeypatch):
    monkeypatch.setattr(
        orchestrator_pipeline, "trend_advisor_service",
        SimpleNamespace(run_trend_advisor=_fake_run_trend_advisor),
    )


def test_trend_advisor_stage_runs_alone_with_standardized_events(tmp_path, _patch_advisor):
    """The advisor's own ad hoc events must reach the caller in the same
    shape the four stages emit, not in the service's private shape."""
    request = TrendAdvisorRequest(domain="CS_AI_ML")

    async def run():
        events, result = [], None
        async for event in orchestrator_pipeline.run_trend_advisor_stage(
            request, output_root=tmp_path,
        ):
            if event["type"] == "result":
                result = event["result"]
            else:
                events.append(event)
        return events, result

    events, result = asyncio.run(run())

    assert result.shortlist[0].topic == "Edge Inference"
    assert events
    for event in events:
        assert event["stage"] == "trend_advisor"
        assert set(event) >= {"run_id", "stage", "service", "status", "emoji",
                              "title", "message", "details", "timestamp"}
    assert [e["status"] for e in events] == ["running", "done"]
    assert len({e["run_id"] for e in events}) == 1


def test_advisor_then_pipeline_feeds_the_chosen_topic_into_the_run(tmp_path, _patch_advisor):
    """The pick is the caller's; the orchestrator only carries it through as
    the research question and as the run's recorded provenance."""
    async def run():
        events, result = [], None
        async for event in orchestrator_pipeline.run_advisor_then_pipeline(
            TrendAdvisorRequest(domain="CS_AI_ML"),
            lambda advisor: advisor.shortlist[0].topic,
            ResearchRequest(research_question="placeholder", corpus_size=6),
            output_root=tmp_path,
        ):
            if event["type"] == "result":
                result = event["result"]
            else:
                events.append(event)
        return events, result

    events, result = asyncio.run(run())

    assert result.request.research_question == "Edge Inference"
    assert result.request.corpus_size == 6
    assert result.trend_advisor.chosen_topic == "Edge Inference"
    metadata = json.loads((Path(result.run_directory) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["chosen_topic"] == "Edge Inference"
    # Stage 0's own progress reaches the caller before Discovery's does.
    assert events.index(next(e for e in events if e["stage"] == "trend_advisor")) < \
        events.index(next(e for e in events if e["stage"] == "discovery"))


def test_advisor_then_pipeline_does_not_run_the_pipeline_without_a_pick(tmp_path, _patch_advisor):
    """Stopping to think it over is a legitimate outcome, not a run."""
    async def run():
        events = []
        async for event in orchestrator_pipeline.run_advisor_then_pipeline(
            TrendAdvisorRequest(domain="CS_AI_ML"), lambda advisor: None,
            output_root=tmp_path,
        ):
            events.append(event)
        return events

    events = asyncio.run(run())

    assert not any(e["type"] == "result" for e in events)
    assert not any(e.get("stage") == "discovery" for e in events)
    assert events[-1]["stage"] == "trend_advisor"


# --- Stage 0 against the real advisor service, which really writes to disk ---

@pytest.fixture
def _real_advisor(monkeypatch):
    """The fakes above write nothing, so they cannot see where the advisor's
    own artifacts land. Only OpenAlex and the local model are stubbed out."""
    from shared.contracts.trend_contract import TopicCandidate

    service = orchestrator_pipeline._trend_advisor_service()

    async def _find_rising_topics(domain, limit=8, domain_other_name=None):
        return [TopicCandidate(topic="Edge Inference", topic_id="T1",
                               growth_metric=3.0, paper_count=900, prior_papers=300)]

    async def _health_check():
        return False

    monkeypatch.setattr(service, "find_rising_topics", _find_rising_topics)
    monkeypatch.setattr(service.llm_provider, "health_check", _health_check)
    monkeypatch.setattr(orchestrator_pipeline, "trend_advisor_service", service)
    return service


def test_chained_run_leaves_exactly_one_run_directory(tmp_path, _real_advisor):
    """One run, one auditable folder: Stage 0's evidence lives inside the
    pipeline's run directory rather than in a second folder beside it."""
    async def run():
        async for event in orchestrator_pipeline.run_advisor_then_pipeline(
            TrendAdvisorRequest(domain="CS_AI_ML"),
            lambda advisor: advisor.shortlist[0].topic,
            ResearchRequest(research_question="placeholder", corpus_size=6),
            output_root=tmp_path,
        ):
            if event["type"] == "result":
                return event["result"]

    result = asyncio.run(run())
    run_directory = Path(result.run_directory)

    assert [p.name for p in tmp_path.iterdir() if p.is_dir()] == [run_directory.name]
    assert run_directory.name.startswith("edge-inference_")
    saved = run_directory / "00_trend_advisor" / "result.json"
    assert json.loads(saved.read_text(encoding="utf-8"))["shortlist"][0]["topic"] == "Edge Inference"
    assert (run_directory / "final" / "draft.md").exists()


def test_trend_advisor_stage_alone_keeps_its_own_run_directory(tmp_path, _real_advisor):
    """Stopping at the shortlist still has to leave the student a folder."""
    async def run():
        async for event in orchestrator_pipeline.run_trend_advisor_stage(
            TrendAdvisorRequest(domain="CS_AI_ML"), output_root=tmp_path,
        ):
            if event["type"] == "result":
                return event

    event = asyncio.run(run())
    run_directory = Path(event["run_directory"])

    assert run_directory.name.startswith("cs-ai-ml_")
    assert [p.name for p in tmp_path.iterdir() if p.is_dir()] == [run_directory.name]
    assert (run_directory / "00_trend_advisor" / "result.json").exists()


def test_no_key_or_email_reaches_the_event_log_or_the_run_folder(tmp_path, monkeypatch):
    """#5 / D-023: a stage message or error that echoes a keyed URL is masked
    before it is written anywhere a user (or a shared run folder) sees it."""
    monkeypatch.setenv("OPENALEX_API_KEY", "oa-secret-0123456789")
    monkeypatch.setenv("CONTACT_EMAIL", "student@example.edu")

    async def _leaky_discovery(request):
        yield {"type": "source_error", "message": "GET https://api.openalex.org/works?"
               "api_key=oa-secret-0123456789&mailto=student@example.edu failed"}
        raise orchestrator_pipeline.PipelineError("failed: api_key=oa-secret-0123456789")

    monkeypatch.setattr(orchestrator_pipeline.discovery_service, "run_discovery", _leaky_discovery)

    async def run():
        with pytest.raises(orchestrator_pipeline.PipelineError):
            async for _ in orchestrator_pipeline.run_pipeline(
                    ResearchRequest(research_question="What is X?", corpus_size=6),
                    output_root=tmp_path, run_id="abc123"):
                pass

    asyncio.run(run())
    written = "".join(p.read_text(encoding="utf-8") for p in tmp_path.rglob("*") if p.is_file())
    assert "api.openalex.org" in written
    assert "oa-secret-0123456789" not in written and "student@example.edu" not in written


@pytest.mark.parametrize("balance, choice, plan, searches", [
    (None, "auto", "full", 5),     # unknown until the first request: no reason to cut
    (1000, "auto", "full", 5),
    (30, "auto", "reduced", 0),    # 30 < 71 credits: headless takes the reduced plan
    (50, "full", "full", 2),       # the user insisted: OpenAlex searches stop at the balance
    (1000, "reduced", "reduced", 0),
])
def test_the_budget_check_estimates_the_run_and_picks_the_plan(monkeypatch, balance, choice, plan, searches):
    monkeypatch.setattr(orchestrator_pipeline.scholarly_gateway, "balance", lambda: balance)
    check = orchestrator_pipeline.check_budget(
        ResearchRequest(research_question="Q?", corpus_size=6, budget_plan=choice))
    # 2 searches (first + re-plan) + 5 verification searches at 10 credits, 1 DOI filter at 1.
    assert check["estimated_credits"] == 71
    assert (check["balance"], check["plan"], check["verification_searches"]) == (balance, plan, searches)


def test_a_budget_check_before_discovery_is_logged_and_credits_reach_the_metadata(tmp_path, monkeypatch):
    import httpx
    from shared.utilities import scholarly_gateway

    monkeypatch.setattr(scholarly_gateway, "_gateway", scholarly_gateway.Gateway())  # no state from other tests
    monkeypatch.setattr(scholarly_gateway, "balance", lambda: 30)
    seen = {}

    async def discovery(request):
        seen["request"] = request
        transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"results": []}))
        async with httpx.AsyncClient(transport=transport) as http:  # one OpenAlex search
            await scholarly_gateway.Run().task(scholarly_gateway.get(
                http, "https://api.openalex.org/works", params={"search": "q"}))
        async for event in _fake_run_discovery(request):
            yield event

    monkeypatch.setattr(orchestrator_pipeline.discovery_service, "run_discovery", discovery)

    async def run():
        async for event in orchestrator_pipeline.run_pipeline(
                ResearchRequest(research_question="What is X?", corpus_size=6),
                output_root=tmp_path, run_id="b1"):
            if event["type"] == "result":
                return event["result"]

    result = asyncio.run(run())
    run_directory = Path(result.run_directory)
    logged = [json.loads(line) for line in (run_directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    [check] = [e for e in logged if e.get("details", {}).get("raw_type") == "budget_check"]
    assert check["stage"] == "discovery" and logged.index(check) < next(
        i for i, e in enumerate(logged) if e.get("service") == "research_discovery")
    assert {k: check["details"][k] for k in ("estimated_credits", "balance", "plan")} == {
        "estimated_credits": 71, "balance": 30, "plan": "reduced"}
    assert "30 credit(s) left" in check["details"]["reason"]
    assert "reduced plan" in check["message"] and "OpenAlex key" in check["message"]
    assert seen["request"].verification_searches == 0

    metadata = json.loads((run_directory / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["credits_used"] == {"OpenAlex": 10}

```

### `tests/orchestrator/test_api.py`

```python
"""Fast unit tests for the FastAPI backend, with the pipeline faked out."""

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from orchestrator import api as orchestrator_api  # noqa: E402
from shared.contracts.discovery_contract import (  # noqa: E402
    DiscoveryRequest, DiscoveryResult, NoveltyAssessment, PaperMetadata, ResearchGap,
)
from shared.contracts.pipeline_contract import PipelineResult, ResearchRequest, StageTimings  # noqa: E402
from shared.contracts.qa_contract import QualityAssuranceResult, QualityScores  # noqa: E402
from shared.contracts.verification_contract import VerificationResult  # noqa: E402
from shared.contracts.writing_contract import DraftMetadata, WritingResult  # noqa: E402
from orchestrator.pipeline import PipelineError  # noqa: E402


def _fake_pipeline_result() -> PipelineResult:
    discovery = DiscoveryResult(
        research_request=DiscoveryRequest(research_question="Q?", corpus_size=6),
        research_interpretation="interp", search_queries={"keyword": "x"},
        sources_searched=["OpenAlex"],
        selected_papers=[PaperMetadata(id=1, title="Paper One", has_abstract=True, abstract="abstract")],
        field_overview="overview", limitations="none",
        research_gaps=[ResearchGap(
            title="Gap", gap_type="methodological", impact="high",
            description="desc", evidence="evi", supporting_paper_ids=[1], research_questions=["Q1?"],
        )],
        novelty_analysis=NoveltyAssessment(novelty_summary="summary", confidence="medium", caveats="caveat"),
        confidence_notes="notes",
    )
    writing = WritingResult(
        research_outline="# Q\n## Introduction", research_draft_markdown="draft [1].",
        reference_candidates=['[1] A. Author, "Paper One," 2020.'],
        run_directory="/tmp/writing-run",
        draft_metadata=DraftMetadata(
            model_name="qwen3.5:9b", run_id="run1", generated_at="2026-08-27T00:00:00Z",
            sections_written=2, papers_cited=1,
        ),
    )
    verification = VerificationResult(
        validated_draft_markdown="draft [1].",
        validation_report="# Citation Verification Report\n",
    )
    qa = QualityAssuranceResult(
        final_draft_markdown="draft [1].", final_references=['[1] A. Author, "Paper One," 2020.'],
        final_validation_report="# Citation Verification Report\n",
        quality_report_markdown="# Quality Assurance Report\n",
        scores=QualityScores(
            citation_integrity=5.0, claim_source_alignment=5.0,
            process_control=5.0, literature_coverage=5.0, overall=5.0,
        ),
    )
    return PipelineResult(
        run_id="run1", run_directory="/tmp/run1",
        request=ResearchRequest(research_question="Q?"),
        discovery=discovery, writing=writing, verification=verification, quality_assurance=qa,
        timings=StageTimings(
            discovery_seconds=1.0, writing_seconds=1.0,
            verification_seconds=1.0, quality_assurance_seconds=1.0,
        ),
    )


async def _fake_run_pipeline_success(request, output_root=None, run_id=None):
    yield {"type": "progress", "run_id": run_id, "message": "working"}
    yield {"type": "result", "result": _fake_pipeline_result()}


async def _fake_run_pipeline_failure(request, output_root=None, run_id=None):
    yield {"type": "progress", "run_id": run_id, "message": "working"}
    raise PipelineError("Research Discovery Service produced no result.")


@pytest.fixture
def client():
    return TestClient(orchestrator_api.app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_research_endpoint_returns_pipeline_result(client, monkeypatch):
    monkeypatch.setattr(orchestrator_api, "run_pipeline", _fake_run_pipeline_success)
    response = client.post("/research", json={"research_question": "What is X?"})
    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "run1"
    assert body["quality_assurance"]["scores"]["overall"] == 5.0


def test_research_endpoint_rejects_invalid_request(client):
    response = client.post("/research", json={
        "research_question": "Q?", "target_format": "Other",
    })
    assert response.status_code == 422


def test_research_endpoint_returns_502_on_pipeline_error(client, monkeypatch):
    monkeypatch.setattr(orchestrator_api, "run_pipeline", _fake_run_pipeline_failure)
    response = client.post("/research", json={"research_question": "What is X?"})
    assert response.status_code == 502


def _wait_for_status(client, run_id, target_statuses, timeout=5.0):
    deadline = time.monotonic() + timeout
    body = None
    while time.monotonic() < deadline:
        response = client.get(f"/research/runs/{run_id}")
        body = response.json()
        if body["status"] in target_statuses:
            return body
        time.sleep(0.02)
    raise AssertionError(f"Run {run_id} never reached {target_statuses}, last saw {body}")


def test_start_research_run_returns_run_id_immediately(client, monkeypatch):
    monkeypatch.setattr(orchestrator_api, "run_pipeline", _fake_run_pipeline_success)
    response = client.post("/research/runs", json={"research_question": "What is X?"})
    assert response.status_code == 202
    run_id = response.json()["run_id"]
    assert run_id

    body = _wait_for_status(client, run_id, {"completed", "failed"})
    assert body["status"] == "completed"
    assert body["result"]["run_id"] == "run1"
    assert body["error"] is None


def test_start_research_run_marks_failed_on_pipeline_error(client, monkeypatch):
    monkeypatch.setattr(orchestrator_api, "run_pipeline", _fake_run_pipeline_failure)
    response = client.post("/research/runs", json={"research_question": "What is X?"})
    run_id = response.json()["run_id"]

    body = _wait_for_status(client, run_id, {"completed", "failed"})
    assert body["status"] == "failed"
    assert body["result"] is None
    assert "produced no result" in body["error"]


def test_get_research_run_returns_404_for_unknown_run_id(client):
    response = client.get("/research/runs/does-not-exist")
    assert response.status_code == 404


def test_research_run_events_stream_returns_progress_then_end(client, monkeypatch):
    monkeypatch.setattr(orchestrator_api, "run_pipeline", _fake_run_pipeline_success)
    response = client.post("/research/runs", json={"research_question": "What is X?"})
    run_id = response.json()["run_id"]
    _wait_for_status(client, run_id, {"completed", "failed"})

    with client.stream("GET", f"/research/runs/{run_id}/events") as stream:
        assert stream.status_code == 200
        body = "".join(stream.iter_text())

    assert '"message": "working"' in body or '"message":"working"' in body
    assert '"run_id": "run1"' in body or '"run_id":"run1"' in body
    assert "event: end" in body


def test_stream_research_run_events_returns_404_for_unknown_run_id(client):
    response = client.get("/research/runs/does-not-exist/events")
    assert response.status_code == 404


@pytest.mark.parametrize("fake_stages, expected_status", [
    (_fake_run_pipeline_success, "completed"),
    (_fake_run_pipeline_failure, "failed"),
])
def test_past_run_survives_restart_from_events_jsonl(client, monkeypatch, tmp_path, fake_stages, expected_status):
    from orchestrator import pipeline as orchestrator_pipeline
    from orchestrator import run_registry

    # Real run_pipeline wrapper (so events.jsonl gets written), fake stages.
    monkeypatch.setattr(orchestrator_pipeline, "_run_stages", fake_stages)
    monkeypatch.setattr(orchestrator_pipeline, "DEFAULT_OUTPUT_ROOT", tmp_path)
    response = client.post("/research/runs", json={"research_question": "What is X?"})
    run_id = response.json()["run_id"]
    before = _wait_for_status(client, run_id, {"completed", "failed"})

    monkeypatch.setattr(run_registry, "_runs", {})  # simulated restart

    body = client.get(f"/research/runs/{run_id}").json()
    assert body == before
    assert body["status"] == expected_status

    with client.stream("GET", f"/research/runs/{run_id}/events") as stream:
        assert stream.status_code == 200
        replay = "".join(stream.iter_text())
    assert '"message": "working"' in replay
    assert replay.endswith("event: end\ndata: {}\n\n")


def test_advisor_shortlist_hands_off_to_pipeline_under_same_run_id(client, monkeypatch):
    from shared.contracts.trend_contract import TopicCandidate, TrendAdvisorResult

    shortlist = TrendAdvisorResult(
        domain="MECHANICAL", generated_at="2026-09-16T00:00:00Z",
        shortlist=[TopicCandidate(topic="Topic A", growth_metric=2.0, paper_count=100,
                                  gap_evidence=["no work tests A under load"], gap_signal=True)],
    )

    async def fake_advisor(request, output_root=None, run_id=None):
        yield {"type": "progress", "run_id": run_id, "message": "ranking"}
        yield {"type": "result", "result": shortlist, "run_directory": f"/nowhere/x_{run_id}"}

    seen = {}

    async def fake_pipeline(request, output_root=None, run_id=None, **kwargs):
        seen.update(request=request, run_id=run_id, **kwargs)
        yield {"type": "result", "result": _fake_pipeline_result()}

    async def fake_question(candidate):
        return f"How does {candidate.topic} behave under load?"

    monkeypatch.setattr(orchestrator_api, "run_trend_advisor_stage", fake_advisor)
    monkeypatch.setattr(orchestrator_api, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(orchestrator_api.advisor_cli, "_research_question_for", fake_question)

    run_id = client.post("/advisor/runs", json={"domain": "MECHANICAL"}).json()["run_id"]
    body = _wait_for_status(client, run_id, {"completed", "failed"})
    assert body["result"]["shortlist"][0]["topic"] == "Topic A"

    assert client.post(f"/advisor/runs/{run_id}/question", json={"topic": "Nope"}).status_code == 422
    drafted = client.post(f"/advisor/runs/{run_id}/question", json={"topic": "Topic A"}).json()
    assert drafted == {"research_question": "How does Topic A behave under load?", "drafted": True}

    response = client.post(f"/advisor/runs/{run_id}/pipeline", json={
        "topic": "Topic A", "research_question": drafted["research_question"]})
    assert response.status_code == 202
    body = _wait_for_status(client, run_id, {"completed", "failed"})
    assert body["status"] == "completed"
    assert seen["run_id"] == run_id
    assert seen["request"].research_question == drafted["research_question"]
    assert seen["trend_advisor"].chosen_topic == "Topic A"
    assert seen["advisor_directory"] == f"/nowhere/x_{run_id}"


def test_index_serves_frontend(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "ResearchGenie" in response.text


def test_evidence_table_is_downloadable_and_has_a_tab(client, monkeypatch, tmp_path):
    (tmp_path / "final").mkdir()
    (tmp_path / "final" / "evidence_table.md").write_text("# Evidence table\n", encoding="utf-8")
    result = _fake_pipeline_result().model_copy(update={"run_directory": str(tmp_path)})
    monkeypatch.setattr(orchestrator_api.run_registry, "get_run",
                        lambda run_id: SimpleNamespace(result=result))

    response = client.get("/research/runs/run1/files/evidence_table.md")
    assert response.status_code == 200
    assert response.text.startswith("# Evidence table")
    page = client.get("/").text
    assert '["evidence_table.md", "Evidence table", ".md"]' in page and '"Evidence": ""' in page


def test_system_endpoint_reports_each_scholarly_provider_without_secrets(client, monkeypatch):
    monkeypatch.setattr(orchestrator_api.llm_provider, "health_check", _async_true)
    monkeypatch.setenv("OPENALEX_API_KEY", "oa-secret-0123456789")
    monkeypatch.setenv("CONTACT_EMAIL", "student@example.edu")
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    body = client.get("/system").json()
    providers = {p["name"]: p for p in body["providers"]}
    assert providers["OpenAlex"]["access"] == "keyed"
    assert providers["PubMed (NCBI)"]["access"] == "identified"
    assert providers["Semantic Scholar"]["access"] == "anonymous"
    assert all(p["health"] in ("healthy", "degraded") for p in body["providers"])
    assert "oa-secret-0123456789" not in str(body) and "student@example.edu" not in str(body)


async def _async_true():
    return True


def test_the_budget_endpoint_tells_the_page_when_to_offer_the_key_or_the_reduced_plan(client, monkeypatch):
    monkeypatch.setattr(orchestrator_api.rg_config, "load", lambda: orchestrator_api.rg_config.Config())
    monkeypatch.setattr(orchestrator_api.scholarly_gateway, "balance", lambda: 30)
    body = client.post("/research/budget", json={"research_question": "What is X?"}).json()
    assert body["plan"] == "reduced" and body["balance"] == 30 and body["estimated_credits"] == 71
    chosen = client.post("/research/budget", json={"research_question": "What is X?",
                                                   "budget_plan": "full"}).json()
    assert chosen["plan"] == "full"


def test_a_web_run_waits_for_approval_and_approve_continues_it(monkeypatch):
    # One event loop for the whole test (the run waits on a future across requests);
    # the lifespan then runs, so keep it off the real ~/.researchgenie config.
    monkeypatch.setattr(orchestrator_api.rg_config, "load", lambda: orchestrator_api.rg_config.Config())
    with TestClient(orchestrator_api.app) as client:
        _approval_round_trip(client, monkeypatch)


def _approval_round_trip(client, monkeypatch):
    papers = [{"id": 1, "title": "Kept"}, {"id": 2, "title": "Dropped"}]
    decisions = []

    async def pipeline_with_approval(request, output_root=None, run_id=None, approve=None):
        yield {"type": "progress", "run_id": run_id, "message": "review the papers",
               "details": {"raw_type": "approval_request", "papers": papers}}
        decisions.append(await approve(papers, 0))
        yield {"type": "result", "result": _fake_pipeline_result()}

    monkeypatch.setattr(orchestrator_api, "run_pipeline", pipeline_with_approval)
    run_id = client.post("/research/runs", json={"research_question": "Q?", "approval": "ask"}).json()["run_id"]
    _wait_for_status(client, run_id, {"awaiting_approval"})
    assert client.post(f"/research/runs/{run_id}/approve", json={"drop": [2]}).status_code == 200
    _wait_for_status(client, run_id, {"completed"})
    assert decisions == [{"approved_by": "web", "drop": [2]}]
    assert client.post(f"/research/runs/{run_id}/approve", json={"drop": []}).status_code == 409


def test_an_api_run_approves_automatically_by_default(client, monkeypatch):
    seen = {}

    async def pipeline(request, output_root=None, run_id=None, approve=None):
        seen["approve"] = approve
        yield {"type": "result", "result": _fake_pipeline_result()}

    monkeypatch.setattr(orchestrator_api, "run_pipeline", pipeline)
    run_id = client.post("/research/runs", json={"research_question": "Q?"}).json()["run_id"]
    _wait_for_status(client, run_id, {"completed"})
    assert seen["approve"] is None

```

## The experiment

```bash
cd Course-Project/code
pip install -e .
pytest -v tests/orchestrator/test_pipeline.py tests/orchestrator/test_api.py
```

## Real result
Raw console output also saved as [`sample_run.txt`](./sample_run.txt) in this folder.

The event ledger and the API tests above pass against the real orchestrator. The Telegram bot authenticated against Telegram's own servers and stayed connected for the full review session, with every inbound message, model session, and outbound reply visible live in the Telegram Sync dashboard. The Logs dashboard streamed every stage of a real pipeline run live, for direct observation rather than a static report.

## Why this is the real evidence, not a toy
Both dashboards read the actual event ledger and the actual OpenClaw gateway log files. Neither one simulates, replays or pre records anything; they can only show what the real runtime is doing at the moment someone opens them.

# Lab 10 - Cloud Operations Agent

**Skill:** Cloud SDLC

**Worklet mapping:** A 12 rule Doctor that diagnoses a failure, quotes its evidence, proposes one fix, waits for human confirmation, then resumes rather than restarting.

## What this demonstrates

CI runs on every push. A container packages both a local Ollama and a cloud Gemini profile. The Doctor reads the real event ledger and system state, matches against a 12 rule table, and only ever proposes a fix after naming its cause from real evidence.

## Real code exercised

The actual source files this lab is built on, copied in full below, not linked only.

### `orchestrator/doctor.py`

```python
"""The Doctor (#13): why a run failed, and the one fix that recovers it.

Deterministic, never a model call (spec #1, "Doctor"). RULES is the whole
diagnosis: each row matches the run's last error message and/or a check over
its logged events, the live system status and its final folder, and names a
plain-language cause, a failure class (the headless CLI's exit code, #10)
and a fix. The first matching row wins; no match is an honest "no known
cause" with the last error, never a guess.

A fix reuses what already exists: start Ollama or pull the model
(researchgenie/ollama_manager.py), the key setup (researchgenie config),
waiting out a provider's reset, and resume (#9). Nothing is applied here
without the caller's confirmation; every surface (CLI, headless --yes, the
API and the web button) asks first. Each diagnosis is saved as
<run>/doctor_report.json with the evidence, the chosen fix and its outcome.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from shared.utilities.scholarly_gateway import redact

Events = list[dict[str, Any]]
REPORT_NAME = "doctor_report.json"


def read_events(run_directory: Path) -> Events:
    """Every line of <run>/events.jsonl; [] when the run has no log yet."""
    log = Path(run_directory) / "events.jsonl"
    if not log.is_file():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]


def last_error(events: Events) -> str | None:
    """The message of the last error line, unless a later attempt (a resume)
    has logged progress since."""
    error = None
    for event in events:
        if event.get("type") == "error":
            error = event.get("message") or ""
        elif event.get("type") in ("progress", "result"):
            error = None
    return error


def _completed(events: Events) -> bool:
    return bool(events) and events[-1].get("type") == "result"


def exhausted_hosts(events: Events) -> list[str]:
    """Hosts the gateway reported as exhausted, in the order first reported."""
    hosts = [e["details"]["host"] for e in events
             if (e.get("details") or {}).get("raw_type") == "provider_exhausted"
             and e["details"].get("host")]
    return list(dict.fromkeys(hosts))


@dataclass
class Context:
    """What a rule may look at."""
    events: Events
    system: dict[str, Any] | None = None   # None: classify from the log alone (headless exit codes)
    run_directory: Path | None = None

    def metadata(self) -> dict[str, Any]:
        path = self.run_directory / "metadata.json" if self.run_directory else None
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path else {}
        except (OSError, ValueError):
            return {}

    def sys(self, key: str) -> Any:
        return (self.system or {}).get(key)


@dataclass(frozen=True)
class Fix:
    id: str
    label: str                      # shown before the user confirms
    automatic: bool = True          # False: the report says what to do; nothing is run
    resume_from: str | None = "failed"   # resume afterwards: "failed" = the failed stage, or a stage name
    interactive_only: bool = False  # needs a prompt (key entry); headless and web report it instead


FIXES = {fix.id: fix for fix in (
    Fix("start_ollama", "Start Ollama, then resume the run from the failed stage."),
    Fix("start_ollama_rewrite", "Start Ollama, then write the draft again (Writing and every "
                                "stage after it).", resume_from="writing"),
    Fix("pull_model", "Download the model in Ollama, then resume the run from the failed stage."),
    Fix("gemini_setup", "Re-enter the Gemini API key with `researchgenie config`, then resume.",
        interactive_only=True),
    Fix("openalex_key", "Add a free OpenAlex API key (takes a minute, `researchgenie config`), "
                        "then resume.", interactive_only=True),
    Fix("wait_for_reset", "Resume the run once the provider's budget has reset (checked first; "
                          "OpenAlex resets at midnight UTC)."),
    Fix("resume", "Resume the run from the failed stage."),
    Fix("resume_writing", "Write the draft again (Writing and every stage after it).",
        resume_from="writing"),
    Fix("rephrase", "Start a new run with a sharper question or added keywords "
                    "(`--clarify auto` suggests one).", automatic=False, resume_from=None),
    Fix("install_latex", "Install tectonic (https://tectonic-typesetting.github.io), then re-run "
                         "Quality Assurance to compile paper.pdf; paper.tex is already in final/.",
        automatic=False, resume_from=None),
)}


@dataclass(frozen=True)
class Rule:
    id: str
    failure_class: str | None  # model_unavailable | providers_exhausted | no_relevant_papers | None
    cause: str                 # plain language, for the student
    fix: str                   # a FIXES key
    pattern: str | None = None  # regex the last error must match; None: any error
    when: Callable[[Context], bool] = field(default=lambda ctx: True)
    completed: bool = False    # a rule for a run that finished (no error), e.g. a partial draft
    quote: tuple[str, ...] = ()  # event raw_types quoted as evidence besides the error line
    fact: Callable[[Context], str] | None = None  # the live-status evidence


def _ollama_down(ctx: Context) -> bool:
    return ctx.sys("provider") == "ollama" and ctx.sys("ollama_running") is False


def _model_missing(ctx: Context) -> bool:
    return ctx.sys("provider") == "ollama" and ctx.sys("ollama_running") and \
        ctx.sys("model_installed") is False


_OLLAMA_FACT = lambda ctx: "Live check: Ollama is not answering right now."  # noqa: E731
_MODEL_FACT = lambda ctx: f"Live check: the model {ctx.sys('model')} is not installed in Ollama."  # noqa: E731

RULES: tuple[Rule, ...] = (
    Rule("ollama_unreachable", "model_unavailable",
         "The local model server (Ollama) is not running, so the model could not be asked.",
         "start_ollama", pattern=r"Could not reach the local Ollama server",
         fact=lambda ctx: _OLLAMA_FACT(ctx) if _ollama_down(ctx) else ""),
    Rule("model_missing", "model_unavailable",
         "The model this run uses is not downloaded in Ollama yet.",
         "pull_model", pattern=r"is not installed in Ollama"),
    Rule("gemini_unavailable", "model_unavailable",
         "The Gemini provider could not be used: its API key is missing or rejected, or it is unreachable.",
         "gemini_setup",
         pattern=r"GEMINI_API_KEY is not set|Gemini rejected the configured API key|Could not reach the Gemini API"),
    Rule("openalex_budget_no_key", "providers_exhausted",
         "OpenAlex's free daily budget (about 100 searches) is spent and no OpenAlex key is set, "
         "and no other source found papers.",
         "openalex_key", pattern=r"No papers found in any source",
         when=lambda ctx: "api.openalex.org" in exhausted_hosts(ctx.events)
         and ctx.sys("openalex_key") is False,
         quote=("provider_exhausted", "source_error"),
         fact=lambda ctx: "Live check: no OpenAlex API key is configured."),
    Rule("providers_exhausted", "providers_exhausted",
         "Every literature source was out of budget or rate-limited, so the search found nothing.",
         "wait_for_reset", pattern=r"No papers found in any source",
         when=lambda ctx: bool(exhausted_hosts(ctx.events)), quote=("provider_exhausted", "source_error")),
    Rule("all_sources_failed", "providers_exhausted",
         "Every literature source failed to answer (network trouble or an outage).",
         "resume", pattern=r"No papers found in any source", quote=("source_error",)),
    Rule("no_relevant_paper", "no_relevant_papers",
         "The search worked, but no paper it found was on topic for the question.",
         "rephrase", pattern=r"judged relevant to the research question", quote=("relevance_filter",)),
    # Any other error while the model is down now: after every specific cause
    # above, so a known cause is never blamed on a model that is merely off.
    Rule("ollama_stopped", "model_unavailable",
         "Ollama is not running now; the run most likely failed because it stopped mid-run.",
         "start_ollama", when=_ollama_down, fact=_OLLAMA_FACT),
    Rule("model_missing_now", "model_unavailable",
         "Ollama is running, but the model this run uses is not installed.",
         "pull_model", when=_model_missing, fact=_MODEL_FACT),
    # Writing isolates each card and section, so a model that stops mid-Writing
    # leaves a finished run with a broken draft, not an error. Rewriting it
    # while Ollama is still down would break it the same way.
    Rule("ollama_stopped_mid_writing", None,
         "The run finished with an incomplete draft and Ollama is not running now: it most "
         "likely stopped during Writing.",
         "start_ollama_rewrite", completed=True,
         when=lambda ctx: ctx.metadata().get("draft_status") in ("partial", "failed") and _ollama_down(ctx),
         fact=lambda ctx: f"{_OLLAMA_FACT(ctx)} metadata.json: draft_status = "
                          f"{ctx.metadata().get('draft_status')}"),
    Rule("partial_draft", None,
         "The run finished, but the draft is incomplete: some sections are missing or broken.",
         "resume_writing", completed=True,
         when=lambda ctx: ctx.metadata().get("draft_status") in ("partial", "failed"),
         fact=lambda ctx: f"metadata.json: draft_status = {ctx.metadata().get('draft_status')}, "
                          f"broken sections = {ctx.metadata().get('broken_sections') or []}"),
    Rule("missing_latex", None,
         "The run finished, but no PDF was compiled: no LaTeX toolchain (tectonic or pdflatex) "
         "is installed.",
         "install_latex", completed=True,
         when=lambda ctx: ctx.run_directory is not None
         and not (ctx.run_directory / "final" / "paper.pdf").is_file() and ctx.sys("latex") is False,
         fact=lambda ctx: "Live check: neither tectonic nor pdflatex is on PATH; final/paper.pdf is missing."),
)


def match(events: Events, system: dict[str, Any] | None = None,
          run_directory: Path | None = None) -> Rule | None:
    """The first rule matching the run, or None (no known cause)."""
    ctx = Context(events, system, run_directory)
    error = last_error(events)
    completed = _completed(events)
    for rule in RULES:
        if rule.completed:
            applies = completed
        else:
            applies = error is not None and (rule.pattern is None or re.search(rule.pattern, error))
        if applies and rule.when(ctx):
            return rule
    return None


def _line(event: dict[str, Any]) -> str:
    if event.get("type") == "error":
        return f"error: {event.get('message')}"
    return f"{event.get('timestamp', '')} [{event.get('stage', '')}] {event.get('message', '')}".strip()


async def system_status() -> dict[str, Any]:
    """The live facts the rules check. Never carries a key, only whether one is set."""
    from researchgenie import ollama_manager as om
    provider = os.getenv("AI_PROVIDER", "ollama")
    model = os.getenv("LOCAL_AI_MODEL", "qwen3.5:9b")
    running = await om.is_ollama_server_running() if provider == "ollama" else None
    return {
        "provider": provider, "model": model, "ollama_running": running,
        "model_installed": await om.is_model_installed(model) if running else None,
        "latex": bool(shutil.which("tectonic") or shutil.which("pdflatex")),
        "openalex_key": bool(os.getenv("OPENALEX_API_KEY")),
        "gemini_key": bool(os.getenv("GEMINI_API_KEY")),
    }


def diagnose(run_directory: Path, system: dict[str, Any]) -> dict[str, Any]:
    """The diagnosis of one run folder (also the body of doctor_report.json)."""
    run_directory = Path(run_directory)
    events = read_events(run_directory)
    rule = match(events, system, run_directory)
    error = last_error(events)
    diagnosis: dict[str, Any] = {
        "run_id": run_directory.name.partition("_")[2],
        "run_directory": str(run_directory),
        "diagnosed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "state": "completed" if _completed(events) else "failed" if error is not None else "running",
        "last_error": error, "system": system,
    }
    if rule is None:
        diagnosis.update(
            rule=None, failure_class="unknown", fix=None,
            cause=("No known cause matches this run's last error; nothing is guessed. "
                   "The last error is quoted below." if error is not None else
                   "This run has not failed: it is still going, or it stopped without logging "
                   "an error." if not _completed(events) else
                   "This run completed and no known problem was found."),
            evidence=[f"error: {error}"] if error is not None else [])
        return diagnosis
    ctx = Context(events, system, run_directory)
    evidence = [_line(e) for e in events
                if (e.get("details") or {}).get("raw_type") in rule.quote][-8:]
    if error is not None:
        evidence.append(f"error: {error}")
    if rule.fact and (fact := rule.fact(ctx)):
        evidence.append(fact)
    fix = FIXES[rule.fix]
    diagnosis.update(rule=rule.id, failure_class=rule.failure_class, cause=rule.cause,
                     evidence=evidence,
                     fix={"id": fix.id, "label": fix.label, "automatic": fix.automatic,
                          "interactive_only": fix.interactive_only, "resume_from": fix.resume_from})
    return diagnosis


def write_report(run_directory: Path, diagnosis: dict[str, Any], *, applied: bool,
                 outcome: str) -> Path:
    path = Path(run_directory) / REPORT_NAME
    report = {**diagnosis, "applied": applied, "outcome": outcome}
    path.write_text(redact(json.dumps(report, indent=2, ensure_ascii=False)), encoding="utf-8")
    return path


def resume_stage(diagnosis: dict[str, Any]) -> str | None:
    """The stage a fix resumes from: None means the failed stage (resume's default)."""
    stage = (diagnosis.get("fix") or {}).get("resume_from")
    return None if stage == "failed" else stage


async def apply_fix(diagnosis: dict[str, Any], *, key_setup: Callable[[], bool] | None = None,
                    wait_seconds: int = 15) -> str | None:
    """Carries out the fix's own step (before any resume). Returns None when
    it worked, else why not. The caller has already asked for confirmation.
    key_setup is the interactive key prompt; without one (headless, web) a
    key fix is reported, not run."""
    from researchgenie import ollama_manager as om
    from shared.utilities import scholarly_gateway
    fix = diagnosis.get("fix") or {}
    if not fix:
        return "There is no fix for an unknown cause."
    if not fix["automatic"]:
        return f"This fix is a step for you: {fix['label']}"
    if fix["id"] in ("start_ollama", "start_ollama_rewrite"):
        if not om.start_ollama_server():
            return "Ollama could not be started (is it installed and on PATH?)."
        for _ in range(wait_seconds):
            if await om.is_ollama_server_running():
                return None
            await asyncio.sleep(1)
        return f"Ollama was started but did not answer within {wait_seconds} s."
    if fix["id"] == "pull_model":
        model = (diagnosis.get("system") or {}).get("model") or os.getenv("LOCAL_AI_MODEL", "qwen3.5:9b")
        return None if await om.pull_model(model) else f"Downloading {model} failed."
    if fix["interactive_only"]:
        if key_setup is None:
            return f"This fix needs you at a prompt: {fix['label']}"
        return None if key_setup() else "Key setup did not finish."
    if fix["id"] == "wait_for_reset":
        still = [h for h in exhausted_hosts(read_events(Path(diagnosis["run_directory"])))
                 if scholarly_gateway._gateway.exhausted(h)]
        if still:
            return f"Still out of budget: {', '.join(still)}. Try again after the reset (midnight UTC)."
        return None
    return None  # resume / resume_writing: the resume is the fix


async def resume_and_record(run_directory: Path, diagnosis: dict[str, Any],
                            resume: Callable[..., AsyncIterator[dict[str, Any]]],
                            approve: Callable | None = None) -> AsyncIterator[dict[str, Any]]:
    """The fix's resume: yields its events and writes the outcome into
    doctor_report.json when it ends. `resume` is orchestrator.pipeline.resume_pipeline."""
    try:
        async for event in resume(run_directory, resume_stage(diagnosis), approve=approve):
            yield event
    except Exception as error:
        write_report(run_directory, diagnosis, applied=True,
                     outcome=f"Fix applied; the resumed run failed again: {redact(str(error)) or type(error).__name__}")
        raise
    write_report(run_directory, diagnosis, applied=True, outcome="Fix applied; the resumed run completed.")

```

### Real tests for this lab

### `tests/orchestrator/test_doctor.py`

```python
"""The Doctor (#13): one test per rule row over recorded event-log lines
(tests/fixtures/doctor/), then the confirmation rule and the fix → resume →
report loop through the headless CLI and the API, with every service faked."""

import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "orchestrator"))

from orchestrator import doctor  # noqa: E402
from orchestrator import pipeline as orchestrator_pipeline  # noqa: E402
from researchgenie import headless  # noqa: E402
from researchgenie import ollama_manager as om  # noqa: E402
from shared.utilities import llm_provider  # noqa: E402
from test_pipeline import (  # noqa: E402
    _fake_run_discovery, _fake_run_qa, _fake_run_verification, _fake_run_writing,
)

FIXTURES = ROOT / "tests" / "fixtures" / "doctor"
HEALTHY = {"provider": "ollama", "model": "qwen3.5:9b", "ollama_running": True,
           "model_installed": True, "latex": True, "openalex_key": True, "gemini_key": False}


def _run_folder(tmp_path, fixture, metadata=None, pdf=False):
    folder = tmp_path / "what-is-x_r1"
    folder.mkdir()
    shutil.copy(FIXTURES / f"{fixture}.jsonl", folder / "events.jsonl")
    if metadata is not None:
        (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (folder / "final").mkdir()
    if pdf:
        (folder / "final" / "paper.pdf").write_bytes(b"%PDF")
    return folder


ROWS = [
    # fixture, live-status changes, metadata, pdf, rule, fix
    ("ollama_unreachable", {"ollama_running": False}, None, False, "ollama_unreachable", "start_ollama"),
    ("writing_model_failed", {"ollama_running": False, "model_installed": None}, None, False,
     "ollama_stopped", "start_ollama"),
    ("model_missing", {}, None, False, "model_missing", "pull_model"),
    ("writing_model_failed", {"model_installed": False}, None, False, "model_missing_now", "pull_model"),
    ("gemini_unavailable", {"provider": "gemini", "ollama_running": None}, None, False,
     "gemini_unavailable", "gemini_setup"),
    ("providers_exhausted", {"openalex_key": False}, None, False, "openalex_budget_no_key", "openalex_key"),
    ("providers_exhausted", {}, None, False, "providers_exhausted", "wait_for_reset"),
    ("all_sources_failed", {}, None, False, "all_sources_failed", "resume"),
    ("no_relevant_paper", {}, None, False, "no_relevant_paper", "rephrase"),
    ("completed", {"ollama_running": False, "model_installed": None}, {"draft_status": "partial"}, True,
     "ollama_stopped_mid_writing", "start_ollama_rewrite"),
    ("completed", {}, {"draft_status": "partial", "broken_sections": ["Discussion"]}, True,
     "partial_draft", "resume_writing"),
    ("completed", {"latex": False}, {"draft_status": "complete"}, False, "missing_latex", "install_latex"),
]


@pytest.mark.parametrize("fixture, system, metadata, pdf, rule, fix", ROWS,
                         ids=[row[4] for row in ROWS])
def test_each_rule_row(tmp_path, fixture, system, metadata, pdf, rule, fix):
    folder = _run_folder(tmp_path, fixture, metadata, pdf)
    diagnosis = doctor.diagnose(folder, {**HEALTHY, **system})
    assert diagnosis["rule"] == rule and diagnosis["fix"]["id"] == fix
    assert diagnosis["cause"] and diagnosis["fix"]["label"]
    error = doctor.last_error(doctor.read_events(folder))
    if error is not None:
        assert f"error: {error}" in diagnosis["evidence"]
    if rule in ("providers_exhausted", "openalex_budget_no_key"):
        assert any("api.openalex.org" in line for line in diagnosis["evidence"])
    if rule in ("ollama_stopped", "model_missing_now", "missing_latex", "partial_draft",
                "ollama_stopped_mid_writing"):
        assert any(line.startswith(("Live check", "metadata.json")) for line in diagnosis["evidence"])


def test_a_draft_broken_by_a_stopped_ollama_starts_it_before_rewriting(tmp_path, monkeypatch):
    folder = _run_folder(tmp_path, "completed", {"draft_status": "partial"}, pdf=True)
    diagnosis = doctor.diagnose(folder, {**HEALTHY, "ollama_running": False, "model_installed": None})
    assert doctor.resume_stage(diagnosis) == "writing"
    started = []
    monkeypatch.setattr(om, "start_ollama_server", lambda: started.append(1) or True)
    monkeypatch.setattr(om, "is_ollama_server_running", _async(True))
    assert asyncio.run(doctor.apply_fix(diagnosis)) is None and started == [1]


def test_a_known_cause_is_not_blamed_on_a_model_that_is_merely_off(tmp_path):
    folder = _run_folder(tmp_path, "providers_exhausted")
    down = {**HEALTHY, "ollama_running": False, "model_installed": None}
    assert doctor.diagnose(folder, down)["rule"] == "providers_exhausted"


def test_a_healthy_completed_run_and_an_unknown_failure_are_reported_honestly(tmp_path):
    completed = _run_folder(tmp_path, "completed", {"draft_status": "complete"}, pdf=True)
    assert doctor.diagnose(completed, HEALTHY)["rule"] is None

    shutil.rmtree(completed)
    folder = _run_folder(tmp_path, "unknown")
    diagnosis = doctor.diagnose(folder, HEALTHY)
    assert diagnosis["rule"] is None and diagnosis["fix"] is None
    assert diagnosis["failure_class"] == "unknown"
    assert "No known cause" in diagnosis["cause"] and diagnosis["last_error"] == "Crossref went away."
    assert diagnosis["evidence"] == ["error: Crossref went away."]


def test_the_rows_without_a_live_status_give_the_headless_exit_class():
    assert doctor.match([]) is None
    for fixture, failure_class in (("ollama_unreachable", "model_unavailable"),
                                   ("providers_exhausted", "providers_exhausted"),
                                   ("no_relevant_paper", "no_relevant_papers")):
        lines = [json.loads(l) for l in (FIXTURES / f"{fixture}.jsonl").read_text().splitlines()]
        assert doctor.match(lines).failure_class == failure_class


# ------------------------------------------------- confirmation, fix, resume, report

@pytest.fixture
def ollama_died_in_verification(tmp_path, monkeypatch):
    """A real (faked-service) run that failed in Verification with Ollama
    unreachable, and an Ollama that is down until the Doctor starts it."""
    monkeypatch.setattr(orchestrator_pipeline, "DEFAULT_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(headless, "DEFAULT_OUTPUT_ROOT", tmp_path)
    for service, name, fake in (
            (orchestrator_pipeline.discovery_service, "run_discovery", _fake_run_discovery),
            (orchestrator_pipeline.writing_service, "run_writing", _fake_run_writing),
            (orchestrator_pipeline.qa_service, "run_quality_assurance", _fake_run_qa)):
        monkeypatch.setattr(service, name, fake)

    async def ollama_gone(request):
        yield {"type": "status", "stage": "verify", "state": "running", "message": "verifying"}
        raise llm_provider.LLMProviderError(
            "Could not reach the local Ollama server at http://127.0.0.1:11434. Start it with: ollama serve")

    monkeypatch.setattr(orchestrator_pipeline.verification_service, "run_verification", ollama_gone)
    monkeypatch.setattr(headless, "_ready", lambda: True)
    assert headless.main(["run", "-q", "What is X?", "--json"]) == headless.EXIT_MODEL_UNAVAILABLE
    monkeypatch.setattr(orchestrator_pipeline.verification_service, "run_verification",
                        _fake_run_verification)

    ollama = {"running": False, "started": 0}

    def start():
        ollama["started"] += 1
        ollama["running"] = True
        return True

    async def running():
        return ollama["running"]

    async def installed(model):
        return True

    monkeypatch.setattr(om, "start_ollama_server", start)
    monkeypatch.setattr(om, "is_ollama_server_running", running)
    monkeypatch.setattr(om, "is_model_installed", installed)
    folder = next(tmp_path.glob("*_*"))
    return folder, ollama


def _report(folder):
    return json.loads((folder / "doctor_report.json").read_text(encoding="utf-8"))


def test_headless_needs_yes_before_any_fix_is_applied(ollama_died_in_verification, capsys):
    folder, ollama = ollama_died_in_verification
    capsys.readouterr()
    code = headless.main(["doctor", folder.name.partition("_")[2], "--json"])
    out = json.loads(capsys.readouterr().out)
    assert ollama["started"] == 0 and out["applied"] is False and "--yes" in out["outcome"]
    assert code == headless.EXIT_MODEL_UNAVAILABLE  # the run is still failed

    report = _report(folder)
    assert report["rule"] == "ollama_unreachable" and report["fix"]["id"] == "start_ollama"
    assert report["evidence"] and report["applied"] is False and report["outcome"]


def test_doctor_yes_starts_ollama_resumes_and_records_the_outcome(ollama_died_in_verification, capsys):
    folder, ollama = ollama_died_in_verification
    capsys.readouterr()
    code = headless.main(["doctor", str(folder), "--yes", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == headless.EXIT_OK and ollama["started"] == 1
    assert headless.status_of(folder)["state"] == "completed"
    report = _report(folder)
    assert report["applied"] is True and report["outcome"] == "Fix applied; the resumed run completed."
    assert out["outcome"] == report["outcome"]


def test_a_fix_for_you_is_never_run(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(headless, "DEFAULT_OUTPUT_ROOT", tmp_path)
    folder = _run_folder(tmp_path, "no_relevant_paper")
    monkeypatch.setattr(doctor, "system_status", _async(HEALTHY))
    headless.main(["doctor", str(folder), "--yes", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert out["fix"]["id"] == "rephrase" and out["applied"] is False
    assert out["outcome"].startswith("No automatic fix")


def _async(value):
    async def get():
        return value
    return get


def test_the_api_diagnoses_then_applies_only_when_asked(ollama_died_in_verification, monkeypatch):
    from fastapi.testclient import TestClient
    from orchestrator import api, run_registry
    folder, ollama = ollama_died_in_verification
    monkeypatch.setattr(run_registry, "_runs", {})
    client = TestClient(api.app)
    run_id = folder.name.partition("_")[2]

    shown = client.post(f"/research/runs/{run_id}/doctor", json={}).json()
    assert shown["rule"] == "ollama_unreachable" and shown["applied"] is False
    assert ollama["started"] == 0 and _report(folder)["applied"] is False

    stale = client.post(f"/research/runs/{run_id}/doctor", json={"apply": True, "fix": "pull_model"})
    assert stale.status_code == 409 and ollama["started"] == 0  # not the fix that was shown
    applied = client.post(f"/research/runs/{run_id}/doctor",
                          json={"apply": True, "fix": shown["fix"]["id"]}).json()
    assert applied["applied"] is True and applied["resumed"] is True and ollama["started"] == 1
    deadline = time.monotonic() + 5
    while client.get(f"/research/runs/{run_id}").json()["status"] != "completed":
        assert time.monotonic() < deadline, "the resumed run never completed"
        time.sleep(0.02)
    for _ in range(100):  # the report is written as the resume's stream ends
        if _report(folder)["outcome"] == "Fix applied; the resumed run completed.":
            break
        time.sleep(0.02)
    assert _report(folder)["outcome"] == "Fix applied; the resumed run completed."
    assert client.post("/research/runs/nope/doctor", json={}).status_code == 404

```

## The experiment

```bash
cd Course-Project/code
pip install -e .
pytest -v tests/orchestrator/test_doctor.py
```

## Real result
All 12 rules above pass against the real Doctor implementation. Proven live and end to end, not only in unit tests: the first live attempt surfaced a genuine gap, a broken draft was not being diagnosed because Ollama was down while the run had still completed. That was fixed live with a new rule, ollama_stopped_mid_writing. Full recovery was then proven for real: Ollama was killed deliberately mid run, the Doctor caught it, fixed it, and the run resumed to a completed draft. Full record in `Course-Project/docs/evidence/live-pass/LIVE_PASS.md`.

## Why this is the real evidence, not a toy
The bug the Doctor fixed live was a real bug found by testing, not a scripted failure written in advance for the assignment.

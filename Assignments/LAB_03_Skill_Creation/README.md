# Lab 3 - Skill Creation

**Skill:** Skills

**Worklet mapping:** Reusable skills built from instructions, tools and a typed contract, not a one off script.

## What this demonstrates

Every ResearchGenie capability sits behind a typed Pydantic contract, and one Agent Skill folder lets any agent runtime, Claude Code or OpenClaw, start a run from chat using only the real headless CLI. A dedicated test fails the build the moment the skill's documented commands drift from the real CLI's own parser.

## Real code exercised

The actual source files this lab is built on, copied in full below, not linked only.

### `shared/contracts/discovery_contract.py`

```python
"""Input/output contract for the Research Discovery Service (pipeline stage 1).

Defined independently of any particular internal implementation so the
orchestrator, the adapter, and Service 2 can all depend on a stable shape.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

GapType = Literal[
    "theoretical", "methodological", "population", "contextual", "temporal",
    "measurement", "evidence_contradiction", "interdisciplinary", "practical",
]
Impact = Literal["high", "medium", "low"]
GapStatus = Literal["open", "partially_addressed", "already_answered"]


class DiscoveryRequest(BaseModel):
    """Input to the Research Discovery Service."""
    research_question: str
    domain: str | None = None
    year_range: tuple[int, int] | None = None
    preferred_databases: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    excluded_topics: list[str] = Field(default_factory=list)
    corpus_size: int = Field(ge=6, le=9)
    # Read the research library: stored excerpts, seeded past papers (#19).
    memory: bool = True
    # OpenAlex verification searches the budget check allowed (#7); the
    # rest search Crossref. None: no cap.
    verification_searches: int | None = Field(default=None, ge=0)


class PaperMetadata(BaseModel):
    id: int
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    citations: int | None = None
    doi: str | None = None
    url: str | None = None
    source: str = ""
    has_abstract: bool = False
    abstract: str | None = None
    relevance_score: float | None = None
    relevance_reason: str | None = None
    # The authors' own Discussion/Limitations/Future-research text, when an
    # open-access full text was actually retrieved (Discovery already fetches
    # this for its own gap mining). Forwarded so Service 2 can write from the
    # authors' stated reasoning and not only from the abstract — see
    # DECISIONS.md D-029/D-039. None whenever no full text was available.
    discussion_excerpt: str | None = None


class ResearchGap(BaseModel):
    title: str
    gap_type: GapType
    impact: Impact
    description: str
    evidence: str
    supporting_paper_ids: list[int] = Field(default_factory=list)
    research_questions: list[str] = Field(default_factory=list)


class AuthorFlaggedGap(BaseModel):
    question: str
    status: GapStatus
    verdict: str
    source_paper_ids: list[int] = Field(default_factory=list)
    evidence_paper_ids: list[int] = Field(default_factory=list)
    recommendation: str


class NoveltyAssessment(BaseModel):
    """Whether/how the research question offers something not already covered
    by the selected corpus. Must be evidence-grounded, not asserted."""
    novelty_summary: str
    supporting_gap_titles: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    caveats: str


class DiscoveryResult(BaseModel):
    """Output of the Research Discovery Service — direct input to Service 2."""
    research_request: DiscoveryRequest
    research_interpretation: str
    search_queries: dict[str, str]
    sources_searched: list[str]
    selected_papers: list[PaperMetadata]
    field_overview: str
    important_findings: list[str] = Field(default_factory=list)
    limitations: str
    future_research_directions: list[str] = Field(default_factory=list)
    research_gaps: list[ResearchGap]
    author_flagged_gaps: list[AuthorFlaggedGap] = Field(default_factory=list)
    novelty_analysis: NoveltyAssessment
    confidence_notes: str

```

### `shared/contracts/pipeline_contract.py`

```python
"""Top-level input/output contract for the full four-stage research pipeline.

This is the orchestrator's own contract, one level above the four
per-service contracts — see PROJECT_NOTES.md's "Inputs" section for the
user-facing fields this maps from.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from shared.contracts.discovery_contract import DiscoveryResult
from shared.contracts.qa_contract import QualityAssuranceResult
from shared.contracts.verification_contract import VerificationResult
from shared.contracts.trend_contract import TrendAdvisorResult
from shared.contracts.writing_contract import TargetFormat, WritingResult

PublicationType = Literal["conference", "journal", "other"]
# ask: an interactive surface (CLI, web page) asked the user before the run
# and recorded the outcome in `clarification`. auto: the pipeline asks the
# model itself and takes the suggestion. skip: no clarify step — the default,
# so headless callers and scripts keep their question unchanged.
ClarifyMode = Literal["ask", "skip", "auto"]
# ask: the run pauses after Discovery so the user can drop papers (the web
# page sets it; the interactive CLI always asks). auto: approved as found —
# the default, so headless runs, scripts and tests never stall (#17).
ApprovalMode = Literal["ask", "auto"]
# The budget check before Discovery (#7). auto: the full plan when the known
# OpenAlex balance covers the run, else the reduced plan (verification
# searches use Crossref) — the default, so headless runs never stall.
# full / reduced: the user's choice, made on an interactive surface or a flag.
BudgetPlan = Literal["auto", "full", "reduced"]


class Clarification(BaseModel):
    """What the clarify step saw and decided. The chosen question is the
    request's research_question; this keeps the original next to it."""
    original_question: str
    decision: Literal["accept", "edit", "keep"]
    # None when the model gave no usable answer (the original was kept).
    specific: bool | None = None
    missing: list[str] = Field(default_factory=list)
    suggested_question: str = ""


class ResearchRequest(BaseModel):
    """The single request that starts a pipeline run."""
    research_question: str
    publication_type: PublicationType = "conference"
    target_venue: str | None = None
    target_format: TargetFormat = "IEEE"
    format_other_name: str | None = None
    deadline: str | None = None
    corpus_size: int = Field(default=8, ge=6, le=9)
    domain: str | None = None
    year_range: tuple[int, int] | None = None
    preferred_databases: list[str] = Field(default_factory=list)
    language: str = "en"
    max_draft_length: int | None = Field(default=None, gt=0)
    keywords: list[str] = Field(default_factory=list)
    excluded_topics: list[str] = Field(default_factory=list)
    clarify: ClarifyMode = "skip"
    clarification: Clarification | None = None
    approval: ApprovalMode = "auto"
    # Discovery reuses the research library: stored discussion excerpts and
    # matching past papers, still screened by the relevance gate (#19).
    # False (`--no-memory`) never reads the library.
    memory: bool = True
    budget_plan: BudgetPlan = "auto"

    @model_validator(mode="after")
    def _require_format_other_name(self) -> "ResearchRequest":
        if self.target_format == "Other" and not self.format_other_name:
            raise ValueError("format_other_name is required when target_format is 'Other'")
        return self


class StageTimings(BaseModel):
    discovery_seconds: float
    writing_seconds: float
    verification_seconds: float
    quality_assurance_seconds: float


class PipelineResult(BaseModel):
    """Aggregate output of one full pipeline run."""
    run_id: str
    run_directory: str
    request: ResearchRequest
    # Present only when the run was started from the optional Trend & Gap
    # Advisor (pipeline stage 0) rather than from a question the user typed.
    # Carried so a finished run answers "why this topic?" as well as "what
    # did it produce?" — see DECISIONS.md D-031.
    trend_advisor: TrendAdvisorResult | None = None
    discovery: DiscoveryResult
    writing: WritingResult
    verification: VerificationResult
    quality_assurance: QualityAssuranceResult
    timings: StageTimings

```

### `shared/contracts/qa_contract.py`

```python
"""Input/output contract for the Quality Assurance Service (pipeline stage 4)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from shared.contracts.discovery_contract import DiscoveryResult
from shared.contracts.verification_contract import VerificationResult
from shared.contracts.writing_contract import WritingResult


class QualityAssuranceRequest(BaseModel):
    """Input to the Quality Assurance Service — all three prior stages'
    outputs, since an independent audit needs the original evidence
    (Discovery), the draft (Writing), and the citation checks (Verification)
    together."""
    discovery: DiscoveryResult
    writing: WritingResult
    verification: VerificationResult


class UnsupportedClaim(BaseModel):
    """A cited sentence whose wording does not overlap enough with its
    cited paper's title/abstract to be considered evidence-aligned."""
    sentence: str
    citation_numbers: list[int]


class MissingCitation(BaseModel):
    """A sentence that reads as a factual/empirical claim but cites no source."""
    sentence: str
    section: str


class QualityScores(BaseModel):
    citation_integrity: float = Field(ge=0.0, le=5.0)
    claim_source_alignment: float = Field(ge=0.0, le=5.0)
    process_control: float = Field(ge=0.0, le=5.0)
    literature_coverage: float = Field(ge=0.0, le=5.0)
    overall: float = Field(ge=0.0, le=5.0)


class QualityAssuranceResult(BaseModel):
    """Output of the Quality Assurance Service — the final pipeline artifact."""
    final_draft_markdown: str
    final_references: list[str] = Field(default_factory=list)
    final_validation_report: str
    quality_report_markdown: str
    scores: QualityScores
    unsupported_claims: list[UnsupportedClaim] = Field(default_factory=list)
    missing_citations: list[MissingCitation] = Field(default_factory=list)
    missing_sections: list[str] = Field(default_factory=list)
    known_limitations: list[str] = Field(default_factory=list)
    audit_metadata: dict[str, Any] = Field(default_factory=dict)

```

### `shared/contracts/trend_contract.py`

```python
"""Input/output contract for the Trend & Gap Advisor Service (pipeline stage 0).

Stage 0 runs *before* Research Discovery and is entirely optional: it turns
"which topic should I even research?" into a ranked, evidence-backed
shortlist, whose chosen entry then auto-fills a ResearchRequest. Defined
here — beside the four existing per-service contracts — so the advisor
service, the advisor CLI, and any future consumer depend on a stable shape
rather than on the service's internals.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

DomainChoice = Literal[
    "CS_AI_ML", "ECE", "EEE", "MECHANICAL", "CIVIL", "CLOUD_DEVOPS", "Other",
]

# Which signal put a topic on the shortlist. "gap_mining" is declared but
# not produced by the current pipeline: every candidate originates from the
# velocity step, and gap mining only annotates those, so a topic can be
# "velocity" or (once gaps corroborate it) "both". It is kept in the literal
# because a future gap-first discovery path would produce it, and widening a
# persisted enum later is a breaking change for every result.json already on
# disk, whereas an unused value costs nothing.
CandidateSource = Literal["velocity", "gap_mining", "both"]

DEFAULT_SHORTLIST_SIZE = 8


class TrendAdvisorRequest(BaseModel):
    """Input to the Trend & Gap Advisor Service."""

    domain: DomainChoice
    # Mirrors WritingRequest.target_format/format_other_name exactly: a domain
    # outside the curated picklist is still usable, it just loses the verified
    # OpenAlex subfield mapping (see trend_advisor/velocity.py::DOMAIN_SUBFIELDS)
    # and falls back to a free-text OpenAlex search, which is less precise.
    domain_other_name: str | None = None
    # 6-9 matches DiscoveryRequest.corpus_size's own convention — the shortlist
    # is meant to be scannable in one screen, not exhaustive.
    shortlist_size: int = Field(default=DEFAULT_SHORTLIST_SIZE, ge=6, le=9)

    @model_validator(mode="after")
    def _require_domain_other_name(self) -> "TrendAdvisorRequest":
        if self.domain == "Other" and not self.domain_other_name:
            raise ValueError("domain_other_name is required when domain is 'Other'")
        return self

    @property
    def domain_label(self) -> str:
        return self.domain_other_name or self.domain


class ExamplePaper(BaseModel):
    """One real, independently checkable paper behind a topic's ranking."""

    title: str
    doi: str | None = None
    year: int | None = None
    citations: int | None = None
    # Carried so the gap-mining swarm can read each paper's own words without
    # re-fetching it; the CLI never prints these, it only prints title/DOI.
    abstract: str | None = None
    # The paper's own Discussion/Limitations/Future-research text, when it
    # could be retrieved from an open-access copy. This — not the abstract —
    # is where authors actually state what remains unsolved, so gap mining
    # prefers it and falls back to the abstract only when it is absent.
    future_text: str | None = None


# Threshold for the "hot but crowded" vs "under-explored gap" split (user
# story 5). Chosen from the real OpenAlex distribution probed while building
# this stage: within one engineering subfield pair over a 3-year window, the
# head topics run 30k-50k works and the long tail runs in the hundreds, so a
# few thousand separates "everyone is already here" from "quiet corner".
CROWDED_PAPER_COUNT = 3000


class TopicCandidate(BaseModel):
    """One shortlisted topic plus every number and quote it was ranked on.

    Every field here is computed from live OpenAlex/arXiv data — nothing is
    asserted by the model. A student must be able to re-run the same queries
    and get the same evidence (user stories 3 and 13).
    """

    topic: str
    # OpenAlex topic id (e.g. "T10028"), so the evidence is re-queryable.
    topic_id: str | None = None
    # recent_papers / max(prior_papers, 1) — >1.0 means the topic is growing.
    growth_metric: float
    # Works published in the recent window; the "crowded vs quiet" axis that
    # growth_metric alone cannot express (user story 5).
    paper_count: int
    # The same count for the preceding window. Together with paper_count this
    # is the whole growth calculation, shown so a student can redo the
    # division themselves.
    prior_papers: int = 0
    example_papers: list[ExamplePaper] = Field(default_factory=list)
    # True when independent papers' own future-work/limitations sections
    # repeatedly flag this as unmet — a different signal from publication
    # volume, and the one that marks a genuine gap rather than a crowd.
    gap_signal: bool = False
    # One concise statement per gap a paper raised, in the extracting model's
    # words rather than verbatim quotation (see gap_mining.GAP_SYSTEM_PROMPT).
    # A topic can contribute several from the same paper, so the count here is
    # not the number of papers that flagged it.
    gap_evidence: list[str] = Field(default_factory=list)
    # Secondary, CS-adjacent-domains-only signal; None when not applicable
    # or when arXiv was unreachable (never silently zero — see spec's
    # "visible, honest failure" requirement).
    arxiv_recent_count: int | None = None
    source: CandidateSource = "velocity"

    @property
    def is_crowded(self) -> bool:
        """A high-volume topic: competing here means competing with many."""
        return self.paper_count >= CROWDED_PAPER_COUNT

    @property
    def signal_label(self) -> str:
        """The one-phrase reading of this topic's signals (user story 5).

        Lives here so the CLI table and the smoke runner cannot disagree
        about what a given combination of flags is called.
        """
        if self.gap_signal:
            return "recurring gap"
        return "hot but crowded" if self.is_crowded else "under-explored"


class TrendAdvisorResult(BaseModel):
    """Output of the Trend & Gap Advisor Service."""

    domain: DomainChoice
    domain_other_name: str | None = None
    shortlist: list[TopicCandidate]
    # The topic the student locked in, when they got that far. Named rather
    # than implied by shortlist position: the orchestrator records this as
    # the run's chosen topic, and a positional convention shared across two
    # modules is one reorder away from attributing the wrong topic to the
    # student with nothing failing. None when the shortlist was produced but
    # no pick was made (user story 10 — stopping to think it over).
    chosen_topic: str | None = None
    generated_at: str
    # Surfaced, not swallowed: an unreachable arXiv or a thin domain must be
    # visible to the student rather than presented as confident output.
    warnings: list[str] = Field(default_factory=list)

    @property
    def domain_label(self) -> str:
        """What to call this domain in output — the same rule as on the
        request, so a result read back from disk labels itself identically."""
        return self.domain_other_name or self.domain

    @property
    def gap_flagged_count(self) -> int:
        return sum(1 for c in self.shortlist if c.gap_signal)

    @property
    def chosen_candidate(self) -> TopicCandidate | None:
        """The shortlist entry the student locked in, matched by name.

        None when nothing was picked, and also when the named topic is not
        in this shortlist — callers must not report a stranger's evidence
        as the chosen topic's.
        """
        return next((c for c in self.shortlist if c.topic == self.chosen_topic), None)

```

### `shared/contracts/verification_contract.py`

```python
"""Input/output contract for the Citation Verification Service (pipeline stage 3)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from shared.contracts.writing_contract import WritingResult


class VerificationRequest(BaseModel):
    """Input to the Citation Verification Service — Service 2's output."""
    writing: WritingResult


class ReferenceCheck(BaseModel):
    """One reference-list entry's verification outcome."""
    reference_entry: str
    doi: str | None = None
    verified: bool
    verification_source: Literal["Crossref", "OpenAlex", "none"] = "none"
    issue: str | None = None


class VerificationResult(BaseModel):
    """Output of the Citation Verification Service — direct input to Service 4."""
    validated_draft_markdown: str
    verified_references: list[ReferenceCheck] = Field(default_factory=list)
    invalid_references: list[ReferenceCheck] = Field(default_factory=list)
    unverifiable_references: list[ReferenceCheck] = Field(default_factory=list)
    missing_references: list[str] = Field(default_factory=list)
    citation_issues: list[str] = Field(default_factory=list)
    validation_report: str

```

### `shared/contracts/writing_contract.py`

```python
"""Input/output contract for the Research Writing Service (pipeline stage 2)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from shared.contracts.discovery_contract import DiscoveryResult

TargetFormat = Literal["IEEE", "Springer", "ACM", "APA", "Other"]

# Extensible: a target format not listed here can still be passed as "Other"
# with format_other_name set — see WritingRequest.
#
# Springer -> "vancouver" (numbered in-text citations, e.g. "(1)", with a
# numbered reference list), not an author-year style. This was a real,
# verified fix (see PAPER_OUTPUT_FINAL_DIAGNOSTIC.md finding #4 /
# DECISIONS.md D-028): the LaTeX template actually used for "Springer"
# (`llncs` — Springer's real, standard Lecture Notes in Computer Science
# class; see latex_export.py) documents numbered citations as its
# convention, but this mapping previously paired it with
# "chicago-author-date" (parenthetical author-year), a mismatched
# combination that was never wired to any actual Springer requirement — it
# was simply the wrong entry from the day this table was first written.
# "vancouver" was already implemented and declared in
# SUPPORTED_CITATION_STYLES but had never been reachable through any real
# target_format before this fix (zero real usage, zero test coverage) —
# both are added here.
_FORMAT_TO_CITATION_STYLE = {
    "IEEE": "ieee",
    "Springer": "vancouver",
    "ACM": "elsevier-harvard",
    "APA": "apa-7",
    "Other": "elsevier-harvard",
}


class WritingRequest(BaseModel):
    """Input to the Research Writing Service — Service 1's output plus the
    venue/format details from the original research request."""
    discovery: DiscoveryResult
    target_format: TargetFormat = "IEEE"
    format_other_name: str | None = None
    output_language: str = "en"
    target_words: int | None = Field(default=None, gt=0)

    @property
    def citation_style(self) -> str:
        return _FORMAT_TO_CITATION_STYLE[self.target_format]


class CitationEntry(BaseModel):
    paper_id: str
    citation_key: str
    in_text_citation: str
    narrative_citation: str
    reference_entry: str


class DraftMetadata(BaseModel):
    """Required disclosure that this is AI-generated synthesis, not a
    verified publication — see PROJECT_NOTES.md safety rules."""
    disclaimer: str = (
        "AI-generated research draft for human review. Not a verified or "
        "published research paper. Claims must be checked against cited "
        "sources before any external use."
    )
    model_name: str
    run_id: str
    generated_at: str
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    papers_cited: int = 0
    sections_written: int = 0
    total_sections: int = 0
    failed_sections: int = 0
    # Explicit completeness classification (see DECISIONS.md D-019, revised
    # by D-026's real-rendered-paper completeness check). This is computed
    # from the ACTUAL RENDERED draft text (services/research-writing/
    # completeness.py), not just the writing graph's internal tag-resolution
    # signal — see that module's docstring for why the two can disagree.
    # "complete": every required section (bookends + outline sections) is
    # present with real, non-placeholder prose (short-but-evidence-limited
    # sections still count as complete — see length_status/
    # evidence_limited_sections below for that distinction, per Fix 7).
    # "partial": the draft is usable but at least one required section is
    # missing, blank, or a failure placeholder.
    # "failed": every required section is broken, or both bookends
    # (Introduction and Conclusion) are — the document is not usable.
    draft_status: Literal["complete", "partial", "failed"] = "complete"
    # Length planning/analytics (Fix 3/7/8, DECISIONS.md D-026). All are
    # None/empty when no length target was requested, preserving the old
    # length-unaware behavior for any caller that doesn't set
    # ResearchRequest.max_draft_length.
    target_words: int | None = None
    actual_words: int = 0
    # "short" / "on_target" / "over_target" — see word_budget.classify_length.
    # None when no target_words was requested.
    length_status: Literal["short", "on_target", "over_target"] | None = None
    # Section titles (as rendered, e.g. "Introduction", "Literature Review")
    # whose content is real but shorter than planned because the available
    # evidence was genuinely too thin to develop further — never because the
    # model was told to pad or fabricate anything (Fix 9 "Case B"). Tracked
    # separately from failed_sections so a short-but-honest paper is never
    # confused with a broken one.
    evidence_limited_sections: list[str] = Field(default_factory=list)
    # Section titles that are missing, blank, or rendered as a genuine
    # failure placeholder (as opposed to evidence-limited) in the final
    # document — the human-readable counterpart to `failed_sections`'s count.
    broken_sections: list[str] = Field(default_factory=list)
    duplicate_sections: list[str] = Field(default_factory=list)


class WritingResult(BaseModel):
    """Output of the Research Writing Service — direct input to Service 3."""
    research_outline: str
    research_draft_markdown: str
    citation_mapping: dict[str, CitationEntry] = Field(default_factory=dict)
    reference_candidates: list[str] = Field(default_factory=list)
    run_directory: str
    draft_metadata: DraftMetadata

```

### `researchgenie/headless.py`

```python
"""Headless, JSON-speaking CLI (#10), for agents, scripts and the skill.

    researchgenie run -q "question" [--every-request-field ...] [--length short]
                      [--no-cache] [--no-memory] [--detach] [--run-id ID] [--json]
    researchgenie status <run id or folder> [--json]
    researchgenie result <run id or folder> [--json]
    researchgenie resume <run id or folder> [--from STAGE] (--json | --detach)
    researchgenie doctor <run id or folder> [--yes] [--json]
    researchgenie library search <query> --json

Never prompts. Progress goes to stderr, the answer to stdout (JSON with
--json). `run --detach` starts the same command in a background process and
prints the run id at once. Command lines are fixed: no shell, no inline
code, user input only ever travels as argv items or pydantic-validated
fields. Exit codes (EXIT_CODES) name the failure class from
orchestrator/doctor.py's rule table.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import typing
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from pydantic import ValidationError
from rich.console import Console
from rich.prompt import Confirm

from orchestrator import decision_brief, doctor
from orchestrator.pipeline import (
    DEFAULT_OUTPUT_ROOT, FINAL_FILES, STAGE_NAMES, find_run_directory, resume_pipeline,
    run_pipeline,
)
from researchgenie import config as rg_config
from shared.contracts.pipeline_contract import ResearchRequest
from shared.utilities.scholarly_gateway import redact

EXIT_OK = 0
EXIT_FAILED = 1            # a failure with no more specific class
EXIT_USAGE = 2             # bad arguments (argparse uses 2 as well)
EXIT_MODEL_UNAVAILABLE = 3
EXIT_PROVIDERS_EXHAUSTED = 4
EXIT_NO_RELEVANT_PAPERS = 5
EXIT_NOT_FOUND = 6         # no run with that id or folder
EXIT_NOT_FINISHED = 7      # `result` asked of a run still going
EXIT_CODES = {"model_unavailable": EXIT_MODEL_UNAVAILABLE,
              "providers_exhausted": EXIT_PROVIDERS_EXHAUSTED,
              "no_relevant_papers": EXIT_NO_RELEVANT_PAPERS}

COMMANDS = ("run", "status", "result", "doctor")
# ResearchRequest fields a flag can't set: the clarify outcome is recorded by
# an interactive surface, never typed in.
_NOT_FLAGS = {"clarification"}


def _year_range(text: str) -> list[int]:
    start, sep, end = text.partition("-")
    if not (sep and start.isdigit() and end.isdigit()):
        raise argparse.ArgumentTypeError("use START-END, e.g. 2015-2024")
    return [int(start), int(end)]


def _run_id(text: str) -> str:
    """A caller-chosen run id becomes part of a folder name: keep it plain."""
    if not re.fullmatch(r"[A-Za-z0-9-]{1,64}", text):
        raise argparse.ArgumentTypeError("use 1-64 letters, digits and hyphens")
    return text


def _add_request_options(parser: argparse.ArgumentParser) -> None:
    """One flag per ResearchRequest field, so the CLI never falls behind the
    contract; pydantic validates the values."""
    for name, info in ResearchRequest.model_fields.items():
        if name in _NOT_FLAGS:
            continue
        flags = ["--" + name.replace("_", "-")]
        options: dict[str, Any] = {"dest": name, "default": None}
        if name == "research_question":
            flags = ["-q", "--question", *flags]
        elif name == "target_format":
            flags = ["--format", *flags]
        if name == "year_range":
            options.update(type=_year_range, metavar="START-END")
        elif info.annotation is bool:  # a default-on switch: --no-<name> turns it off
            flags = ["--no-" + name.replace("_", "-")]
            options["action"] = "store_false"
        elif typing.get_origin(info.annotation) is list:
            options.update(action="append", metavar=name.rstrip("s").upper(),
                           help="repeat for more than one")
        elif typing.get_origin(info.annotation) is typing.Literal:
            # "ask" (clarify, approval) needs someone to answer; never headless.
            options["choices"] = [c for c in typing.get_args(info.annotation) if c != "ask"]
        parser.add_argument(*flags, **options)
    parser.add_argument("--length", choices=("short", "standard", "detailed"),
                        help="a paper length preset (sets --max-draft-length)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="researchgenie", description="Headless ResearchGenie.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="start a research run without prompts")
    _add_request_options(run)
    run.add_argument("--detach", action="store_true", help="print the run id and return at once")
    run.add_argument("--run-id", type=_run_id,
                     help="choose the run id, so `status` works while the run is going")

    for name, text in (("status", "stage, state and last event"),
                       ("result", "final files, scores, verification counts, draft status")):
        sub = commands.add_parser(name, help=text)
        sub.add_argument("run", help="run id or run folder")

    resume = commands.add_parser("resume", help="resume a failed run from the failed stage")
    resume.add_argument("run", help="run id or run folder")
    resume.add_argument("--from", dest="from_stage", choices=STAGE_NAMES)
    resume.add_argument("--detach", action="store_true")

    doctor_cmd = commands.add_parser("doctor", help="diagnose a run; apply the fix once confirmed")
    doctor_cmd.add_argument("run", help="run id or run folder")
    doctor_cmd.add_argument("--yes", action="store_true",
                            help="apply the proposed fix without asking (required with --json)")

    library_cmd = commands.add_parser("library", help="search the research library")
    library_cmd.add_argument("action", choices=["search"])
    library_cmd.add_argument("query", nargs="+")

    for sub in commands.choices.values():
        sub.add_argument("--json", action="store_true", help="machine-readable output")
        sub.add_argument("--no-cache", action="store_true",
                         help="don't answer scholarly requests from the local cache")
    return parser


def is_headless(args: list[str]) -> bool:
    """run/status/result/doctor always; resume only with --json or --detach,
    library only with --json (bare, they stay the interactive commands)."""
    if args[:1] == ["resume"]:
        return "--json" in args or "--detach" in args
    if args[:1] == ["library"]:
        return "--json" in args
    return bool(args) and args[0] in COMMANDS


def _request(args: argparse.Namespace) -> ResearchRequest:
    fields = {name: value for name, value in vars(args).items()
              if name in ResearchRequest.model_fields and value is not None}
    if args.length and "max_draft_length" not in fields:
        from researchgenie.cli import _LENGTH_PRESETS
        fields["max_draft_length"] = _LENGTH_PRESETS[args.length]
    return ResearchRequest(**fields)


def _resolve(target: str) -> Path | None:
    path = Path(target)
    return path if (path / "events.jsonl").is_file() or (path / "00_request.json").is_file() \
        else find_run_directory(target, DEFAULT_OUTPUT_ROOT)


def _detached_log(run_id: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / f".detached-{run_id}.log"


def status_of(run_directory: Path) -> dict[str, Any]:
    events = doctor.read_events(run_directory)
    state = "running"
    for event in events:  # the last word wins: a resume logs progress after an error
        if event.get("type") == "result":
            state = "completed"
        elif event.get("type") == "error":
            state = "failed"
        elif event.get("type") == "progress":
            state = "running"
    last = next((e for e in reversed(events) if e.get("type") in ("progress", "error")), {})
    # An error line has no stage of its own: it failed in the last one that ran.
    stage = next((e["stage"] for e in reversed(events) if e.get("stage")), None)
    status: dict[str, Any] = {
        "run_id": run_directory.name.partition("_")[2], "state": state,
        "stage": stage, "run_directory": str(run_directory),
        "last_event": {k: last[k] for k in ("type", "stage", "status", "message", "timestamp")
                       if k in last},
    }
    if state == "failed":
        rule = doctor.match(events)
        status["error"] = doctor.last_error(events)
        status["failure_class"] = rule.failure_class if rule else "unknown"
    return status


def result_of(run_directory: Path) -> dict[str, Any]:
    status = status_of(run_directory)
    events = doctor.read_events(run_directory)
    final = next((e["result"] for e in reversed(events) if e.get("type") == "result"), None)
    out = {**status, "final_files": {name: str(run_directory / "final" / name)
                                     for name in FINAL_FILES
                                     if (run_directory / "final" / name).is_file()}}
    if status["state"] == "completed" and final:
        draft = final["writing"]["draft_metadata"]
        verification = final["verification"]
        out.update(
            scores=final["quality_assurance"]["scores"],
            verification={"verified": len(verification["verified_references"]),
                          "invalid": len(verification["invalid_references"]),
                          "unverifiable": len(verification["unverifiable_references"])},
            draft_status=draft["draft_status"], length_status=draft.get("length_status"),
            actual_words=draft.get("actual_words"), target_words=draft.get("target_words"),
        )
    if status["state"] != "running":  # the one next action (#12); a failed run's points to the Doctor
        brief = decision_brief.build_brief(run_directory)
        path = decision_brief.find_brief(run_directory)
        out["decision_brief"] = {"path": str(path) if path else None,
                                 "next_action": brief["next_action"], "markdown": brief["markdown"]}
    return out


def _exit_code(report: dict[str, Any]) -> int:
    if report["state"] == "completed":
        return EXIT_OK
    if report["state"] == "failed":
        return EXIT_CODES.get(report.get("failure_class"), EXIT_FAILED)
    return EXIT_NOT_FINISHED


def _print(data: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return
    for key, value in data.items():
        shown = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
        print(f"{key}: {shown}")


async def _drive(events) -> str | None:
    """Runs an event stream to the end, progress to stderr. The error message, or None."""
    try:
        async for event in events:
            if event.get("type") == "progress":
                print(f"[{event['stage']}] {event['message']}", file=sys.stderr, flush=True)
    except Exception as error:  # PipelineError, a model error, or anything a stage raised
        return redact(str(error)) or type(error).__name__
    return None


def _detach(argv: list[str], run_id: str) -> None:
    """Starts this same command in the background, without --detach."""
    rest = [a for a in argv if a != "--detach"]
    if rest[:1] == ["run"]:
        rest += ["--run-id", run_id]
    elif "--json" not in rest:
        rest.append("--json")  # without it the child would be the interactive resume
    child = [sys.executable, "-m", "researchgenie.cli", *rest]
    log = _detached_log(run_id)
    log.parent.mkdir(parents=True, exist_ok=True)
    options: dict[str, Any] = {"stdin": subprocess.DEVNULL, "stdout": log.open("ab"),
                               "stderr": subprocess.STDOUT}
    if sys.platform == "win32":
        options["creationflags"] = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                                    | subprocess.CREATE_NO_WINDOW)
    else:
        options["start_new_session"] = True
    subprocess.Popen(child, **options)


def _ready() -> bool:
    """The model is usable now (starting Ollama if it isn't running), with
    the checks' messages on stderr."""
    from researchgenie.cli import _ensure_ready
    return _ensure_ready(Console(stderr=True, legacy_windows=False), rg_config.load())


def _key_setup(fix_id: str) -> Callable[[], bool]:
    """The interactive key prompt a key fix runs (the same steps as `researchgenie config`)."""
    def setup() -> bool:
        from researchgenie.setup_wizard import run_setup_wizard, setup_scholarly_access
        console = Console(legacy_windows=False)
        if fix_id == "gemini_setup":
            config = run_setup_wizard(console)
        else:
            config = rg_config.load()
            setup_scholarly_access(console, config)
            rg_config.save(config)
        rg_config.apply_to_environment(config)
        return config.setup_complete or fix_id != "gemini_setup"
    return setup


async def _doctor(directory: Path, args: argparse.Namespace) -> int:
    """Diagnose, ask (or take --yes), apply, resume; the report records each step."""
    diagnosis = doctor.diagnose(directory, await doctor.system_status())
    fix = diagnosis["fix"]
    if not args.json:
        print(f"Cause: {diagnosis['cause']}")
        for line in diagnosis["evidence"]:
            print(f"  {line}")
        if fix:
            print(f"Proposed fix: {fix['label']}")
    if fix is None:
        applied, outcome = False, "No fix proposed."
    elif not fix["automatic"]:
        applied, outcome = False, f"No automatic fix; to do: {fix['label']}"
    elif not (args.yes or (not args.json and Confirm.ask("Apply this fix?", default=False))):
        applied, outcome = False, "Not applied: not confirmed" + (" (headless needs --yes)." if args.json else ".")
    else:
        problem = await doctor.apply_fix(
            diagnosis, key_setup=None if args.json else _key_setup(fix["id"]))
        if problem:
            applied, outcome = False, problem
        else:
            await _drive(doctor.resume_and_record(directory, diagnosis, resume_pipeline))
            report = json.loads((directory / doctor.REPORT_NAME).read_text(encoding="utf-8"))
            _print({**report, "report_path": str(directory / doctor.REPORT_NAME)}, args.json)
            return _exit_code(result_of(directory))
    path = doctor.write_report(directory, diagnosis, applied=applied, outcome=outcome)
    _print({**diagnosis, "applied": applied, "outcome": outcome, "report_path": str(path)}, args.json)
    return _exit_code(result_of(directory))


def main(argv: list[str]) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as stop:  # argparse already printed why
        return EXIT_USAGE if stop.code else EXIT_OK
    if args.no_cache:
        os.environ["RESEARCHGENIE_NO_CACHE"] = "1"
    rg_config.apply_scholarly(rg_config.load())

    def fail(code: int, message: str, **extra) -> int:
        _print({"error": message, **extra}, args.json)
        return code

    if args.command == "run":
        try:
            request = _request(args)
        except ValidationError as error:
            return fail(EXIT_USAGE, str(error))
        run_id = args.run_id or uuid4().hex
        if args.run_id and find_run_directory(run_id, DEFAULT_OUTPUT_ROOT):
            return fail(EXIT_USAGE, f"Run id {run_id!r} is already used.")
        if args.detach:
            _detach(argv, run_id)
            _print({"run_id": run_id, "state": "started",
                    "log": str(_detached_log(run_id))}, args.json)
            return EXIT_OK
        if not _ready():
            return fail(EXIT_MODEL_UNAVAILABLE, "The configured model is not ready.",
                        run_id=run_id, failure_class="model_unavailable")
        error = asyncio.run(_drive(run_pipeline(request, run_id=run_id)))
        directory = find_run_directory(run_id, DEFAULT_OUTPUT_ROOT)
        if directory is None:  # failed before the run folder existed
            return fail(EXIT_FAILED, error or "The run left no folder.", run_id=run_id)
        report = result_of(directory)
        _print(report, args.json)
        return _exit_code(report)

    if args.command == "library":
        from shared.utilities import library
        query = " ".join(args.query)
        _print({"runs": library.search_runs(query), "papers": library.search_papers(query)}, args.json)
        return EXIT_OK

    directory = _resolve(args.run)
    if directory is None:
        if args.command == "status" and _detached_log(args.run).is_file():
            _print({"run_id": args.run, "state": "starting",
                    "log": str(_detached_log(args.run))}, args.json)
            return EXIT_OK
        return fail(EXIT_NOT_FOUND, f"No run {args.run!r} under {DEFAULT_OUTPUT_ROOT}.")

    if args.command == "doctor":
        return asyncio.run(_doctor(directory, args))
    if args.command == "status":
        _print(status_of(directory), args.json)
        return EXIT_OK
    if args.command == "result":
        report = result_of(directory)
        _print(report, args.json)
        return _exit_code(report)

    # resume
    if args.detach:
        _detach(argv, directory.name.partition("_")[2])
        _print({"run_id": directory.name.partition("_")[2], "state": "resuming"}, args.json)
        return EXIT_OK
    if not _ready():
        return fail(EXIT_MODEL_UNAVAILABLE, "The configured model is not ready.",
                    failure_class="model_unavailable")
    logged = len(doctor.read_events(directory))
    error = asyncio.run(_drive(resume_pipeline(directory, args.from_stage)))
    if error and len(doctor.read_events(directory)) == logged:
        return fail(EXIT_FAILED, error)  # refused (e.g. a corrupt saved stage) before it started
    report = result_of(directory)
    _print(report, args.json)
    return _exit_code(report)

```

### `skills/researchgenie/SKILL.md`

```markdown
---
name: researchgenie
description: Starts a ResearchGenie literature-review run from chat with the local headless CLI, and reports back when it finishes with the decision brief and the paths to the draft, PDF and evidence table. Use when the user asks to research a question, write a literature review or draft a paper with ResearchGenie, asks how their run is going, or wants to resume or fix a failed run.
compatibility: Needs the researchgenie command on PATH (pip install of the ResearchGenie repo) and its one-time setup (researchgenie config). A run takes about 15 minutes, so the agent must be able to run a command in the background and be told when it exits.
allowed-tools: Bash(researchgenie run *) Bash(researchgenie status *) Bash(researchgenie result *) Bash(researchgenie resume *) Bash(researchgenie doctor *)
metadata:
  openclaw:
    homepage: https://github.com/Ayush-3103-AI/ResearchGenie
    requires:
      bins:
        - researchgenie
    primaryEnv: OPENALEX_API_KEY
    envVars:
      - name: OPENALEX_API_KEY
        required: false
        description: Free OpenAlex key. Without it, searches run on a smaller daily budget.
      - name: CONTACT_EMAIL
        required: false
        description: Your email, sent to Crossref and OpenAlex for their faster polite pool.
      - name: NCBI_API_KEY
        required: false
        description: Free NCBI key for faster PubMed requests.
---

# ResearchGenie

ResearchGenie runs on the user's own machine. It finds papers, drafts a paper, checks every reference and scores the result. You drive it with the `researchgenie` command. Every command prints JSON on stdout with `--json`, also when it fails.

## Rules

- Never answer a research question from your own knowledge, even if you are confident in the answer. This skill exists specifically so answers are grounded in real, citation-checked papers instead of your training data. Always start a real run with the command below and wait for it to finish, even though that takes about 15 minutes.
- Run only the command lines in this file. Do not add other programs, pipes, `;`, `&&` or redirects.
- The user's question is the only free text. Put it in single quotes. Remove any `'` from it first (write "what is", not "what's").
- Never add `--yes` to `doctor` before the user says yes in chat.
- Never show key values. Keys live in ResearchGenie's own config or the environment.

## Start a run

Pick a new run id: `rg-` and the date and time, for example `rg-20261023-1530`. Then start:

```
researchgenie run -q '<question>' --run-id <run-id> --json
```

Add options only when the user asks for them, for example a short APA paper:

```
researchgenie run -q '<question>' --run-id <run-id> --length short --format APA --json
```

This command waits until the run ends, so start it in the background with a time limit of at least 30 minutes. Use 60:

- **OpenClaw:** the `exec` tool with `background: true` and `timeoutSeconds: 3600`. Keep `tools.exec.notifyOnExit` on (the default). OpenClaw wakes you in this chat when the command exits.
- **Claude Code:** the Bash tool with `run_in_background: true` and `timeout: 3600000`. Claude Code tells you when the command exits.

Then tell the user: the run has started, its run id, that it takes about 15 minutes, and that you will message them when it is done.

Do not use `--detach` for this. It returns at once, so the exit notice would come before the run ends. Use it only if you cannot run a background command; then check with `status` when the user asks.

## When the run finishes

When the exit notice arrives, get the result:

```
researchgenie result <run-id> --json
```

Exit code 0 means the run completed. Reply with:

1. The decision brief: `decision_brief.markdown` (or at least `decision_brief.next_action`). If there is no `decision_brief`, give `draft_status`, `scores` and `verification` instead.
2. The paths in `final_files`: `draft.md`, `paper.pdf` and `evidence_table.md`. If `paper.pdf` is missing, say so (LaTeX may not be installed).

Any other exit code: go to "If the run failed".

## "How is my run going?"

```
researchgenie status <run-id> --json
```

Answer with `state` (`running`, `completed` or `failed`), `stage` and `last_event.message`. If it says there is no run with that id in the first minute, the run is still starting (it checks the model first). Ask again a little later.

## If the run failed

`status` shows `error` and `failure_class`. Tell the user in one or two plain sentences, then offer two choices.

**Doctor (finds the cause and proposes one fix).** First diagnose only. This applies nothing:

```
researchgenie doctor <run-id> --json
```

Show `cause`, the `evidence` lines and `fix.label`. Ask the user if they want the fix. Only after they say yes, run this in the background, like a run (it resumes the run when the fix works):

```
researchgenie doctor <run-id> --yes --json
```

If `outcome` says the fix needs the user at a prompt or is a step for them (for example adding a key with `researchgenie config`), tell them to do it on their machine. Do not try it yourself.

**Resume (starts again from the failed stage).** Run it in the background, like a run:

```
researchgenie resume <run-id> --json
```

To restart from an earlier stage (`discovery`, `writing`, `verification` or `quality_assurance`):

```
researchgenie resume <run-id> --from writing --json
```

If the run stopped before it made a run folder (exit code 3 at the start), `doctor` and `resume` find no run. Tell the user to start Ollama or run `researchgenie config` on their machine, then start a new run.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success (for `result`: the run completed) |
| 1 | Failed, no more specific cause |
| 2 | Bad arguments, or the run id is already used |
| 3 | The model is unavailable (Ollama not running, model missing, Gemini key problem) |
| 4 | Every literature source was exhausted or failed |
| 5 | No relevant paper was found |
| 6 | No run with that id |
| 7 | `result` asked of a run that is still going |

```

### Real tests for this lab

### `tests/researchgenie/test_headless.py`

```python
"""The headless CLI (#10), driven through its own surface (argv in, stdout
JSON and an exit code out) over faked pipeline services."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "orchestrator"))

from orchestrator import pipeline as orchestrator_pipeline  # noqa: E402
from researchgenie import headless  # noqa: E402
from shared.utilities import llm_provider, response_cache  # noqa: E402
from test_pipeline import (  # noqa: E402
    _fake_run_discovery, _fake_run_qa, _fake_run_verification, _fake_run_writing,
)


@pytest.fixture(autouse=True)
def outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator_pipeline, "DEFAULT_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(headless, "DEFAULT_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(orchestrator_pipeline.discovery_service, "run_discovery", _fake_run_discovery)
    monkeypatch.setattr(orchestrator_pipeline.writing_service, "run_writing", _fake_run_writing)
    monkeypatch.setattr(orchestrator_pipeline.verification_service, "run_verification", _fake_run_verification)
    monkeypatch.setattr(orchestrator_pipeline.qa_service, "run_quality_assurance", _fake_run_qa)
    monkeypatch.setattr(headless, "_ready", lambda: True)
    return tmp_path


def _cli(capsys, *argv):
    code = headless.main(list(argv))
    out = capsys.readouterr().out
    return code, json.loads(out) if out.strip().startswith("{") else out


def _failing_discovery(message, exhausted=False):
    async def discovery(request, approve=None):
        if exhausted:
            yield {"type": "provider_exhausted", "host": "api.openalex.org",
                   "remedy": "Add a key", "message": "OpenAlex is out of budget"}
        if "Ollama" in message:
            raise llm_provider.LLMProviderError(message)
        raise orchestrator_pipeline.PipelineError(message)
        yield  # pragma: no cover
    return discovery


def test_run_takes_every_request_field_and_returns_the_result_as_json(capsys, outputs):
    code, report = _cli(
        capsys, "run", "-q", "What is X?", "--corpus-size", "7", "--format", "Springer",
        "--publication-type", "journal", "--target-venue", "Nature", "--deadline", "2026-12-01",
        "--domain", "biology", "--year-range", "2015-2024", "--preferred-databases", "pubmed",
        "--language", "en", "--length", "short", "--keywords", "fasting", "--keywords", "memory",
        "--excluded-topics", "mice", "--clarify", "skip", "--approval", "auto", "--no-memory",
        "--json")
    assert code == headless.EXIT_OK
    assert report["state"] == "completed" and report["draft_status"] == "complete"
    assert report["scores"]["overall"] == 5.0
    assert report["verification"] == {"verified": 0, "invalid": 0, "unverifiable": 0}
    assert Path(report["final_files"]["draft.md"]).is_file()

    saved = json.loads((Path(report["run_directory"]) / "00_request.json").read_text())
    assert saved["corpus_size"] == 7 and saved["target_format"] == "Springer"
    assert saved["year_range"] == [2015, 2024] and saved["keywords"] == ["fasting", "memory"]
    assert saved["excluded_topics"] == ["mice"] and saved["preferred_databases"] == ["pubmed"]
    assert saved["publication_type"] == "journal" and saved["target_venue"] == "Nature"
    assert saved["max_draft_length"] > 0 and saved["approval"] == "auto"
    assert saved["memory"] is False

    code, status = _cli(capsys, "status", report["run_id"], "--json")
    assert code == 0 and status["state"] == "completed"
    assert status["stage"] == "quality_assurance" and status["last_event"]["message"]


def test_a_headless_run_short_of_openalex_budget_takes_the_reduced_plan_and_says_so(capsys, monkeypatch):
    monkeypatch.setattr(orchestrator_pipeline.scholarly_gateway, "balance", lambda: 30)
    code = headless.main(["run", "-q", "What is X?", "--json"])
    err = capsys.readouterr().err
    assert code == headless.EXIT_OK
    assert "Running the reduced plan" in err and "OpenAlex key" in err


@pytest.mark.parametrize("message, exhausted, exit_code, failure_class", [
    ("Could not reach the local Ollama server at http://127.0.0.1:11434. Start it with: ollama serve",
     False, headless.EXIT_MODEL_UNAVAILABLE, "model_unavailable"),
    ("No papers found in any source. Try rephrasing the question or enabling more sources.",
     True, headless.EXIT_PROVIDERS_EXHAUSTED, "providers_exhausted"),
    ("No paper in the search results was judged relevant to the research question. Try again.",
     False, headless.EXIT_NO_RELEVANT_PAPERS, "no_relevant_papers"),
    ("Crossref went away.", False, headless.EXIT_FAILED, "unknown"),
])
def test_each_failure_class_has_its_own_exit_code(capsys, monkeypatch, message, exhausted,
                                                  exit_code, failure_class):
    monkeypatch.setattr(orchestrator_pipeline.discovery_service, "run_discovery",
                        _failing_discovery(message, exhausted))
    code, report = _cli(capsys, "run", "-q", "What is X?", "--json")
    assert code == exit_code
    assert report["state"] == "failed" and report["failure_class"] == failure_class
    assert report["error"] == message

    code, status = _cli(capsys, "status", report["run_id"], "--json")
    assert code == 0 and status["failure_class"] == failure_class
    code, _ = _cli(capsys, "result", report["run_id"], "--json")
    assert code == exit_code


def test_an_unready_model_stops_before_the_run(capsys, monkeypatch):
    monkeypatch.setattr(headless, "_ready", lambda: False)
    code, report = _cli(capsys, "run", "-q", "What is X?", "--json")
    assert code == headless.EXIT_MODEL_UNAVAILABLE and report["failure_class"] == "model_unavailable"


def test_detach_prints_the_run_id_and_starts_a_fixed_background_command(capsys, monkeypatch):
    started = []

    class FakePopen:
        def __init__(self, argv, **options):
            started.append((argv, options))

    monkeypatch.setattr(headless.subprocess, "Popen", FakePopen)
    code, report = _cli(capsys, "run", "-q", "What is X?; rm -rf /", "--detach", "--json")
    assert code == 0 and report["state"] == "started" and report["run_id"]
    [(argv, options)] = started
    assert argv[:3] == [sys.executable, "-m", "researchgenie.cli"]
    assert "--detach" not in argv and argv[-2:] == ["--run-id", report["run_id"]]
    assert "What is X?; rm -rf /" in argv  # one argv item, never parsed by a shell
    assert not options.get("shell")

    code, status = _cli(capsys, "status", report["run_id"], "--json")
    assert code == 0 and status["state"] == "starting"


def test_a_chosen_run_id_is_plain_and_used_once(capsys):
    code, report = _cli(capsys, "run", "-q", "What is X?", "--run-id", "rg-20261023-1530", "--json")
    assert code == 0 and report["run_id"] == "rg-20261023-1530"
    code, status = _cli(capsys, "status", "rg-20261023-1530", "--json")
    assert code == 0 and status["state"] == "completed"
    code, refused = _cli(capsys, "run", "-q", "What is Y?", "--run-id", "rg-20261023-1530", "--json")
    assert code == headless.EXIT_USAGE and "already used" in refused["error"]
    assert headless.main(["run", "-q", "What is X?", "--run-id", "../escape", "--json"]) == headless.EXIT_USAGE


def test_resume_is_available_headless(capsys, monkeypatch):
    async def verification_fails(request):
        yield {"type": "status", "stage": "verify", "state": "running", "message": "verifying"}
        raise orchestrator_pipeline.PipelineError("Crossref went away.")

    monkeypatch.setattr(orchestrator_pipeline.verification_service, "run_verification",
                        verification_fails)
    code, failed = _cli(capsys, "run", "-q", "What is X?", "--json")
    assert code == headless.EXIT_FAILED and failed["stage"] == "verification"

    monkeypatch.setattr(orchestrator_pipeline.verification_service, "run_verification",
                        _fake_run_verification)
    code, report = _cli(capsys, "resume", failed["run_id"], "--json")
    assert code == 0 and report["state"] == "completed"

    code, refused = _cli(capsys, "resume", failed["run_id"], "--json")  # nothing left to resume
    assert code == headless.EXIT_FAILED and "already has a saved result" in refused["error"]


def test_unknown_and_unfinished_runs(capsys, outputs):
    code, report = _cli(capsys, "status", "nope", "--json")
    assert code == headless.EXIT_NOT_FOUND and "nope" in report["error"]

    running = outputs / "what-is-x_r1"
    running.mkdir()
    (running / "events.jsonl").write_text(json.dumps(
        {"type": "progress", "stage": "writing", "status": "running", "message": "writing"}) + "\n")
    code, status = _cli(capsys, "status", "r1", "--json")
    assert code == 0 and status["state"] == "running" and status["stage"] == "writing"
    code, report = _cli(capsys, "result", str(running), "--json")
    assert code == headless.EXIT_NOT_FINISHED and report["final_files"] == {}


@pytest.mark.parametrize("argv", [
    ["run", "--json"],                                    # no question
    ["run", "-q", "Q?", "--approval", "ask"],             # nobody to ask
    ["run", "-q", "Q?", "--clarify", "ask"],
    ["run", "-q", "Q?", "--corpus-size", "40", "--json"],  # out of the contract's range
    ["run", "-q", "Q?", "--year-range", "2015"],
])
def test_bad_arguments_exit_2(capsys, argv):
    assert headless.main(argv) == headless.EXIT_USAGE


def test_library_search_speaks_json(capsys, outputs):
    code, report = _cli(capsys, "run", "-q", "Does sleep aid memory?", "--json")
    code, found = _cli(capsys, "library", "search", "sleep", "memory", "--json")
    assert code == 0 and [r["run_id"] for r in found["runs"]] == [report["run_id"]]


def test_which_command_lines_are_headless():
    assert headless.is_headless(["run", "-q", "Q"]) and headless.is_headless(["status", "x"])
    assert headless.is_headless(["resume", "x", "--json"])
    assert not headless.is_headless(["resume", "x"]) and not headless.is_headless([])
    assert not headless.is_headless(["library", "search", "x"])
    assert headless.is_headless(["library", "search", "x", "--json"])


def test_no_cache_and_cache_clear(capsys, monkeypatch):
    _cli(capsys, "status", "nope", "--no-cache")
    assert response_cache.reads_enabled() is False
    monkeypatch.delenv("RESEARCHGENIE_NO_CACHE")

    from researchgenie import cli
    monkeypatch.setattr(sys, "argv", ["researchgenie", "--no-cache", "status", "nope", "--json"])
    with pytest.raises(SystemExit) as stop:  # headless, not the interactive app
        cli.main()
    assert stop.value.code == headless.EXIT_NOT_FOUND and not response_cache.reads_enabled()
    monkeypatch.delenv("RESEARCHGENIE_NO_CACHE")

    response_cache.put("GET x?", 200, "text/plain", b"x", None, 0.0)
    from researchgenie import cli
    monkeypatch.setattr(sys, "argv", ["researchgenie", "cache", "clear"])
    cli.main()
    assert response_cache.get("GET x?", 0.0) is None

```

### `tests/test_agent_skill.py`

```python
"""The agent skill (#21) follows the Agent Skills spec, and every command
line in it is one the headless CLI accepts, so the skill can't drift."""

import re
import shlex
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from researchgenie import headless  # noqa: E402

SKILL_DIR = ROOT / "skills" / "researchgenie"
TEXT = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
PLACEHOLDERS = {"<question>": "What is X?", "<run-id>": "rg-20261023-1530"}


def test_frontmatter_follows_the_agent_skills_spec():
    _, front, body = TEXT.split("---\n", 2)
    meta = yaml.safe_load(front)
    assert set(meta) <= {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
    assert meta["name"] == SKILL_DIR.name
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", meta["name"]) and len(meta["name"]) <= 64
    assert 0 < len(meta["description"]) <= 1024
    assert 0 < len(meta["compatibility"]) <= 500
    assert meta["metadata"]["openclaw"]["requires"]["bins"] == ["researchgenie"]
    assert body.strip()


def test_every_command_line_parses_with_the_headless_cli():
    lines = [line for block in re.findall(r"```\n(.*?)```", TEXT, re.S) for line in block.splitlines()]
    assert len(lines) >= 7
    for line in lines:
        for placeholder, value in PLACEHOLDERS.items():
            line = line.replace(placeholder, value)
        program, *argv = shlex.split(line)
        assert program == "researchgenie", line
        assert headless.is_headless(argv), line
        headless.build_parser().parse_args(argv)  # SystemExit fails the test

```

## The experiment

```bash
cd Course-Project/code
pip install -e .
pytest -v tests/researchgenie/test_headless.py tests/test_agent_skill.py
```

## Real result
Every command line documented in SKILL.md above is parsed by the real headless CLI's own argument parser inside the test. If a future change to the CLI breaks a documented command, this test fails immediately, not at a demo.

## Why this is the real evidence, not a toy
This exact skill file is also installed and checked live in OpenClaw with `openclaw skills check`, see Lab 6, using this file as is, not a simplified copy.

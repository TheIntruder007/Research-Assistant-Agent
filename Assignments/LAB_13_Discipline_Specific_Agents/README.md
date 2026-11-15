# Lab 13 - Discipline-Specific Agents

**Skill:** Deep Research

**Worklet mapping:** A domain worklet for different engineering disciplines, not one narrow fixed topic.

## What this demonstrates

The Trend and Gap Advisor covers six fields, CS/AI/ML, ECE, EEE, Mechanical, Civil and Cloud/DevOps, ranking topics by measured growth, papers in the last 3 years divided by the 3 before, not the model's opinion. A confirmed pick hands straight into a full pipeline run under one run id.

## Real code exercised

The actual source files this lab is built on, copied in full below, not linked only.

### `services/trend-advisor/trend_advisor/velocity.py`

```python
"""Step 1 — publication-velocity analysis: which subfields are actually growing.

A standalone, tool-using agent step (course skill 1): it decides its own
queries from a domain alone, calls real scholarly-database tools, and returns
evidence. It has no dependency on the chat loop, the service wrapper, the
orchestrator, or the CLI, and can be run and tested entirely on its own.

The whole ranking is arithmetic over live OpenAlex counts — the model is
never asked which topics are rising, because a model's answer to that is
unverifiable and drifts between runs. Likewise the domain -> subfield table
below is hardcoded and verified against the live OpenAlex taxonomy rather
than inferred at runtime (spec user story 19).
"""

from __future__ import annotations

import asyncio
import datetime
import re
import sys
from itertools import zip_longest
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "research-discovery"))

from discovery import fulltext  # noqa: E402
from discovery.sources import arxiv, openalex  # noqa: E402
from shared.contracts.trend_contract import ExamplePaper, TopicCandidate  # noqa: E402

# Verified against https://api.openalex.org/fields/{17,21,22} on 2026-09-12 —
# every id below was read back from OpenAlex's own subfield list for its
# field, not guessed. Each domain lists the subfields a student in that
# department would recognize as theirs; overlaps (2208 in both ECE and EEE,
# 1712 in both CS and Cloud/DevOps) are deliberate, because the departments
# really do overlap there.
DOMAIN_SUBFIELDS: dict[str, list[str]] = {
    # Artificial Intelligence, Computer Vision, CS Applications, Software
    "CS_AI_ML": ["1702", "1707", "1706", "1712"],
    # Electrical & Electronic Eng, Computer Networks & Comms, Signal
    # Processing, Hardware & Architecture
    "ECE": ["2208", "1705", "1711", "1708"],
    # Electrical & Electronic Eng, Energy Eng & Power Technology, Renewable
    # Energy, Control & Systems Eng
    "EEE": ["2208", "2102", "2105", "2207"],
    # Mechanical Eng, Mechanics of Materials, Aerospace Eng, Industrial &
    # Manufacturing Eng
    "MECHANICAL": ["2210", "2211", "2202", "2209"],
    # Civil & Structural Eng, Building & Construction, Architecture, Ocean Eng
    "CIVIL": ["2205", "2215", "2216", "2212"],
    # Information Systems, Software, Computer Networks & Comms, Hardware &
    # Architecture
    "CLOUD_DEVOPS": ["1710", "1712", "1705", "1708"],
}

# Domains whose work genuinely preprints on arXiv in volume. Everything else
# gets no arXiv signal at all rather than a misleadingly thin one.
ARXIV_DOMAINS = frozenset({"CS_AI_ML", "ECE", "CLOUD_DEVOPS"})

WINDOW_YEARS = 3
# Below this many works in the recent window, a growth ratio is noise: three
# papers where there was one is a 3x "rise" that means nothing. Set well
# above that floor so the shortlist only contains topics a student could
# actually assemble a corpus in.
MIN_RECENT_WORKS = 200
# A domain that cannot produce at least this many rankable topics is a
# failure to report, not a short list to present (spec user story 18).
MIN_TOPICS_FOR_SHORTLIST = 3
EXAMPLE_PAPERS_PER_TOPIC = 5
# arXiv's Atom API ranks by relevance and has no count endpoint, so the
# preprint signal is a sample of this many top hits, not a census. That is
# why it never enters growth_metric — see _growth_key.
ARXIV_SEARCH_LIMIT = 50
# How many papers per topic get a full-text fetch for their Discussion /
# Limitations / Future-research sections. Matches gap_mining's
# MAX_PAPERS_PER_TOPIC — fetching text for papers the swarm will never read
# would be pure latency. Retrieval reuses Discovery's existing extractor.
GAP_TEXT_PAPERS_PER_TOPIC = 2
FULLTEXT_CONCURRENCY = 8


class VelocityUnavailable(RuntimeError):
    """OpenAlex was unreachable, or the domain yielded too little data to
    rank honestly. Raised instead of returning an empty or padded shortlist,
    so the advisor never shows a student confidence it does not have."""


def windows(current_year: int | None = None) -> tuple[tuple[int, int], tuple[int, int]]:
    """The (recent, prior) year windows to compare.

    The current year is excluded entirely: indexing lags, so a partial year
    always looks like a collapse in publication volume and would invert
    every ranking.
    """
    year = current_year or datetime.date.today().year
    last_complete = year - 1
    recent = (last_complete - WINDOW_YEARS + 1, last_complete)
    prior = (recent[0] - WINDOW_YEARS, recent[0] - 1)
    return recent, prior


def _normalize(name: str) -> str:
    """The cross-source matching key. OpenAlex and arXiv spell the same topic
    and the same paper title with different case, punctuation and hyphenation,
    so raw string equality would treat one thing as two."""
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def _growth_key(candidate: TopicCandidate) -> tuple[float, int]:
    """Growth first, then how much recent work exists across *both* sources.

    arXiv only ever moves the volume half of the key. Its count comes from a
    relevance-ranked sample rather than a full corpus, so folding it into
    growth_metric would put a sampled number inside the one ratio a student is
    told they can re-derive from OpenAlex themselves.
    """
    return (candidate.growth_metric,
            candidate.paper_count + (candidate.arxiv_recent_count or 0))


def rank_by_growth(*, recent: dict[str, tuple[str, int]],
                   prior: dict[str, tuple[str, int]],
                   limit: int) -> list[TopicCandidate]:
    """Rank topics by how much faster they are published now than before.

    growth = recent / max(prior, 1). A topic absent from the prior window
    divides by 1, which makes a genuinely new topic rank by its own recent
    volume rather than by an infinity.
    """
    candidates = []
    for topic_id, (name, recent_count) in recent.items():
        if recent_count < MIN_RECENT_WORKS:
            continue
        prior_count = prior.get(topic_id, ("", 0))[1]
        candidates.append(TopicCandidate(
            topic=name or topic_id,
            topic_id=topic_id,
            growth_metric=round(recent_count / max(prior_count, 1), 2),
            paper_count=recent_count,
            prior_papers=prior_count,
            source="velocity",
        ))
    candidates.sort(key=_growth_key, reverse=True)
    return candidates[:limit]


def merge_sources(primary: list[TopicCandidate],
                  secondary: list[TopicCandidate]) -> list[TopicCandidate]:
    """Fold a second source's candidates into the first by normalized topic.

    A topic both sources found stays ONE shortlist entry. The primary keeps
    its own paper_count: OpenAlex counts a whole indexed corpus and arXiv a
    sample of preprints, so adding them would report a volume neither source
    measured. The secondary's count lands in arxiv_recent_count instead, where
    it is labelled for what it is. Example papers union, deduplicated by
    title, and interleaved rather than appended: gap mining reads only the
    first MAX_PAPERS_PER_TOPIC papers that have text, and every OpenAlex
    paper has one (works_for_topic filters on has_abstract), so appending
    put the preprints permanently out of the swarm's reach.
    """
    merged: dict[str, TopicCandidate] = {}
    for candidate in primary + secondary:
        existing = merged.get(_normalize(candidate.topic))
        if existing is None:
            merged[_normalize(candidate.topic)] = candidate
            continue
        if candidate.arxiv_recent_count is not None:
            existing.arxiv_recent_count = candidate.arxiv_recent_count
        seen = {_normalize(p.title) for p in existing.example_papers}
        added = []
        for paper in candidate.example_papers:
            if _normalize(paper.title) not in seen:
                seen.add(_normalize(paper.title))
                added.append(paper)
        existing.example_papers = [
            paper
            for pair in zip_longest(existing.example_papers, added)
            for paper in pair if paper is not None
        ]
    return list(merged.values())


async def _arxiv_candidate(http: httpx.AsyncClient, topic: str,
                           year_range: tuple[int, int]) -> TopicCandidate | None:
    """arXiv's own evidence for one topic: how many of its top hits fall in
    the same window OpenAlex was counted over, plus the preprints themselves
    as example papers. Reuses Discovery's existing arXiv connector rather
    than adding a second client. Returns None (never an empty candidate) when
    arXiv fails, so an outage is never mistaken for an absence of preprints.

    `year_range` is the OpenAlex recent window, upper bound included: the two
    counts are added in _growth_key, so an open-ended arXiv filter would let
    the current partial year through on one side of a sum whose other side
    deliberately excludes it.

    growth_metric stays 0.0 rather than being computed from the sample: a
    relevance-ranked top-50 cannot support a recent-vs-prior ratio, and
    inventing one would let a sampled number outrank counted ones.
    paper_count stays 0 for the same reason it is never summed in
    merge_sources: it is OpenAlex's measured volume, and arXiv has not
    measured it. The sample lands in arxiv_recent_count alone.
    """
    start, end = year_range
    try:
        papers = await arxiv.search(http, {"arxiv": topic},
                                    limit=ARXIV_SEARCH_LIMIT)
    except (httpx.HTTPError, ValueError):
        return None
    recent = [p for p in papers if p.year and start <= p.year <= end]
    return TopicCandidate(
        topic=topic,
        growth_metric=0.0,
        paper_count=0,
        arxiv_recent_count=len(recent),
        example_papers=[
            ExamplePaper(title=p.title, doi=p.doi, year=p.year,
                         citations=p.citations, abstract=p.abstract)
            for p in recent[:EXAMPLE_PAPERS_PER_TOPIC]
        ],
    )


async def _attach_future_work_text(http: httpx.AsyncClient,
                                   works_by_topic: list[list]) -> None:
    """Populate `paper.future_text` for the papers gap mining will read.

    Authors state what is still unsolved in a paper's Discussion /
    Limitations / Future-research sections, not in its abstract — so mining
    abstracts alone would weaken the one signal this stage exists to
    provide. Reuses Discovery's `fulltext` module (Europe PMC JATS, then an
    open-access PDF), which never raises and simply leaves `future_text`
    unset when no open-access copy is reachable; gap mining then falls back
    to the abstract for that paper.
    """
    targets = [
        paper
        for works in works_by_topic
        for paper in works[:GAP_TEXT_PAPERS_PER_TOPIC]
        if fulltext.is_candidate(paper)
    ]
    if not targets:
        return
    sem = asyncio.Semaphore(FULLTEXT_CONCURRENCY)

    async def one(paper) -> None:
        async with sem:
            await fulltext.fetch_future_sections(http, paper)

    await asyncio.gather(*(one(paper) for paper in targets))


async def find_rising_topics(
    domain: str, *, limit: int = 8, domain_other_name: str | None = None,
    transport: httpx.BaseTransport | None = None,
    current_year: int | None = None,
) -> list[TopicCandidate]:
    """Rank the fastest-growing topics in `domain`, each with real example papers.

    Raises VelocityUnavailable when OpenAlex cannot be reached or the domain
    is too thin to rank. `transport` is a test seam (httpx.MockTransport);
    production callers leave it None.
    """
    subfields = DOMAIN_SUBFIELDS.get(domain)
    label = domain_other_name or domain
    search = None if subfields else label
    recent_window, prior_window = windows(current_year)

    async with httpx.AsyncClient(timeout=60, transport=transport,
                                 follow_redirects=True) as http:
        try:
            recent, prior = await asyncio.gather(
                openalex.topic_counts(http, year_range=recent_window,
                                      subfield_ids=subfields, search=search),
                openalex.topic_counts(http, year_range=prior_window,
                                      subfield_ids=subfields, search=search),
            )
        except httpx.HTTPError as error:
            raise VelocityUnavailable(
                f"Could not reach OpenAlex to analyse publication velocity: {error}"
            ) from None

        ranked = rank_by_growth(recent=recent, prior=prior, limit=limit)
        if len(ranked) < MIN_TOPICS_FOR_SHORTLIST:
            raise VelocityUnavailable(
                f"OpenAlex returned too few rankable topics for '{label}' "
                f"({len(ranked)} with at least {MIN_RECENT_WORKS} recent works). "
                "No shortlist can be produced honestly for this domain."
            )

        # One round-trip per topic, issued together: these are independent
        # reads of the same API that already served the two count queries
        # above concurrently.
        per_topic_works = await asyncio.gather(*(
            openalex.works_for_topic(http, candidate.topic_id,
                                     year_range=recent_window,
                                     limit=EXAMPLE_PAPERS_PER_TOPIC)
            for candidate in ranked
        ))
        await _attach_future_work_text(http, per_topic_works)
        for candidate, works in zip(ranked, per_topic_works):
            candidate.example_papers = [
                ExamplePaper(title=w.title, doi=w.doi, year=w.year,
                             citations=w.citations, abstract=w.abstract,
                             future_text=getattr(w, "future_text", None))
                for w in works
            ]

        if domain in ARXIV_DOMAINS:
            # Deliberately sequential, unlike the OpenAlex reads above: arXiv
            # asks callers to space requests out, and this is a secondary
            # source not worth risking a rate-limit ban over.
            from_arxiv = []
            for candidate in ranked:
                found = await _arxiv_candidate(http, candidate.topic,
                                               recent_window)
                if found is not None:
                    from_arxiv.append(found)
            ranked = merge_sources(ranked, from_arxiv)
            ranked.sort(key=_growth_key, reverse=True)
            ranked = ranked[:limit]
            # Merging collapses topics whose names normalize alike, so the
            # shortlist can come out thinner than what was already checked
            # above. Re-check rather than under-deliver quietly.
            if len(ranked) < MIN_TOPICS_FOR_SHORTLIST:
                raise VelocityUnavailable(
                    f"Merging the OpenAlex and arXiv candidates for "
                    f"'{label}' left only {len(ranked)} "
                    "distinct topic(s). No shortlist can be produced honestly "
                    "for this domain."
                )

    return ranked

```

### `services/trend-advisor/service.py`

```python
"""Public entry point for the Trend & Gap Advisor Service (pipeline stage 0).

Runs the two evidence steps — publication velocity (trend_advisor/velocity.py)
and the cross-paper future-work gap-mining swarm (trend_advisor/gap_mining.py)
— and adapts them to the shared TrendAdvisorResult contract, in the same
progress-events-then-final-result shape as the four existing pipeline
services, so the orchestrator and the advisor CLI can treat this stage
uniformly with the rest.

This stage is optional and strictly additive: nothing in Services 1-4 calls
it, and a normal `researchgenie` run never reaches it.
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.contracts.trend_contract import (  # noqa: E402
    TrendAdvisorRequest, TrendAdvisorResult,
)
from shared.utilities import llm_provider, scholarly_gateway  # noqa: E402
from shared.utilities.run_paths import slugify  # noqa: E402

sys.path.insert(0, str(ROOT / "services" / "research-writing"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trend_advisor.gap_mining import mine_gaps  # noqa: E402
from trend_advisor.velocity import (  # noqa: E402
    ARXIV_DOMAINS, VelocityUnavailable, find_rising_topics,
)
from writing.adapters.language_model import close_language_model  # noqa: E402
from writing.adapters.ollama_model import create_ollama_model  # noqa: E402

DEFAULT_OUTPUT_ROOT = ROOT / "outputs"

# Each gap-mining worker only reads one abstract, so it needs nothing like
# the writing service's budget — see gap_mining.WORKER_MAX_TOKENS for why a
# small budget is also the fast choice on this hardware (D-009).
_GAP_MODEL_MAX_TOKENS = 1200


class TrendAdvisorError(RuntimeError):
    """The stage could not produce an honest shortlist. Raised rather than
    yielding an empty or padded result — the whole point of this stage is
    evidence the student can check, so "no evidence" must be visible."""


async def run_trend_advisor(
    request: TrendAdvisorRequest, *, output_root: Path | None = None,
    run_id: str | None = None, model=None,
) -> AsyncIterator[dict]:
    """Yields progress events, then a final
    {"type": "result", "result": TrendAdvisorResult} event.

    Raises TrendAdvisorError when no honest shortlist can be produced (most
    often OpenAlex unreachable, or a free-text domain too thin to rank).

    `model` is a test/advanced seam for the gap-mining swarm's LanguageModel;
    production callers leave it None and get the local Ollama adapter.
    """
    run_id = run_id or uuid4().hex
    warnings: list[str] = []

    yield {"type": "status", "stage": "velocity", "state": "running",
           "message": f"Analysing publication velocity for {request.domain_label}..."}
    # Its own gateway run (#4): the advisor's OpenAlex and arXiv calls share
    # the process-wide pacing with any pipeline run beside it, trip their own
    # breaker, and report rate_limited / provider_exhausted here as they happen.
    gateway_run = scholarly_gateway.Run()
    velocity = gateway_run.task(find_rising_topics(
        request.domain, limit=request.shortlist_size,
        domain_other_name=request.domain_other_name,
    ))
    async for event in gateway_run.relay(velocity):
        yield event
    if (cached := gateway_run.cache_event()) is not None:
        yield cached
    try:
        candidates = velocity.result()
    except VelocityUnavailable as error:
        yield {"type": "status", "stage": "velocity", "state": "failed",
               "message": str(error)}
        raise TrendAdvisorError(str(error)) from None
    yield {"type": "status", "stage": "velocity", "state": "done",
           "message": f"Ranked {len(candidates)} rising topic(s) by publication growth."}

    if request.domain == "Other":
        warnings.append(
            f"'{request.domain_other_name}' is a free-text domain, so it has no "
            "curated OpenAlex subfield mapping — topics were matched by text "
            "search and may be less precise than for a listed domain."
        )
    # Only the CS-adjacent domains query arXiv at all, so a missing count
    # elsewhere means "never asked", not "asked and failed" — warning on it
    # would report an outage that did not happen.
    if request.domain in ARXIV_DOMAINS and any(
            c.arxiv_recent_count is None for c in candidates):
        warnings.append("arXiv was unreachable, so the preprint signal is missing "
                        "for some topics. The OpenAlex ranking is unaffected.")

    owns_model = model is None
    if owns_model:
        # Checked before the swarm rather than inferred from failed workers:
        # with the provider down, every worker fails, every topic comes back
        # with no gap evidence, and "no recurring gaps found" would read as a
        # finding instead of an outage. Degrade visibly, not silently.
        if not await llm_provider.health_check():
            warnings.append(
                "The local model was unavailable, so no future-work/limitations "
                "mining ran — topics are ranked on publication velocity alone, "
                "and the absence of gap flags below means 'not checked', not "
                "'no gaps exist'."
            )
            yield {"type": "status", "stage": "gap_mining", "state": "failed",
                   "message": warnings[-1]}
            model = None
        else:
            model = create_ollama_model(max_tokens=_GAP_MODEL_MAX_TOKENS)

    if model is not None:
        yield {"type": "status", "stage": "gap_mining", "state": "running",
               "message": f"Mining future-work and limitations across "
                          f"{len(candidates)} topic(s) in parallel..."}
        try:
            candidates = await mine_gaps(candidates, model=model)
        finally:
            if owns_model:
                await close_language_model(model)
        flagged = sum(1 for c in candidates if c.gap_signal)
        yield {"type": "status", "stage": "gap_mining", "state": "done",
               "message": f"{flagged} topic(s) flagged as recurring gaps by "
                          "multiple independent papers."}

    result = TrendAdvisorResult(
        domain=request.domain,
        domain_other_name=request.domain_other_name,
        # Gap-flagged topics first: a recurring, publicly-admitted unmet need
        # is a stronger reason to pick a topic than raw growth, which is why
        # this stage mines gaps at all. Within each group the velocity
        # ranking is preserved.
        shortlist=sorted(candidates, key=lambda c: c.gap_signal, reverse=True),
        generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        warnings=warnings,
    )

    # Step 2 (memory): every run keeps its own full, timestamped record, the
    # same convention the four pipeline stages already follow.
    root = output_root or DEFAULT_OUTPUT_ROOT
    run_directory = root / f"{slugify(request.domain_label)}_{run_id}"
    result_path = run_directory / "00_trend_advisor" / "result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    yield {"type": "result", "result": result, "run_directory": str(run_directory)}

```

### Real tests for this lab

### `tests/services/test_trend_advisor_service.py`

```python
"""Service-boundary tests for the Trend & Gap Advisor.

Exercises external behaviour at `run_trend_advisor()` — the same level
tests/services/test_discovery_service.py works at — with OpenAlex mocked at
the transport layer and the gap-mining model faked, so no network and no
Ollama are required.
"""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "research-discovery"))
sys.path.insert(0, str(ROOT / "services" / "research-writing"))
sys.path.insert(0, str(ROOT / "services" / "trend-advisor"))

from shared.contracts.trend_contract import TrendAdvisorRequest, TrendAdvisorResult  # noqa: E402
from trend_advisor import velocity  # noqa: E402


def _load_service():
    """Every service's entry point is named service.py, so a plain
    `import service` resolves to whichever one a sibling test module imported
    first (see DECISIONS.md D-010). Load this one by explicit file path under
    a unique name, exactly as orchestrator/pipeline.py does."""
    path = ROOT / "services" / "trend-advisor" / "service.py"
    spec = importlib.util.spec_from_file_location("test_trend_advisor_service_mod", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


service = _load_service()
TrendAdvisorError = service.TrendAdvisorError
run_trend_advisor = service.run_trend_advisor

RECENT = {"T1": ("Rising Topic", 900), "T2": ("Flat Topic", 800), "T3": ("Third Topic", 700)}
PRIOR = {"T1": ("Rising Topic", 100), "T2": ("Flat Topic", 790), "T3": ("Third Topic", 400)}


def _openalex_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if "arxiv.org" in str(request.url):
            # The CS-adjacent domains also query arXiv for a secondary signal.
            return httpx.Response(200, text=(
                '<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
                "<title>A Preprint</title><summary>Some text.</summary>"
                "<published>2024-01-01T00:00:00Z</published>"
                "<id>http://arxiv.org/abs/2401.00001</id>"
                "</entry></feed>"
            ))
        if "group_by" in params:
            start = int(params["filter"].split("publication_year:")[1].split("-")[0])
            counts = RECENT if start >= 2023 else PRIOR
            return httpx.Response(200, json={"group_by": [
                {"key": f"https://openalex.org/{tid}", "key_display_name": name, "count": n}
                for tid, (name, n) in counts.items()
            ]})
        topic = params["filter"].split("primary_topic.id:")[1].split(",")[0]
        return httpx.Response(200, json={"results": [
            {
                "display_name": f"{topic} Paper {i}", "publication_year": 2024,
                "cited_by_count": 10, "doi": f"https://doi.org/10.1000/{topic}-{i}",
                "abstract_inverted_index": {"Future": [0], "work": [1], "is": [2], "needed": [3]},
                "authorships": [{"author": {"display_name": "A. Author"}}],
                "primary_location": {"source": {"display_name": "A Journal"}},
            }
            for i in range(2)
        ]})
    return httpx.MockTransport(handler)


class _FakeModel:
    """Reports one gap per paper, so every topic clears the recurrence bar."""

    def __init__(self):
        self.calls = 0

    async def generate_structured(self, *, system_prompt, user_prompt, response_model):
        self.calls += 1
        title = user_prompt.split("Paper title: ")[1].split("\n")[0]
        return response_model.model_validate(
            {"gap_statements": [f"Gap stated by {title}."]}
        )


@pytest.fixture(autouse=True)
def _mock_openalex(monkeypatch):
    """Pin the advisor's OpenAlex calls to the mock transport and to a fixed
    'today', so the year windows the assertions rely on never drift."""
    real = velocity.find_rising_topics

    async def patched(domain, **kwargs):
        kwargs.setdefault("transport", _openalex_transport())
        kwargs.setdefault("current_year", 2026)
        return await real(domain, **kwargs)

    monkeypatch.setattr(velocity, "find_rising_topics", patched)
    monkeypatch.setattr(service, "find_rising_topics", patched)


async def _collect(request, **kwargs):
    events, result = [], None
    async for event in run_trend_advisor(request, **kwargs):
        if event["type"] == "result":
            result = event["result"]
        else:
            events.append(event)
    return events, result


def test_produces_a_ranked_shortlist_with_evidence(tmp_path):
    events, result = asyncio.run(_collect(
        TrendAdvisorRequest(domain="CS_AI_ML", shortlist_size=6),
        output_root=tmp_path, model=_FakeModel(),
    ))

    assert isinstance(result, TrendAdvisorResult)
    assert result.domain == "CS_AI_ML"
    assert len(result.shortlist) == 3
    top = result.shortlist[0]
    assert top.growth_metric > 1
    assert top.example_papers and top.example_papers[0].doi
    # Both stages ran and both reported done.
    assert [e["stage"] for e in events if e["state"] == "done"] == ["velocity", "gap_mining"]


def test_gap_flagged_topics_are_listed_before_ungapped_ones(tmp_path):
    class _OneTopicOnly(_FakeModel):
        async def generate_structured(self, *, system_prompt, user_prompt, response_model):
            title = user_prompt.split("Paper title: ")[1].split("\n")[0]
            gaps = [f"Gap from {title}."] if title.startswith("T3") else []
            return response_model.model_validate({"gap_statements": gaps})

    _, result = asyncio.run(_collect(
        TrendAdvisorRequest(domain="CIVIL"), output_root=tmp_path, model=_OneTopicOnly(),
    ))

    assert result.shortlist[0].topic == "Third Topic"
    assert result.shortlist[0].gap_signal is True
    assert result.shortlist[0].source == "both"
    assert all(c.gap_signal is False for c in result.shortlist[1:])


def test_run_output_is_saved_to_its_own_timestamped_folder(tmp_path):
    _, result = asyncio.run(_collect(
        TrendAdvisorRequest(domain="MECHANICAL"), output_root=tmp_path,
        run_id="abc123", model=_FakeModel(),
    ))

    saved = tmp_path / "mechanical_abc123" / "00_trend_advisor" / "result.json"
    assert saved.exists()
    assert json.loads(saved.read_text(encoding="utf-8"))["shortlist"][0]["topic"] == \
        result.shortlist[0].topic


def test_two_runs_keep_two_separate_records(tmp_path):
    """Step 2 (memory) claims every run keeps its own full record. Without a
    distinct run id per run the second call would overwrite the first, and a
    student comparing this week's shortlist with last week's would find only
    one of them."""
    request = TrendAdvisorRequest(domain="MECHANICAL")
    _, first = asyncio.run(_collect(request, output_root=tmp_path, model=_FakeModel()))
    _, second = asyncio.run(_collect(request, output_root=tmp_path, model=_FakeModel()))

    saved = sorted(tmp_path.glob("mechanical_*/00_trend_advisor/result.json"))
    assert len(saved) == 2, "the second run overwrote the first run's record"

    # Run ids are random, so the two files sort in no meaningful order —
    # each is checked on its own rather than paired with a call by position.
    expected = {len(first.shortlist), len(second.shortlist)}
    for path in saved:
        record = json.loads(path.read_text(encoding="utf-8"))
        # Each record is the full shortlist with its evidence, not a stub.
        assert len(record["shortlist"]) in expected
        assert all(topic["example_papers"] for topic in record["shortlist"])
        assert record["generated_at"]


def test_free_text_domain_warns_that_results_are_less_precise(tmp_path):
    _, result = asyncio.run(_collect(
        TrendAdvisorRequest(domain="Other", domain_other_name="synthetic biology"),
        output_root=tmp_path, model=_FakeModel(),
    ))

    assert any("free-text domain" in w for w in result.warnings)
    assert result.domain_other_name == "synthetic biology"


def test_unreachable_openalex_raises_rather_than_returning_an_empty_shortlist(tmp_path, monkeypatch):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("openalex unreachable", request=request)

    real = velocity.find_rising_topics

    async def patched(domain, **kwargs):
        return await real(domain, transport=httpx.MockTransport(refuse), **kwargs)

    monkeypatch.setattr(service, "find_rising_topics", patched)

    with pytest.raises(TrendAdvisorError):
        asyncio.run(_collect(TrendAdvisorRequest(domain="EEE"),
                             output_root=tmp_path, model=_FakeModel()))


def test_no_result_file_is_written_when_the_stage_fails(tmp_path, monkeypatch):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("openalex unreachable", request=request)

    real = velocity.find_rising_topics

    async def patched(domain, **kwargs):
        return await real(domain, transport=httpx.MockTransport(refuse), **kwargs)

    monkeypatch.setattr(service, "find_rising_topics", patched)

    with pytest.raises(TrendAdvisorError):
        asyncio.run(_collect(TrendAdvisorRequest(domain="EEE"),
                             output_root=tmp_path, model=_FakeModel()))
    assert list(tmp_path.iterdir()) == []


def test_model_outage_is_reported_rather_than_read_as_no_gaps_found(tmp_path, monkeypatch):
    """With the provider down every worker fails, so every topic comes back
    with no gap evidence. That must not be presented as a finding."""

    async def down():
        return False

    monkeypatch.setattr(service.llm_provider, "health_check", down)

    events, result = asyncio.run(_collect(
        TrendAdvisorRequest(domain="CIVIL"), output_root=tmp_path,
    ))

    assert any("model was unavailable" in w for w in result.warnings)
    assert any(e["stage"] == "gap_mining" and e["state"] == "failed" for e in events)
    # The velocity half of the stage still delivered.
    assert len(result.shortlist) == 3
    assert all(c.source == "velocity" for c in result.shortlist)


def test_no_arxiv_warning_for_a_domain_that_never_queries_arxiv(tmp_path):
    """Only the CS-adjacent domains use arXiv. Warning that it was
    "unreachable" for Mechanical reports an outage that never happened."""
    _, result = asyncio.run(_collect(
        TrendAdvisorRequest(domain="MECHANICAL"), output_root=tmp_path,
        model=_FakeModel(),
    ))

    assert all(c.arxiv_recent_count is None for c in result.shortlist)
    assert not any("arXiv" in w for w in result.warnings)


def test_a_throttled_openalex_call_is_retried_and_shown_as_a_progress_event(tmp_path, monkeypatch):
    """The advisor's OpenAlex calls run inside its own gateway run (#4): a 429
    is retried rather than failing the stage, and the wait is reported."""
    real = velocity.find_rising_topics
    inner = _openalex_transport()
    throttled = []

    def handler(request):
        if "group_by" in request.url.params and not throttled:
            throttled.append(request)
            return httpx.Response(429, headers={"Retry-After": "1"})
        return inner.handle_request(request)

    async def patched(domain, **kwargs):
        return await real(domain, transport=httpx.MockTransport(handler), current_year=2026, **kwargs)

    monkeypatch.setattr(service, "find_rising_topics", patched)
    events, result = asyncio.run(_collect(TrendAdvisorRequest(domain="MECHANICAL"),
                                          output_root=tmp_path, model=_FakeModel()))

    assert throttled and result.shortlist
    assert any(e["type"] == "rate_limited" for e in events)

```

### `tests/services/test_trend_velocity.py`

```python
"""Unit tests for the Trend & Gap Advisor's publication-velocity ranking.

The ranking math is the only genuinely novel reasoning in this step (the
HTTP layer is Discovery's already-tested OpenAlex connector), so it is
tested here against mocked OpenAlex responses with known counts — no
network, no model. The live counterpart lives in
test_trend_velocity_live.py and skips without network.
"""

import asyncio
import re
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "research-discovery"))
sys.path.insert(0, str(ROOT / "services" / "trend-advisor"))

from shared.contracts.trend_contract import ExamplePaper, TopicCandidate  # noqa: E402
from trend_advisor.gap_mining import MAX_PAPERS_PER_TOPIC, mine_gaps  # noqa: E402
from trend_advisor.velocity import (  # noqa: E402
    DOMAIN_SUBFIELDS, MIN_RECENT_WORKS, MIN_TOPICS_FOR_SHORTLIST,
    VelocityUnavailable, _arxiv_candidate, _growth_key, find_rising_topics,
    merge_sources, rank_by_growth, windows,
)


def _groups(counts: dict[str, tuple[str, int]]) -> dict:
    return {"group_by": [
        {"key": f"https://openalex.org/{tid}", "key_display_name": name, "count": n}
        for tid, (name, n) in counts.items()
    ]}


def _work(title: str) -> dict:
    return {
        "display_name": title, "publication_year": 2024, "cited_by_count": 10,
        "doi": "https://doi.org/10.1000/x", "abstract_inverted_index": {"An": [0], "abstract": [1]},
        "authorships": [{"author": {"display_name": "A. Author"}}],
        "primary_location": {"source": {"display_name": "A Journal"}},
    }


def _arxiv_feed(*entries: tuple[str, int]) -> str:
    body = "".join(
        f"<entry><title>{title}</title><summary>Preprint text.</summary>"
        f"<published>{year}-01-01T00:00:00Z</published>"
        f"<id>http://arxiv.org/abs/{year}.0000{i}</id></entry>"
        for i, (title, year) in enumerate(entries)
    )
    return f'<feed xmlns="http://www.w3.org/2005/Atom">{body}</feed>'


def _transport(recent: dict, prior: dict,
               arxiv: tuple = (("A Preprint", 2024),),
               works: tuple = ("A Paper",)) -> httpx.MockTransport:
    """Answers the two group_by window queries by the year range in the
    filter, every per-topic works query with `works`, and the CS-adjacent
    domains' arXiv queries with a fixed preprint feed."""
    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if "arxiv.org" in str(request.url):
            return httpx.Response(200, text=_arxiv_feed(*arxiv))
        if "group_by" in params:
            start = params["filter"].split("publication_year:")[1].split("-")[0]
            body = _groups(recent) if int(start) >= 2023 else _groups(prior)
            return httpx.Response(200, json=body)
        return httpx.Response(200, json={"results": [_work(t) for t in works]})
    return httpx.MockTransport(handler)


# --- rank_by_growth: the ranking math itself -------------------------------

def test_growth_is_recent_over_prior():
    ranked = rank_by_growth(
        recent={"T1": ("Growing", 600)}, prior={"T1": ("Growing", 200)}, limit=5,
    )
    assert ranked[0].growth_metric == pytest.approx(3.0)
    assert ranked[0].paper_count == 600
    assert ranked[0].paper_count == 600 and ranked[0].prior_papers == 200


def test_ranks_by_growth_not_by_volume():
    """A huge, flat topic must lose to a smaller, fast-growing one —
    otherwise the shortlist just re-lists whatever is already biggest."""
    ranked = rank_by_growth(
        recent={"BIG": ("Crowded", 50_000), "FAST": ("Rising", 900)},
        prior={"BIG": ("Crowded", 49_000), "FAST": ("Rising", 150)},
        limit=5,
    )
    assert [c.topic for c in ranked] == ["Rising", "Crowded"]


def test_drops_topics_below_the_recent_volume_floor():
    """A topic with almost no recent work has an enormous but meaningless
    growth ratio (3 works vs 1) and must not top the shortlist."""
    ranked = rank_by_growth(
        recent={"NOISE": ("Noise", MIN_RECENT_WORKS - 1), "REAL": ("Real", 800)},
        prior={"NOISE": ("Noise", 1), "REAL": ("Real", 400)},
        limit=5,
    )
    assert [c.topic for c in ranked] == ["Real"]


def test_brand_new_topic_with_no_prior_work_is_ranked_not_divided_by_zero():
    ranked = rank_by_growth(
        recent={"NEW": ("Brand New", 900)}, prior={}, limit=5,
    )
    assert ranked[0].growth_metric == pytest.approx(900.0)
    assert ranked[0].prior_papers == 0


def test_shrinking_topic_keeps_its_real_sub_one_growth_metric():
    ranked = rank_by_growth(
        recent={"OLD": ("Fading", 500)}, prior={"OLD": ("Fading", 2000)}, limit=5,
    )
    assert ranked[0].growth_metric == pytest.approx(0.25)


def test_limit_truncates_to_the_requested_shortlist_size():
    recent = {f"T{i}": (f"Topic {i}", 500 + i * 100) for i in range(12)}
    prior = {f"T{i}": (f"Topic {i}", 300) for i in range(12)}
    assert len(rank_by_growth(recent=recent, prior=prior, limit=6)) == 6


def test_crowded_flag_separates_hot_from_quiet():
    ranked = rank_by_growth(
        recent={"BIG": ("Crowded", 40_000), "SMALL": ("Quiet", 700)},
        prior={"BIG": ("Crowded", 10_000), "SMALL": ("Quiet", 200)},
        limit=5,
    )
    by_topic = {c.topic: c for c in ranked}
    assert by_topic["Crowded"].is_crowded is True
    assert by_topic["Quiet"].is_crowded is False


# --- windows: which years get compared ------------------------------------

def test_windows_exclude_the_incomplete_current_year():
    recent, prior = windows(current_year=2026)
    assert recent == (2023, 2025)
    assert prior == (2020, 2022)
    assert recent[0] > prior[1]


# --- find_rising_topics: the step's public entry point ---------------------

def test_find_rising_topics_returns_candidates_with_example_papers():
    transport = _transport(
        recent={"T1": ("Rising", 900), "T2": ("Flat", 800), "T3": ("Steady", 600)},
        prior={"T1": ("Rising", 100), "T2": ("Flat", 790), "T3": ("Steady", 400)},
    )
    candidates = asyncio.run(find_rising_topics("MECHANICAL", limit=6, transport=transport))
    assert candidates[0].topic == "Rising"
    assert candidates[0].topic_id == "T1"
    assert candidates[0].example_papers[0].title == "A Paper"
    assert candidates[0].source == "velocity"
    # Mechanical is not CS-adjacent, so arXiv is never asked at all.
    assert all(c.arxiv_recent_count is None for c in candidates)


def test_every_curated_domain_maps_to_real_subfield_ids():
    for domain, subfields in DOMAIN_SUBFIELDS.items():
        assert subfields, f"{domain} has no subfield mapping"
        assert all(s.isdigit() and len(s) == 4 for s in subfields), domain


def test_unreachable_openalex_fails_loudly_rather_than_returning_nothing():
    """The spec's hard requirement: never present fabricated confidence."""
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("openalex unreachable", request=request)

    with pytest.raises(VelocityUnavailable):
        asyncio.run(find_rising_topics("CIVIL", limit=6,
                                       transport=httpx.MockTransport(refuse)))


def test_domain_yielding_too_few_topics_fails_loudly():
    transport = _transport(recent={}, prior={})
    with pytest.raises(VelocityUnavailable):
        asyncio.run(find_rising_topics("EEE", limit=6, transport=transport))


def test_other_domain_searches_free_text_instead_of_subfields():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if "group_by" in params:
            seen.append(dict(params))
            start = params["filter"].split("publication_year:")[1].split("-")[0]
            counts = (
                {"T9": ("Bio Topic", 900), "T8": ("Other Bio", 700), "T7": ("More Bio", 500)}
                if int(start) >= 2023
                else {"T9": ("Bio Topic", 100), "T8": ("Other Bio", 600), "T7": ("More Bio", 480)}
            )
            return httpx.Response(200, json=_groups(counts))
        return httpx.Response(200, json={"results": [_work("A Paper")]})

    candidates = asyncio.run(find_rising_topics(
        "Other", limit=6, transport=httpx.MockTransport(handler),
        domain_other_name="synthetic biology",
    ))
    assert candidates[0].topic == "Bio Topic"
    assert all(q.get("search") == "synthetic biology" for q in seen)
    assert all("primary_topic.subfield.id" not in q["filter"] for q in seen)


# --- merge_sources: OpenAlex + arXiv without double-counting ---------------

def _arxiv_side(topic: str, count: int, *titles: str) -> TopicCandidate:
    return TopicCandidate(
        topic=topic, growth_metric=0.0, paper_count=0,
        arxiv_recent_count=count,
        example_papers=[ExamplePaper(title=t) for t in titles],
    )


def test_a_topic_found_in_both_sources_is_merged_not_double_counted():
    """The whole point of the merge: two sources finding the same topic must
    produce one shortlist entry whose paper_count is still OpenAlex's count,
    not the two corpora added together."""
    openalex = rank_by_growth(
        recent={"T1": ("Federated Learning", 900)},
        prior={"T1": ("Federated Learning", 300)}, limit=5,
    )
    merged = merge_sources(
        openalex, [_arxiv_side("federated-learning", 40, "A Preprint")],
    )

    assert len(merged) == 1
    assert merged[0].paper_count == 900
    assert merged[0].growth_metric == pytest.approx(3.0)
    assert merged[0].arxiv_recent_count == 40
    assert [p.title for p in merged[0].example_papers] == ["A Preprint"]


def test_the_same_paper_from_both_sources_is_listed_once():
    openalex = rank_by_growth(
        recent={"T1": ("Scaling", 900)}, prior={"T1": ("Scaling", 300)}, limit=5,
    )
    openalex[0].example_papers = [ExamplePaper(title="Scaling Laws for Models")]
    merged = merge_sources(
        openalex,
        [_arxiv_side("Scaling", 5, "scaling laws for models.", "A Second Preprint")],
    )

    assert [p.title for p in merged[0].example_papers] == [
        "Scaling Laws for Models", "A Second Preprint",
    ]


def test_a_topic_only_the_second_source_found_is_kept():
    openalex = rank_by_growth(
        recent={"T1": ("Known", 900)}, prior={"T1": ("Known", 300)}, limit=5,
    )
    merged = merge_sources(openalex, [_arxiv_side("Only On arXiv", 12)])

    assert {c.topic for c in merged} == {"Known", "Only On arXiv"}


def test_cs_domain_merges_arxiv_evidence_into_the_same_topic_entry():
    transport = _transport(
        recent={"T1": ("Rising", 900), "T2": ("Flat", 800), "T3": ("Steady", 600)},
        prior={"T1": ("Rising", 100), "T2": ("Flat", 790), "T3": ("Steady", 400)},
        arxiv=(("A Preprint", 2024), ("An Older Preprint", 2019)),
    )
    candidates = asyncio.run(find_rising_topics(
        "CS_AI_ML", limit=6, transport=transport, current_year=2026,
    ))

    assert [c.topic for c in candidates] == ["Rising", "Steady", "Flat"]
    top = candidates[0]
    assert top.paper_count == 900
    # Only the preprint inside the recent window counts.
    assert top.arxiv_recent_count == 1
    assert [p.title for p in top.example_papers] == ["A Paper", "A Preprint"]


def _arxiv_only(*entries: tuple[str, int],
                year_range: tuple[int, int] = (2023, 2025)) -> TopicCandidate:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, text=_arxiv_feed(*entries)))

    async def run():
        async with httpx.AsyncClient(transport=transport) as http:
            return await _arxiv_candidate(http, "Some Topic", year_range)

    return asyncio.run(run())


def test_arxiv_sample_is_counted_once_in_the_ranking_key():
    """Regression: paper_count carried the same sample as arxiv_recent_count,
    so _growth_key added an arXiv-only topic's volume to itself — and the
    sample was shown to the student as an OpenAlex 'Recent papers' count."""
    candidate = _arxiv_only(("One", 2024), ("Two", 2024))

    assert candidate.arxiv_recent_count == 2
    assert candidate.paper_count == 0
    assert _growth_key(candidate)[1] == 2
    assert candidate.is_crowded is False


def test_arxiv_window_matches_the_openalex_window_at_both_ends():
    """The two counts are summed, so the arXiv filter has to exclude the
    incomplete current year exactly as windows() does — otherwise one side
    of the sum spans four years and the other three."""
    candidate = _arxiv_only(("Too Old", 2022), ("In Window", 2024),
                            ("Current Year", 2026))

    assert candidate.arxiv_recent_count == 1
    assert [p.title for p in candidate.example_papers] == ["In Window"]


def test_merge_collapsing_topics_below_the_floor_fails_loudly():
    """Two OpenAlex topics that normalize to one key become one entry, which
    can push the shortlist under MIN_TOPICS_FOR_SHORTLIST after the merge —
    the guard that ran before it must run again."""
    names = ["Edge AI", "Edge-AI", "edge ai"]
    assert len(names) >= MIN_TOPICS_FOR_SHORTLIST
    transport = _transport(
        recent={f"T{i}": (n, 900 - i) for i, n in enumerate(names)},
        prior={f"T{i}": (n, 300) for i, n in enumerate(names)},
    )

    with pytest.raises(VelocityUnavailable):
        asyncio.run(find_rising_topics("CS_AI_ML", limit=6, transport=transport,
                                       current_year=2026))


class _PromptRecorder:
    def __init__(self):
        self.prompts: list[str] = []

    async def generate_structured(self, *, system_prompt, user_prompt, response_model):
        self.prompts.append(user_prompt)
        return response_model()


def test_an_arxiv_preprint_actually_reaches_a_gap_mining_worker():
    """D-035's claim, end to end. Gap mining reads only the first
    MAX_PAPERS_PER_TOPIC papers that have text, and every OpenAlex paper has
    text (works_for_topic filters on has_abstract), so appending the
    preprints put them permanently out of the swarm's reach."""
    transport = _transport(
        recent={"T1": ("Rising", 900), "T2": ("Flat", 800), "T3": ("Steady", 600)},
        prior={"T1": ("Rising", 100), "T2": ("Flat", 790), "T3": ("Steady", 400)},
        arxiv=(("A Preprint", 2024),),
        works=("A Paper", "Another Paper"),
    )
    candidates = asyncio.run(find_rising_topics(
        "CS_AI_ML", limit=6, transport=transport, current_year=2026,
    ))
    model = _PromptRecorder()
    asyncio.run(mine_gaps(candidates, model=model))

    mined = [p.split("Paper title: ")[1].split("\n")[0] for p in model.prompts]
    assert len(mined) == len(candidates) * MAX_PAPERS_PER_TOPIC
    # Every topic spent one of its two worker slots on the preprint.
    assert sorted(set(mined)) == ["A Paper", "A Preprint"]


def test_arxiv_volume_breaks_a_tie_between_equally_growing_topics():
    """arxiv_recent_count has to actually move the ranking, or the second
    source is decoration."""
    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if "arxiv.org" in str(request.url):
            entries = (("Beta Preprint", 2024), ("Another Beta Preprint", 2024))
            return httpx.Response(200, text=_arxiv_feed(
                *(entries if "Beta" in params["search_query"] else ()),
            ))
        if "group_by" in params:
            start = int(params["filter"].split("publication_year:")[1].split("-")[0])
            counts = (
                {"T1": ("Alpha", 600), "T2": ("Beta", 600), "T3": ("Gamma", 500)}
                if start >= 2023
                else {"T1": ("Alpha", 300), "T2": ("Beta", 300), "T3": ("Gamma", 480)}
            )
            return httpx.Response(200, json=_groups(counts))
        return httpx.Response(200, json={"results": [_work("A Paper")]})

    candidates = asyncio.run(find_rising_topics(
        "CS_AI_ML", limit=6, transport=httpx.MockTransport(handler),
        current_year=2026,
    ))

    by_topic = {c.topic: c for c in candidates}
    assert by_topic["Alpha"].growth_metric == by_topic["Beta"].growth_metric
    assert [c.topic for c in candidates][:2] == ["Beta", "Alpha"]


# --- the domain actually steers the query ----------------------------------

def _per_domain_transport() -> httpx.MockTransport:
    """OpenAlex, but every topic is named after the subfield it was asked
    for — so a domain whose mapping never reaches the wire comes back with
    another domain's topics, and the assertions below catch it."""
    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if "arxiv.org" in str(request.url):
            return httpx.Response(200, text=_arxiv_feed(("A Preprint", 2024)))
        if "group_by" in params:
            filters = params["filter"]
            subfields = re.findall(r"subfields/(\d{4})", filters)
            recent = int(filters.split("publication_year:")[1].split("-")[0]) >= 2023
            return httpx.Response(200, json=_groups({
                f"T{sid}": (f"Topic of subfield {sid}", (900 - i * 100) if recent else 300)
                for i, sid in enumerate(subfields)
            }))
        return httpx.Response(200, json={"results": [_work("A Paper")]})
    return httpx.MockTransport(handler)


@pytest.mark.parametrize("domain", sorted(DOMAIN_SUBFIELDS))
def test_every_curated_domain_returns_a_non_empty_ranked_shortlist(domain):
    candidates = asyncio.run(find_rising_topics(
        domain, limit=8, transport=_per_domain_transport()))

    assert candidates, f"{domain} produced no candidates"
    growths = [c.growth_metric for c in candidates]
    assert growths == sorted(growths, reverse=True), f"{domain} came back unranked"
    assert all(c.example_papers for c in candidates)
    # Every topic it returned came from a subfield this domain actually maps to.
    assert {c.topic_id.removeprefix("T") for c in candidates} <= set(DOMAIN_SUBFIELDS[domain])


def test_two_domains_produce_different_top_candidates():
    """The domain -> subfield table is only worth having if the domain
    changes the answer. A table that collapsed to one field, or a domain
    argument dropped on its way to the query, both survive every other test
    in this file."""
    transport = _per_domain_transport()
    mechanical = asyncio.run(find_rising_topics("MECHANICAL", limit=8, transport=transport))
    civil = asyncio.run(find_rising_topics("CIVIL", limit=8, transport=transport))

    assert mechanical[0].topic != civil[0].topic
    assert {c.topic for c in mechanical}.isdisjoint({c.topic for c in civil})

```

### `tests/shared/test_trend_contract.py`

```python
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.contracts.trend_contract import (  # noqa: E402
    TopicCandidate, TrendAdvisorRequest, TrendAdvisorResult,
)


@pytest.mark.parametrize("domain", ["Biotech", "cs_ai_ml", "OTHER"])
def test_unlisted_domain_is_rejected(domain):
    with pytest.raises(ValidationError):
        TrendAdvisorRequest(domain=domain)


def test_other_without_domain_other_name_is_rejected():
    with pytest.raises(ValidationError):
        TrendAdvisorRequest(domain="Other")


def test_other_with_domain_other_name_is_accepted():
    req = TrendAdvisorRequest(domain="Other", domain_other_name="Biotech")
    assert req.domain_label == "Biotech"


def test_fixed_domain_is_accepted():
    req = TrendAdvisorRequest(domain="MECHANICAL")
    assert req.domain_label == "MECHANICAL"


# --- Derived labels and lookups the CLI, smoke runner and orchestrator share ---

def _candidate(**kwargs) -> TopicCandidate:
    base = dict(topic="Edge Inference", topic_id="T1", growth_metric=2.0, paper_count=100)
    return TopicCandidate(**{**base, **kwargs})


def _result(**kwargs) -> TrendAdvisorResult:
    base = dict(domain="CS_AI_ML", generated_at="2026-09-13T00:00:00Z", shortlist=[])
    return TrendAdvisorResult(**{**base, **kwargs})


def test_a_gap_flagged_topic_reads_as_a_gap_even_when_it_is_also_crowded():
    """The gap signal is the reason this stage mines gaps at all, so it wins
    over raw volume in the one phrase the student is shown."""
    assert _candidate(gap_signal=True, paper_count=90_000).signal_label == "recurring gap"


def test_volume_alone_separates_a_crowded_topic_from_a_quiet_one():
    assert _candidate(paper_count=90_000).signal_label == "hot but crowded"
    assert _candidate(paper_count=10).signal_label == "under-explored"


def test_a_free_text_domain_labels_itself_by_its_own_name():
    result = _result(domain="Other", domain_other_name="synthetic biology")
    assert result.domain_label == "synthetic biology"
    assert _result().domain_label == "CS_AI_ML"


def test_gap_flagged_count_counts_topics_not_pieces_of_evidence():
    result = _result(shortlist=[
        _candidate(topic="A", gap_signal=True, gap_evidence=["one", "two", "three"]),
        _candidate(topic="B"),
    ])
    assert result.gap_flagged_count == 1


def test_the_chosen_candidate_is_found_by_name():
    result = _result(shortlist=[_candidate(topic="A"), _candidate(topic="B", gap_signal=True)],
                     chosen_topic="B")
    assert result.chosen_candidate.gap_signal is True


def test_a_chosen_topic_that_is_not_on_the_shortlist_resolves_to_nothing():
    """Never another topic's evidence: a name the shortlist doesn't carry
    must come back empty rather than falling through to a neighbour."""
    result = _result(shortlist=[_candidate(topic="A")], chosen_topic="Not Listed")
    assert result.chosen_candidate is None


def test_no_pick_resolves_to_nothing():
    assert _result(shortlist=[_candidate(topic="A")]).chosen_candidate is None

```

## The experiment

```bash
cd Course-Project/code
pip install -e .
pytest -v tests/services/test_trend_advisor_service.py tests/services/test_trend_velocity.py tests/shared/test_trend_contract.py
```

## Real result
Raw console output also saved as [`sample_run.txt`](./sample_run.txt) in this folder.

Every curated domain above returns a non empty ranked shortlist, and two domains are shown to produce genuinely different top candidates, proving the ranking is not hard coded per field. A live CS/AI/ML advisor run picked a real rising topic and handed it directly into a full pipeline run with no human retyping anything, recorded in `Course-Project/docs/evidence/live-pass/`.

## Why this is the real evidence, not a toy
The example gap statement the advisor produced live, what specific similarity exists between fake images generated by GANs and those from diffusion models, came from real publication data, not a seeded example.

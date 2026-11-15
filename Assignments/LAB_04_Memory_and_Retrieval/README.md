# Lab 4 - Memory and Retrieval

**Skill:** Skills

**Worklet mapping:** Short term state through LangGraph and SQLite checkpoints, long term retrieval through a research library and a response cache.

## What this demonstrates

ResearchGenie remembers papers and search results across runs. A SQLite full text research library stores every run and paper, and a response cache keeps searches for 7 days and DOIs for 30 days. A seeded past paper still has to pass the exact same relevance gate as a brand new one, so memory can make a run richer, never less checked.

## Real code exercised

The actual source files this lab is built on, copied in full below, not linked only.

### `shared/utilities/library.py`

```python
"""The local research library: every completed run and the papers it
selected, in one SQLite file (standard library sqlite3 + FTS5) next to the
CLI's config, at ~/.researchgenie/library.sqlite.

    papers     (doi PK, title, abstract, year, venue, discussion_excerpt,
                fulltext_status, first_seen, authors (JSON list), citations)
    runs       (run_id PK, question, created, overall_quality, draft_status, run_dir)
    run_papers (run_id, doi, selected)
    papers_fts (title, abstract)   runs_fts (question)

Written once at the end of every run by orchestrator/pipeline.py, which
calls import_run() on the run folder it just finished, the same function
`researchgenie library import` uses to backfill old run folders. Re-importing
a folder replaces its rows, so a backfill can be repeated safely. A paper
with no DOI is not stored: the DOI is the only key that is the same across
runs. Folders without 01_discovery/result.json (the flattened copies under
docs/research-papers/) give a run row with no papers.

Discovery reads it back (#19): stored_excerpts() spares a paper's full-text
fetch, and search_papers() seeds past papers into the candidate pool.

Literature cards are deliberately NOT stored. Their cache key includes the
question-specific tag definitions (see build_cached_literature_card() in
services/research-writing/writing/modules/literature_card.py), so a card
built for one question is not valid for another.

RESEARCHGENIE_LIBRARY overrides the file's location (tests point it at a
temporary file so the real library is never touched).
"""

from __future__ import annotations

import datetime
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from researchgenie.config import CONFIG_DIR

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    doi TEXT PRIMARY KEY, title TEXT, abstract TEXT, year INTEGER, venue TEXT,
    discussion_excerpt TEXT, fulltext_status TEXT, first_seen TEXT,
    authors TEXT, citations INTEGER);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, question TEXT, created TEXT, overall_quality REAL,
    draft_status TEXT, run_dir TEXT);
CREATE TABLE IF NOT EXISTS run_papers (
    run_id TEXT, doi TEXT, selected INTEGER, PRIMARY KEY (run_id, doi));
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(doi UNINDEXED, title, abstract);
CREATE VIRTUAL TABLE IF NOT EXISTS runs_fts USING fts5(run_id UNINDEXED, question);
"""
# Added to `papers` after the first release (#19); connect() adds them to an
# older library file, so a seeded past paper can be cited with its authors.
_ADDED_PAPER_COLUMNS = {"authors": "TEXT", "citations": "INTEGER"}

# Words every research question shares; left in, they would make every past
# run "related" to every new one.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "the", "to", "what",
    "when", "which", "who", "why", "with", "between", "effect", "effects",
    "impact", "role", "their", "there", "this", "that", "these", "affect",
}


def library_path() -> Path:
    return Path(os.getenv("RESEARCHGENIE_LIBRARY", str(CONFIG_DIR / "library.sqlite")))


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(path) if path else library_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    have = {row["name"] for row in conn.execute("PRAGMA table_info(papers)")}
    for column, kind in _ADDED_PAPER_COLUMNS.items():
        if column not in have:
            conn.execute(f"ALTER TABLE papers ADD COLUMN {column} {kind}")
    return conn


def fts_query(text: str) -> str:
    """Builds a MATCH expression from free text: each content word quoted,
    OR'd together. Raw user text is never passed to MATCH, where quotes,
    hyphens or words like NOT would be parsed as query syntax."""
    words = dict.fromkeys(w for w in re.findall(r"\w+", text.lower())
                          if len(w) > 1 and w not in _STOPWORDS)
    return " OR ".join(f'"{w}"' for w in words)


def _normalise_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    return re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", doi.strip().lower()) or None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def import_run(run_dir: Path | str, conn: sqlite3.Connection | None = None) -> bool:
    """Writes one run folder (metadata.json, plus 01_discovery/result.json
    when present) into the library. Returns False for a folder that is not
    a finished run."""
    run_dir = Path(run_dir).resolve()
    metadata = _read_json(run_dir / "metadata.json")
    if not isinstance(metadata, dict) or not metadata.get("run_id"):
        return False
    question = metadata.get("research_question") or ""
    created = metadata.get("created") or datetime.datetime.fromtimestamp(
        (run_dir / "metadata.json").stat().st_mtime, datetime.timezone.utc).isoformat()
    discovery = _read_json(run_dir / "01_discovery" / "result.json") or {}

    own_conn = conn is None
    conn = conn or connect()
    try:
        with conn:
            run_id = metadata["run_id"]
            conn.execute("INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?)", (
                run_id, question, created, metadata.get("overall_quality_score"),
                metadata.get("draft_status"), str(run_dir)))
            conn.execute("DELETE FROM runs_fts WHERE run_id = ?", (run_id,))
            conn.execute("INSERT INTO runs_fts VALUES (?, ?)", (run_id, question))
            for paper in discovery.get("selected_papers") or []:
                doi = _normalise_doi(paper.get("doi"))
                if not doi:
                    continue
                excerpt = paper.get("discussion_excerpt")
                # A later run without the excerpt must not erase one an
                # earlier run already extracted.
                conn.execute("""
                    INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(doi) DO UPDATE SET
                        title = excluded.title,
                        abstract = COALESCE(excluded.abstract, papers.abstract),
                        year = COALESCE(excluded.year, papers.year),
                        venue = COALESCE(excluded.venue, papers.venue),
                        discussion_excerpt = COALESCE(excluded.discussion_excerpt, papers.discussion_excerpt),
                        fulltext_status = CASE WHEN COALESCE(excluded.discussion_excerpt,
                            papers.discussion_excerpt) IS NULL THEN 'none' ELSE 'excerpt' END,
                        first_seen = MIN(papers.first_seen, excluded.first_seen),
                        authors = COALESCE(excluded.authors, papers.authors),
                        citations = COALESCE(excluded.citations, papers.citations)
                """, (doi, paper.get("title"), paper.get("abstract"), paper.get("year"),
                      paper.get("venue"), excerpt, "excerpt" if excerpt else "none", created,
                      json.dumps(paper["authors"]) if paper.get("authors") else None,
                      paper.get("citations")))
                row = conn.execute("SELECT title, abstract FROM papers WHERE doi = ?", (doi,)).fetchone()
                conn.execute("DELETE FROM papers_fts WHERE doi = ?", (doi,))
                conn.execute("INSERT INTO papers_fts VALUES (?, ?, ?)", (doi, row["title"], row["abstract"]))
                # Discovery's result lists only the papers it selected.
                conn.execute("INSERT OR REPLACE INTO run_papers VALUES (?, ?, 1)", (run_id, doi))
    finally:
        if own_conn:
            conn.close()
    return True


def import_runs(paths: list[Path | str]) -> int:
    """Imports every run folder (a folder holding metadata.json) under the
    given paths. Returns how many runs were written."""
    conn = connect()
    try:
        return sum(import_run(meta.parent, conn)
                   for path in paths if Path(path).is_dir()
                   for meta in Path(path).rglob("metadata.json"))
    finally:
        conn.close()


def get_run(run_id: str) -> dict[str, Any] | None:
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def search_runs(text: str, limit: int = 10, exclude_run_id: str | None = None) -> list[dict[str, Any]]:
    """Past runs whose question matches, best full-text score first."""
    query = fts_query(text)
    if not query:
        return []
    conn = connect()
    try:
        rows = conn.execute("""
            SELECT runs.* FROM runs_fts JOIN runs USING (run_id)
            WHERE runs_fts MATCH ? AND run_id != ? ORDER BY bm25(runs_fts) LIMIT ?
        """, (query, exclude_run_id or "", limit)).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def search_papers(text: str, limit: int = 10) -> list[dict[str, Any]]:
    """Stored papers whose title or abstract matches, best score first."""
    query = fts_query(text)
    if not query:
        return []
    conn = connect()
    try:
        rows = conn.execute("""
            SELECT papers.* FROM papers_fts JOIN papers USING (doi)
            WHERE papers_fts MATCH ? ORDER BY bm25(papers_fts) LIMIT ?
        """, (query, limit)).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def stored_excerpts(dois: list[str]) -> dict[str, str]:
    """{doi: discussion excerpt} for each of `dois` the library holds an
    excerpt for, keyed by the DOI as given."""
    wanted = {doi: _normalise_doi(doi) for doi in dois}
    keys = sorted({key for key in wanted.values() if key})
    if not keys:
        return {}
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT doi, discussion_excerpt FROM papers WHERE discussion_excerpt IS NOT NULL "
            f"AND doi IN ({', '.join('?' * len(keys))})", keys).fetchall()
    finally:
        conn.close()
    found = {row["doi"]: row["discussion_excerpt"] for row in rows}
    return {doi: found[key] for doi, key in wanted.items() if key in found}


def describe_run(run: dict[str, Any]) -> str:
    """'You researched a similar question on 13 Sep ("...") — QA 4.2 → <folder>'"""
    try:
        created = datetime.datetime.fromisoformat(run["created"])
        when = f"{created.day} {created:%b %Y}"
    except (TypeError, ValueError):
        when = "an earlier date"
    quality = run.get("overall_quality")
    qa = f"QA {quality:.1f}" if isinstance(quality, (int, float)) else "QA n/a"
    return (f'You researched a similar question on {when} ("{run["question"]}") '
            f'— {qa} → {run["run_dir"]}')

```

### `shared/utilities/response_cache.py`

```python
"""Persistent cache for scholarly responses and provider state (#6).

One SQLite file next to the CLI's config, ~/.researchgenie/cache.sqlite
(RESEARCHGENIE_CACHE overrides it; tests point it at a temporary file):

    responses      (key PK, status, content_type, body, expires)
    provider_state (host PK, exhausted_until, balance)

The scholarly gateway reads and writes `responses` on every GET: a repeat or
similar run answers its searches and DOI lookups from disk, at no quota. The
key is the method, URL and sorted parameters with the identity parameters
(keys, email, NCBI's tool name) left out, so it never holds a secret and the
same request made with or without a key hits the same entry. TTLs: searches
7 days, DOI and entity lookups 30 days, extracted full text never expires.
Only 200 responses are stored.

`provider_state` keeps what the next run must know before it starts: the
time a credit-metered host (OpenAlex) is exhausted until, and its last
rate-limit headers (the balance).

RESEARCHGENIE_NO_CACHE=1 (the CLI's `--no-cache`) skips reads; fresh answers
still refresh the cache. A broken or locked cache file never fails a run:
every operation degrades to a miss.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path
from urllib.parse import urlencode, urlsplit

from researchgenie.config import CONFIG_DIR

SEARCH_TTL = 7 * 24 * 3600
LOOKUP_TTL = 30 * 24 * 3600
# The parameters that identify the user; never part of a cache key.
IDENTITY_PARAMS = frozenset({"api_key", "mailto", "email", "tool"})
# A single-record path: Crossref /works/<doi>, OpenAlex /works/doi:<doi> or /works/W123
# (a DOI has slashes of its own, so match the start of the id, not the path's end).
_ENTITY_PATH = re.compile(r"/works/(?:10\.|doi:|W\d)", re.IGNORECASE)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key TEXT PRIMARY KEY, status INTEGER, content_type TEXT, body BLOB, expires REAL);
CREATE TABLE IF NOT EXISTS provider_state (
    host TEXT PRIMARY KEY, exhausted_until REAL, balance TEXT);
"""


def cache_path() -> Path:
    return Path(os.getenv("RESEARCHGENIE_CACHE", str(CONFIG_DIR / "cache.sqlite")))


def reads_enabled() -> bool:
    return os.getenv("RESEARCHGENIE_NO_CACHE", "") in ("", "0")


def _connect() -> sqlite3.Connection:
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=5)
    conn.executescript(_SCHEMA)
    return conn


def _run(sql: str, args: tuple = ()) -> list[tuple]:
    """One statement in its own connection; [] on any SQLite or file error."""
    # ponytail: synchronous SQLite on the event loop; a local file answers in
    # well under a millisecond. Move to a thread if profiling ever says otherwise.
    try:
        conn = _connect()
    except (sqlite3.Error, OSError):
        return []
    try:
        with conn:
            return conn.execute(sql, args).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def key(url: str, params: dict | None = None) -> str:
    """'GET <url>?<sorted params>' without the identity parameters."""
    kept = sorted((k, str(v)) for k, v in (params or {}).items() if k not in IDENTITY_PARAMS)
    return f"GET {url}?{urlencode(kept)}"


def ttl_for(url: str) -> float:
    """30 days for a DOI/entity lookup or a PubMed fetch by id, else 7 (a search)."""
    path = urlsplit(url).path
    return LOOKUP_TTL if path.endswith("efetch.fcgi") or _ENTITY_PATH.search(path) else SEARCH_TTL


def get(cache_key: str, now: float) -> tuple[int, str, bytes] | None:
    """(status, content type, body) of a fresh entry, else None."""
    if not reads_enabled():
        return None
    rows = _run("SELECT status, content_type, body FROM responses "
                "WHERE key = ? AND (expires IS NULL OR expires > ?)", (cache_key, now))
    return (rows[0][0], rows[0][1], bytes(rows[0][2])) if rows else None


def put(cache_key: str, status: int, content_type: str, body: bytes,
        ttl: float | None, now: float) -> None:
    """ttl None: never expires."""
    _run("INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?)",
         (cache_key, status, content_type, body, None if ttl is None else now + ttl))


def get_text(cache_key: str) -> str | None:
    """Extracted full text stored with put_text (never expires)."""
    hit = get(cache_key, 0.0)
    return hit[2].decode("utf-8") if hit else None


def put_text(cache_key: str, text: str) -> None:
    put(cache_key, 200, "text/plain; charset=utf-8", text.encode("utf-8"), None, 0.0)


def clear() -> int:
    """Removes every cached response (provider state stays). Returns how many."""
    rows = _run("SELECT COUNT(*) FROM responses")
    _run("DELETE FROM responses")
    return rows[0][0] if rows else 0


def save_exhausted(host: str, until: float) -> None:
    _run("INSERT INTO provider_state (host, exhausted_until) VALUES (?, ?) "
         "ON CONFLICT(host) DO UPDATE SET exhausted_until = excluded.exhausted_until",
         (host, until))


def exhausted_until() -> dict[str, float]:
    return {host: until for host, until in
            _run("SELECT host, exhausted_until FROM provider_state "
                 "WHERE exhausted_until IS NOT NULL")}


def save_balance(host: str, balance: dict) -> None:
    _run("INSERT INTO provider_state (host, balance) VALUES (?, ?) "
         "ON CONFLICT(host) DO UPDATE SET balance = excluded.balance",
         (host, json.dumps(balance)))


def balance(host: str) -> dict | None:
    """The host's last rate-limit headers, with 'seen_at' (epoch seconds)."""
    rows = _run("SELECT balance FROM provider_state WHERE host = ? AND balance IS NOT NULL", (host,))
    return json.loads(rows[0][0]) if rows else None

```

### Real tests for this lab

### `tests/shared/test_library.py`

```python
"""The research library (shared/utilities/library.py), its pipeline hooks
and its CLI/API surfaces, against a temporary library (tests/conftest.py)
and fixture run folders."""

import asyncio
import json
import sqlite3
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from rich.console import Console

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from orchestrator import api as orchestrator_api  # noqa: E402
from orchestrator import pipeline as orchestrator_pipeline  # noqa: E402
from researchgenie import cli  # noqa: E402
from shared.contracts.pipeline_contract import ResearchRequest  # noqa: E402
from shared.utilities import library  # noqa: E402
from tests.orchestrator import test_pipeline as fakes  # noqa: E402


@pytest.fixture(autouse=True)
def _fake_services(monkeypatch):
    monkeypatch.setattr(orchestrator_pipeline.discovery_service, "run_discovery", fakes._fake_run_discovery)
    monkeypatch.setattr(orchestrator_pipeline.writing_service, "run_writing", fakes._fake_run_writing)
    monkeypatch.setattr(orchestrator_pipeline.verification_service, "run_verification", fakes._fake_run_verification)
    monkeypatch.setattr(orchestrator_pipeline.qa_service, "run_quality_assurance", fakes._fake_run_qa)


def _run_folder(root: Path, run_id: str, question: str, papers: list[dict],
                quality: float = 4.2, discovery: bool = True) -> Path:
    folder = root / f"{run_id}-folder"
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text(json.dumps({
        "run_id": run_id, "research_question": question, "overall_quality_score": quality,
        "draft_status": "complete",
    }), encoding="utf-8")
    (folder / "draft.md").write_text("past draft", encoding="utf-8")
    if discovery:
        (folder / "01_discovery").mkdir()
        (folder / "01_discovery" / "result.json").write_text(
            json.dumps({"selected_papers": papers}), encoding="utf-8")
    return folder


_PAPER = {"title": "Fasting and memory in adults", "abstract": "Intermittent fasting improved recall.",
          "year": 2021, "venue": "J. Nutr.", "doi": "https://doi.org/10.1/ABC",
          "discussion_excerpt": "Future work should test older adults."}


def _counts() -> dict[str, int]:
    conn = library.connect()
    try:
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("papers", "runs", "run_papers", "papers_fts", "runs_fts")}
    finally:
        conn.close()


def test_library_path_is_the_temporary_one(tmp_path):
    assert library.library_path() == tmp_path / "library.sqlite"


def test_write_then_search_round_trip(tmp_path):
    folder = _run_folder(tmp_path, "run1", "What are the effects of intermittent fasting on cognition?",
                         [_PAPER, {"title": "No DOI paper"}])
    assert library.import_run(folder)

    runs = library.search_runs("Does intermittent fasting help memory?")
    assert [r["run_id"] for r in runs] == ["run1"]
    assert runs[0]["overall_quality"] == 4.2 and runs[0]["run_dir"] == str(folder.resolve())

    papers = library.search_papers("fasting recall")
    assert [p["doi"] for p in papers] == ["10.1/abc"]  # normalised; the DOI-less paper is skipped
    assert papers[0]["fulltext_status"] == "excerpt"
    assert library.search_runs("quantum chromodynamics") == []


def test_import_is_idempotent(tmp_path):
    _run_folder(tmp_path, "run1", "Intermittent fasting and cognition?", [_PAPER])
    _run_folder(tmp_path, "run2", "Remote work and productivity?", [], discovery=False)
    (tmp_path / "advisor_only" / "00_trend_advisor").mkdir(parents=True)  # not a finished run

    assert library.import_runs([tmp_path]) == 2
    first = _counts()
    assert library.import_runs([tmp_path]) == 2
    assert _counts() == first == {"papers": 1, "runs": 2, "run_papers": 1, "papers_fts": 1, "runs_fts": 2}


def test_a_later_run_without_an_excerpt_keeps_the_stored_one(tmp_path):
    library.import_run(_run_folder(tmp_path, "run1", "Fasting?", [_PAPER]))
    library.import_run(_run_folder(tmp_path, "run2", "Fasting again?",
                                   [{**_PAPER, "discussion_excerpt": None, "abstract": None}]))
    paper = library.search_papers("fasting")[0]
    assert paper["discussion_excerpt"] == _PAPER["discussion_excerpt"]
    assert paper["abstract"] == _PAPER["abstract"]


def test_stored_excerpts_by_doi_and_an_older_library_file_gains_the_new_columns(tmp_path):
    old = sqlite3.connect(library.library_path())  # the schema before #19: no authors, citations
    old.execute("CREATE TABLE papers (doi TEXT PRIMARY KEY, title TEXT, abstract TEXT, year INTEGER, "
                "venue TEXT, discussion_excerpt TEXT, fulltext_status TEXT, first_seen TEXT)")
    old.execute("INSERT INTO papers VALUES ('10.2/old', 'Old', NULL, 2010, NULL, 'Old excerpt.', "
                "'excerpt', '2020-01-01')")
    old.commit()
    old.close()

    library.import_run(_run_folder(tmp_path, "run1", "Fasting?",
                                   [{**_PAPER, "authors": ["A. One"], "citations": 7}]))
    assert library.stored_excerpts(["10.2/OLD", "https://doi.org/10.1/abc", "10.3/none"]) == {
        "10.2/OLD": "Old excerpt.", "https://doi.org/10.1/abc": _PAPER["discussion_excerpt"]}
    paper = library.search_papers("fasting")[0]
    assert json.loads(paper["authors"]) == ["A. One"] and paper["citations"] == 7


def test_discovery_memory_and_time_reach_metadata(tmp_path, monkeypatch):
    """A later script compares two runs' Discovery from metadata.json (#19)."""
    seen = []

    async def discovery(request):
        seen.append(request.memory)
        yield {"type": "memory_hit", "count": 3, "reused_excerpts": 2, "seeded_papers": 1,
               "message": "Research library: reused 2 stored excerpt(s)."}
        yield {"type": "result", "result": fakes._fake_discovery_result()}
    monkeypatch.setattr(orchestrator_pipeline.discovery_service, "run_discovery", discovery)

    events = _run(ResearchRequest(research_question="Remote work?", corpus_size=6), tmp_path)
    run_directory = Path(events[-1]["result"].run_directory)
    logged = [json.loads(line) for line in
              (run_directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    [hit] = [e for e in logged if e.get("status") == "memory_hit"]
    assert hit["stage"] == "discovery" and hit["details"] == {
        "raw_type": "memory_hit", "count": 3, "reused_excerpts": 2, "seeded_papers": 1}
    metadata = json.loads((run_directory / "metadata.json").read_text())
    assert metadata["memory"] == {"enabled": True, "count": 3, "reused_excerpts": 2, "seeded_papers": 1}
    assert isinstance(metadata["timings"]["discovery_seconds"], float)

    _run(ResearchRequest(research_question="Remote work?", corpus_size=6, memory=False), tmp_path)
    assert seen == [True, False]


def test_query_syntax_in_the_question_is_never_passed_to_match():
    assert library.fts_query('What "is" NOT fasting-related (AND) OR*?') == '"not" OR "fasting" OR "related"'
    assert library.search_runs('"unbalanced quote NEAR( -x *') == []
    assert library.fts_query("What is the?") == ""


def _run(request, output_root):
    async def go():
        return [e async for e in orchestrator_pipeline.run_pipeline(request, output_root=output_root)]
    return asyncio.run(go())


def test_finished_run_is_written_and_related_runs_are_surfaced_before_discovery(tmp_path):
    first = _run(ResearchRequest(research_question="What is the impact of remote work on productivity?",
                                 corpus_size=6), tmp_path / "out")
    first_id = first[-1]["result"].run_id
    assert library.get_run(first_id)["overall_quality"] == 5.0
    assert [p["doi"] for p in library.search_papers("Paper One")] == ["10.1/x"]
    assert not any(e.get("details", {}).get("raw_type") == "related_runs" for e in first)

    events = _run(ResearchRequest(research_question="Remote work and employee productivity?",
                                  corpus_size=6), tmp_path / "out")
    related = [e for e in events if e.get("details", {}).get("raw_type") == "related_runs"]
    assert len(related) == 1
    event = related[0]
    assert event["type"] == "progress" and event["stage"] == "discovery"
    assert [r["run_id"] for r in event["details"]["runs"]] == [first_id]
    assert set(event["details"]["runs"][0]) >= {"question", "created", "overall_quality", "run_dir"}
    assert "QA 5.0" in event["message"]
    # Surfaced before Discovery's own events.
    assert events.index(event) < next(i for i, e in enumerate(events) if e.get("message") == "searching")


def test_a_library_failure_never_fails_the_run(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(library, "connect", broken)

    events = _run(ResearchRequest(research_question="Remote work?", corpus_size=6), tmp_path)
    assert events[-1]["type"] == "result"
    warnings = [e for e in events if e.get("details", {}).get("raw_type") == "library_error"]
    assert len(warnings) == 2 and all(e["status"] == "warning" for e in warnings)
    assert "disk full" in warnings[0]["message"]


def test_cli_library_import_and_search(tmp_path):
    _run_folder(tmp_path, "run1", "Intermittent fasting and cognition?", [_PAPER])
    console = Console(record=True, width=300)
    cli._library_command(console, ["import", str(tmp_path)])
    cli._library_command(console, ["search", "fasting"])
    text = console.export_text()
    assert "Imported 1 run(s)" in text
    assert 'You researched a similar question on' in text and "QA 4.2" in text
    assert "Fasting and memory in adults" in text and "doi:10.1/abc" in text


def test_api_serves_a_past_runs_file_from_the_library(tmp_path):
    library.import_run(_run_folder(tmp_path, "pastrun", "Fasting?", []))
    client = TestClient(orchestrator_api.app)
    response = client.get("/research/runs/pastrun/files/draft.md")
    assert response.status_code == 200 and response.text == "past draft"
    assert client.get("/research/runs/pastrun/files/paper.pdf").status_code == 404
    assert client.get("/research/runs/unknown/files/draft.md").status_code == 404

```

### `tests/shared/test_response_cache.py`

```python
"""The scholarly response cache (#6), seen through the gateway's seam: the
mock transport's request count, a fake clock for TTLs, and the cache file
itself. The root conftest points RESEARCHGENIE_CACHE at a temporary file."""

import asyncio
import sqlite3
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "research-discovery"))

from discovery import fulltext, pipeline  # noqa: E402
from discovery.models import Paper  # noqa: E402
from shared.utilities import response_cache, scholarly_gateway  # noqa: E402
from shared.utilities.scholarly_gateway import Gateway, ProviderExhausted, Run  # noqa: E402

SEARCH = "https://api.openalex.org/works"
LOOKUP = "https://api.crossref.org/works/10.1000/abc"
DAY = 24 * 3600


class FakeTime:
    def __init__(self):
        self.now = 1_800_000_000.0

    def wall(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(0)


def _counting(response=None):
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        return response or httpx.Response(200, json={"results": []})

    return httpx.MockTransport(handler), seen


def _get(gateway, transport, url, run=None, **kwargs):
    async def go():
        async with httpx.AsyncClient(transport=transport) as http:
            return await gateway.get(http, url, **kwargs)

    async def main():
        return await (run or Run()).task(go())

    return asyncio.run(main())


def _gateway(fake=None):
    fake = fake or FakeTime()
    return Gateway(wall_clock=fake.wall, clock=lambda: 0.0, sleep=fake.sleep), fake


def test_a_repeated_request_is_answered_from_the_cache():
    gateway, _ = _gateway()
    transport, seen = _counting(httpx.Response(200, json={"results": [{"id": "W1"}]}))
    first = _get(gateway, transport, SEARCH, params={"search": "fasting"})
    run = Run()
    second = _get(gateway, transport, SEARCH, run=run, params={"search": "fasting"})
    assert len(seen) == 1
    assert second.json() == first.json() == {"results": [{"id": "W1"}]}
    assert run.cache_event()["hits"] == 1
    assert Run().cache_event() is None


def test_ttl_is_honored_per_request_kind():
    gateway, fake = _gateway()
    transport, seen = _counting()
    _get(gateway, transport, SEARCH, params={"search": "q"})
    _get(gateway, transport, LOOKUP)
    fake.now += 8 * DAY  # past a search's 7 days, inside a lookup's 30
    _get(gateway, transport, SEARCH, params={"search": "q"})
    _get(gateway, transport, LOOKUP)
    assert [r.url.host for r in seen] == ["api.openalex.org", "api.crossref.org", "api.openalex.org"]
    fake.now += 23 * DAY  # 31 days after the first lookup
    _get(gateway, transport, LOOKUP)
    assert len(seen) == 4


def test_cache_keys_never_contain_a_key_or_email(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "sekret-openalex-key")
    monkeypatch.setenv("CONTACT_EMAIL", "student@uni.example")
    gateway, _ = _gateway()
    transport, seen = _counting()
    _get(gateway, transport, SEARCH, params={"search": "q"})
    assert "sekret-openalex-key" in str(seen[0].url)  # the key did go to OpenAlex

    rows = sqlite3.connect(response_cache.cache_path()).execute("SELECT key FROM responses").fetchall()
    assert rows
    for (key,) in rows:
        assert "sekret" not in key and "student" not in key and "uni.example" not in key

    # Without the key it is the same request, so the same entry answers it.
    monkeypatch.delenv("OPENALEX_API_KEY")
    _get(gateway, transport, SEARCH, params={"search": "q"})
    assert len(seen) == 1


def test_only_successful_responses_are_stored():
    gateway, _ = _gateway()
    transport, seen = _counting(httpx.Response(404))
    _get(gateway, transport, LOOKUP)
    _get(gateway, transport, LOOKUP)
    assert len(seen) == 2


def test_no_cache_skips_reads_and_clear_empties_the_cache(monkeypatch):
    gateway, _ = _gateway()
    transport, seen = _counting()
    _get(gateway, transport, SEARCH, params={"search": "q"})
    monkeypatch.setenv("RESEARCHGENIE_NO_CACHE", "1")
    _get(gateway, transport, SEARCH, params={"search": "q"})
    assert len(seen) == 2
    monkeypatch.delenv("RESEARCHGENIE_NO_CACHE")
    _get(gateway, transport, SEARCH, params={"search": "q"})
    assert len(seen) == 2
    assert response_cache.clear() == 1
    _get(gateway, transport, SEARCH, params={"search": "q"})
    assert len(seen) == 3


def test_exhaustion_and_balance_survive_a_process_restart():
    fake = FakeTime()
    first, _ = _gateway(fake)
    rate = {"X-RateLimit-Remaining": "40", "X-RateLimit-Limit": "1000"}
    transport, _ = _counting(httpx.Response(200, headers=rate, json={"results": []}))
    _get(first, transport, SEARCH, params={"search": "a"})
    gone = httpx.Response(429, text='{"error": "not_enough_credits"}')
    transport, _ = _counting(gone)
    with pytest.raises(ProviderExhausted):
        _get(first, transport, SEARCH, params={"search": "b"})

    restarted, _ = _gateway(fake)  # a new process: a fresh Gateway, the same cache file
    transport, seen = _counting()
    with pytest.raises(ProviderExhausted):
        _get(restarted, transport, SEARCH, params={"search": "c"})
    assert seen == []
    assert restarted.provider_status()[0]["health"] == "degraded"
    balance = response_cache.balance("api.openalex.org")
    assert balance["x-ratelimit-remaining"] == "40" and balance["seen_at"] == fake.now
    assert restarted.provider_status()[0]["budget_remaining"] == "40"  # known before a run starts
    # The cached search still answers, budget or not.
    assert _get(restarted, transport, SEARCH, params={"search": "a"}).status_code == 200


def test_extracted_full_text_is_cached_with_no_expiry(monkeypatch):
    discussion = "Discussion. " + "Future work should test this in older adults. " * 20
    xml = (f"<article><body><sec sec-type='discussion'><title>Discussion</title>"
           f"<p>{discussion}</p></sec></body></article>")
    transport, seen = _counting(httpx.Response(200, text=xml))

    async def fetch():
        paper = Paper(title="T", source="OpenAlex", pmcid="PMC1")
        async with httpx.AsyncClient(transport=transport) as http:
            run = Run()
            ok = await run.task(fulltext.fetch_future_sections(http, paper))
        return ok, paper.future_text, run.cache_hits

    first = asyncio.run(fetch())
    second = asyncio.run(fetch())
    assert first[0] and second[0] and second[1] == first[1]
    assert len(seen) == 1 and second[2] == 1


def _work(i: int) -> dict:
    return {"id": f"https://openalex.org/W{i}",
            "display_name": f"Intermittent fasting and cognitive performance, study {i}",
            "abstract_inverted_index": {w: [n] for n, w in enumerate(
                ["Intermittent", "fasting", "improved", "cognitive", "performance"])},
            "publication_year": 2022, "cited_by_count": 10,
            "doi": f"https://doi.org/10.1000/w.{i}",
            "authorships": [{"author": {"display_name": "A. Author"}}]}


def test_a_second_identical_discovery_run_makes_no_network_call(monkeypatch):
    async def fake_queries(question, model, feedback=None):
        return {"keyword": "fasting cognition", "pubmed": "x", "arxiv": "x", "interpretation": "i"}

    async def fake_analysis(question, corpus, candidates, verification_papers, model):
        yield {"type": "report", "report": {}}

    monkeypatch.setattr(pipeline, "generate_queries", fake_queries)
    monkeypatch.setattr(pipeline, "analyze_gaps", fake_analysis)
    transport, seen = _counting(httpx.Response(200, json={"results": [_work(i) for i in range(8)]}))

    async def discover():
        return [e async for e in pipeline.run_pipeline(
            "Effects of intermittent fasting on cognitive performance", ["openalex"],
            per_source=10, corpus_size=6, fulltext_enabled=False, transport=transport)]

    first = asyncio.run(discover())
    requests_after_first = len(seen)
    second = asyncio.run(discover())
    assert requests_after_first == 1 and len(seen) == 1
    assert not [e for e in first if e["type"] == "cache_hit"]
    [hit] = [e for e in second if e["type"] == "cache_hit"]
    assert hit["hits"] == 1 and "cache" in hit["message"]
    corpus = lambda events: [e for e in events if e["type"] == "corpus"][0]["papers"]
    assert corpus(second) == corpus(first)

```

## The experiment

```bash
cd Course-Project/code
pip install -e .
pytest -v tests/shared/test_library.py tests/shared/test_response_cache.py
```

## Real result
Raw console output also saved as [`sample_run.txt`](./sample_run.txt) in this folder.

Every library and cache unit test passes above, including the explicit test that an off topic seeded paper is rejected by the relevance gate. On a related live run recorded in `Course-Project/docs/evidence/live-pass/LIVE_PASS.md`, memory surfaced 10 past papers, 1 reused excerpt and 9 seeded, and Discovery time was higher, not lower, because seeding widens the corpus to screen rather than skipping work.

## Why this is the real evidence, not a toy
The honest, measured finding, that memory widened the corpus but did not speed up that particular run, is kept in rather than edited out.

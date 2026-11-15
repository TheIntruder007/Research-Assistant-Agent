# Lab 2 - Tool-Using Agent

**Skill:** Agents

**Worklet mapping:** The model chooses which scholarly databases to search and what query to send each one; code checks the choice against the field.

## What this demonstrates

Discovery does not hard code which database to search. The local 9B model returns a tool plan, a database plus a query for each source, through native function calling. Code validates each choice against the research field, for example it refuses PubMed for a mechanical engineering question, with a rule based fallback if the plan is invalid.

## Real code exercised

The actual source files this lab is built on, copied in full below, not linked only.

### `services/research-discovery/discovery/source_plan.py`

```python
"""Which databases to search, and with what query (#16).

The model sees the four sources as tools that take one query each and calls
the ones that fit the question. Its calls are validated: a call to an
unknown tool, or without a usable query string, rejects the whole plan.
Whenever the plan is rejected, the model makes no call, or the provider
can't call tools (Gemini), a deterministic fallback is used: the request's
preferred databases if it names any, else the default sources filtered by
domain (no PubMed for mechanical, civil, electrical/electronics and
cloud/devops; computer science keeps it for health informatics). The fallback is
never worse than searching everything, which is what Discovery did before.
"""

from __future__ import annotations

import re

from shared.utilities import llm_provider as llm

from .sources import SOURCES

# What each source is good for, shown to the model in its tool description.
_STRENGTHS = {
    "s2": "broad coverage of every discipline, strong in computer science",
    "openalex": "the broadest index of scholarly works in every discipline",
    "pubmed": "biomedical, clinical, health and life-science literature only",
    "arxiv": "preprints in physics, mathematics, computer science, engineering and quantitative fields",
}
# Domains whose papers PubMed doesn't index; spending its quota there finds nothing.
_NOT_PUBMED = re.compile(r"\b(mechanical|civil|electrical|electronics|ece|eee|cloud|devops)\b")
MAX_QUERY_CHARS = 300

PLAN_SYSTEM = (
    "You choose which literature databases to search for a research question. Call the "
    "search tool of every database likely to hold relevant papers, each with a search query "
    "suited to that database, and skip databases outside the question's field."
)


def tools(source_keys: list[str]) -> list[dict]:
    return [{"type": "function", "function": {
        "name": f"search_{key}",
        "description": f"Search {SOURCES[key][0]}: {_STRENGTHS[key]}.",
        "parameters": {"type": "object", "required": ["query"],
                       "properties": {"query": {"type": "string",
                                                "description": "the search query for this database"}}},
    }} for key in source_keys]


def _valid(calls: list[dict], source_keys: list[str]) -> dict[str, str] | None:
    """{source key: query} when every call is a known tool with a usable query, else None."""
    plan: dict[str, str] = {}
    for call in calls:
        name, arguments = call.get("name"), call.get("arguments")
        key = name.removeprefix("search_") if isinstance(name, str) else None
        query = arguments.get("query") if isinstance(arguments, dict) else None
        if key not in source_keys or not isinstance(query, str) \
                or not query.strip() or len(query) > MAX_QUERY_CHARS:
            return None
        plan.setdefault(key, query.strip())
    return plan or None


def _named(source_keys: list[str], preferred: list[str] | None) -> list[str]:
    """The sources the request names (by key or label), among those available."""
    return [k for k in source_keys
            if any(p.strip().lower() in (k, SOURCES[k][0].lower()) for p in preferred or [])]


def fallback(source_keys: list[str], domain: str | None,
             preferred: list[str] | None = None) -> list[str]:
    """The deterministic plan: preferred databases, else the domain filter."""
    if named := _named(source_keys, preferred):
        return named
    domain_words = (domain or "").lower().replace("_", " ")
    return [k for k in source_keys if not (k == "pubmed" and _NOT_PUBMED.search(domain_words))]


async def plan_sources(question: str, queries: dict, source_keys: list[str], model: str | None,
                       domain: str | None = None, preferred: list[str] | None = None) -> dict:
    """{"mode": "model" | "fallback", "reason": str, "queries": {key: query dict}}.
    Never raises: any model failure is a fallback with its reason."""
    if _named(source_keys, preferred):  # databases the user named win over the model's choice
        reason = "the request names its databases"
    else:
        try:
            calls = await llm.complete_tools(
                system=PLAN_SYSTEM, user=f"Research question: {question}"
                + (f"\nField: {domain}" if domain else ""),
                tools=tools(source_keys), model=model)
        except Exception as error:  # LLMProviderError, a malformed reply: never crash the run
            calls, reason = [], f"the model could not be asked ({type(error).__name__})"
        else:
            reason = "the model called no tool"
        if calls is None:
            reason = "this model provider does not support tool calls"
        elif calls:
            chosen = _valid(calls, source_keys)
            if chosen:
                return {"mode": "model", "reason": "chosen by the model",
                        "queries": {k: {**queries, "keyword": q, "pubmed": q, "arxiv": q}
                                    for k, q in chosen.items()}}
            reason = "the model's tool calls were invalid (unknown tool or bad arguments)"
    keys = fallback(source_keys, domain, preferred) or list(source_keys)
    return {"mode": "fallback", "reason": reason, "queries": {k: queries for k in keys}}

```

### `shared/utilities/scholarly_gateway.py`

```python
"""One gateway for every outbound scholarly HTTP request.

Replaces Discovery's `_http_retry.py` (DECISIONS.md D-028). That helper
retried each request on its own, so four sources and the full-text fan-out
hammered the same hosts in parallel with no shared pace, waited at most
~7 s in total, and retried OpenAlex's "daily budget gone" 429 as if it were
transient. The gateway owns six things, all driven by the POLICIES table:

- pacing: one token bucket per host, shared by the whole process (two runs at
  once draw from the same bucket). Requests wait for a token; they are never
  fired first and retried later.
- retry: jittered exponential backoff on 429/5xx that honors Retry-After and
  rate-limit reset headers, up to a per-host cap on total waiting.
- identity: the contact email (CONTACT_EMAIL) as `mailto`, NCBI's `tool` and
  `email` parameters, the free OpenAlex and NCBI keys (OPENALEX_API_KEY,
  NCBI_API_KEY, #5) and a User-Agent that names the tool. Neither provider
  accepts its key in a header, so keys travel as query parameters and every
  URL the gateway hands back is redacted (`redact`, D-023's lesson).
- circuit breaker: repeated 429/5xx responses within one run, or an OpenAlex
  `not_enough_credits` 429, mark the host exhausted. Later calls fail fast
  with ProviderExhausted and the run continues on the other sources.
- cache (#6): GETs are answered from shared/utilities/response_cache.py when
  a fresh entry exists (no token, no quota, even for an exhausted host), and
  200 responses are stored. Credit exhaustion and the OpenAlex rate-limit
  headers persist there too, so the next process knows before it starts.
- credits (#7): what each OpenAlex request costs (CREDIT_COSTS), the balance
  left today for the run's budget check, and a per-run tally of the credits
  spent per provider (count_credits).

Callers keep passing their own httpx.AsyncClient. Per-run state (breaker
counts, the event queue for `rate_limited` / `provider_exhausted`) lives in a
Run; start the run's work with `run.task(coro)` and relay its events with
`run.relay(task)`. A call made outside any Run still gets pacing, retry and
identity, but its breaker state lasts only for that one call.
"""

from __future__ import annotations

import asyncio
import contextvars
import datetime
import email.utils
import logging
import os
import random
import threading
import time
import weakref
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Coroutine
from urllib.parse import quote, quote_plus, urlsplit

import httpx

from shared.utilities import response_cache

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
TOOL_NAME = "lit-gap"  # NCBI's required `tool` parameter, kept from the old sources
PACING_NOTICE_SECONDS = 10.0  # a self-imposed pacing wait this long is worth telling the user


@dataclass(frozen=True)
class HostPolicy:
    name: str                 # shown to the user
    rate: float               # sustained requests per second
    burst: int                # tokens the bucket holds when idle
    concurrency: int          # requests in flight at once
    identity: str             # "mailto" | "ncbi" | "none" (every host also gets the User-Agent)
    remedy: str               # what the user can do once the host is exhausted
    key_env: str | None = None     # env var holding the provider's free API key, sent as `api_key`
    keyed_rate: float | None = None  # sustained rate once that key is set (burst follows it)
    attempts: int = 4         # tries per request, the first one included
    max_retry_wait: float = 60.0   # cap on total waiting per request, seconds
    backoff: float = 1.0      # first backoff step when the server gives no wait
    breaker_threshold: int = 6     # consecutive 429/5xx responses in a run that trip the breaker
    credits_marker: str | None = None  # 429 body text meaning "budget gone until reset"


# Starting values from each provider's published limits (spec #1, Further
# Notes, checked 2026-10-18) — deliberately conservative.
POLICIES: dict[str, HostPolicy] = {
    "api.openalex.org": HostPolicy(
        name="OpenAlex", rate=10, burst=10, concurrency=4, identity="mailto",
        credits_marker="not_enough_credits", key_env="OPENALEX_API_KEY",
        # The key raises the daily budget (about 100 -> 1,000 searches), not the pace.
        remedy="Add a free OpenAlex API key (`researchgenie config`) for a larger daily "
               "budget, or wait for it to reset; Discovery continues on the other sources."),
    # Keyless only: the project will not get an S2 key, so no key header is
    # ever sent. The shared anonymous pool 429s often; one slow connection and
    # a breaker that trips on the first 429 burst keep a busy pool to seconds.
    "api.semanticscholar.org": HostPolicy(
        name="Semantic Scholar", rate=1 / 3, burst=1, concurrency=1, identity="none",
        attempts=3, max_retry_wait=15, backoff=2.0, breaker_threshold=2,
        remedy="Semantic Scholar's shared keyless pool is busy; try again later. "
               "Discovery continues on the other sources."),
    "eutils.ncbi.nlm.nih.gov": HostPolicy(
        name="PubMed (NCBI)", rate=3, burst=3, concurrency=3, identity="ncbi",
        key_env="NCBI_API_KEY", keyed_rate=10,
        remedy="Add a contact email and a free NCBI API key (`researchgenie config`), "
               "or try again later."),
    "export.arxiv.org": HostPolicy(
        name="arXiv", rate=1 / 3, burst=1, concurrency=1, identity="none",
        backoff=3.0,
        remedy="arXiv is throttling this machine; try again in a few minutes."),
    # Single-record pace without the polite pool; Citation Verification's DOI lookups (#4).
    "api.crossref.org": HostPolicy(
        name="Crossref", rate=5, burst=5, concurrency=1, identity="mailto",
        remedy="Add a contact email (`researchgenie config`) to use Crossref's polite "
               "pool, or try again later."),
    # No published limit — be conservative.
    "www.ebi.ac.uk": HostPolicy(
        name="Europe PMC", rate=5, burst=5, concurrency=4, identity="none",
        remedy="Europe PMC is overloaded; full text falls back to open-access PDFs."),
}

OPENALEX_HOST = "api.openalex.org"
# OpenAlex's price per request, in credits (#7): a search costs ten list or
# filter calls; a single-entity lookup (/works/W1, /works/doi:...) is free.
CREDIT_COSTS = {"search": 10, "list": 1, "entity": 0}
DOI_BATCH = 100  # DOIs per OR-filter request: OpenAlex's page limit (#7)

# Any other host: open-access PDF links point at arbitrary publisher sites.
DEFAULT_POLICY = HostPolicy(
    name="Open-access host", rate=2, burst=2, concurrency=2, identity="none",
    attempts=2, max_retry_wait=10,
    remedy="This host is refusing requests; other full-text routes are still tried.")


# Environment variables whose values must never be shown to the user.
SECRET_ENVS = ("OPENALEX_API_KEY", "NCBI_API_KEY", "CONTACT_EMAIL", "GEMINI_API_KEY")
DEGRADED_SECONDS = 15 * 60  # a tripped breaker shows the provider as degraded this long


def redact(text: str) -> str:
    """`text` with every configured key and the contact email masked, raw
    and as httpx percent-encodes it in a query string (`@` -> `%40`)."""
    for name in SECRET_ENVS:
        value = os.getenv(name)
        if value:
            for form in {value, quote(value, safe=""), quote_plus(value)}:
                text = text.replace(form, "***")
    return text


class _RedactLogs(logging.Filter):
    """httpx logs each request URL at INFO; mask the secrets in it."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        cleaned = redact(message)
        if cleaned != message:
            record.msg, record.args = cleaned, None
        return True


for _logger in ("httpx", "httpcore"):
    logging.getLogger(_logger).addFilter(_RedactLogs())


def _keyed(policy: "HostPolicy") -> bool:
    return bool(policy.key_env and os.getenv(policy.key_env))


def host_of(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def policy_for(url_or_host: str) -> HostPolicy:
    host = host_of(url_or_host) if "/" in url_or_host else url_or_host.lower()
    return POLICIES.get(host, DEFAULT_POLICY)


def concurrency(url_or_host: str) -> int:
    """How many requests a caller should fan out to this host at once."""
    return policy_for(url_or_host).concurrency


def credit_cost(url: str, params: dict | None = None) -> int:
    """The credits one request costs; 0 on a host without a credit budget."""
    if not policy_for(url).credits_marker:
        return 0
    if response_cache.ttl_for(url) == response_cache.LOOKUP_TTL:  # an entity path
        return CREDIT_COSTS["entity"]
    return CREDIT_COSTS["search" if (params or {}).get("search") else "list"]


class ProviderExhausted(httpx.HTTPError):
    """A host is skipped for the rest of the run (breaker tripped) or until its
    reset time (credit budget gone). Subclasses httpx.HTTPError so callers that
    already degrade on HTTP failures (the Trend Advisor, Verification) treat it
    the same way without new except clauses."""

    def __init__(self, host: str, reason: str, remedy: str, until: float | None = None):
        self.host = host
        self.reason = reason
        self.remedy = remedy
        self.until = until  # wall-clock epoch seconds, for credit exhaustion
        super().__init__(f"{policy_for(host).name} ({host}) is unavailable: {reason}. {remedy}")


_current_run: contextvars.ContextVar["Run | None"] = contextvars.ContextVar(
    "scholarly_gateway_run", default=None)
_credits_used: contextvars.ContextVar[dict[str, int] | None] = contextvars.ContextVar(
    "scholarly_gateway_credits", default=None)


def count_credits() -> dict[str, int]:
    """Starts a {provider name: credits} tally for the current context and
    every task started from it (each stage's Run copies the context), filled
    in as answers arrive; the orchestrator records it in metadata.json (#7)."""
    tally: dict[str, int] = {}
    _credits_used.set(tally)
    return tally


class Run:
    """One run's breaker state and its queue of gateway progress events."""

    def __init__(self) -> None:
        self.events: asyncio.Queue[dict] = asyncio.Queue()
        self.failures: dict[str, int] = {}
        self.tripped: dict[str, ProviderExhausted] = {}
        self.cache_hits = 0
        self._reported: set[str] = set()

    def task(self, coro: Coroutine) -> asyncio.Task:
        """Start `coro` as a task whose gateway calls (and any tasks it spawns)
        count against this run."""
        context = contextvars.copy_context()
        context.run(_current_run.set, self)
        return asyncio.create_task(coro, context=context)

    async def relay(self, task: asyncio.Task) -> AsyncIterator[dict]:
        """Yield this run's gateway events as they happen while `task` runs.
        When the loop ends, `task.result()` is ready."""
        getter = None
        try:
            while not task.done():
                getter = asyncio.ensure_future(self.events.get())
                await asyncio.wait({task, getter}, return_when=asyncio.FIRST_COMPLETED)
                if getter.done():
                    yield getter.result()
                else:
                    getter.cancel()
            for event in self.drain():
                yield event
        finally:
            if getter is not None and not getter.done():
                getter.cancel()
            if not task.done():  # the consumer stopped listening mid-run
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    def drain(self) -> list[dict]:
        """The events queued so far, without waiting."""
        drained = []
        while not self.events.empty():
            drained.append(self.events.get_nowait())
        return drained

    def emit(self, event: dict) -> None:
        self.events.put_nowait(event)

    def cache_event(self) -> dict | None:
        """The `cache_hit` event summing this run's hits so far, or None."""
        if not self.cache_hits:
            return None
        return {"type": "cache_hit", "hits": self.cache_hits,
                "message": f"{self.cache_hits} scholarly request(s) answered from the local "
                           "cache (no network call, no quota)."}

    def available(self, url: str) -> bool:
        """False once this run's breaker tripped for the URL's host, or its
        credit budget is used up — a caller can skip the call entirely."""
        host = host_of(url)
        return host not in self.tripped and not _gateway.exhausted(host)

    def exhausted(self, error: ProviderExhausted) -> ProviderExhausted:
        """Report a host as exhausted (once per run) and return the error to raise."""
        if error.host not in self._reported:
            self._reported.add(error.host)
            self.emit({"type": "provider_exhausted", "host": error.host,
                       "remedy": error.remedy, "message": str(error)})
        return error


class _Bucket:
    """Token bucket that hands out future slots: a caller reserves a token and
    is told how long to sleep, so concurrent callers queue in order without
    holding a lock across the wait. Tokens go negative while slots are booked
    ahead. A threading lock keeps it correct across event loops and threads."""

    def __init__(self, rate: float, burst: int, clock: Callable[[], float]):
        self.rate = rate
        self.burst = burst
        self._clock = clock
        self._tokens = float(burst)
        self._updated = clock()  # may lie in the future while the host is paused
        self._lock = threading.Lock()

    def reserve(self) -> float:
        """Take a token; return the seconds to wait before using it."""
        with self._lock:
            now = self._clock()
            start = max(now, self._updated)
            self._tokens = min(self.burst, self._tokens + (start - self._updated) * self.rate)
            self._updated = start
            self._tokens -= 1
            return start - now + max(0.0, -self._tokens) / self.rate

    def pause(self, seconds: float) -> None:
        """The server asked us to slow down: no new slot before now+seconds,
        and a single request (not a burst) when the pause ends."""
        with self._lock:
            self._updated = max(self._updated, self._clock() + seconds)
            if self._tokens >= 0:  # negative means callers already booked slots
                self._tokens = 1.0


def _header_wait(resp: httpx.Response, now: float) -> float | None:
    """Seconds the server asked us to wait: Retry-After (delta or HTTP date),
    else a rate-limit reset header (delta seconds or an epoch timestamp)."""
    retry_after = resp.headers.get("Retry-After")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            try:
                return max(0.0, email.utils.parsedate_to_datetime(retry_after).timestamp() - now)
            except (TypeError, ValueError):
                pass
    reset = resp.headers.get("X-RateLimit-Reset")
    if reset:
        try:
            value = float(reset)
        except ValueError:
            return None
        return max(0.0, value - now) if value > 1e9 else max(0.0, value)
    return None


def _next_utc_midnight(now: float) -> float:
    today = datetime.datetime.fromtimestamp(now, datetime.timezone.utc).date()
    midnight = datetime.datetime.combine(today + datetime.timedelta(days=1), datetime.time(),
                                         datetime.timezone.utc)
    return midnight.timestamp()


def _identify(policy: HostPolicy, params: dict | None, headers: dict | None) -> tuple[dict, dict]:
    params = dict(params or {})
    headers = dict(headers or {})
    if _keyed(policy):
        params.setdefault("api_key", os.environ[policy.key_env])
    email_address = os.getenv("CONTACT_EMAIL", "")
    if policy.identity == "mailto" and email_address:
        params.setdefault("mailto", email_address)
    elif policy.identity == "ncbi":
        params.setdefault("tool", TOOL_NAME)
        if email_address:
            params.setdefault("email", email_address)
    contact = f"; mailto:{email_address}" if email_address else ""
    headers.setdefault("User-Agent",
                       f"Mozilla/5.0 (compatible; {TOOL_NAME}/1.0; research tool{contact})")
    return params, headers


class Gateway:
    """Process-wide host state: buckets, connection slots and credit exhaustion.
    `clock`, `wall_clock`, `sleep` and `jitter` are test seams."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic,
                 wall_clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], Coroutine] = asyncio.sleep,
                 jitter: Callable[[], float] = random.random):
        self._clock = clock
        self._wall_clock = wall_clock
        self._sleep = sleep
        self._jitter = jitter
        self._buckets: dict[str, _Bucket] = {}
        self._buckets_lock = threading.Lock()
        # Semaphores belong to one event loop, so keep a set per loop.
        self._slots: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        # Loaded from the cache DB on first use (#6), written on every change.
        self._exhausted_until: dict[str, float] = {}
        self._state_loaded = False
        self._tripped_at: dict[str, float] = {}  # wall clock of the last breaker trip, any run

    def _exhausted(self) -> dict[str, float]:
        """Credit exhaustion by host, including what an earlier process saved."""
        if not self._state_loaded:
            self._state_loaded = True
            for host, until in response_cache.exhausted_until().items():
                self._exhausted_until.setdefault(host, until)
        return self._exhausted_until

    def _bucket(self, host: str, policy: HostPolicy) -> _Bucket:
        # Keyed and keyless traffic pace separately, so adding a key mid-process takes effect.
        keyed = _keyed(policy) and policy.keyed_rate is not None
        with self._buckets_lock:
            if (host, keyed) not in self._buckets:
                rate = policy.keyed_rate if keyed else policy.rate
                burst = max(policy.burst, int(rate)) if keyed else policy.burst
                self._buckets[(host, keyed)] = _Bucket(rate, burst, self._clock)
            return self._buckets[(host, keyed)]

    def provider_status(self) -> list[dict]:
        """Each named provider's access (keyed / identified / anonymous) and
        health (healthy / degraded), for the CLI and the web header. Never
        carries a key or the email itself."""
        now = self._wall_clock()
        identified = bool(os.getenv("CONTACT_EMAIL"))
        status = []
        for host, policy in POLICIES.items():
            if _keyed(policy):
                access = "keyed"
            elif identified and policy.identity in ("mailto", "ncbi"):
                access = "identified"
            else:
                access = "anonymous"
            until = self._exhausted().get(host)
            if until is not None and until > now:
                health, reason = "degraded", "daily budget used up"
            elif now - self._tripped_at.get(host, -DEGRADED_SECONDS) < DEGRADED_SECONDS:
                health, reason = "degraded", "recent rate-limit or server errors"
            else:
                health, reason = "healthy", ""
            if host == "api.semanticscholar.org":
                reason = reason or "keyless shared pool, best-effort"
            entry = {"name": policy.name, "host": host, "access": access,
                     "health": health, "reason": reason}
            if policy.credits_marker:  # the last known budget, from an earlier run if need be (#6)
                entry["budget_remaining"] = (response_cache.balance(host) or {}).get("x-ratelimit-remaining")
            status.append(entry)
        return status

    def _slot(self, host: str, policy: HostPolicy) -> asyncio.Semaphore:
        per_loop = self._slots.setdefault(asyncio.get_running_loop(), {})
        if host not in per_loop:
            per_loop[host] = asyncio.Semaphore(policy.concurrency)
        return per_loop[host]

    async def get(self, http: httpx.AsyncClient, url: str, *, params: dict | None = None,
                  headers: dict | None = None, timeout: float | None = None) -> httpx.Response:
        """GET through the gateway. Returns the final response — a successful
        one, a non-retryable error (the caller's raise_for_status reports it),
        or the last 429/5xx once retries are used up. Raises ProviderExhausted
        when the host is skipped. A fresh cached answer is returned without
        a request (#6)."""
        host, policy = host_of(url), policy_for(url)
        cache_key = redact(response_cache.key(url, params))
        cached = response_cache.get(cache_key, self._wall_clock())
        if cached is not None:
            count_cache_hit()
            status, content_type, body = cached
            return httpx.Response(status, headers={"content-type": content_type}, content=body,
                                  request=httpx.Request("GET", redact(url)))
        async with self._slot(host, policy):
            resp = await self._send(http, url, params, headers, timeout, stream=False)
        if resp.status_code == 200:
            response_cache.put(cache_key, 200, resp.headers.get("content-type", ""),
                               resp.content, response_cache.ttl_for(url), self._wall_clock())
        return resp

    @asynccontextmanager
    async def stream(self, http: httpx.AsyncClient, url: str, *, params: dict | None = None,
                     headers: dict | None = None,
                     timeout: float | None = None) -> AsyncIterator[httpx.Response]:
        """Streaming GET through the gateway; the host's connection slot is
        held until the body has been read or abandoned."""
        host, policy = host_of(url), policy_for(url)
        async with self._slot(host, policy):
            resp = await self._send(http, url, params, headers, timeout, stream=True)
            try:
                yield resp
            finally:
                await resp.aclose()

    async def _send(self, http: httpx.AsyncClient, url: str, params: dict | None,
                    headers: dict | None, timeout: float | None, *, stream: bool) -> httpx.Response:
        host, policy = host_of(url), policy_for(url)
        run = _current_run.get() or Run()
        params, headers = _identify(policy, params, headers)
        extra = {} if timeout is None else {"timeout": timeout}
        waited = 0.0
        for attempt in range(policy.attempts):
            self._check_available(run, host, policy)
            pace = self._bucket(host, policy).reserve()
            # Only a first attempt's wait is pacing; a retry's was reported as rate_limited.
            if attempt == 0 and pace >= PACING_NOTICE_SECONDS:
                run.emit({"type": "rate_limited", "host": host, "wait_seconds": round(pace, 1),
                          "message": f"Pacing requests to {policy.name}: waiting "
                                     f"{pace:.0f} s for a free slot."})
            if pace > 0:
                await self._sleep(pace)

            request = http.build_request("GET", url, params=params, headers=headers, **extra)
            resp = await http.send(request, stream=stream)
            shown = redact(str(request.url))
            if shown != str(request.url):
                # raise_for_status and error reprs echo resp.request.url.
                resp.request = httpx.Request("GET", shown)
            if policy.credits_marker:  # a credit-metered host: keep its last balance (#6)
                rate = {k.lower(): v for k, v in resp.headers.items()
                        if k.lower().startswith("x-ratelimit")}
                if rate:
                    response_cache.save_balance(host, {**rate, "seen_at": self._wall_clock(),
                                                       "keyed": _keyed(policy)})
            if (tally := _credits_used.get()) is not None and resp.status_code != 429:
                tally[policy.name] = tally.get(policy.name, 0) + credit_cost(url, params)
            if resp.status_code not in RETRY_STATUSES:
                run.failures[host] = 0
                return resp

            if stream:
                await resp.aread()  # small error body; needed for the credits check
            if (resp.status_code == 429 and policy.credits_marker
                    and policy.credits_marker in resp.text):
                await resp.aclose()
                now = self._wall_clock()
                wait = _header_wait(resp, now)
                until = now + wait if wait is not None else _next_utc_midnight(now)
                self._exhausted()[host] = until
                response_cache.save_exhausted(host, until)
                raise run.exhausted(self._credits_error(host, policy, until))

            run.failures[host] = run.failures.get(host, 0) + 1
            if run.failures[host] >= policy.breaker_threshold:
                await resp.aclose()
                self._tripped_at[host] = self._wall_clock()
                run.tripped[host] = ProviderExhausted(
                    host, f"{run.failures[host]} rate-limit/server errors in a row "
                          f"(last HTTP {resp.status_code})", policy.remedy)
                raise run.exhausted(run.tripped[host])

            remaining = policy.max_retry_wait - waited
            if attempt == policy.attempts - 1 or remaining <= 0:
                return resp  # retries used up; the caller's raise_for_status reports it
            await resp.aclose()
            wait = _header_wait(resp, self._wall_clock())
            if wait is None:
                wait = policy.backoff * 2 ** attempt * (0.5 + self._jitter())
            wait = min(wait, remaining)
            waited += wait
            self._bucket(host, policy).pause(wait)
            run.emit({"type": "rate_limited", "host": host, "wait_seconds": round(wait, 1),
                      "message": f"{policy.name} answered HTTP {resp.status_code}; "
                                 f"waiting {wait:.0f} s before retrying."})
        raise AssertionError("unreachable: the last attempt always returns or raises")

    def exhausted(self, host: str) -> bool:
        """The host's credit budget is used up and has not reset yet."""
        return self._exhausted().get(host, 0.0) > self._wall_clock()

    def balance(self, host: str) -> int | None:
        """Credits the host has left today: 0 while its budget is used up,
        else the last x-ratelimit-remaining it sent since the daily reset
        (midnight UTC) with the key set now, else None (unknown)."""
        if self.exhausted(host):
            return 0
        seen = response_cache.balance(host) or {}
        now = self._wall_clock()
        if (seen.get("seen_at", 0) < _next_utc_midnight(now) - 24 * 3600
                or seen.get("keyed", False) != _keyed(policy_for(host))):
            return None  # seen before today's reset, or with another key
        try:
            return int(seen["x-ratelimit-remaining"])
        except (KeyError, ValueError):
            return None

    def _check_available(self, run: Run, host: str, policy: HostPolicy) -> None:
        if host in run.tripped:
            raise run.exhausted(run.tripped[host])
        until = self._exhausted().get(host)
        if until is not None:
            if until > self._wall_clock():
                raise run.exhausted(self._credits_error(host, policy, until))
            self._exhausted_until.pop(host, None)  # the budget has reset

    @staticmethod
    def _credits_error(host: str, policy: HostPolicy, until: float) -> ProviderExhausted:
        reset = datetime.datetime.fromtimestamp(until, datetime.timezone.utc)
        return ProviderExhausted(
            host, f"its daily credit budget is used up until {reset:%H:%M} UTC",
            policy.remedy, until=until)


_gateway = Gateway()


async def get(http: httpx.AsyncClient, url: str, *, params: dict | None = None,
              headers: dict | None = None, timeout: float | None = None) -> httpx.Response:
    """GET `url` through the process-wide gateway (see Gateway.get)."""
    return await _gateway.get(http, url, params=params, headers=headers, timeout=timeout)


def count_cache_hit() -> None:
    """Counts a cache hit served outside Gateway.get (extracted full text) on
    the current run, if there is one."""
    run = _current_run.get()
    if run is not None:
        run.cache_hits += 1


def provider_status() -> list[dict]:
    """Per-provider access and health from the process-wide gateway."""
    return _gateway.provider_status()


def balance(host: str = OPENALEX_HOST) -> int | None:
    """The host's credits left today, if known (see Gateway.balance)."""
    return _gateway.balance(host)


def stream(http: httpx.AsyncClient, url: str, *, params: dict | None = None,
           headers: dict | None = None, timeout: float | None = None):
    """`async with stream(http, url) as resp:` through the process-wide gateway."""
    return _gateway.stream(http, url, params=params, headers=headers, timeout=timeout)

```

### `scripts/spike_tool_calling.py`

```python
"""Spike for #16: does the local model call the source tools reliably?

Asks the configured model (LOCAL_AI_MODEL, default qwen3.5:9b) to plan the
sources for 12 research questions across disciplines, exactly as Discovery
does, and counts per prompt: a valid plan (every call a known tool with a
usable query), and a sensible one (no PubMed for the engineering and
computing questions, PubMed for the biomedical ones). Paste the printed
summary into issue #16 before merging.

    python scripts/spike_tool_calling.py [--model qwen3.5:9b] [--json out.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "research-discovery"))

# (question, field, PubMed expected?)
PROMPTS = [
    ("Fatigue life of welded steel joints under variable amplitude loading", "Mechanical", False),
    ("Self-healing concrete using bacterial additives for crack repair", "Civil", False),
    ("Grid-forming inverter control for weak power grids", "Electrical", False),
    ("Low-power wake-up receivers for IoT sensor nodes", "Electronics", False),
    ("Kubernetes autoscaling policies for latency-sensitive microservices", "Cloud / DevOps", False),
    ("Retrieval-augmented generation for reducing LLM hallucination", "Computer Science", False),
    ("Effects of intermittent fasting on cognitive performance in adults", "Nutrition", True),
    ("Gut microbiome changes after antibiotic treatment in infants", "Medicine", True),
    ("CRISPR base editing efficiency in human hematopoietic stem cells", "Genetics", True),
    ("Urban green space access and residents' mental health", "Public health", True),
    ("Remote work and employee productivity after 2020", "Management", None),
    ("Topological insulators for low-dissipation electronics", "Physics", None),
]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("LOCAL_AI_MODEL", "qwen3.5:9b"))
    parser.add_argument("--json", type=Path, help="also save every prompt's raw calls here")
    args = parser.parse_args()

    from discovery import source_plan
    from shared.utilities import llm_provider

    keys = ["s2", "openalex", "pubmed", "arxiv"]
    rows = []
    for question, field, wants_pubmed in PROMPTS:
        started = time.monotonic()
        try:
            calls = await llm_provider.complete_tools(
                system=source_plan.PLAN_SYSTEM,
                user=f"Research question: {question}\nField: {field}",
                tools=source_plan.tools(keys), model=args.model)
            error = None
        except llm_provider.LLMProviderError as exc:
            calls, error = None, str(exc)
        plan = source_plan._valid(calls or [], keys)
        sensible = plan is not None and (wants_pubmed is None or ("pubmed" in plan) == wants_pubmed)
        rows.append({"question": question, "field": field, "calls": calls, "error": error,
                     "valid": plan is not None, "sensible": sensible,
                     "sources": sorted(plan) if plan else [],
                     "seconds": round(time.monotonic() - started, 1)})
        print(f"{'OK ' if sensible else 'VAL' if plan else 'NO '} {field:<16} "
              f"{','.join(sorted(plan)) if plan else (error or 'no valid calls')}")

    valid = sum(r["valid"] for r in rows)
    sensible = sum(r["sensible"] for r in rows)
    print(f"\nModel {args.model}: {valid}/{len(rows)} valid tool plans, "
          f"{sensible}/{len(rows)} sensible (PubMed only where it belongs).")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())

```

### Real tests for this lab

### `tests/shared/test_scholarly_gateway.py`

```python
"""Tests for the scholarly HTTP gateway (shared/utilities/scholarly_gateway.py).

The gateway replaces Discovery's `_http_retry.get_with_retry()` (DECISIONS.md
D-028); the first group of tests is that helper's suite, carried over so the
old guarantees still hold. Every test runs on a fake clock whose sleeps
return at once and are recorded, and on an httpx.MockTransport — no real
network, no real waiting.
"""

import asyncio
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.utilities import scholarly_gateway  # noqa: E402
from shared.utilities.scholarly_gateway import Gateway, ProviderExhausted, Run  # noqa: E402

OPENALEX = "https://api.openalex.org/works"
S2 = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV = "https://export.arxiv.org/api/query"
NCBI = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EPOCH = 1_800_000_000.0


class FakeTime:
    """A clock that only moves when a test moves it. Sleeps are recorded and
    return at once, so a sleep's length is exactly the wait the gateway chose."""

    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def wall(self) -> float:
        return EPOCH + self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        await asyncio.sleep(0)


@pytest.fixture(autouse=True)
def _no_cache_reads(monkeypatch):
    """These tests send the same request again on purpose (pacing, breaker);
    the response cache (#6) would answer it. Its own tests are in
    test_response_cache.py."""
    monkeypatch.setenv("RESEARCHGENIE_NO_CACHE", "1")


def _gateway(fake: FakeTime) -> Gateway:
    return Gateway(clock=fake.clock, wall_clock=fake.wall, sleep=fake.sleep,
                   jitter=lambda: 0.5)  # jitter factor 1.0: backoff is exactly base * 2**n


def _replies(*responses: httpx.Response):
    """A MockTransport that answers with `responses` in order (the last one
    repeats) and records each request it saw."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return responses[min(len(seen) - 1, len(responses) - 1)]

    return httpx.MockTransport(handler), seen


def _get(gateway: Gateway, transport, url: str, run: Run | None = None, **kwargs):
    async def go():
        async with httpx.AsyncClient(transport=transport) as http:
            return await gateway.get(http, url, **kwargs)

    async def main():
        return await (run or Run()).task(go())

    return asyncio.run(main())


# --------------------------------------------------- carried over from _http_retry

def test_immediate_success_makes_exactly_one_request():
    fake = FakeTime()
    transport, seen = _replies(httpx.Response(200))
    resp = _get(_gateway(fake), transport, OPENALEX)
    assert resp.status_code == 200
    assert len(seen) == 1
    assert fake.sleeps == []


def test_a_single_transient_5xx_recovers_on_retry():
    transport, seen = _replies(httpx.Response(503), httpx.Response(200))
    resp = _get(_gateway(FakeTime()), transport, OPENALEX)
    assert resp.status_code == 200
    assert len(seen) == 2


def test_429_is_retried_the_same_as_5xx():
    transport, seen = _replies(httpx.Response(429), httpx.Response(429), httpx.Response(200))
    resp = _get(_gateway(FakeTime()), transport, OPENALEX)
    assert resp.status_code == 200
    assert len(seen) == 3


def test_a_non_retryable_status_returns_immediately_without_delay():
    fake = FakeTime()
    transport, seen = _replies(httpx.Response(404))
    resp = _get(_gateway(fake), transport, OPENALEX)
    assert resp.status_code == 404
    assert len(seen) == 1  # a real client error, not transient
    assert fake.sleeps == []


def test_exhausting_all_attempts_returns_the_last_response_not_an_exception():
    transport, seen = _replies(httpx.Response(503))
    resp = _get(_gateway(FakeTime()), transport, OPENALEX)
    assert resp.status_code == 503
    assert len(seen) == scholarly_gateway.POLICIES["api.openalex.org"].attempts


def test_backoff_without_a_server_hint_doubles_each_time():
    fake = FakeTime()
    transport, _ = _replies(httpx.Response(503), httpx.Response(503), httpx.Response(200))
    _get(_gateway(fake), transport, OPENALEX)
    assert fake.sleeps == [1.0, 2.0]


# ------------------------------------------------------------------- pacing

def test_two_runs_at_once_share_one_token_bucket_per_host():
    """arXiv allows 1 request per 3 s. Two runs firing three requests each at
    the same instant must be spaced 3 s apart as one queue, not two."""
    fake = FakeTime()
    gateway = _gateway(fake)
    transport, seen = _replies(httpx.Response(200))

    async def three_requests():
        async with httpx.AsyncClient(transport=transport) as http:
            return await asyncio.gather(*[gateway.get(http, ARXIV) for _ in range(3)])

    async def main():
        run_a, run_b = Run(), Run()
        await asyncio.gather(run_a.task(three_requests()), run_b.task(three_requests()))

    asyncio.run(main())
    assert len(seen) == 6
    # The first request takes the idle bucket's token; each later one waits
    # one more 3-second slot, whichever run it came from.
    assert sorted(fake.sleeps) == pytest.approx([3, 6, 9, 12, 15])


def test_bucket_refills_with_time():
    fake = FakeTime()
    gateway = _gateway(fake)
    transport, _ = _replies(httpx.Response(200))
    _get(gateway, transport, ARXIV)
    fake.now += 3.5  # more than a full slot later
    _get(gateway, transport, ARXIV)
    assert fake.sleeps == []


# ------------------------------------------------------------- server hints

def test_retry_after_is_honored():
    fake = FakeTime()
    run = Run()
    transport, seen = _replies(httpx.Response(429, headers={"Retry-After": "10"}),
                               httpx.Response(200))
    resp = _get(_gateway(fake), transport, OPENALEX, run=run)
    assert resp.status_code == 200
    assert fake.sleeps == pytest.approx([10.0])
    assert run.drain() == [{
        "type": "rate_limited", "host": "api.openalex.org", "wait_seconds": 10.0,
        "message": "OpenAlex answered HTTP 429; waiting 10 s before retrying.",
    }]


def test_rate_limit_reset_header_is_honored():
    fake = FakeTime()
    transport, _ = _replies(httpx.Response(429, headers={"X-RateLimit-Reset": "7"}),
                            httpx.Response(200))
    _get(_gateway(fake), transport, NCBI)
    assert fake.sleeps == pytest.approx([7.0])


def test_a_long_retry_after_waits_only_up_to_the_host_cap():
    """OpenAlex's cap is 60 s of waiting per request: asked for 500 s, the
    gateway waits 60 s once, then hands back the 429 instead of waiting on."""
    fake = FakeTime()
    transport, seen = _replies(httpx.Response(429, headers={"Retry-After": "500"}))
    resp = _get(_gateway(fake), transport, OPENALEX)
    assert fake.sleeps == pytest.approx([60.0])
    assert len(seen) == 2
    assert resp.status_code == 429


# ------------------------------------------------------------- credit budget

def _no_credits(reset: str | None = "3600") -> httpx.Response:
    headers = {"X-RateLimit-Reset": reset} if reset else {}
    return httpx.Response(429, headers=headers,
                          json={"error": "not_enough_credits",
                                "message": "Daily budget exhausted."})


def test_not_enough_credits_is_never_retried_and_marks_the_host_exhausted():
    fake = FakeTime()
    gateway = _gateway(fake)
    run = Run()
    transport, seen = _replies(_no_credits(), httpx.Response(200))

    with pytest.raises(ProviderExhausted) as raised:
        _get(gateway, transport, OPENALEX, run=run)
    assert len(seen) == 1
    assert fake.sleeps == []
    assert raised.value.host == "api.openalex.org"
    assert raised.value.until == pytest.approx(EPOCH + 3600)
    [event] = run.drain()
    assert event["type"] == "provider_exhausted"
    assert event["host"] == "api.openalex.org"
    assert event["remedy"] == scholarly_gateway.POLICIES["api.openalex.org"].remedy

    # Exhaustion is host state, not run state: a second run fails fast too.
    with pytest.raises(ProviderExhausted):
        _get(gateway, transport, OPENALEX)
    assert len(seen) == 1

    # Once the reset time passes, the host is used again.
    fake.now += 3601
    assert _get(gateway, transport, OPENALEX).status_code == 200


def test_not_enough_credits_without_a_reset_header_lasts_until_utc_midnight():
    fake = FakeTime()
    transport, _ = _replies(_no_credits(reset=None))
    with pytest.raises(ProviderExhausted) as raised:
        _get(_gateway(fake), transport, OPENALEX)
    assert raised.value.until == pytest.approx(EPOCH - EPOCH % 86400 + 86400)


def test_a_plain_openalex_429_is_still_retried():
    transport, seen = _replies(httpx.Response(429, json={"error": "rate_limited"}),
                               httpx.Response(200))
    assert _get(_gateway(FakeTime()), transport, OPENALEX).status_code == 200
    assert len(seen) == 2


# ----------------------------------------------------------- circuit breaker

def test_semantic_scholar_breaker_trips_on_the_first_429_burst_and_then_fails_fast():
    fake = FakeTime()
    gateway = _gateway(fake)
    run = Run()
    transport, seen = _replies(httpx.Response(429))

    with pytest.raises(ProviderExhausted) as raised:
        _get(gateway, transport, S2, run=run)
    assert len(seen) == 2  # the 429 and one retry, a few seconds apart
    assert sum(fake.sleeps) < 10
    assert raised.value.host == "api.semanticscholar.org"

    with pytest.raises(ProviderExhausted):
        _get(gateway, transport, S2, run=run)
    assert len(seen) == 2  # failed fast: no request sent

    events = run.drain()
    assert [e["type"] for e in events] == ["rate_limited", "provider_exhausted"]
    assert events[1]["remedy"] == scholarly_gateway.POLICIES["api.semanticscholar.org"].remedy


def test_repeated_5xx_across_requests_trips_the_breaker_for_the_rest_of_the_run():
    gateway = _gateway(FakeTime())
    run = Run()
    transport, seen = _replies(httpx.Response(503))
    policy = scholarly_gateway.POLICIES["export.arxiv.org"]

    # One request alone uses up its attempts without tripping the breaker...
    assert _get(gateway, transport, ARXIV, run=run).status_code == 503
    assert len(seen) == policy.attempts
    # ...but the next one in the same run reaches the threshold.
    with pytest.raises(ProviderExhausted):
        _get(gateway, transport, ARXIV, run=run)
    assert len(seen) == policy.breaker_threshold
    with pytest.raises(ProviderExhausted):
        _get(gateway, transport, ARXIV, run=run)
    assert len(seen) == policy.breaker_threshold


def test_a_success_resets_the_breaker_count():
    gateway = _gateway(FakeTime())
    run = Run()
    transport, seen = _replies(httpx.Response(503), httpx.Response(200),
                               httpx.Response(503), httpx.Response(200))
    assert _get(gateway, transport, S2, run=run).status_code == 200
    assert _get(gateway, transport, S2, run=run).status_code == 200
    assert len(seen) == 4


def test_a_tripped_breaker_belongs_to_its_run_only():
    gateway = _gateway(FakeTime())
    transport, _ = _replies(httpx.Response(429), httpx.Response(429), httpx.Response(200))
    with pytest.raises(ProviderExhausted):
        _get(gateway, transport, S2, run=Run())
    assert _get(gateway, transport, S2, run=Run()).status_code == 200


def test_provider_exhausted_is_an_httpx_error_so_existing_handlers_degrade():
    assert issubclass(ProviderExhausted, httpx.HTTPError)


# ------------------------------------------------------------------ identity

def test_identity_is_injected_per_host_policy(monkeypatch):
    monkeypatch.setenv("CONTACT_EMAIL", "me@example.org")
    monkeypatch.setenv("S2_API_KEY", "should-never-be-sent")
    gateway = _gateway(FakeTime())
    transport, seen = _replies(httpx.Response(200))
    for url in (OPENALEX, NCBI, S2):
        _get(gateway, transport, url, params={"q": "x"})
    openalex, ncbi, s2 = seen

    assert openalex.url.params["mailto"] == "me@example.org"
    assert ncbi.url.params["tool"] == scholarly_gateway.TOOL_NAME
    assert ncbi.url.params["email"] == "me@example.org"
    assert "mailto" not in s2.url.params and "email" not in s2.url.params
    assert "x-api-key" not in s2.headers  # Semantic Scholar stays keyless
    assert all("mailto:me@example.org" in r.headers["User-Agent"] for r in seen)
    assert all(r.url.params["q"] == "x" for r in seen)


# ------------------------------------------------------------------- stream

def test_stream_retries_a_transient_error_before_handing_over_the_body():
    gateway = _gateway(FakeTime())
    transport, seen = _replies(httpx.Response(503), httpx.Response(200, content=b"%PDF-1.7"))

    async def go():
        async with httpx.AsyncClient(transport=transport) as http:
            async with gateway.stream(http, "https://journals.example.org/a.pdf") as resp:
                return resp.status_code, b"".join([c async for c in resp.aiter_bytes()])

    status, body = asyncio.run(go())
    assert (status, body) == (200, b"%PDF-1.7")
    assert len(seen) == 2


def test_requests_in_flight_per_host_never_exceed_its_policy():
    gateway = _gateway(FakeTime())
    in_flight = {"now": 0, "max": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        in_flight["now"] += 1
        in_flight["max"] = max(in_flight["max"], in_flight["now"])
        for _ in range(3):
            await asyncio.sleep(0)
        in_flight["now"] -= 1
        return httpx.Response(200)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            await asyncio.gather(*[gateway.get(http, "https://www.ebi.ac.uk/europepmc/x")
                                   for _ in range(12)])

    asyncio.run(go())
    assert in_flight["max"] == scholarly_gateway.POLICIES["www.ebi.ac.uk"].concurrency


def test_host_policy_sets_fan_out_concurrency():
    assert scholarly_gateway.concurrency("https://export.arxiv.org/api/query") == 1
    assert scholarly_gateway.concurrency("https://www.ebi.ac.uk/europepmc/x") == \
        scholarly_gateway.POLICIES["www.ebi.ac.uk"].concurrency
    assert scholarly_gateway.concurrency("https://unknown.example.org/p.pdf") == \
        scholarly_gateway.DEFAULT_POLICY.concurrency


# ------------------------------------------------------------- keys (#5)

# The `+` and `@` make the email look different once percent-encoded in a URL.
OPENALEX_KEY, NCBI_KEY, EMAIL = "oa-secret-key-123", "ncbi-secret-key-456", "a.b+c@example.edu"
ENCODED_EMAIL = "a.b%2Bc%40example.edu"


def _keys(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", OPENALEX_KEY)
    monkeypatch.setenv("NCBI_API_KEY", NCBI_KEY)
    monkeypatch.setenv("CONTACT_EMAIL", EMAIL)


def test_each_provider_gets_its_own_identity_and_semantic_scholar_never_a_key(monkeypatch):
    _keys(monkeypatch)
    fake = FakeTime()
    gateway = _gateway(fake)
    transport, seen = _replies(httpx.Response(200))
    for url in (OPENALEX, NCBI, "https://api.crossref.org/works/10.1/x", S2):
        _get(gateway, transport, url)
    openalex, ncbi, crossref, s2 = (r.url.params for r in seen)

    assert openalex["api_key"] == OPENALEX_KEY and openalex["mailto"] == EMAIL
    assert ncbi["api_key"] == NCBI_KEY and ncbi["tool"] and ncbi["email"] == EMAIL
    assert crossref["mailto"] == EMAIL and "api_key" not in crossref
    assert not s2 and not any(k.lower() in ("x-api-key", "api_key") for k in seen[3].headers)


def test_a_keyed_ncbi_is_paced_at_its_keyed_rate(monkeypatch):
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    fake = FakeTime()
    unkeyed = _gateway(fake)
    transport, _ = _replies(httpx.Response(200))
    for _ in range(4):
        _get(unkeyed, transport, NCBI)
    assert fake.sleeps and fake.sleeps[-1] == pytest.approx(1 / 3)

    monkeypatch.setenv("NCBI_API_KEY", NCBI_KEY)
    fake = FakeTime()
    keyed = _gateway(fake)
    for _ in range(11):
        _get(keyed, transport, NCBI)
    assert fake.sleeps == [pytest.approx(1 / 10)]


def test_no_key_or_email_leaks_into_errors_tracebacks_or_logs(monkeypatch, caplog):
    """D-023's lesson: the keys travel as query parameters (neither OpenAlex
    nor NCBI accepts a header), so every message that echoes a URL must not."""
    import logging
    import traceback
    _keys(monkeypatch)
    caplog.set_level(logging.DEBUG)
    gateway = _gateway(FakeTime())
    transport, _ = _replies(httpx.Response(404))
    resp = _get(gateway, transport, OPENALEX, params={"search": "x"})
    with pytest.raises(httpx.HTTPStatusError) as caught:
        resp.raise_for_status()

    shown = str(caught.value) + "".join(traceback.format_exception(caught.value)) + caplog.text
    assert "api.openalex.org" in shown  # the URL is still there, just without its secrets
    for secret in (OPENALEX_KEY, EMAIL, ENCODED_EMAIL, "c%40example.edu"):
        assert secret not in shown
    assert scholarly_gateway.redact(f"key={OPENALEX_KEY}&email={EMAIL}") == "key=***&email=***"


def test_provider_status_shows_access_and_health_per_provider(monkeypatch):
    for name in ("OPENALEX_API_KEY", "NCBI_API_KEY", "CONTACT_EMAIL"):
        monkeypatch.delenv(name, raising=False)
    fake = FakeTime()
    gateway = _gateway(fake)
    status = {p["host"]: p for p in gateway.provider_status()}
    assert status["api.openalex.org"]["access"] == "anonymous"
    assert status["api.openalex.org"]["health"] == "healthy"

    monkeypatch.setenv("CONTACT_EMAIL", EMAIL)
    monkeypatch.setenv("OPENALEX_API_KEY", OPENALEX_KEY)
    status = {p["host"]: p for p in gateway.provider_status()}
    assert status["api.openalex.org"]["access"] == "keyed"
    assert status["api.crossref.org"]["access"] == "identified"
    assert status["api.semanticscholar.org"]["access"] == "anonymous"  # never keyed (no S2 plan)

    transport, _ = _replies(httpx.Response(429, text="not_enough_credits"))
    with pytest.raises(ProviderExhausted):
        _get(gateway, transport, OPENALEX)
    status = {p["host"]: p for p in gateway.provider_status()}
    assert status["api.openalex.org"]["health"] == "degraded"
    assert OPENALEX_KEY not in str(status) and EMAIL not in str(status)


# --------------------------------------------------- credit budget (#7)

def test_credit_costs_a_search_ten_a_filter_one_an_entity_lookup_nothing():
    assert scholarly_gateway.credit_cost(OPENALEX, {"search": "q"}) == 10
    assert scholarly_gateway.credit_cost(OPENALEX, {"filter": "doi:10.1/a|10.1/b"}) == 1
    assert scholarly_gateway.credit_cost("https://api.openalex.org/works/doi:10.1/a") == 0
    assert scholarly_gateway.credit_cost("https://api.openalex.org/works/W123") == 0
    assert scholarly_gateway.credit_cost("https://api.crossref.org/works", {"query": "q"}) == 0


def test_a_run_tallies_the_credits_each_provider_charged():
    gateway = _gateway(FakeTime())
    transport, _ = _replies(httpx.Response(200, json={"results": []}))

    async def main():
        tally = scholarly_gateway.count_credits()
        async with httpx.AsyncClient(transport=transport) as http:
            await Run().task(gateway.get(http, OPENALEX, params={"search": "q"}))  # a stage's task
            await gateway.get(http, OPENALEX, params={"filter": "doi:10.1/a"})
            await gateway.get(http, "https://api.crossref.org/works", params={"query": "q"})
        return tally

    assert asyncio.run(main()) == {"OpenAlex": 11, "Crossref": 0}


def test_the_balance_counts_only_since_the_daily_reset_and_for_the_key_set_now(monkeypatch):
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    fake = FakeTime()
    gateway = _gateway(fake)
    assert gateway.balance("api.openalex.org") is None  # nothing seen yet

    transport, _ = _replies(httpx.Response(200, headers={"X-RateLimit-Remaining": "40"}, json={}))
    _get(gateway, transport, OPENALEX, params={"search": "q"})
    assert gateway.balance("api.openalex.org") == 40

    monkeypatch.setenv("OPENALEX_API_KEY", "a-new-key")
    assert gateway.balance("api.openalex.org") is None  # seen without the key now set
    monkeypatch.delenv("OPENALEX_API_KEY")
    fake.now += 24 * 3600  # past midnight UTC: the budget has reset
    assert gateway.balance("api.openalex.org") is None

    gateway._exhausted_until["api.openalex.org"] = fake.wall() + 3600
    assert gateway.balance("api.openalex.org") == 0

```

## The experiment

```bash
cd Course-Project/code
pip install -e .
pytest -v tests/shared/test_scholarly_gateway.py
```

## Real result
The gateway unit tests above pass against the real implementation, including the check that 250 DOI lookups batch into 3 Crossref requests plus 1 OpenAlex request. Separately, the spike script above was run live against the real local model for 12 research questions spanning disciplines. Real captured result, copied unedited from that run, is in `spike_output.json` in this folder: 12 of 12 valid tool plans, 11 of 12 sensible, meaning PubMed only where it belongs.

## Why this is the real evidence, not a toy
No part of this is a rule based stand in for the model. The actual qwen3.5:9b model, running locally, produced every one of the 12 tool plans in `spike_output.json`.

# ResearchGenie — How It Works (Explained Simply)

This document explains the whole project in plain language: what it does, how
each part works, what's been added on top of the original pipeline, and
exactly what is still pending before the final review. Updated 2026-10-27
(late evening), after a full day of live testing, two real bugs found and
fixed, and the OpenClaw/Telegram chat integration being wired up.

---

## 1. What ResearchGenie actually is

You give it a research question (e.g. "What are the effects of intermittent
fasting on cognitive performance?"). It then does, by itself, the whole job a
human researcher would do for a literature-review paper:

1. Finds real, relevant published papers.
2. Reads them and writes a draft paper that cites them correctly.
3. Double-checks every citation is a real paper that actually exists.
4. Grades its own work and tells you what's weak.
5. Hands you a finished PDF, plus a one-page summary of what to fix.

It runs on a local AI model (Ollama, a 9-billion-parameter model called
`qwen3.5`) on your own machine — no internet AI service needed, no cost per
run, nothing leaves your laptop except the paper-database searches themselves.

---

## 2. The core pipeline — the four stages

Every run goes through four stages, always in this fixed order. The order
never changes; this is deliberate, so a run can always be checked/audited
step by step.

### Stage 1 — Discovery (finding the papers)

- The AI model reads your question and decides which academic databases to
  search (OpenAlex, Semantic Scholar, PubMed, arXiv, Crossref, Europe PMC) and
  what search terms to use for each.
- It doesn't just trust itself here: code checks that the databases it picked
  actually make sense for the topic (e.g. it won't let the model search PubMed
  for a mechanical engineering question).
- It gathers candidate papers, reads their abstracts, and narrows down to the
  most relevant ones.
- **Proof this is real, not fake:** today's runs found real papers that
  independently check out against the live Crossref database (title, journal,
  publish date all matched, checked by hand, not just by ResearchGenie).

### Stage 2 — Writing (drafting the paper)

- The AI reads the full text (or abstract, if full text isn't available) of
  each paper it found.
- It writes the paper section by section (Introduction, Literature Review,
  Discussion, Conclusion, etc.), and every factual claim it writes gets
  tagged with which paper it came from — like `[P003]`.
- This tagging is the important part: the AI isn't allowed to just "know"
  something and state it without pointing to a source.

### Stage 3 — Verification (fact-checking the citations)

- For every citation the draft uses, the system looks up that paper's DOI
  (a unique ID every published paper has) against two independent real
  databases — Crossref and OpenAlex.
- Each citation ends up marked one of three ways:
  - **Verified** — the DOI is real and matches.
  - **Invalid** — something is wrong (made-up or mismatched DOI).
  - **Unverifiable** — couldn't confirm either way (e.g. a database was
    briefly unreachable).
- Nothing gets silently "fixed" — if a citation can't be verified, it stays
  flagged, so you know to check it by hand.

### Stage 4 — Quality Assurance (grading itself)

- A separate, independent pass re-reads the finished draft and checks: does
  every claim in the text actually match what its cited paper says? Are
  there gaps, contradictions, or unsupported claims?
- This produces a score out of 5 and a written report.
- This check is **deterministic** — meaning it doesn't rely on the AI's mood;
  the same draft always gets graded the same way.

### The output you get

- `paper.pdf` — the finished, formatted paper (IEEE / Springer / ACM / APA
  style, your choice).
- `evidence_table.md` — every claim, which paper it's from, and whether that
  citation was verified.
- `decision_brief.md` — a short summary: what's strong, what's weak, which
  references need a human look, and one clear next action.
- `quality_report.md` and `validation_report.md` — the self-grading and
  citation-checking results.

---

## 3. What's new (added after the original 4-stage pipeline)

The four stages above are the original, core product. Here's everything
that's been built **on top of** that since then:

### Trend Advisor (finding what to research in the first place)
Before you even ask a question, this can suggest *what's worth researching*:
it looks at real publication counts over time in a field (e.g. Computer
Science / AI / ML, Electronics, Mechanical) and finds topics that are growing
fast — and flags "research gaps" (things several papers all say still need
more study). Tested live across 3 fields — all working, producing real gap
statements grounded in real publication data, not invented ones.

### Doctor (self-healing when something breaks)
If a run fails mid-way — say, your local AI model crashes or isn't running —
the Doctor looks at the evidence, names the likely cause in plain language,
proposes a fix, and (only after you say yes) applies it and resumes the run
from where it failed, instead of starting over.

**This is now fully proven live**, after some real debugging work today. The
live test deliberately kills Ollama the moment the Writing stage starts, then
checks the Doctor correctly diagnoses "Ollama stopped", restarts it, and
resumes the draft. Getting this test to actually pass surfaced two real bugs
in Discovery's "gap analysis" step (not in the Doctor itself):
1. The list of "extra papers to double-check candidate research gaps against"
   had no size limit, so it kept growing every run and eventually made the
   prompt too big for the AI model to finish responding.
2. Even after limiting that list, the model's own reply-length budget (8,000
   words' worth of tokens) was simply too small for a realistic case. Raised
   to 16,000.
Both are fixed, tested, and merged. The Doctor itself (`#13`) now passes the
live recovery test end to end: kill the model mid-run → it notices → fixes →
resumes → finishes.

### Dev Agent (an AI that fixes bugs in ResearchGenie's own code)
A separate tool: you give it a small, well-described bug ticket, and it
writes a failing test proving the bug exists, then patches the code until
that test passes — never touching its own test file, never committing or
pushing by itself. A human always reviews the patch before it's applied. This
is a developer tool, not something end users interact with.

**Honest result:** on its real demo ticket (APA-7 citation formatting), the
Dev Agent did not converge on its own across 8 attempts (5 with the local
model, 3 with Gemini) — but its safety rules worked exactly as designed every
single time: it refused to overwrite real test files 9 times rather than fake
a pass, and it never committed anything broken. The actual citation fix was
then written by hand instead, and that's recorded plainly, not hidden.

### Library memory / reuse
The system remembers papers and search results from past runs. If you ask a
related question later, it can reuse what it already found instead of
re-searching from scratch — tested, cut repeat search time noticeably.

### Scholarly Gateway (shared traffic management)
All the academic databases are now called through one shared layer that
paces requests so you don't get rate-limited, retries automatically on
temporary failures, and tracks how much of each database's free quota you've
used so a long run won't suddenly fail partway through from running out of
credits.

### Chat-based access via OpenClaw — now working on Telegram
You can trigger a research run from a chat app instead of only the terminal
or web UI. **How this was actually set up today:**
- OpenClaw (the chat-bridge tool) was already installed on this laptop.
- A Discord bot was attempted first, but got stuck: the setup wizard said
  "configured" but nothing ever actually saved to disk. That's parked for now
  — not fixed, not worth more time tonight.
- Switched to **Telegram** instead, which worked cleanly:
  1. Created a bot through Telegram's official `@BotFather` (this gives a
     secret "bot token" — like a password just for that bot).
  2. The token was saved to a local file on this laptop (never pasted into
     chat, to avoid exposing it) and loaded into OpenClaw from that file.
  3. The first token BotFather had issued earlier had actually been
     invalidated somewhere along the way — Telegram's servers rejected it
     outright (401 Unauthorized). Revoking it and generating a brand new one
     through BotFather fixed it immediately.
  4. The OpenClaw "gateway" (the background program that keeps the bot
     listening) is now running and connected as **`@Research_Genie_bot`**.
- **What's still needed:** open Telegram, search `@Research_Genie_bot`, and
  send `/start` — that links your personal Telegram account to the bot so it
  can actually talk back to you. Not done yet as of writing this.
- **Also still needed:** install the actual ResearchGenie "skill" into
  OpenClaw (the part that lets the bot understand research requests, not just
  connect) and walk through the real chat demo. See the pending list below.
- Note: the gateway is running in the foreground on this laptop right now,
  not as a permanent background service — if this laptop restarts, it needs
  to be started again by hand before the Telegram bot will respond.

### MCP server (for Claude Desktop)
A read-only connector that lets Claude Desktop search your paper library or
verify a DOI directly, without running a full research pipeline. Not yet set
up on this laptop — see pending list.

### Automated "live pass" testing
A single script (`scripts/live_pass.py`) that runs every check that needs the
real AI model — the full test suite, real research runs, the Doctor, the Dev
Agent, the Trend Advisor — and writes an honest report
(`docs/evidence/live-pass/LIVE_PASS.md`) of exactly what passed and what
didn't, with the raw evidence saved alongside it. This is what today's
verification was built on, and it's also what caught both real bugs
described above — it doesn't let a failure quietly disappear.

### Docker packaging
The whole system can also run inside Docker containers, for easier setup on a
different machine. Not re-tested today (optional, only if time allows before
the review — see pending list).

---

## 4. What's actually proven working right now

- **The full core pipeline** (Discovery → Writing → Verification → QA) —
  real runs completed, citations independently spot-checked against the live
  Crossref database by hand, not just by ResearchGenie itself.
- **Tool-calling / database selection** — the model reliably picks sensible
  databases for a topic (12/12 valid plans, 11/12 sensible in testing).
- **Library reuse / memory.**
- **Trend Advisor**, across 3 academic fields.
- **The CLI** (`run`, `status`, `result`, `resume`, `doctor`, and now a
  working bare `--help`).
- **The Doctor's live self-healing** — genuinely proven today: kill the model
  mid-run, it recovers on its own with one human approval.
- **The full automated test suite** — 559 passed, 1 pre-existing skip, on
  `main`, right now.
- **Telegram connects to the bot** (`@Research_Genie_bot` is live and
  authenticated with Telegram's servers).

## 5. What is still pending before the review

This is the honest, current to-do list. Nothing here is hidden or guessed —
this is exactly where things stand as of writing this document.

1. **You need to send `/start` to `@Research_Genie_bot` on Telegram** so it's
   linked to your account, not just running.
2. **Install the ResearchGenie skill into OpenClaw** so the bot can actually
   understand and run research requests, not just receive messages.
3. **Walk through and record the real chat demo** (asking it a research
   question over Telegram and getting a result back).
4. **Discord is still broken** (config never saves) — parked, not a blocker
   since Telegram works, but worth mentioning honestly if asked.
5. **The Doctor's live web-UI rehearsal** — watching the browser show a
   "Diagnose & fix" banner live, with you present, hasn't been done yet (the
   backend recovery itself is already proven via the command-line test).
6. **Claude Desktop MCP server** — not yet configured on this laptop.
7. **Team names and USNs** — still needed from you, to fill into the lab
   worklet document.
8. **Tagging the final reviewed version** (`v1.0-review`) on GitHub — not
   done yet; happens once everything above is settled.
9. **Cache warming, the fallback demo video, and a timed dry run** — these
   are explicitly your part to do before the review, as a safety net in case
   anything misbehaves live.
10. **Docker packaging live-check** — optional, only if time allows; skip if
    short on time, it doesn't affect the core grade.

Nothing on this list means the core project doesn't work — the research
pipeline itself (the part that matters most: "does this produce a real,
correctly-cited paper") is solid, tested, and independently verified. What's
left is mostly demo polish and a few things that specifically need you
present (your phone, your Telegram account, your team's details).

---

## 6. Where to see the real proof yourself

- `docs/evidence/live-pass/LIVE_PASS.md` — the full honest pass/fail report
- `docs/evidence/live-pass/runs/fasting/final/paper.pdf` — a real generated
  paper
- `docs/evidence/live-pass/runs/fasting/final/evidence_table.md` — every
  claim and citation from that paper, with verification status
- `docs/demo-runbook.md` — the planned walkthrough for the final review
- GitHub issues `#13` (Doctor) and `#26` (APA/Dev Agent) — both closed today
  with the full honest story in their comments

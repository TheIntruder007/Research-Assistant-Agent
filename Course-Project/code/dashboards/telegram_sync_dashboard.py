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

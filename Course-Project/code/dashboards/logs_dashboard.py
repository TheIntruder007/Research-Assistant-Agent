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

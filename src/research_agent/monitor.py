"""Local monitoring dashboard for the research agent.

Read-only page on 127.0.0.1:8765: bot status, live query activity, per-stage
LLM timings/cost, and pending reviews. Data comes from Postgres (query_log,
messages, pending_approvals) and logs/usage.jsonl; this module never writes to
either. Run alongside the bot:

    python -m research_agent.monitor
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import psutil
from flask import Flask, jsonify

from research_agent.config import load_settings
from research_agent.db import WAREHOUSE_TABLES, connect
from research_agent.llm.usage import usage_totals

HOST = "127.0.0.1"
PORT = 8765
IN_FLIGHT_WINDOW_S = 600
EVAL_CHANNEL = "C_EVAL"
ACTIVITY_LIMIT = 25

app = Flask(__name__)
_conn = None


def get_conn():
    """Reusable connection; reconnects if the database dropped it."""
    global _conn
    settings = load_settings()
    if not settings.database_url:
        return None
    if _conn is not None:
        try:
            _conn.execute("SELECT 1")
            return _conn
        except Exception:
            _conn = None
    try:
        _conn = connect(settings.database_url)
    except Exception:
        _conn = None
    return _conn


def load_usage_entries(path: Path, limit: int = 0) -> list[dict]:
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    entries.sort(key=lambda entry: entry["ts"])
    return entries if not limit else entries[-limit:]


def bot_running() -> tuple[bool, int | None]:
    """True if a research_agent.app process exists, plus its start time."""
    for proc in psutil.process_iter(["cmdline", "create_time"]):
        try:
            cmdline = proc.info["cmdline"] or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if any("research_agent.app" in part for part in cmdline):
            return True, int(proc.info["create_time"])
    return False, None


def infer_stage(tasks: list[str]) -> str:
    """Stage of an in-flight run from the LLM tasks already completed."""
    if "synthesize" in tasks:
        return "writing answer"
    if "sql" in tasks:
        return "querying warehouse"
    if "scope" in tasks:
        return "scoping"
    return "queued"


def _label(channel_id: str) -> str:
    return "eval" if channel_id == EVAL_CHANNEL else "user"


def status_payload(settings) -> dict:
    running, created = bot_running()
    calls, cost = usage_totals(Path(settings.usage_log_path))
    payload: dict = {
        "bot_running": running,
        "bot_uptime_s": int(time.time() - created) if created else None,
        "tables": {},
        "pending_reviews": None,
        "calls": calls,
        "cost_usd": round(cost, 2),
    }
    conn = get_conn()
    if conn is None:
        return payload
    try:
        rows = conn.execute(
            "SELECT relname, reltuples FROM pg_class WHERE relname = ANY(%s)",
            (list(WAREHOUSE_TABLES),),
        ).fetchall()
        payload["tables"] = {name: max(int(tuples), 0) for name, tuples in rows}
        payload["pending_reviews"] = conn.execute(
            "SELECT count(*) FROM pending_approvals WHERE status = 'pending'"
        ).fetchone()[0]
    except Exception:
        pass
    return payload


IN_FLIGHT_SQL = """
SELECT m.created_at, m.channel_id, m.thread_ts, left(m.content, 300)
FROM messages m
WHERE m.role = 'user'
  AND m.created_at > now() - interval '10 minutes'
  AND NOT EXISTS (
      SELECT 1 FROM messages a
      WHERE a.channel_id = m.channel_id AND a.thread_ts = m.thread_ts
        AND a.role = 'assistant' AND a.created_at > m.created_at
  )
ORDER BY m.created_at DESC
LIMIT 10
"""


def activity_payload(conn, usage_entries: list[dict], now: float | None = None) -> dict:
    now = now or time.time()
    payload: dict = {"queries": [], "messages": [], "approvals": [], "in_flight": []}
    if conn is None:
        return payload
    try:
        for row in conn.execute(
            "SELECT created_at, channel_id, left(question, 300), sources,"
            " sql IS NOT NULL FROM query_log ORDER BY created_at DESC LIMIT %s",
            (ACTIVITY_LIMIT,),
        ).fetchall():
            payload["queries"].append(
                {
                    "when": row[0].timestamp(),
                    "channel": row[1],
                    "label": _label(row[1]),
                    "question": row[2],
                    "sources": row[3] or "",
                    "has_sql": row[4],
                }
            )
        for row in conn.execute(
            "SELECT created_at, channel_id, role, left(content, 300) FROM messages"
            " ORDER BY created_at DESC LIMIT 20"
        ).fetchall():
            payload["messages"].append(
                {
                    "when": row[0].timestamp(),
                    "channel": row[1],
                    "label": _label(row[1]),
                    "role": row[2],
                    "content": row[3],
                }
            )
        for row in conn.execute(
            "SELECT id, origin_channel_id, question, reason, turns, created_at"
            " FROM pending_approvals WHERE status = 'pending'"
            " ORDER BY created_at DESC LIMIT 10"
        ).fetchall():
            payload["approvals"].append(
                {
                    "id": row[0],
                    "channel": row[1],
                    "question": row[2],
                    "reason": row[3],
                    "turns": row[4],
                    "age_s": now - row[5].timestamp(),
                }
            )
        for row in conn.execute(IN_FLIGHT_SQL).fetchall():
            started = row[0].timestamp()
            tasks = [entry["task"] for entry in usage_entries if entry["ts"] >= started]
            payload["in_flight"].append(
                {
                    "question": row[3],
                    "channel": row[1],
                    "label": _label(row[1]),
                    "stage": infer_stage(tasks),
                    "elapsed": round(now - started),
                }
            )
    except Exception:
        pass
    return payload


def usage_payload(usage_entries: list[dict], recent: int = 15) -> dict:
    per_model: dict[str, dict] = defaultdict(
        lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    )
    for entry in usage_entries:
        model = per_model[entry["model"]]
        model["calls"] += 1
        model["input_tokens"] += entry["input_tokens"]
        model["output_tokens"] += entry["output_tokens"]
        model["cost_usd"] += entry["cost_usd"]
    models = [{"model": name, **stats} for name, stats in sorted(per_model.items())]
    return {"models": models, "recent": usage_entries[-recent:][::-1]}


@app.get("/api/status")
def api_status():
    return jsonify(status_payload(load_settings()))


@app.get("/api/activity")
def api_activity():
    settings = load_settings()
    entries = load_usage_entries(Path(settings.usage_log_path))
    return jsonify(activity_payload(get_conn(), entries))


@app.get("/api/usage")
def api_usage():
    settings = load_settings()
    return jsonify(usage_payload(load_usage_entries(Path(settings.usage_log_path))))


@app.get("/")
def page():
    return PAGE, 200, {"Content-Type": "text/html; charset=utf-8"}


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Research agent monitor</title>
<style>
  :root { --bg:#0f1115; --card:#171a21; --line:#262b36; --text:#d7dce4; --muted:#8b93a3;
          --green:#3fb96f; --red:#e05656; --amber:#e0a13f; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui, sans-serif; }
  header { padding:14px 20px; border-bottom:1px solid var(--line); display:flex; align-items:baseline; gap:12px; }
  header h1 { font-size:16px; margin:0; }
  .live { color:var(--green); font-size:12px; }
  main { padding:16px 20px; max-width:1200px; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; }
  .card h3 { margin:0 0 6px; font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; }
  .big { font-size:20px; margin:4px 0; }
  .small { font-size:12px; color:var(--muted); margin:0; word-break:break-word; }
  .muted { color:var(--muted); }
  .dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; }
  .dot.up { background:var(--green); } .dot.down { background:var(--red); }
  h2 { font-size:14px; margin:22px 0 8px; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; }
  table { width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--line); border-radius:8px; font-size:13px; }
  th, td { text-align:left; padding:6px 10px; border-bottom:1px solid var(--line); white-space:nowrap; }
  th { color:var(--muted); font-weight:500; font-size:12px; }
  td.q { white-space:normal; max-width:420px; }
  .badge { display:inline-block; padding:1px 7px; border-radius:10px; font-size:11px; }
  .badge.user { background:#233a5c; color:#9db8e8; }
  .badge.eval { background:#3a2f1c; color:#d8b56a; }
  .badge.stage { background:#2a2f3c; color:var(--text); }
  .badge.ok { background:#1c3a2a; color:#7fd9a6; }
  .stale { border-left:3px solid var(--red); padding-left:8px; }
  .inflight { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 12px; margin-bottom:8px; display:flex; gap:10px; align-items:center; }
  .inflight .q { flex:1; }
  .model-haiku { color:#9db8e8; } .model-sonnet { color:#e8c39d; }
  .model-deepseek { color:#b39de8; } .model-gemini { color:#9de8c9; }
  .model-other { color:var(--muted); }
  #updated { color:var(--muted); font-size:12px; margin-top:18px; }
</style>
</head>
<body>
<header>
  <h1>Research agent monitor</h1>
  <span class="live" id="live">connecting…</span>
</header>
<main>
  <div class="cards" id="status-cards"></div>

  <h2>In flight</h2>
  <div id="in-flight"><p class="muted">Loading…</p></div>

  <h2>Recent queries</h2>
  <div id="activity"></div>

  <h2>Stage timeline (last LLM calls)</h2>
  <div id="timeline"></div>

  <h2>Usage by model</h2>
  <div id="usage"></div>

  <h2>Pending reviews</h2>
  <div id="approvals"></div>

  <p id="updated"></p>
</main>
<script>
const byId = id => document.getElementById(id);
function esc(s){ return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function clock(ts){ return new Date(ts * 1000).toLocaleTimeString(); }
function dur(s){
  if (s == null) return '—';
  if (s < 60) return Math.round(s) + 's';
  return Math.floor(s / 60) + 'm' + Math.round(s % 60) + 's';
}
function modelClass(m){
  const id = m.toLowerCase();
  if (id.includes('haiku')) return 'model-haiku';
  if (id.includes('sonnet')) return 'model-sonnet';
  if (id.includes('deepseek')) return 'model-deepseek';
  if (id.includes('gemini')) return 'model-gemini';
  return 'model-other';
}
function shortModel(m){ return m.split('/').pop(); }

function renderStatus(s){
  const up = s.bot_running
    ? '<span class="dot up"></span>up'
    : '<span class="dot down"></span>down';
  const tables = Object.entries(s.tables).length
    ? Object.entries(s.tables).map(([k, v]) => k + ': ' + v.toLocaleString()).join(' &middot; ')
    : '—';
  byId('status-cards').innerHTML =
    '<div class="card"><h3>Bot</h3><p class="big">' + up + '</p>' +
      '<p class="small">uptime ' + dur(s.bot_uptime_s) + '</p></div>' +
    '<div class="card"><h3>Warehouse</h3><p class="small">' + esc(tables) + '</p></div>' +
    '<div class="card"><h3>Pending reviews</h3><p class="big">' + (s.pending_reviews ?? '—') + '</p></div>' +
    '<div class="card"><h3>LLM usage</h3><p class="big">' + s.calls + ' calls</p>' +
      '<p class="small">≈ $' + s.cost_usd.toFixed(2) + '</p></div>';
}

function renderInFlight(a){
  if (!a.in_flight.length){ byId('in-flight').innerHTML = '<p class="muted">No query in flight.</p>'; return; }
  byId('in-flight').innerHTML = a.in_flight.map(f =>
    '<div class="inflight' + (f.elapsed > 180 ? ' stale' : '') + '">' +
      '<span class="badge stage">' + esc(f.stage) + '</span>' +
      '<span class="q">' + esc(f.question) + '</span>' +
      '<span class="muted">' + esc(f.channel) + ' &middot; ' + dur(f.elapsed) + ' elapsed</span>' +
    '</div>').join('');
}

function renderActivity(a){
  if (!a.queries.length){ byId('activity').innerHTML = '<p class="muted">No queries yet.</p>'; return; }
  byId('activity').innerHTML = '<table><tr><th>When</th><th>Type</th><th>Question</th><th>Sources</th><th>SQL</th></tr>' +
    a.queries.map(q =>
      '<tr><td>' + clock(q.when) + '</td>' +
      '<td><span class="badge ' + q.label + '">' + q.label + '</span></td>' +
      '<td class="q" title="' + esc(q.question) + '">' + esc(q.question) + '</td>' +
      '<td class="muted">' + esc(q.sources) + '</td>' +
      '<td>' + (q.has_sql ? '<span class="badge ok">yes</span>' : '—') + '</td></tr>'
    ).join('') + '</table>';
}

function renderTimeline(u){
  if (!u.recent.length){ byId('timeline').innerHTML = '<p class="muted">No LLM calls yet.</p>'; return; }
  byId('timeline').innerHTML = '<table><tr><th>When</th><th>Task</th><th>Model</th><th>In</th><th>Out</th><th>Latency</th><th>Cost</th></tr>' +
    u.recent.map(e =>
      '<tr><td>' + clock(e.ts) + '</td>' +
      '<td>' + esc(e.task) + '</td>' +
      '<td class="' + modelClass(e.model) + '">' + esc(shortModel(e.model)) + '</td>' +
      '<td>' + e.input_tokens.toLocaleString() + '</td>' +
      '<td>' + e.output_tokens.toLocaleString() + '</td>' +
      '<td>' + dur(e.latency_ms / 1000) + '</td>' +
      '<td>$' + e.cost_usd.toFixed(4) + '</td></tr>'
    ).join('') + '</table>';
}

function renderUsage(u){
  if (!u.models.length){ byId('usage').innerHTML = '<p class="muted">No usage yet.</p>'; return; }
  byId('usage').innerHTML = '<table><tr><th>Model</th><th>Calls</th><th>Input</th><th>Output</th><th>Cost</th></tr>' +
    u.models.map(m =>
      '<tr><td class="' + modelClass(m.model) + '">' + esc(shortModel(m.model)) + '</td>' +
      '<td>' + m.calls + '</td>' +
      '<td>' + m.input_tokens.toLocaleString() + '</td>' +
      '<td>' + m.output_tokens.toLocaleString() + '</td>' +
      '<td>$' + m.cost_usd.toFixed(2) + '</td></tr>'
    ).join('') + '</table>';
}

function renderApprovals(a){
  if (!a.approvals.length){ byId('approvals').innerHTML = '<p class="muted">None open.</p>'; return; }
  byId('approvals').innerHTML = '<table><tr><th>Age</th><th>Question</th><th>Reason</th><th>Turns</th></tr>' +
    a.approvals.map(p =>
      '<tr><td>' + dur(p.age_s) + '</td>' +
      '<td class="q">' + esc(p.question) + '</td>' +
      '<td class="q" title="' + esc(p.reason) + '">' + esc(p.reason) + '</td>' +
      '<td>' + p.turns + '</td></tr>'
    ).join('') + '</table>';
}

async function refresh(){
  const started = Date.now();
  try {
    const [s, a, u] = await Promise.all([
      fetch('/api/status').then(r => r.json()),
      fetch('/api/activity').then(r => r.json()),
      fetch('/api/usage').then(r => r.json())
    ]);
    renderStatus(s); renderInFlight(a); renderActivity(a);
    renderTimeline(u); renderUsage(u); renderApprovals(a);
    byId('live').textContent = 'live · ' + Math.round((Date.now() - started) / 10) / 100 + 's';
    byId('updated').textContent = 'last updated ' + new Date().toLocaleTimeString();
  } catch (err) {
    byId('live').textContent = 'unreachable: ' + err;
  }
}
refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>
"""


def main() -> None:
    settings = load_settings()
    if not settings.database_url:
        print("WARNING: DATABASE_URL not set; warehouse status will be unavailable")
    print(f"Monitor on http://{HOST}:{PORT} (Ctrl+C to stop)")
    app.run(host=HOST, port=PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()

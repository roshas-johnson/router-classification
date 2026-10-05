"""
Local server for human routing-labeling.

Serves human_label.html and saves each selection to eval-data/human_labels.json
on every click (POST /save). The deck = the 300 assignment=="human" rows PLUS 50
random GPT-labeled rows mixed in for blind judge-validation (their ids are kept in
eval-data/_gpt_overlap_ids.json and are NOT marked in the UI). Blind: no Gemini/Jev
outputs are ever sent to the page.

Run:  .venv/Scripts/python.exe scripts/human_label_server.py
Then open the printed http://127.0.0.1:8899 URL.
"""

import csv
import json
import random
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval-data"
SET = EVAL / "golden_labeling_set.json"
GPT = EVAL / "golden_llm_labels.csv"
LABELS = EVAL / "human_labels.json"
OVERLAP = EVAL / "_gpt_overlap_ids.json"
HTML = Path(__file__).resolve().parent / "human_label.html"
PORT = 8899
KEEP = ["id", "brand", "turn_index", "current_message", "conversation_history",
        "previous_agent", "available_agents"]

# ---- build the labeling deck: 300 human + 50 hidden GPT-overlap rows ----
rows_all = json.loads(SET.read_text(encoding="utf-8"))["labeling_set"]
by_id = {r["id"]: r for r in rows_all}
human = [r for r in rows_all if r.get("assignment") == "human"]

if OVERLAP.exists():                                   # reuse a fixed overlap across restarts
    overlap = json.loads(OVERLAP.read_text(encoding="utf-8"))
else:
    gpt_ids = [r["id"] for r in csv.DictReader(open(GPT, encoding="utf-8")) if r.get("id")]
    human_ids = {r["id"] for r in human}
    pool = [i for i in gpt_ids if i not in human_ids]
    random.seed(7)
    overlap = random.sample(pool, min(50, len(pool)))
    OVERLAP.write_text(json.dumps(overlap), encoding="utf-8")

deck_rows = human + [by_id[i] for i in overlap if i in by_id]
random.seed(13)
random.shuffle(deck_rows)

# Reorder: unlabeled rows first, ~1 hidden-validation row per 2 human rows, so the
# GPT-6 judge gets validated within the first ~100 labels. Labels persist by id, so
# reordering never loses progress; already-labeled rows move to the end.
_overlap = set(overlap)
_done = set()
if LABELS.exists():
    try:
        _done = set(json.loads(LABELS.read_text(encoding="utf-8")).keys())
    except Exception:
        _done = set()
_un_human = [r for r in deck_rows if r["id"] not in _overlap and r["id"] not in _done]
_un_hidden = [r for r in deck_rows if r["id"] in _overlap and r["id"] not in _done]
_labeled = [r for r in deck_rows if r["id"] in _done]


def _interleave(hu, hd, per=2):
    out, i, j = [], 0, 0
    while i < len(hu) or j < len(hd):
        for _ in range(per):
            if i < len(hu):
                out.append(hu[i]); i += 1
        if j < len(hd):
            out.append(hd[j]); j += 1
    return out


ordered = _interleave(_un_human, _un_hidden, per=2) + _labeled
DECK = [{k: r.get(k) for k in KEEP} for r in ordered]   # strip assignment etc. (blind)

_lock = threading.Lock()


def load_labels():
    if LABELS.exists():
        try:
            return json.loads(LABELS.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/data":
            self._send(200, DECK)
        elif self.path == "/load":
            self._send(200, load_labels())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/save":
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                return self._send(400, {"error": "bad json"})
            rid = body.get("id")
            if not rid:
                return self._send(400, {"error": "no id"})
            with _lock:
                labels = load_labels()
                labels[rid] = {
                    "gold_agent": body.get("gold_agent"),
                    "unsure": bool(body.get("unsure")),
                    "notes": body.get("notes", ""),
                }
                LABELS.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
                count = len(labels)
            self._send(200, {"ok": True, "saved": rid, "count": count})
        else:
            self._send(404, {"error": "not found"})

    def log_message(self, *a):
        pass  # keep console quiet


if __name__ == "__main__":
    done = len(load_labels())
    print(f"Deck: {len(DECK)} rows ({len(human)} human + {len(overlap)} hidden validation). "
          f"Already labeled: {done}.", flush=True)
    ip = lan_ip()
    print(f"This device : http://127.0.0.1:{PORT}", flush=True)
    print(f"Same WiFi   : http://{ip}:{PORT}   <-- open this on other devices", flush=True)
    print("(saves to eval-data/human_labels.json on every click)", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()

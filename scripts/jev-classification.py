"""
Jev classification run — route every WhatsApp row in routing_dataset.json through
the Jev API (Jev only) and save results to eval-data/Jev-classification-result.csv.

- One row = one routing classification, processed in CHRONOLOGICAL order
  (routing_dataset.json is stored oldest-first: Sep 19 -> Sep 21 IST).
- Per-row Jev inputs: current_message + conversation_history + previous_agent (state),
  and the brand's agent names + descriptions as the choice criteria. Criteria are
  resolved per brand from brands_level_ai_info.json (NOT stored in the dataset).
- Retry with backoff on transient API errors (429 / 5xx / network).
- Progress-save / resume: each result is flushed to the CSV as it completes; on start
  rows whose id already exists in the CSV are skipped, so a restart continues.

Usage:
  python jev-classification.py            # full run (resumes if CSV exists)
  python jev-classification.py --limit 3  # process only the next 3 pending rows
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "eval-data" / "routing_dataset.json"
BRANDS = ROOT / "brands_level_ai_info.json"
OUT = ROOT / "eval-data" / "Jev-classification-result.csv"

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
TIMEOUT = 60
MAX_RETRIES = 5

INSTRUCTIONS = (
    "Select the single agent whose description best fits the customer's current WhatsApp "
    "message. Base the decision on the current message; use the conversation history and "
    "previous agent only to resolve short or ambiguous follow-ups, not to override a clear "
    "current intent. Keep the previous agent if the message still fits it; switch when it "
    "clearly belongs to another agent. Match strictly against the descriptions, and route "
    "to a human only on an explicit request."
)

COLUMNS = ["id", "sender", "brand", "current_message", "jev_output", "confidence",
           "response_time", "probabilities", "input_tokens", "output_tokens"]

load_dotenv(ROOT / ".env")
API_KEY = os.environ["JEV_API_KEY"]


def load_criteria_by_brand():
    data = json.loads(BRANDS.read_text(encoding="utf-8"))
    return {b["brand"]: {a["agentName"]: a["description"] for a in b["agentsInfo"]} for b in data}


def load_done_ids():
    done = set()
    if OUT.exists():
        with open(OUT, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("id"):
                    done.add(r["id"])
    return done


def call_jev(row, criteria, session):
    body = {
        "state": {
            "current_message": row["current_message"],
            "conversation_history": row["conversation_history"],
            "previous_agent": row["previous_agent"],
        },
        "model": MODEL,
        "questions": {"agent": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}},
    }
    headers = {"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"}
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.time()
            resp = session.post(API_URL, headers=headers, json=body, timeout=TIMEOUT)
            dt = time.time() - t0
            if resp.status_code == 200:
                return resp.json(), dt
            if resp.status_code == 429 or resp.status_code >= 500:      # transient
                last_err = "http_%d" % resp.status_code
                time.sleep(min(2 ** attempt, 30))
                continue
            raise RuntimeError("http_%d: %s" % (resp.status_code, resp.text[:200]))   # non-transient
        except requests.RequestException as e:
            last_err = "%s: %s" % (type(e).__name__, str(e)[:120])
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError("failed after %d retries (%s)" % (MAX_RETRIES, last_err))


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    rows = json.loads(DATASET.read_text(encoding="utf-8"))["routing_dataset"]  # chronological, oldest-first
    criteria_by_brand = load_criteria_by_brand()
    done = load_done_ids()
    new_file = not OUT.exists()

    session = requests.Session()
    processed = 0
    skipped = len(done)
    print("total dataset rows: %d | already done: %d" % (len(rows), skipped), flush=True)

    with open(OUT, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
            f.flush()
        for row in rows:
            if row["id"] in done:
                continue
            criteria = criteria_by_brand[row["brand"]]
            data, dt = call_jev(row, criteria, session)
            ans = data.get("answers", {}).get("agent", {})
            usage = data.get("usage") or {}
            writer.writerow({
                "id": row["id"],
                "sender": row["sender"],
                "brand": row["brand"],
                "current_message": row["current_message"],
                "jev_output": ans.get("choice"),
                "confidence": ans.get("confidence"),
                "response_time": round(dt, 3),
                "probabilities": json.dumps(ans.get("probabilities") or {}, ensure_ascii=False),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
            })
            f.flush()                      # progress-save per row (resume-safe)
            processed += 1
            if processed % 50 == 0:
                print("  processed %d (total in csv %d)" % (processed, skipped + processed), flush=True)
            if limit is not None and processed >= limit:
                print("hit --limit %d, stopping" % limit, flush=True)
                break

    print("DONE: processed %d this run, %d were already done, %d total in CSV"
          % (processed, skipped, skipped + processed), flush=True)


if __name__ == "__main__":
    main()

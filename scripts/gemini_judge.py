"""
Gemini++ runner — re-route the golden set with Gemini, THIS TIME giving it the
agent descriptions (the production prompt omitted them). Same real data, same
description-aware prompt as the GPT-6 judge -> a fair "Gemini with descriptions"
baseline to compare against Jev and the human gold.

- Model: gemini-3-flash-preview (the production model) via the OpenAI-compatible
  endpoint. Verify the exact id in your Google AI Studio; if the preview is gone,
  use the closest current id (e.g. gemini-3.5-flash).
- Runs on ALL 1,500 golden rows (agreements + all disagreements, incl. the human
  high-conf ones) so we can compare vs Jev, GPT-6, and human gold.
- Blind (not shown Gemini's silver or Jev's answers), retry+backoff, resume-safe.

Key: put GEMINI_API_KEY (or GOOGLE_API_KEY) in .env.

Usage:
  python scripts/gemini_judge.py --limit 5   # smoke test
  python scripts/gemini_judge.py             # full 1,500 (resumes)
"""

import csv
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import (OpenAI, RateLimitError, APIConnectionError, APITimeoutError,
                    InternalServerError)

ROOT = Path(__file__).resolve().parents[1]
SET = ROOT / "eval-data" / "golden_labeling_set.json"
OUT = ROOT / "eval-data" / "gemini_plus_labels.csv"

MODEL = "gemini-3-flash-preview"   # <-- production model; verify/adjust in Google AI Studio
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MAX_RETRIES = 5
COLUMNS = ["id", "brand", "gemini_plus_agent", "gemini_plus_reason", "valid_choice", "model"]

# identical prompt to the GPT-6 judge, so this is a fair "Gemini + descriptions" test
SYSTEM_PROMPT = (
    "You are an expert customer-support routing judge for a multi-brand WhatsApp assistant. "
    "Your job is to decide which ONE agent should handle the customer's CURRENT message, "
    "choosing only from the agents provided for this brand. You are a careful reference "
    "labeler, not a fast production router - reason carefully and correctly.\n\n"
    "Rules:\n"
    "1. Base the decision primarily on the CURRENT message. Use the conversation history and "
    "the previous agent only to interpret short or ambiguous follow-ups (e.g. 'yes', 'ok', a "
    "bare number) - never to override a clear new intent.\n"
    "2. Keep the previous agent if the current message still fits its responsibility; switch "
    "when the message clearly belongs to a different agent.\n"
    "3. Match strictly against the provided agent descriptions. Do not invent agents or rely on "
    "knowledge beyond the descriptions.\n"
    "4. Route to a human/handoff/connect agent ONLY when the customer explicitly asks to talk "
    "to a human.\n"
    "5. You MUST choose exactly one agent, using its exact agentName as written. If the message "
    "is ambiguous, choose the single best fit - never refuse or return anything outside the list.\n\n"
    "Return ONLY a JSON object: {\"agent\": \"<exact agentName>\", \"reason\": \"<one concise sentence>\"}."
)


def build_user_prompt(row):
    agents = row.get("available_agents") or []
    agent_lines = "\n".join(f"- {a['agentName']}: {a['description']}" for a in agents)
    hist = row.get("conversation_history") or []
    hist_lines = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in hist) if hist \
        else "(none - this is the first available turn)"
    prev = row.get("previous_agent") or "none"
    return (
        f"BRAND: {row.get('brand')}\n\n"
        f"AVAILABLE AGENTS (choose exactly one, by its exact name):\n{agent_lines}\n\n"
        f"CONVERSATION HISTORY (oldest to newest):\n{hist_lines}\n\n"
        f"PREVIOUS AGENT (handled the prior turn): {prev}\n\n"
        f"CURRENT CUSTOMER MESSAGE:\n{row.get('current_message')}\n\n"
        "Return the JSON now."
    )


def parse_json(text):
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        t = m.group(0)
    return json.loads(t)


def call_gemini(client, row):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(row)},
    ]
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(model=MODEL, messages=messages)
            data = parse_json(resp.choices[0].message.content)
            return data.get("agent"), data.get("reason")
        except (json.JSONDecodeError, KeyError, AttributeError):
            last = "bad_json"
            time.sleep(min(2 ** attempt, 20))
        except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as e:
            last = f"{type(e).__name__}: {str(e)[:120]}"
            time.sleep(min(2 ** attempt, 30))
        # other errors (BadRequestError, AuthenticationError, ...) propagate immediately
    raise RuntimeError(f"failed after {MAX_RETRIES} retries ({last})")


def load_done():
    done = set()
    if OUT.exists():
        with open(OUT, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("id"):
                    done.add(r["id"])
    return done


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

    load_dotenv(ROOT / ".env")
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env")
    client = OpenAI(api_key=key, base_url=BASE_URL)

    rows = json.loads(SET.read_text(encoding="utf-8"))["labeling_set"]   # all 1,500
    done = load_done()
    allowed = {r["id"]: {a["agentName"] for a in (r.get("available_agents") or [])} for r in rows}
    new_file = not OUT.exists()
    processed, skipped = 0, len(done)
    print(f"rows to label: {len(rows)} | already done: {skipped} | model: {MODEL}", flush=True)

    with open(OUT, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader(); f.flush()
        for row in rows:
            if row["id"] in done:
                continue
            agent, reason = call_gemini(client, row)
            valid = agent in allowed.get(row["id"], set())
            writer.writerow({"id": row["id"], "brand": row["brand"],
                             "gemini_plus_agent": agent, "gemini_plus_reason": reason,
                             "valid_choice": valid, "model": MODEL})
            f.flush()
            processed += 1
            if not valid:
                print(f"  [warn] id={row['id']} returned '{agent}' not in brand agent set", flush=True)
            if processed % 50 == 0:
                print(f"  processed {processed} (total in csv {skipped + processed})", flush=True)
            if limit is not None and processed >= limit:
                print(f"hit --limit {limit}, stopping", flush=True)
                break

    print(f"DONE: processed {processed}, already-done {skipped}, total {skipped + processed}", flush=True)


if __name__ == "__main__":
    main()

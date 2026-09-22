"""
Build eval-data/routing_dataset.json from the WhatsApp LangSmith router logs.

Rules (from the spec):
  - WhatsApp only (sender = "<phone>-<brand>").
  - One row per WhatsApp user message (= one router run).
  - id = LangSmith run_id.  conversation identity = sender (phone+brand).
  - STRICT chronological processing, one message at a time (global time-sorted
    single pass; per-conversation last-agent map).  Never batch a sender together.
  - conversation_history = the run's own inputs.history (roles system->assistant);
    it is the production history strictly before the current message.
  - previous_agent = LLM_OUTPUT of the immediately preceding WA turn in the same
    conversation; null for the first in-window turn.
  - LLM_OUTPUT = raw outputs.output (silver label). Never invent / never modify source.

Source files are READ only. Deliverable = eval-data/routing_dataset.json.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "test_data" / "router_runs_2026-09-19_2026-09-21.json"
OUT = ROOT / "eval-data" / "routing_dataset.json"

ROLE_MAP = {"user": "user", "system": "assistant", "assistant": "assistant"}


def is_wa(sender: str) -> bool:
    parts = sender.split("-")
    return len(parts) == 2 and parts[0].isdigit() and parts[1] != ""


def brand_of(sender: str) -> str:
    return sender.split("-", 1)[1]


def map_history(hist):
    out = []
    for m in (hist or []):
        if not isinstance(m, dict):
            continue
        role = ROLE_MAP.get(m.get("role"), m.get("role"))
        out.append({"role": role, "content": m.get("content")})
    return out


def main():
    logs = json.loads(LOGS.read_text(encoding="utf-8"))
    wa = [r for r in logs if is_wa((r.get("inputs") or {}).get("sender") or "")]

    # strict chronological order; start_time is unique across all rows
    wa_sorted = sorted(wa, key=lambda r: (r["start_time"], r.get("end_time") or "", r["id"]))

    last_agent = {}   # conv -> previous turn's LLM_OUTPUT
    turn_no = {}      # conv -> running turn index
    rows = []
    for r in wa_sorted:
        sender = r["inputs"]["sender"]
        conv = sender                      # sender already encodes phone+brand
        prev = last_agent.get(conv, None)  # None(null) for the first in-window turn
        ti = turn_no.get(conv, 0) + 1
        turn_no[conv] = ti

        llm_output = (r.get("outputs") or {}).get("output")  # raw, faithful

        rows.append({
            "id": r["id"],
            "sender": sender,
            "brand": brand_of(sender),
            "turn_index": ti,
            "current_message": r["inputs"]["query"],
            "conversation_history": map_history(r["inputs"].get("history")),
            "previous_agent": prev,
            "LLM_OUTPUT": llm_output,
        })
        last_agent[conv] = llm_output      # next turn of this conv sees this agent

    meta = {
        "channel": "whatsapp",
        "note": ("Inputs to Jev/Laya = current_message + conversation_history + "
                 "previous_agent + available_agents. available_agents is NOT stored here "
                 "- resolved at runtime from brand (+ channel=whatsapp) via "
                 "brands_level_ai_info.json (internal routing_agent.py descriptions + that "
                 "brand's external aiInfo flows). NOT passed to the model: id, sender, brand, "
                 "LLM_OUTPUT. LLM_OUTPUT is the SILVER label (raw router output). History "
                 "roles anonymized (system->assistant)."),
        "source_logs": LOGS.name,
        "window": "2026-09-19..2026-09-21 IST (WhatsApp only)",
        "conversation_identity": "sender (phone+brand)",
        "row_count": len(rows),
        "conversation_count": len(turn_no),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"meta": meta, "routing_dataset": rows},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(rows)} rows / {len(turn_no)} conversations -> {OUT}")


if __name__ == "__main__":
    main()

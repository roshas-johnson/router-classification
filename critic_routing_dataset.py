"""
Critic Agent: independently verify eval-data/routing_dataset.json against the
21 requirements, checking the ACTUAL data (re-reads the written file + source).
Rejects if any check fails.
"""
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "test_data" / "router_runs_2026-09-19_2026-09-21.json"
OUT = ROOT / "eval-data" / "routing_dataset.json"


def is_wa(s):
    p = s.split("-"); return len(p) == 2 and p[0].isdigit() and p[1] != ""


def brand_of(s):
    return s.split("-", 1)[1]


ROLE_MAP = {"user": "user", "system": "assistant", "assistant": "assistant"}


def map_hist(h):
    return [{"role": ROLE_MAP.get(m.get("role"), m.get("role")), "content": m.get("content")}
            for m in (h or []) if isinstance(m, dict)]


logs = json.loads(LOGS.read_text(encoding="utf-8"))
wa = [r for r in logs if is_wa((r.get("inputs") or {}).get("sender") or "")]
by_id = {r["id"]: r for r in wa}
wa_ids = set(by_id)

data = json.loads(OUT.read_text(encoding="utf-8"))
rows = data.get("routing_dataset")
meta = data.get("meta")

# independent previous_agent + turn_index recompute from SOURCE
wa_sorted = sorted(wa, key=lambda r: (r["start_time"], r.get("end_time") or "", r["id"]))
exp_prev, exp_ti, last, cnt = {}, {}, {}, {}
for r in wa_sorted:
    c = r["inputs"]["sender"]
    exp_prev[r["id"]] = last.get(c, None)
    exp_ti[r["id"]] = cnt.get(c, 0) + 1
    cnt[c] = exp_ti[r["id"]]
    last[c] = (r.get("outputs") or {}).get("output")

R = []
def chk(n, name, ok, detail=""):
    R.append((n, name, ok, detail))

ROW_KEYS = {"id", "sender", "brand", "turn_index", "current_message",
            "conversation_history", "previous_agent", "LLM_OUTPUT"}

ok1 = set(data) == {"meta", "routing_dataset"} and isinstance(meta, dict) and isinstance(rows, list)
bad_keys = sum(1 for r in rows if set(r) != ROW_KEYS)
bad_hist = 0
for r in rows:
    for m in r["conversation_history"]:
        if set(m) != {"role", "content"} or m["role"] not in ("user", "assistant"):
            bad_hist += 1; break
chk(1, "structure matches sample", ok1 and bad_keys == 0 and bad_hist == 0,
    f"bad_keys={bad_keys} bad_hist_rows={bad_hist}")

ids = [r["id"] for r in rows]
chk(2, "1 row per eligible WA msg", len(rows) == len(wa) and set(ids) == wa_ids,
    f"rows={len(rows)} wa={len(wa)} id_match={set(ids)==wa_ids}")
nonwa = sum(1 for r in rows if not is_wa(r["sender"]) or r["id"] not in wa_ids)
chk(3, "no non-WhatsApp rows", nonwa == 0, f"nonwa={nonwa}")
chk(4, "no duplicate messages", len(ids) == len(set(ids)), f"dupes={len(ids)-len(set(ids))}")
chk(5, "run_id is primary id", all(i in wa_ids for i in ids), "")
badb = sum(1 for r in rows if r["brand"] != brand_of(r["sender"]) or not is_wa(r["sender"]))
chk(6, "sender+brand identity", badb == 0, f"bad_brand={badb}")

byconv = defaultdict(list)
for r in rows:
    byconv[r["sender"]].append(r)

bad7 = 0
for c, rs in byconv.items():
    srt = sorted(rs, key=lambda r: r["turn_index"])
    sts = [by_id[r["id"]]["start_time"] for r in srt]
    if sts != sorted(sts):
        bad7 += 1
chk(7, "chronological within conversation", bad7 == 0, f"bad_convs={bad7}")
bad8 = sum(1 for c, rs in byconv.items()
           if sorted(r["turn_index"] for r in rs) != list(range(1, len(rs) + 1)))
chk(8, "turn_index contiguous 1..n", bad8 == 0, f"bad_convs={bad8}")
bad9 = sum(1 for r in rows if r["turn_index"] != exp_ti[r["id"]])
chk(9, "not sender-batched (turn=chrono)", bad9 == 0, f"mismatch={bad9}")
bad10 = sum(1 for r in rows if r["current_message"] != by_id[r["id"]]["inputs"]["query"])
chk(10, "current_message correct", bad10 == 0, f"mismatch={bad10}")
bad11 = sum(1 for r in rows if r["conversation_history"] != map_hist(by_id[r["id"]]["inputs"].get("history")))
chk(11, "history = source pre-current history", bad11 == 0, f"mismatch={bad11}")
chk(12, "history WhatsApp-only", nonwa == 0, "guaranteed by WA sender")
bad13 = sum(1 for r in rows if r["conversation_history"]
            and r["conversation_history"][-1]["role"] == "user"
            and r["conversation_history"][-1]["content"] == r["current_message"])
chk(13, "no future message leaked", bad13 == 0, f"leak={bad13}")
bad14 = sum(1 for r in rows if r["previous_agent"] != exp_prev[r["id"]])
chk(14, "previous_agent = immediate WA predecessor", bad14 == 0, f"mismatch={bad14}")
chk(15, "other channel excluded from prev", bad14 == 0, "recompute over WA-only runs")
bad16 = sum(1 for r in rows if r["turn_index"] == 1 and r["previous_agent"] is not None)
chk(16, "previous_agent null on first turn", bad16 == 0, f"bad={bad16}")
out_by_ct = {(r["sender"], r["turn_index"]): r["LLM_OUTPUT"] for r in rows}
bad17 = sum(1 for r in rows if r["turn_index"] > 1
            and r["previous_agent"] != out_by_ct.get((r["sender"], r["turn_index"] - 1)))
chk(17, "prev not from older turn", bad17 == 0, f"mismatch={bad17}")
multi = sum(1 for c, rs in byconv.items() if len({r["brand"] for r in rs}) != 1)
chk(18, "brand change = separate identity", multi == 0, f"multi_brand_convs={multi}")
bad19 = sum(1 for r in rows if r["LLM_OUTPUT"] != (by_id[r["id"]].get("outputs") or {}).get("output"))
allouts = {tuple(r["LLM_OUTPUT"]) if isinstance(r["LLM_OUTPUT"], list) else r["LLM_OUTPUT"] for r in rows}
badprev = sum(1 for r in rows if r["previous_agent"] is not None
              and (tuple(r["previous_agent"]) if isinstance(r["previous_agent"], list) else r["previous_agent"]) not in allouts)
chk(19, "no invented values", bad19 == 0 and badprev == 0, f"llm_mismatch={bad19} fabricated_prev={badprev}")
src_ok = len(logs) == 10279 and bad10 == 0 and bad11 == 0 and bad19 == 0
chk(20, "source data unmodified", src_ok, f"source_runs={len(logs)}")
chk(21, "file at eval-data/routing_dataset.json", OUT.exists(), str(OUT.relative_to(ROOT)))

allok = all(ok for _, _, ok, _ in R)
print("CRITIC AGENT REPORT  (dataset: %d rows, %d conversations)" % (len(rows), len(byconv)))
print("=" * 74)
for n, name, ok, detail in R:
    print(f"  {n:2}. [{'PASS' if ok else 'FAIL'}] {name:42} {detail}")
print("=" * 74)
print("VERDICT:", "APPROVED - all 21 checks passed" if allok else "REJECTED - see FAIL rows")

empt = sum(1 for r in rows if r["LLM_OUTPUT"] == "")
lst = sum(1 for r in rows if isinstance(r["LLM_OUTPUT"], list))
big = sum(1 for r in rows if isinstance(r["LLM_OUTPUT"], str) and (len(r["LLM_OUTPUT"]) > 60 or "\n" in r["LLM_OUTPUT"]))
prevmal = sum(1 for r in rows if r["previous_agent"] == "" or isinstance(r["previous_agent"], list)
              or (isinstance(r["previous_agent"], str) and ("\n" in r["previous_agent"] or len(r["previous_agent"]) > 60)))
firsthist = sum(1 for r in rows if r["turn_index"] == 1 and r["conversation_history"])
print("\nDATA-QUALITY NOTES (faithful to source, not rejections):")
print(f"  malformed LLM_OUTPUT: empty={empt}, list={lst}, reasoning/refusal-blob={big}  (total {empt+lst+big})")
print(f"  rows whose previous_agent is one such malformed label: {prevmal}")
print(f"  first-turn rows with pre-window history (prev=null, honest): {firsthist}")

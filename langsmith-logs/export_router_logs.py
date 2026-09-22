"""
Export raw `router` runs from LangSmith to JSON.

This file ONLY exports. No cleaning, no field mapping, no sessionization, no
previous_agent, no brand/channel/description enrichment — those happen in
separate files.

Each row is the router run as LangSmith returns it (untouched):
  id, trace_id, start/end time, latency, inputs, outputs, error, tokens, cost, extra.

RUN  (LANGSMITH_API_KEY is read from the project-root .env)
  python export_router_logs.py --start 2026-09-19 --end 2026-09-21 --limit 20   # smoke test
  python export_router_logs.py --start 2026-09-19 --end 2026-09-21              # full window

--start / --end are LOCAL calendar dates, both inclusive, interpreted with
--utc-offset (default +05:30 = IST). LangSmith stores times in UTC.
"""

import argparse
import itertools
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

# Load keys from the project-root .env (one level up from this file), so the
# script works no matter which folder you run it from.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

PROJECT = "llm-bot-expert-memory"
ROUTER_RUN_NAME = "router"


def parse_offset(s):
    # "+05:30" -> timezone(+5:30)
    sign = -1 if s.startswith("-") else 1
    h, m = s.lstrip("+-").split(":")
    return timezone(sign * timedelta(hours=int(h), minutes=int(m)))


def window_utc(start_date, end_date, tz):
    # local dates (inclusive) -> [start_utc, end_utc) in UTC
    start = end = None
    if start_date:
        start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=tz).astimezone(timezone.utc)
    if end_date:
        end = (datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=tz)
               + timedelta(days=1)).astimezone(timezone.utc)
    return start, end


def export(project, out_file, limit=None, start_date=None, end_date=None, utc_offset="+05:30"):
    if not (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")):
        raise SystemExit("LANGSMITH_API_KEY not found. Add it to the project-root .env")

    start_utc, end_utc = window_utc(start_date, end_date, parse_offset(utc_offset))
    fmt = lambda d: d.strftime("%Y-%m-%dT%H:%M:%S")
    clauses = [f'eq(name, "{ROUTER_RUN_NAME}")']
    if start_utc:
        clauses.append(f'gte(start_time, "{fmt(start_utc)}")')
    if end_utc:
        clauses.append(f'lt(start_time, "{fmt(end_utc)}")')
    run_filter = clauses[0] if len(clauses) == 1 else f"and({', '.join(clauses)})"
    print(f"window (UTC): {start_utc} -> {end_utc}")
    print(f"filter: {run_filter}")

    client = Client()  # reads LANGSMITH_API_KEY from env

    # The API caps a single query's page size at 100, so let the SDK paginate
    # internally (no `limit` here) and cap the TOTAL we consume with islice.
    runs = client.list_runs(
        project_name=project,
        filter=run_filter,
        is_root=True,            # depth-0 parent `router` run only
        start_time=start_utc,    # index-level lower bound
    )
    if limit:
        runs = itertools.islice(runs, limit)

    def as_float(x):
        return float(x) if x is not None else None

    rows, outside = [], 0
    for r in runs:
        # guard: skip anything the server-side filter let through outside the window
        if (start_utc and r.start_time and r.start_time < start_utc) or \
           (end_utc and r.start_time and r.start_time >= end_utc):
            outside += 1
            continue
        rows.append({
            "id": str(r.id),
            "trace_id": str(r.trace_id),
            "start_time": r.start_time.isoformat() if r.start_time else None,
            "end_time": r.end_time.isoformat() if r.end_time else None,
            "latency_s": (r.end_time - r.start_time).total_seconds()
                         if (r.start_time and r.end_time) else None,
            "inputs": r.inputs,          # raw: query, history, sender, llm
            "outputs": r.outputs,        # raw: {"output": "<AgentName>"}
            "error": r.error,
            # tokens
            "total_tokens": r.total_tokens,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "reasoning_tokens": (r.completion_token_details or {}).get("reasoning"),
            "cache_read_tokens": (r.prompt_token_details or {}).get("cache_read"),
            # cost (USD, computed by LangSmith)
            "total_cost": as_float(r.total_cost),
            "prompt_cost": as_float(r.prompt_cost),
            "completion_cost": as_float(r.completion_cost),
            "extra": r.extra,            # raw metadata (model name, etc.)
        })
        if len(rows) % 1000 == 0:
            print(f"  ...{len(rows)} runs", flush=True)

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False, default=str)

    print(f"exported {len(rows)} router runs -> {out_file}")
    if outside:
        print(f"  skipped {outside} runs outside the window (server filter leaked)")


def main():
    p = argparse.ArgumentParser(description="Export raw LangSmith router runs to JSON.")
    p.add_argument("--project", default=PROJECT, help="LangSmith project name")
    p.add_argument("--start", help="local start date YYYY-MM-DD (inclusive)")
    p.add_argument("--end", help="local end date YYYY-MM-DD (inclusive)")
    p.add_argument("--utc-offset", default="+05:30", help="local UTC offset (default IST +05:30)")
    p.add_argument("--limit", type=int, default=None, help="max runs to pull (omit for all)")
    p.add_argument("--out", default="router_runs_raw.json", help="output JSON path")
    args = p.parse_args()
    export(args.project, args.out, args.limit, args.start, args.end, args.utc_offset)


if __name__ == "__main__":
    main()

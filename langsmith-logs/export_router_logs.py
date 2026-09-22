"""
Export raw `router` runs from LangSmith to JSON.

This file ONLY exports. No cleaning, no field mapping, no sessionization, no
previous_agent, no brand/channel/description enrichment, no language tagging —
all of that happens in separate files later.

Each row is the router run as LangSmith returns it (untouched):
  id, trace_id, start_time, inputs, outputs, error, token counts, extra/metadata.

RUN
  set LANGSMITH_API_KEY=...          (EU / self-hosted: also set LANGSMITH_ENDPOINT)
  python export_router_logs.py --limit 20     # small test export first
  python export_router_logs.py                # full export
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

# Load keys from the project-root .env (one level up from this file), so the
# script works no matter which folder you run it from.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

PROJECT = "llm-bot-expert-memory"
ROUTER_RUN_NAME = "router"


def export(project, out_file, limit=None):
    if not (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")):
        raise SystemExit("LANGSMITH_API_KEY not found. Add it to the project-root .env")

    client = Client()  # reads LANGSMITH_API_KEY from env

    runs = client.list_runs(
        project_name=project,
        filter=f'eq(name, "{ROUTER_RUN_NAME}")',
        is_root=True,            # depth-0 parent `router` run only
        limit=limit,             # None = all
    )

    rows = []
    for r in runs:
        rows.append({
            "id": str(r.id),
            "trace_id": str(r.trace_id),
            "start_time": r.start_time.isoformat() if r.start_time else None,
            "inputs": r.inputs,          # raw: query, history, sender, llm
            "outputs": r.outputs,        # raw: {"output": "<AgentName>"}
            "error": r.error,
            "total_tokens": r.total_tokens,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "extra": r.extra,            # raw metadata (model name, etc.)
        })

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False, default=str)

    print(f"exported {len(rows)} router runs -> {out_file}")


def main():
    p = argparse.ArgumentParser(description="Export raw LangSmith router runs to JSON.")
    p.add_argument("--project", default=PROJECT, help="LangSmith project name")
    p.add_argument("--limit", type=int, default=None, help="max runs to pull (omit for all)")
    p.add_argument("--out", default="router_runs_raw.json", help="output JSON path")
    args = p.parse_args()
    export(args.project, args.out, args.limit)


if __name__ == "__main__":
    main()

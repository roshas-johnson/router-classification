"""
Build brands_level_ai_info.json — per-brand WhatsApp agents + descriptions.

Pipeline
  1. Read brands from the LangSmith logs (WhatsApp senders only: <phone>-<brand>).
  2. For each unique brand, GET https://<brand>.bobot.in/kv/aiInfo  (once, cached).
  3. For every agent in channels.whatsapp, resolve a description:
        external_flows[].description   (priority)
          -> internal-agent-descriptions.json
          -> "NO_DISC_FOUND"
  4. Validate, then write brands_level_ai_info.json in this (parent) directory.

Source files are only READ, never modified. API failures are recorded per brand
and do not stop the run. Descriptions are never invented.
"""

import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "test_data" / "router_runs_2026-09-19_2026-09-21.json"
INTERNAL_DESC = ROOT / "bob-level-data" / "internal-agent-descriptions.json"
OUT = ROOT / "brands_level_ai_info.json"
ERR_OUT = ROOT / "brands_level_ai_info.errors.json"

AIINFO_URL = "https://{brand}.bobot.in/kv/aiInfo"
NO_DISC = "NO_DISC_FOUND"
TIMEOUT = 15


# ---------- brand extraction (WhatsApp only) ----------
def is_whatsapp_sender(sender: str) -> bool:
    # WhatsApp sender = "<phone>-<brand>": exactly 2 hyphen-parts, first all digits.
    # (UUID senders from other channels have 6 parts and are excluded.)
    parts = sender.split("-")
    return len(parts) == 2 and parts[0].isdigit() and parts[1] != ""


def brand_from_sender(sender: str) -> str:
    # everything after the first hyphen -> brand, e.g. "918057059746-zop" -> "zop"
    return sender.split("-", 1)[1]


def unique_brands_from_logs(logs) -> list:
    brands, seen = [], set()
    for r in logs:
        sender = (r.get("inputs") or {}).get("sender") or ""
        if is_whatsapp_sender(sender):
            b = brand_from_sender(sender)
            if b not in seen:
                seen.add(b)
                brands.append(b)
    return brands


# ---------- aiInfo fetch ----------
def unwrap_config(payload):
    # aiInfo may return the config directly or wrapped; find the object holding channels/external_flows
    if isinstance(payload, dict):
        if "channels" in payload or "external_flows" in payload:
            return payload
        for k in ("value", "data", "result", "aiInfo"):
            v = payload.get(k)
            if isinstance(v, dict) and ("channels" in v or "external_flows" in v):
                return v
    return None


def fetch_ai_info(brand, session, errors):
    url = AIINFO_URL.format(brand=brand)
    try:
        resp = session.get(url, timeout=TIMEOUT)
    except requests.RequestException as e:
        errors[brand] = f"request_error: {type(e).__name__}: {e}"
        return None
    if resp.status_code != 200:
        errors[brand] = f"http_{resp.status_code}"
        return None
    try:
        payload = resp.json()
    except ValueError:
        errors[brand] = "invalid_json"
        return None
    cfg = unwrap_config(payload)
    if cfg is None:
        errors[brand] = "no_channels_or_external_flows_in_response"
        return None
    return cfg


# ---------- description resolution ----------
def external_desc_map(cfg) -> dict:
    out = {}
    for f in (cfg.get("external_flows") or []):
        if isinstance(f, dict):
            name = f.get("agent_name")
            if isinstance(name, str) and name:
                out[name] = f.get("description")
    return out


def whatsapp_agents(cfg) -> list:
    wa = ((cfg.get("channels") or {}).get("whatsapp")) or []
    return [a for a in wa if isinstance(a, str) and a]


def resolve_description(agent, ext, internal):
    # External AI agent description -> Internal -> NO_DISC_FOUND
    if agent in ext and ext[agent]:
        return ext[agent]
    if agent in internal and internal[agent]:
        return internal[agent]
    return NO_DISC


def build_agents_info(cfg, internal):
    ext = external_desc_map(cfg)
    return [{"agentName": a, "description": resolve_description(a, ext, internal)}
            for a in whatsapp_agents(cfg)]


# ---------- validation (runs before writing) ----------
def validate(output, cfgs, internal):
    assert isinstance(output, list), "output must be a list"
    for entry in output:
        brand = entry.get("brand")
        assert isinstance(brand, str) and brand, f"bad brand: {entry!r}"
        assert isinstance(entry.get("agentsInfo"), list), f"{brand}: agentsInfo must be a list"

        expected = whatsapp_agents(cfgs[brand])          # #6 no skipped agents
        got = [a["agentName"] for a in entry["agentsInfo"]]
        assert got == expected, f"{brand}: agent list mismatch {got} != {expected}"

        ext = external_desc_map(cfgs[brand])
        for a in entry["agentsInfo"]:
            name, desc = a.get("agentName"), a.get("description")
            assert isinstance(name, str) and name, f"{brand}: empty agentName"
            assert isinstance(desc, str) and desc, f"{brand}/{name}: empty description"
            if name in ext and ext[name]:               # #5 external overrides internal
                assert desc == ext[name], f"{brand}/{name}: external description not used"
            elif name in internal and internal[name]:
                assert desc == internal[name], f"{brand}/{name}: internal description not used"
            else:
                assert desc == NO_DISC, f"{brand}/{name}: expected {NO_DISC}, got {desc!r}"
    return True


def main():
    logs = json.loads(LOGS.read_text(encoding="utf-8"))
    internal = json.loads(INTERNAL_DESC.read_text(encoding="utf-8"))
    brands = unique_brands_from_logs(logs)
    print(f"brands from WhatsApp logs: {len(brands)}")

    session = requests.Session()
    session.headers.update({"User-Agent": "nostrana-ai-info-builder/1.0"})

    output, cfgs, errors = [], {}, {}
    for i, brand in enumerate(brands, 1):
        cfg = fetch_ai_info(brand, session, errors)      # one call per unique brand
        if cfg is None:
            print(f"  [{i}/{len(brands)}] {brand:22} FAILED: {errors[brand]}")
            continue
        cfgs[brand] = cfg
        agents_info = build_agents_info(cfg, internal)
        output.append({"brand": brand, "agentsInfo": agents_info})
        n_nodisc = sum(1 for a in agents_info if a["description"] == NO_DISC)
        print(f"  [{i}/{len(brands)}] {brand:22} ok: {len(agents_info)} wa-agents"
              + (f", {n_nodisc} {NO_DISC}" if n_nodisc else ""))

    validate(output, cfgs, internal)
    print("VALIDATION PASSED")

    OUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(output)} brands -> {OUT.name}")

    if errors:
        ERR_OUT.write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"recorded {len(errors)} failed brands -> {ERR_OUT.name}")
        for b, why in errors.items():
            print(f"    - {b}: {why}")

    total_agents = sum(len(e["agentsInfo"]) for e in output)
    total_nodisc = sum(1 for e in output for a in e["agentsInfo"] if a["description"] == NO_DISC)
    print(f"summary: {len(output)} brands ok, {len(errors)} failed, "
          f"{total_agents} agents, {total_nodisc} NO_DISC_FOUND")


if __name__ == "__main__":
    sys.exit(main())

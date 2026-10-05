"""Assemble comparison.ipynb (4-way router eval: LLM silver vs Jev vs Laya x2)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "routercmp", "display_name": "router-cmp", "language": "python"}
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md(r"""# Router A/B evaluation — V1 (WhatsApp)
**LLM (Gemini) silver** vs **Jev** vs **Laya-multilingual** vs **Laya-typed-decisions**

> **What "agreement" means here:** the silver label is the *LLM router's own output*, so agreement measures whether a typed model can **reproduce the current router**. High agreement -> safe drop-in. Where a typed model *confidently disagrees*, it may actually be **right** and the LLM wrong -> those are V2 gold-review targets, not automatically errors.

Metrics: agreement (overall / per-brand / per-language), confusion, confidence calibration, cost, latency.""")

code(r"""%matplotlib inline
import json, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (9, 5)
plt.rcParams["figure.dpi"] = 110
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)

ROOT = Path.cwd()
if not (ROOT / "eval-data").exists() and (ROOT.parent / "eval-data").exists():
    ROOT = ROOT.parent
EVAL = ROOT / "eval-data"
LOGS = ROOT / "test_data" / "router_runs_2026-09-19_2026-09-21.json"
BRANDS = ROOT / "brands_level_ai_info.json"
print("ROOT:", ROOT)""")

code(r"""# ---------- helpers ----------
def canon(name):
    # collapse ShoppingAgentVn -> ShoppingAgent; strip whitespace (approved normalization)
    if not isinstance(name, str):
        return name
    n = name.strip()
    return re.sub(r"^ShoppingAgentV\d+$", "ShoppingAgent", n)

def base_cat(name):
    # coarse cross-brand category: canon + drop trailing 'WA' channel suffix
    if not isinstance(name, str):
        return name
    return re.sub(r"WA$", "", canon(name))

def is_malformed(x):
    # silver the LLM failed to emit cleanly (empty / reasoning-blob / refusal)
    if not isinstance(x, str):
        return True
    x = x.strip()
    return (x == "") or ("\n" in x) or (len(x) > 60)

def top_prob(js):
    try:
        p = json.loads(js) if isinstance(js, str) else js
        return max(p.values()) if p else np.nan
    except Exception:
        return np.nan""")

code(r"""# ---------- load silver (LLM) + logs (cost/latency) + brand agent sets ----------
ds = json.load(open(EVAL / "routing_dataset.json", encoding="utf-8"))["routing_dataset"]
df = pd.DataFrame([{"id": r["id"], "sender": r["sender"], "brand": r["brand"],
                    "turn_index": r["turn_index"], "current_message": r["current_message"],
                    "previous_agent": r["previous_agent"], "silver_raw": r["LLM_OUTPUT"]} for r in ds])

logs = json.load(open(LOGS, encoding="utf-8"))
llm = pd.DataFrame([{"id": r["id"], "llm_latency": r.get("latency_s"), "llm_cost": r.get("total_cost")} for r in logs])
df = df.merge(llm, on="id", how="left")

brands = json.load(open(BRANDS, encoding="utf-8"))
brand_agents = {b["brand"]: set(canon(a["agentName"]) for a in b["agentsInfo"]) for b in brands}
print("dataset rows:", len(df), "| brands:", len(brand_agents))""")

code(r"""# ---------- auto-detect result arms and merge ----------
ARM_SPECS = [
    ("Jev",               "Jev-classification-result.csv",   "jev_output",  "probabilities"),
    ("Laya-multilingual", "Laya-multilingual-result.csv",    "laya_output", "distribution"),
    ("Laya-typed",        "Laya-typed-decisions-result.csv", "laya_output", "distribution"),
]
ARMS = []
for name, fname, outc, probc in ARM_SPECS:
    p = EVAL / fname
    if not p.exists():
        print("MISSING (skipped):", fname); continue
    d = pd.read_csv(p)
    keep = ["id", outc, "confidence", "response_time", probc] + (["input_tokens"] if "input_tokens" in d.columns else [])
    d = d[keep].rename(columns={outc: f"{name}__out", "confidence": f"{name}__conf",
                                "response_time": f"{name}__rt", probc: f"{name}__prob",
                                "input_tokens": f"{name}__intok"})
    df = df.merge(d, on="id", how="left")
    ARMS.append(name)
print("arms present:", ARMS)""")

code(r"""# ---------- normalize + scoring set + language + agree flags ----------
df["silver"] = df["silver_raw"].map(canon)
for name in ARMS:
    df[f"{name}__out_c"] = df[f"{name}__out"].map(canon)
    df[f"{name}__conf"] = pd.to_numeric(df[f"{name}__conf"], errors="coerce")
    if df[f"{name}__conf"].isna().all():
        df[f"{name}__conf"] = df[f"{name}__prob"].map(top_prob)
    df[f"{name}__rt"] = pd.to_numeric(df[f"{name}__rt"], errors="coerce")

df["silver_malformed"] = df["silver_raw"].map(is_malformed)
df["silver_routable"] = df.apply(lambda r: isinstance(r["silver"], str) and r["silver"] in brand_agents.get(r["brand"], set()), axis=1)
df["scored"] = (~df["silver_malformed"]) & (df["silver_routable"])
for name in ARMS:
    df[f"{name}__agree"] = df[f"{name}__out_c"] == df["silver"]

n = len(df)
print(f"total rows          : {n}")
print(f"malformed silver    : {int(df['silver_malformed'].sum())}  (LLM self-failures -> excluded, reported separately)")
print(f"non-routable silver : {int((~df['silver_routable'] & ~df['silver_malformed']).sum())}  (label not in brand agent set)")
print(f"SCORED set          : {int(df['scored'].sum())}  ({100*df['scored'].mean():.1f}%)")
scored = df[df["scored"]].copy()
COLORS = {"Jev": "#4C78A8", "Laya-multilingual": "#F58518", "Laya-typed": "#54A24B"}""")

md(r"""## 1. Overall agreement with the LLM router""")
code(r"""ov = pd.Series({name: scored[f"{name}__agree"].mean() for name in ARMS}).sort_values(ascending=False)
print("Overall agreement with LLM silver (scored set, n=%d):" % len(scored))
print((ov * 100).round(2))
ax = (ov * 100).plot(kind="bar", color=[COLORS.get(k, "#888") for k in ov.index])
ax.set_ylabel("Agreement (%)"); ax.set_ylim(0, 100)
ax.set_title(f"Overall agreement vs LLM router  (n={len(scored):,})")
for i, v in enumerate(ov * 100):
    ax.text(i, v + 1.2, f"{v:.1f}%", ha="center", fontweight="bold")
plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(EVAL / "fig_overall_agreement.png"); plt.show()""")

md(r"""## 2. Per-brand agreement (top brands by volume + macro average)""")
code(r"""top_brands = scored["brand"].value_counts().head(10).index.tolist()
rows = []
for b in top_brands:
    sub = scored[scored["brand"] == b]
    row = {"brand": b, "n": len(sub)}
    for name in ARMS:
        row[name] = sub[f"{name}__agree"].mean() * 100
    rows.append(row)
pb = pd.DataFrame(rows).set_index("brand")
print("Per-brand agreement % (top 10 brands):")
print(pb.assign(**{k: pb[k].round(1) for k in ARMS}))
macro = {name: scored.groupby("brand")[f"{name}__agree"].mean().mean() * 100 for name in ARMS}
print("\nMacro-avg across ALL brands:", {k: round(v, 1) for k, v in macro.items()})
ax = pb[ARMS].plot(kind="bar", color=[COLORS.get(k, "#888") for k in ARMS])
ax.set_ylabel("Agreement (%)"); ax.set_ylim(0, 100); ax.set_title("Per-brand agreement (top 10 brands)")
ax.legend(title="model", bbox_to_anchor=(1.01, 1)); plt.xticks(rotation=45, ha="right")
plt.tight_layout(); plt.savefig(EVAL / "fig_per_brand.png"); plt.show()""")

md(r"""## 3. Confusion matrices (base categories)
Rows = LLM silver, cols = model prediction. Names collapsed to base categories (drop `WA`, `Vn`) so brands line up; cells are counts, color is row-normalized.""")
code(r"""def confusion(arm, top=12):
    s = scored["silver"].map(base_cat); m = scored[f"{arm}__out_c"].map(base_cat)
    cats = s.value_counts().head(top).index.tolist()
    s2 = s.where(s.isin(cats), "other"); m2 = m.where(m.isin(cats), "other")
    order = cats + (["other"] if ("other" in set(s2) | set(m2)) else [])
    return pd.crosstab(s2, m2).reindex(index=order, columns=order, fill_value=0)

for arm in ARMS:
    cm = confusion(arm)
    cmn = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cmn, annot=cm.values, fmt="d", cmap="Blues", cbar_kws={"label": "row-normalized"})
    plt.title(f"{arm}: LLM silver (rows) vs {arm} prediction (cols)")
    plt.ylabel("LLM silver"); plt.xlabel(f"{arm} prediction"); plt.tight_layout()
    plt.savefig(EVAL / f"fig_confusion_{arm}.png"); plt.show()""")

md(r"""## 4. Confidence calibration
Does higher model confidence mean higher agreement? If yes, a **confidence gate** (use typed model when confident, else fall back to LLM) becomes viable.""")
code(r"""bins = np.linspace(0, 1, 11)
plt.figure()
for name in ARMS:
    sub = scored.dropna(subset=[f"{name}__conf"]).copy()
    sub["cb"] = pd.cut(sub[f"{name}__conf"], bins, include_lowest=True)
    g = sub.groupby("cb", observed=True)[f"{name}__agree"].mean() * 100
    centers = [iv.mid for iv in g.index]
    plt.plot(centers, g.values, marker="o", label=name, color=COLORS.get(name, "#888"))
plt.plot([0, 1], [0, 100], "k--", alpha=.3, label="perfect calibration")
plt.xlabel("model confidence"); plt.ylabel("agreement with silver (%)"); plt.ylim(0, 100)
plt.title("Confidence calibration"); plt.legend(); plt.tight_layout()
plt.savefig(EVAL / "fig_calibration.png"); plt.show()
print("Mean confidence on agree vs disagree:")
for name in ARMS:
    a = scored[scored[f"{name}__agree"]][f"{name}__conf"].mean()
    d = scored[~scored[f"{name}__agree"]][f"{name}__conf"].mean()
    print(f"  {name:18} agree={a:.3f}  disagree={d:.3f}")""")

md(r"""## 5. Cost & latency""")
code(r"""COST = {"LLM (Gemini)": df["llm_cost"].sum()}
if "Jev__intok" in df.columns:
    COST["Jev"] = df["Jev__intok"].sum() / 1e6 * 0.042   # ~$0.042 / M input tokens, output free
for name in ARMS:
    if name.startswith("Laya"):
        COST[name] = 0.0
cost_s = pd.Series(COST)
print(f"Total cost for {len(df):,} messages (USD):"); print(cost_s.round(4))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
cost_s.plot(kind="bar", ax=axes[0], color="#B279A2")
axes[0].set_title(f"Total cost (USD) for {len(df):,} routes"); axes[0].set_ylabel("USD")
for i, v in enumerate(cost_s):
    axes[0].text(i, v, f"${v:,.2f}", ha="center", va="bottom")
axes[0].set_xticklabels(cost_s.index, rotation=20, ha="right")

lat_rows = [("LLM (Gemini)", df["llm_latency"].median(), df["llm_latency"].quantile(.9))]
for name in ARMS:
    lat_rows.append((name, df[f"{name}__rt"].median(), df[f"{name}__rt"].quantile(.9)))
lat = pd.DataFrame(lat_rows, columns=["arm", "median_s", "p90_s"]).set_index("arm")
print("\nLatency (seconds):"); print(lat.round(3))
lat.plot(kind="bar", ax=axes[1], color=["#72B7B2", "#E45756"])
axes[1].set_title("Latency: median & p90 (s)"); axes[1].set_ylabel("seconds")
axes[1].set_xticklabels(lat.index, rotation=20, ha="right")
plt.tight_layout(); plt.savefig(EVAL / "fig_cost_latency.png"); plt.show()""")

md(r"""## 6. Disagreement analysis
- Where the typed models **agree with each other but differ from the LLM** -> the LLM may be the outlier.
- **High-confidence disagreements** -> the shortlist for V2 human (gold) review.""")
code(r"""if len(ARMS) >= 2:
    first = scored[f"{ARMS[0]}__out_c"]
    same = pd.Series(True, index=scored.index)
    for name in ARMS[1:]:
        same &= (scored[f"{name}__out_c"] == first)
    diff_llm = same & (first != scored["silver"])
    print(f"All typed models agree with each other but DIFFER from LLM: {int(diff_llm.sum())} "
          f"({100*diff_llm.mean():.1f}% of scored) -> candidate LLM mistakes / V2 targets")

for name in ARMS:
    hc = scored[(~scored[f"{name}__agree"]) & (scored[f"{name}__conf"] >= 0.8)]
    print(f"\n{name}: {len(hc)} high-confidence (>=0.80) disagreements. Sample:")
    cols = ["brand", "current_message", "silver", f"{name}__out_c", f"{name}__conf"]
    print(hc[cols].head(5).to_string(index=False, max_colwidth=45))""")

code(r"""# pairwise agreement between all routers (incl. LLM)
labels = ["silver"] + [f"{n}__out_c" for n in ARMS]
names = ["LLM"] + ARMS
M = np.zeros((len(names), len(names)))
for i, a in enumerate(labels):
    for j, b in enumerate(labels):
        M[i, j] = (scored[a] == scored[b]).mean() * 100
mm = pd.DataFrame(M, index=names, columns=names)
plt.figure(figsize=(6, 5))
sns.heatmap(mm, annot=True, fmt=".1f", cmap="Greens", vmin=0, vmax=100)
plt.title("Pairwise agreement (%) between routers"); plt.tight_layout()
plt.savefig(EVAL / "fig_pairwise.png"); plt.show()""")

md(r"""## 7. Summary & decision inputs""")
code(r"""summary = []
for name in ARMS:
    per = scored.groupby("brand")[f"{name}__agree"].mean()
    summary.append({"arm": name,
                    "agreement_%": round(scored[f"{name}__agree"].mean() * 100, 2),
                    "macro_brand_%": round(per.mean() * 100, 2),
                    "cost_full_USD": round(COST.get(name, 0.0), 4),
                    "latency_median_s": round(df[f"{name}__rt"].median(), 3),
                    "latency_p90_s": round(df[f"{name}__rt"].quantile(.9), 3),
                    "mean_conf": round(scored[f"{name}__conf"].mean(), 3)})
summary.append({"arm": "LLM (Gemini)", "agreement_%": 100.0, "macro_brand_%": 100.0,
                "cost_full_USD": round(df["llm_cost"].sum(), 2),
                "latency_median_s": round(df["llm_latency"].median(), 3),
                "latency_p90_s": round(df["llm_latency"].quantile(.9), 3), "mean_conf": np.nan})
summ = pd.DataFrame(summary).set_index("arm")
print("=== V1 SUMMARY ==="); print(summ)
summ.to_csv(EVAL / "comparison_summary.csv")

keep = ["id", "sender", "brand", "turn_index", "current_message", "previous_agent",
        "silver_raw", "silver", "silver_malformed", "silver_routable", "scored"]
for name in ARMS:
    keep += [f"{name}__out", f"{name}__out_c", f"{name}__conf", f"{name}__rt", f"{name}__agree"]
df[keep].to_csv(EVAL / "comparison_rows.csv", index=False)
print("\nsaved -> eval-data/: comparison_summary.csv, comparison_rows.csv, fig_*.png")""")

nb["cells"] = cells
from pathlib import Path
out = Path(__file__).resolve().parents[1] / "notebooks" / "comparison.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(cells), "cells")

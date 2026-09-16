"""Generate authoritative v4 scores + all figure source_data CSVs for CardioRAG writer.

Reads the v4 generation artifacts (dev 384 + ext 96 = 480 rows) and their
fast_judge scores, then emits:

  outputs/scores/llm_judge_full.csv           (480 rows, dataset-tagged)
  outputs/scores/summary_full.csv             (config x metric, correct 480-row summary)
  outputs/scores/ablation_table_v4.csv        (4 config ablation, N=120 each)
  outputs/scores/wilcoxon_posthoc_v4.csv      (paired Wilcoxon, all 120 scenarios)
  outputs/figures/source_data/figure1_kg_stats.csv
  outputs/figures/source_data/figure3_metrics_heatmap.csv
  outputs/figures/source_data/figure4_ablation.csv
  outputs/figures/source_data/figure5_safety.csv

Temporary helper script — deleted after use.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARCHIVE_ROOT = PROJECT_ROOT / "最终数据"
ROOT = PROJECT_ROOT
SCORES = ARCHIVE_ROOT / "01_primary_v4" / "scores"
SD = ARCHIVE_ROOT / "05_figures_and_tables" / "source_data"

CONFIGS = ["vanilla", "naive_rag", "lightrag_generic", "cardiorag_full"]
CONFIG_LABELS = {
    "vanilla": "C1 Vanilla (no RAG)",
    "naive_rag": "C2 Naive RAG",
    "lightrag_generic": "C3 LightRAG-Generic",
    "cardiorag_full": "C4 CardioRAG Full",
}

DATASET_MAP = {
    "dev": ("dev-guideline-internal", "development"),
    "ext": ("heldout-guideline-synthetic", "held_out_synthetic"),
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    dev_gen = load_jsonl(ROOT / "outputs/raw_generations/deepseek_full_zh.jsonl")
    ext_gen = load_jsonl(ROOT / "outputs/raw_generations/deepseek_external_full.jsonl")
    dev_j = load_jsonl(SCORES / "llm_judge.jsonl")
    ext_j = load_jsonl(SCORES / "llm_judge_external.jsonl")

    assert len(dev_gen) == 384 and len(ext_gen) == 96
    assert len(dev_j) == 384 and len(ext_j) == 96

    # ── 1. llm_judge_full.csv (480 rows, dataset-tagged) ─────────────────
    rows = []
    for j, ds_id, ds_role in [(x, *DATASET_MAP["dev"]) for x in dev_j] + \
                             [(x, *DATASET_MAP["ext"]) for x in ext_j]:
        rows.append({
            "scenario_id": j["scenario_id"], "config": j["config"],
            "dataset_id": ds_id, "dataset_role": ds_role,
            "clinical_accuracy": j["clinical_accuracy"],
            "patient_centeredness": j["patient_centeredness"],
            "safety": j["safety"], "citation_correctness": j["citation_correctness"],
            "hallucination": j["hallucination"], "clinical_harm": j["clinical_harm"],
            "rationale": j.get("rationale", ""),
        })
    with open(SCORES / "llm_judge_full.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] llm_judge_full.csv: {len(rows)} rows")

    # ── 2. summary_full.csv (config x metric, correct) ───────────────────
    by_cfg: dict[str, list[dict]] = {c: [r for r in rows if r["config"] == c] for c in CONFIGS}
    summary_lines = ["config,metric,dataset_role,n,mean,std"]
    metrics = ["clinical_accuracy", "patient_centeredness", "safety", "citation_correctness"]
    for cfg in CONFIGS:
        sub = by_cfg[cfg]
        for m in metrics:
            vals = [r[m] for r in sub]
            mean = sum(vals) / len(vals)
            sd = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            summary_lines.append(f"{cfg},{m},all,{len(vals)},{mean:.4f},{sd:.4f}")
    with open(SCORES / "summary_full.csv", "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"[ok] summary_full.csv (config x metric x dataset)")

    # ── 3. ablation_table_v4.csv ─────────────────────────────────────────
    abl = []
    for cfg in CONFIGS:
        sub = by_cfg[cfg]
        acc = [r["clinical_accuracy"] for r in sub]
        pt = [r["patient_centeredness"] for r in sub]
        sa = [r["safety"] for r in sub]
        ci = [r["citation_correctness"] for r in sub]
        abl.append({
            "Config": CONFIG_LABELS[cfg], "N": len(sub),
            "Accuracy (mean)": f"{sum(acc)/len(acc):.2f}",
            "Accuracy (SD)": f"{_sd(acc):.2f}",
            "Patient-Centered (mean)": f"{sum(pt)/len(pt):.2f}",
            "Patient-Centered (SD)": f"{_sd(pt):.2f}",
            "Safety (mean)": f"{sum(sa)/len(sa):.2f}",
            "Safety (SD)": f"{_sd(sa):.2f}",
            "Citation (mean)": f"{sum(ci)/len(ci):.2f}",
            "Citation (SD)": f"{_sd(ci):.2f}",
            "Hallucination Rate": f"{sum(1 for r in sub if r['hallucination'])/len(sub)*100:.1f}%",
            "Clinical Harm Rate": f"{sum(1 for r in sub if r['clinical_harm'])/len(sub)*100:.1f}%",
        })
    with open(SCORES / "ablation_table_v4.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(abl[0].keys()))
        w.writeheader()
        w.writerows(abl)
    print(f"[ok] ablation_table_v4.csv")

    # ── 4. wilcoxon_posthoc_v4.csv (paired over all 120 scenarios) ───────
    try:
        from scipy import stats
        acc_by_scen: dict[tuple[str, str], int] = {(r["scenario_id"], r["config"]): r["clinical_accuracy"] for r in rows}
        scen_ids = sorted({r["scenario_id"] for r in rows})
        posthoc = []
        for i, c1 in enumerate(CONFIGS):
            for c2 in CONFIGS[i + 1:]:
                v1 = [acc_by_scen[(s, c1)] for s in scen_ids]
                v2 = [acc_by_scen[(s, c2)] for s in scen_ids]
                stat, p = stats.wilcoxon(v1, v2)
                p_corr = min(p * 6, 1.0)
                posthoc.append({
                    "Comparison": f"{c1} vs {c2}", "Statistic": f"{stat:.1f}",
                    "p_raw": f"{p:.4f}", "p_corrected": f"{p_corr:.4f}",
                    "Significant": "YES" if p_corr < 0.05 else "no",
                })
        with open(SCORES / "wilcoxon_posthoc_v4.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(posthoc[0].keys()))
            w.writeheader()
            w.writerows(posthoc)
        print(f"[ok] wilcoxon_posthoc_v4.csv ({len(posthoc)} comparisons, N={len(scen_ids)} paired)")
    except ImportError:
        print("[skip] scipy unavailable — wilcoxon_posthoc_v4.csv not written")

    # ── Safety event tallies from generation artifacts ────────────────────
    dev_by_cfg = {c: [g for g in dev_gen if g["config"] == c] for c in CONFIGS}
    ext_by_cfg = {c: [g for g in ext_gen if g["config"] == c] for c in CONFIGS}
    blocked_ids = {g["scenario_id"] for g in dev_gen if g.get("blocked")} | \
                  {g["scenario_id"] for g in ext_gen if g.get("blocked")}
    dev_blocked_ids = {g["scenario_id"] for g in dev_gen if g.get("blocked")}
    heldout_blocked_ids = {g["scenario_id"] for g in ext_gen if g.get("blocked")}
    total_response_n = len(rows)
    total_oob_n = sum(1 for g in dev_gen + ext_gen if g.get("oob_flagged"))
    total_harm_n = sum(1 for r in rows if r["clinical_harm"])
    print(f"  [info] pre-blocked scenarios: {len(blocked_ids)} (shared across configs)")

    # ── 5. figure5_safety.csv ────────────────────────────────────────────
    f5 = []
    for cfg in CONFIGS:
        devs, exts = dev_by_cfg[cfg], ext_by_cfg[cfg]
        jdev = [r for r in rows if r["config"] == cfg and r["dataset_role"] == "development"]
        jext = [r for r in rows if r["config"] == cfg and r["dataset_role"] == "held_out_synthetic"]
        dev_harm_rate = sum(1 for r in jdev if r["clinical_harm"]) / len(jdev)
        heldout_harm_rate = sum(1 for r in jext if r["clinical_harm"]) / len(jext)
        f5.append({
            "config": cfg,
            "blocked_n": sum(1 for g in devs if g.get("blocked")) + sum(1 for g in exts if g.get("blocked")),
            "oob_flagged_n": sum(1 for g in devs if g.get("oob_flagged")) + sum(1 for g in exts if g.get("oob_flagged")),
            "hallucination_n": sum(1 for r in jdev if r["hallucination"]) + sum(1 for r in jext if r["hallucination"]),
            "clinical_harm_n": sum(1 for r in jdev if r["clinical_harm"]) + sum(1 for r in jext if r["clinical_harm"]),
            "dev_harm_n": sum(1 for r in jdev if r["clinical_harm"]),
            "dev_harm_den": len(jdev),
            "heldout_harm_n": sum(1 for r in jext if r["clinical_harm"]),
            "heldout_harm_den": len(jext),
            "dev_harm_rate": round(dev_harm_rate, 4),
            "heldout_harm_rate": round(heldout_harm_rate, 4),
            "input_screened_dev_n": len(dev_blocked_ids),
            "input_emergency_dev_n": 12,
            "input_screened_heldout_n": len(heldout_blocked_ids),
            "input_emergency_heldout_n": 4,
            "input_screened_overall_n": len(blocked_ids),
            "input_emergency_overall_n": 16,
            "post_generation_oob_n": sum(1 for g in devs + exts if g.get("oob_flagged")),
            "response_total_n": len(jdev) + len(jext),
            "text_harm_total_n": sum(1 for r in jdev + jext if r["clinical_harm"]),
        })
    with open(SD / "figure5_safety.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(f5[0].keys()))
        w.writeheader()
        w.writerows(f5)
    print(f"[ok] figure5_safety.csv")

    # ── 6. figure3_metrics_heatmap.csv (config x metric x dataset) ───────
    f3 = []
    for cfg in CONFIGS:
        for ds_role in ("development", "held_out_synthetic"):
            sub = [r for r in rows if r["config"] == cfg and r["dataset_role"] == ds_role]
            for m in metrics:
                vals = [r[m] for r in sub]
                f3.append({
                    "row_label": f"{CONFIG_LABELS[cfg]} · {ds_role.replace('development', 'dev').replace('held_out_synthetic', 'held-out')}",
                    "config": cfg, "dataset_role": ds_role, "metric": m,
                    "value": round(sum(vals) / len(vals), 3),
                })
    with open(SD / "figure3_metrics_heatmap.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["row_label", "config", "dataset_role", "metric", "value"])
        w.writeheader()
        w.writerows(f3)
    print(f"[ok] figure3_metrics_heatmap.csv ({len(f3)} rows)")

    # ── 7. figure4_ablation.csv (wide: method + metrics for forest/bars) ─
    f4 = []
    for cfg in CONFIGS:
        sub = by_cfg[cfg]
        acc = [r["clinical_accuracy"] for r in sub]
        acc_mean = sum(acc) / len(acc)
        acc_sd = _sd(acc)
        pt = [r["patient_centeredness"] for r in sub]
        sa = [r["safety"] for r in sub]
        ci = [r["citation_correctness"] for r in sub]
        f4.append({
            "method": CONFIG_LABELS[cfg],
            "estimate": round(acc_mean, 3),
            "sd_low": round(max(1.0, acc_mean - acc_sd), 3),
            "sd_high": round(min(5.0, acc_mean + acc_sd), 3),
            "sd": round(acc_sd, 3),
            "patient_centeredness": round(sum(pt) / len(pt), 3),
            "safety": round(sum(sa) / len(sa), 3),
            "citation_correctness": round(sum(ci) / len(ci), 3),
            "hallucination_rate": round(sum(1 for r in sub if r["hallucination"]) / len(sub), 4),
        })
    with open(SD / "figure4_ablation.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(f4[0].keys()))
        w.writeheader()
        w.writerows(f4)
    print(f"[ok] figure4_ablation.csv")

    # ── 7b. figure2 x_label column (keep dataset_id/dataset_role for F5) ──
    f2_path = SD / "figure2_performance_by_dataset.csv"
    if f2_path.exists():
        with open(f2_path, encoding="utf-8", newline="") as f:
            f2_rows = list(csv.DictReader(f))
        role_abbr = {"development": "dev", "held_out_synthetic": "held-out"}
        for r in f2_rows:
            r["x_label"] = f"{role_abbr.get(r['dataset_role'], r['dataset_role'])}·{r['method']}"
        with open(f2_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(f2_rows[0].keys()))
            w.writeheader()
            w.writerows(f2_rows)
        print(f"[ok] figure2_performance_by_dataset.csv (+x_label col, {len(f2_rows)} rows)")

    # ── 8. figure1_kg_stats.csv (CardioKG entity/relation distribution) ─
    graphml = ROOT / "lightrag_index" / "graph_chunk_entity_relation.graphml"
    if graphml.exists():
        G = nx.read_graphml(graphml)
        ent_types = Counter()
        for _, d in G.nodes(data=True):
            et = d.get("entity_type") or "Unknown"
            ent_types[str(et)] += 1
        rel_types = Counter()
        for _, _, d in G.edges(data=True):
            rt = d.get("relation_type") or d.get("relation") or "Unknown"
            rel_types[str(rt)] += 1
        rows1 = [{"entity_type": k, "node_count": v} for k, v in
                 sorted(ent_types.items(), key=lambda kv: -kv[1])]
        with open(SD / "figure1_kg_stats.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["entity_type", "node_count"])
            w.writeheader()
            w.writerows(rows1)
        print(f"[ok] figure1_kg_stats.csv: {G.number_of_nodes()} nodes / {G.number_of_edges()} edges")
        print(f"     entity types: {dict(ent_types)}")
        print(f"     top relations: {rel_types.most_common(8)}")
    else:
        print(f"[skip] graphml not found: {graphml}")

    print("\nALL SOURCE DATA READY.")


def _sd(vals: list) -> float:
    mean = sum(vals) / len(vals)
    return (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5


if __name__ == "__main__":
    sys.exit(main())

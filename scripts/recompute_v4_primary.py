"""Recompute frozen v4 summaries without model/API calls.

Inputs are copied v4 raw generations and heuristic scores. Outputs are written
only to the supplied output directory; source artifacts are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

CONFIGS = ["vanilla", "naive_rag", "lightrag_generic", "cardiorag_full"]
METRICS = ["clinical_accuracy", "patient_centeredness", "safety", "citation_correctness"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root = args.root
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    raw = read_jsonl(root / "01_primary_v4/raw_generations/deepseek_full_zh.jsonl")
    raw += read_jsonl(root / "01_primary_v4/raw_generations/deepseek_external_full.jsonl")
    score_rows = list(csv.DictReader((root / "01_primary_v4/scores/llm_judge_full.csv").open(encoding="utf-8-sig")))
    raw_key = {(r["scenario_id"], r["config"]) for r in raw}
    score_key = {(r["scenario_id"], r["config"]) for r in score_rows}
    if len(raw) != 480 or len(score_rows) != 480 or raw_key != score_key:
        raise SystemExit(f"v4 closure failed: raw={len(raw)} scores={len(score_rows)} keys={len(raw_key)}/{len(score_key)}")
    if len(raw_key) != 480:
        raise SystemExit("duplicate raw scenario_id/config keys")

    df = pd.DataFrame(score_rows)
    for m in METRICS:
        df[m] = pd.to_numeric(df[m], errors="raise")
    df["hallucination"] = df["hallucination"].astype(str).str.lower().eq("true")
    df["clinical_harm"] = df["clinical_harm"].astype(str).str.lower().eq("true")

    records = []
    for cfg in CONFIGS:
        sub = df[df.config == cfg].sort_values("scenario_id")
        row = {"Config": cfg, "N": len(sub)}
        for m in METRICS:
            vals = sub[m].to_numpy(dtype=float)
            row[f"{m}_mean"] = round(float(vals.mean()), 6)
            row[f"{m}_sd_population"] = round(float(vals.std(ddof=0)), 6)
        row["hallucination_n"] = int(sub.hallucination.sum())
        row["hallucination_rate"] = round(float(sub.hallucination.mean()), 6)
        row["clinical_harm_n"] = int(sub.clinical_harm.sum())
        row["clinical_harm_rate"] = round(float(sub.clinical_harm.mean()), 6)
        records.append(row)
    pd.DataFrame(records).to_csv(out / "v4_primary_summary_recomputed.csv", index=False)

    wide = df.pivot(index="scenario_id", columns="config", values="clinical_accuracy").sort_index()
    if list(wide.columns) != CONFIGS:
        wide = wide[CONFIGS]
    fried = friedmanchisquare(*(wide[c].to_numpy() for c in CONFIGS))
    pairwise = []
    for i, left in enumerate(CONFIGS):
        for right in CONFIGS[i + 1 :]:
            res = wilcoxon(wide[left], wide[right], alternative="two-sided", zero_method="wilcox")
            pairwise.append({
                "comparison": f"{left} vs {right}",
                "n_paired": len(wide),
                "statistic": float(res.statistic),
                "p_raw": float(res.pvalue),
                "bonferroni_m": 6,
                "p_corrected": min(float(res.pvalue) * 6, 1.0),
                "alpha_adjusted": 0.05 / 6,
                # Compare the uncorrected p against the Bonferroni-adjusted alpha.
                # Comparing p_corrected with alpha_adjusted would apply the factor twice.
                "significant_at_adjusted_alpha": bool(float(res.pvalue) < 0.05 / 6),
            })
    (out / "v4_primary_friedman.json").write_text(json.dumps({
        "n_scenarios": len(wide), "configs": CONFIGS,
        "statistic": float(fried.statistic), "p_value": float(fried.pvalue),
        "pairing": "explicit scenario_id alignment", "sd_definition": "population SD (ddof=0)",
    }, indent=2), encoding="utf-8")
    pd.DataFrame(pairwise).to_csv(out / "v4_primary_wilcoxon_recomputed.csv", index=False)

    audit = {
        "raw_rows": len(raw), "score_rows": len(score_rows), "unique_pairs": len(raw_key),
        "rows_per_config": {cfg: int((df.config == cfg).sum()) for cfg in CONFIGS},
        "api_error_rows": sum(str(r.get("answer_zh", "")).startswith("[API_ERR") for r in raw),
        "empty_answer_rows": sum(not str(r.get("answer_zh", "")).strip() for r in raw),
        "raw_oob_flagged_rows": sum(bool(r.get("oob_flagged", False)) for r in raw),
        "blocked_rows": sum(bool(r.get("blocked", False)) for r in raw),
        "friedman": {"statistic": float(fried.statistic), "p_value": float(fried.pvalue)},
        "significant_pair_count_at_alpha_0.008333": sum(x["significant_at_adjusted_alpha"] for x in pairwise),
        "notes": [
            "This is the frozen v4 heuristic primary analysis; no model/API calls were made.",
            "Wilcoxon pairing is explicit by scenario_id.",
            "The existing source score file is preserved; outputs are derived copies.",
            "The complete LLM-judge file is sensitivity-only and is not included in these primary summaries.",
        ],
    }
    (out / "v4_primary_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

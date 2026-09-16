"""Build frozen-v4 source data for disease-stratified means, paired differences, and missed-emergency table."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = PROJECT_ROOT / "最终数据"
SD = ARCHIVE / "05_figures_and_tables" / "source_data"
TABLES = ARCHIVE / "05_figures_and_tables" / "tables"

CONFIGS = ["vanilla", "naive_rag", "lightrag_generic", "cardiorag_full"]
CONFIG_LABELS = {
    "vanilla": "C1 Vanilla",
    "naive_rag": "C2 Naive RAG",
    "lightrag_generic": "C3 LightRAG-Generic",
    "cardiorag_full": "C4 CardioRAG Full",
}
DISEASE_LABELS = {"CAD": "Coronary artery disease", "HF": "Heart failure", "AF": "Atrial fibrillation"}
MISSED = [
    ("AF-031", "Hypotension with chest tightness during an atrial-fibrillation episode (BP 80/50)"),
    ("AF-039", "Brief loss of consciousness and recently frequent atrial fibrillation"),
    ("CAD-040", "Recurrent chest tightness for 3 days; pain waking the patient that morning"),
    ("HF-031", "Sudden severe dyspnoea, inability to lie flat, and diaphoresis"),
    ("HF-032", "Overnight 2 kg weight gain with worse oedema and breathlessness"),
    ("HF-039", "Acute confusion and incoherent speech in a patient with heart failure"),
]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    scores = list(csv.DictReader((ARCHIVE / "01_primary_v4/scores/llm_judge_full.csv").open(encoding="utf-8-sig")))
    scen = {}
    internal_ids = set()
    for split, fn in [("development", "internal.jsonl"), ("held_out_synthetic", "external.jsonl")]:
        for row in load_jsonl(ARCHIVE / "03_datasets_and_guidelines/scenarios" / fn):
            scen[row["scenario_id"]] = {**row, "split": split}
            if split == "development":
                internal_ids.add(row["scenario_id"])

    for r in scores:
        r["clinical_accuracy"] = float(r["clinical_accuracy"])
        r["citation_correctness"] = float(r["citation_correctness"])
        r["disease"] = scen[r["scenario_id"]]["disease"]

    # Figure 6: disease x config means
    f6 = []
    for disease in ["CAD", "HF", "AF"]:
        for cfg in CONFIGS:
            vals_acc = [r["clinical_accuracy"] for r in scores if r["config"] == cfg and r["disease"] == disease]
            vals_cite = [r["citation_correctness"] for r in scores if r["config"] == cfg and r["disease"] == disease]
            n = len(vals_acc)
            acc_mean = sum(vals_acc) / n
            cite_mean = sum(vals_cite) / n
            acc_sd = (sum((v - acc_mean) ** 2 for v in vals_acc) / n) ** 0.5
            cite_sd = (sum((v - cite_mean) ** 2 for v in vals_cite) / n) ** 0.5
            f6.append({
                "disease": disease,
                "disease_label": DISEASE_LABELS[disease],
                "config": cfg,
                "config_label": CONFIG_LABELS[cfg],
                "n_scenarios": n,
                "guideline_concordance_mean": round(acc_mean, 4),
                "guideline_concordance_sd": round(acc_sd, 4),
                "citation_correctness_mean": round(cite_mean, 4),
                "citation_correctness_sd": round(cite_sd, 4),
            })
    SD.mkdir(parents=True, exist_ok=True)
    with (SD / "figure6_disease_stratified.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(f6[0].keys()))
        w.writeheader()
        w.writerows(f6)

    # Figure 7: paired differences
    by = defaultdict(dict)
    for r in scores:
        by[r["scenario_id"]][r["config"]] = r
    f7 = []
    for sid, d in sorted(by.items()):
        disease = d["vanilla"]["disease"]
        for metric, key in [("guideline_concordance", "clinical_accuracy"), ("citation_correctness", "citation_correctness")]:
            for contrast, left, right in [
                ("C4 minus C1", "cardiorag_full", "vanilla"),
                ("C4 minus C3", "cardiorag_full", "lightrag_generic"),
            ]:
                f7.append({
                    "scenario_id": sid,
                    "disease": disease,
                    "disease_label": DISEASE_LABELS[disease],
                    "split": "development" if sid in internal_ids else "held_out_synthetic",
                    "metric": metric,
                    "contrast": contrast,
                    "difference": round(d[left][key] - d[right][key], 4),
                })
    with (SD / "figure7_paired_differences.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(f7[0].keys()))
        w.writeheader()
        w.writerows(f7)

    # Table 5
    lines = [
        "# Table 5. Development emergency-routing scenarios not captured by the input screener",
        "",
        "| Scenario ID | Disease | Split | Presenting features |",
        "|---|---|---|---|",
    ]
    lines_zh = [
        "# 表 5. 开发集中输入筛查未捕获的急诊路由场景",
        "",
        "| 场景编号 | 病种 | 分层 | 主要表现 |",
        "|---|---|---|---|",
    ]
    zh_feat = {
        "AF-031": "房颤发作时低血压伴胸闷（血压 80/50）",
        "AF-039": "短暂意识丧失，近期房颤频繁",
        "CAD-040": "反复胸闷 3 天，今早痛醒",
        "HF-031": "突发严重呼吸困难、不能平卧、大汗",
        "HF-032": "一夜体重增加 2 kg，水肿和气促加重",
        "HF-039": "心衰患者急性意识模糊、言语紊乱",
    }
    zh_dis = {"CAD": "冠心病", "HF": "心力衰竭", "AF": "心房颤动"}
    for sid, feat in MISSED:
        s = scen[sid]
        dis = s["disease"]
        split = "development"
        lines.append(f"| {sid} | {DISEASE_LABELS[dis]} | {split} | {feat} |")
        lines_zh.append(f"| {sid} | {zh_dis[dis]} | 开发集 | {zh_feat[sid]} |")
    note = (
        "\nThese 6 of 12 development emergency-routing scenarios did not match the shared input-side keyword list. "
        "The held-out synthetic split recalled 4 of 4. Identifiers and features are reported from the frozen scenario file; no model was rerun.\n"
    )
    note_zh = (
        "\n开发集 12 个急诊路由场景中有 6 个未匹配共用输入侧关键词表。留出合成集召回 4/4。"
        "编号和表现来自冻结场景文件，未重跑模型。\n"
    )
    (TABLES / "table5_missed_emergency.md").write_text("\n".join(lines) + note, encoding="utf-8")
    (TABLES / "table5_missed_emergency.zh.md").write_text("\n".join(lines_zh) + note_zh, encoding="utf-8")
    print("wrote figure6_disease_stratified.csv", len(f6))
    print("wrote figure7_paired_differences.csv", len(f7))
    print("wrote table5_missed_emergency.md")


if __name__ == "__main__":
    main()

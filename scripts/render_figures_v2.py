"""CardioRAG figure re-rendering v2 — fixes all QC issues.

Usage:  .venv\\Scripts\\python.exe scripts/render_figures_v2.py

Fixes over original render:
  F1a: system architecture flowchart drawn with matplotlib patches
  F1b: correct entity-type names from source CSV
  F2:  shortened x-axis labels + 35° rotation
  F3:  single heatmap, no (a)/(b) sub-labels
  F4:  wider canvas, clean layout
    F5:  split-level input-screening and response-level text audit

"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
SD = ROOT / "data" / "figure_source"
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Segoe UI", "Arial", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def _read_csv(name: str) -> list[dict]:
    with open(SD / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf", "svg", "tiff"):
        fig.savefig(OUT / f"{stem}.{ext}", facecolor="white", dpi=600)
    print(f"  [ok] {stem}.*  ({4} formats)")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — System architecture + CardioKG entity distribution
# ═══════════════════════════════════════════════════════════════════════════
def render_figure1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.09, 4.33),
                                    gridspec_kw={"width_ratios": [1.3, 1]})

    # ---- (a) System architecture flowchart ----
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.axis("off")
    ax1.set_title("(a) System architecture", fontsize=13, fontweight="bold", loc="left", pad=10)

    def box(x, y, w, h, text, fc="#E8EEF6", ec="#4C72B0", fontsize=9, bold=False):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                               facecolor=fc, edgecolor=ec, linewidth=1.5)
        ax1.add_patch(rect)
        weight = "bold" if bold else "normal"
        ax1.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                 fontsize=fontsize, fontweight=weight, wrap=True,
                 multialignment="center")

    def arrow(x1, y1, x2, y2, label="", color="#333"):
        ax1.annotate("", xy=(x2, y2), xytext=(x1, y1),
                      arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax1.text(mx + 0.15, my, label, fontsize=7.5, color=color, va="center")

    # Boxes (top to bottom)
    box(3.5, 10.5, 3, 0.9, "Patient question\n(Chinese / English)", fc="#FFF3CD", fontsize=9, bold=True)

    box(3.5, 8.8, 3, 0.9, "CV-Guardrail\n(pre-generation check)", fc="#F8D7DA", ec="#C44E52", fontsize=9, bold=True)

    # Emergency branch
    box(0.3, 7.0, 2.6, 0.8, "Emergency route\n→ Local services", fc="#F5C6CB", ec="#C44E52", fontsize=8)
    arrow(4.2, 8.8, 2.2, 7.8, "blocked", color="#C44E52")

    # Pass branch
    box(3.5, 7.0, 3, 0.8, "LightRAG hybrid retrieval\n(CardioKG graph + vector)", fc="#D4EDDA", ec="#55A868", fontsize=9)
    arrow(5.0, 8.8, 5.0, 7.8, "pass", color="#55A868")

    box(3.5, 5.2, 3, 0.8, "DeepSeek-chat\ngeneration (t = 0.3)", fc="#CCE5FF", ec="#4C72B0", fontsize=9)
    arrow(5.0, 7.0, 5.0, 6.0, "")

    box(3.5, 3.4, 3, 0.9, "CV-Guardrail\n(post-generation OOB check)", fc="#F8D7DA", ec="#C44E52", fontsize=9, bold=True)
    arrow(5.0, 5.2, 5.0, 4.3, "", color="#333")

    # OOB branch
    box(7.0, 3.4, 2.6, 0.8, "OOB refusal\n→ Consult clinician", fc="#F5C6CB", ec="#C44E52", fontsize=8)
    arrow(6.5, 3.85, 7.0, 3.85, "OOB", color="#C44E52")

    # Final answer
    box(3.5, 1.6, 3, 0.9, "Final answer\n+ Disclaimer", fc="#D4EDDA", ec="#55A868", fontsize=9, bold=True)
    arrow(5.0, 3.4, 5.0, 2.5, "pass", color="#55A868")

    # CardioKG side annotation
    # CardioKG counts are reported separately from the architecture flow.
    ax1.text(0.3, 11.6, "CardioKG registry\n(401 entities\n501 relations)\nindex: 369 nodes / 489 edges", fontsize=7.5,
             color="#55A868", fontstyle="italic", va="top")

    # C4 annotation
    ax1.text(9.8, 0.3, "C4 CardioRAG Full\n(all layers active)",
             fontsize=8, color="#888", ha="right", va="bottom", fontstyle="italic")

    # ---- (b) CardioKG entity types ----
    rows = _read_csv("figure1_kg_stats.csv")
    names = [r["entity_type"] for r in rows]
    counts = [int(r["node_count"]) for r in rows]

    # Prettify names: CamelCase → spaced
    pretty = {
        "Drug": "Drug", "GuidelineStatement": "Guideline\nstatement",
        "Procedure": "Procedure", "DrugClass": "Drug\nclass",
        "Biomarker": "Biomarker", "PatientFactor": "Patient\nfactor",
        "Contraindication": "Contra-\nindication", "RiskFactor": "Risk\nfactor",
        "Disease": "Disease", "Symptom": "Symptom", "Stratification": "Stratification",
    }
    labels = [pretty.get(n, n) for n in names]

    bars = ax2.barh(range(len(names)), counts, color=PALETTE[0], edgecolor="white", height=0.7)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlabel("Number of entities", fontsize=11)
    ax2.set_title("(b) Persisted CardioKG index entity types  (N = 369)", fontsize=13, fontweight="bold", loc="left", pad=10)
    ax2.invert_yaxis()
    # Add count labels at bar end
    for bar, cnt in zip(bars, counts):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 str(cnt), va="center", fontsize=8.5, color="#333")
    ax2.set_xlim(0, max(counts) * 1.15)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(w_pad=3)
    _save(fig, "figure1_cardiorag_architecture")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — Routing metrics by dataset × method
# ═══════════════════════════════════════════════════════════════════════════
def render_figure2():
    rows = _read_csv("figure2_performance_by_dataset.csv")

    # Short labels: "dev·Vanilla", "ext·Full" etc.
    method_short = {
        "pure_llm": "C1", "naive_rag": "C2",
        "lightrag": "C3", "full_system": "C4",
    }
    role_abbr = {"development": "Dev", "held_out_synthetic": "Held-out"}
    x_labels = [method_short.get(r["method"], r["method"]) for r in rows]
    split_labels = [role_abbr.get(r["dataset_role"], r["dataset_role"][:3]) for r in rows]

    sens = [float(r["sensitivity_high"]) for r in rows]
    spec = [float(r["specificity_high"]) for r in rows]
    f1 = [float(r["f1_high"]) for r in rows]
    binary_acc = [float(r["binary_routing_accuracy"]) for r in rows]
    kappa = [float(r["cohens_kappa"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.09, 4.72))
    x = np.arange(len(x_labels))
    w = 0.18

    # (a) Routing metrics
    ax1.bar(x - 1.5 * w, sens, w, label="Sensitivity", color=PALETTE[0])
    ax1.bar(x - 0.5 * w, spec, w, label="Specificity", color=PALETTE[1])
    ax1.bar(x + 0.5 * w, f1, w, label="F1", color=PALETTE[2])
    ax1.bar(x + 1.5 * w, binary_acc, w, label="Binary routing accuracy", color=PALETTE[3])
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontsize=8.5)
    ax1.set_ylabel("Score (0–1)", fontsize=10)
    ax1.set_ylim(0, 1.15)
    ax1.set_title("(a) High-risk routing metrics", fontsize=11, fontweight="bold", loc="left", pad=8)
    ax1.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False)
    ax1.axhline(1.0, color="#999", linewidth=0.5, linestyle="--")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.text(1.5, 1.09, "Development", ha="center", va="bottom", fontsize=8, color="#555")
    ax1.text(5.5, 1.09, "Held-out synthetic", ha="center", va="bottom", fontsize=8, color="#555")
    ax1.axvline(3.5, color="#ccc", linewidth=1, linestyle="-")

    # (b) Agreement
    ax2.bar(x - 0.2, binary_acc, 0.35, label="Binary routing accuracy", color=PALETTE[3])
    ax2.bar(x + 0.2, kappa, 0.35, label="Cohen's κ", color=PALETTE[4])
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, fontsize=8.5)
    ax2.set_ylabel("Score (0–1)", fontsize=10)
    ax2.set_ylim(0, 1.15)
    ax2.set_title("(b) Agreement and binary routing accuracy", fontsize=11, fontweight="bold", loc="left", pad=8)
    ax2.legend(fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False)
    ax2.axhline(1.0, color="#999", linewidth=0.5, linestyle="--")
    ax2.axvline(3.5, color="#ccc", linewidth=1, linestyle="-")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(w_pad=2.5)
    _save(fig, "figure2_performance_by_dataset")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — Judge score heatmap (single panel, no sub-labels)
# ═══════════════════════════════════════════════════════════════════════════
def render_figure3():
    rows = _read_csv("figure3_metrics_heatmap.csv")

    # Pivot: rows = configuration × split, cols = metric
    metrics = ["clinical_accuracy", "patient_centeredness", "safety", "citation_correctness"]
    row_labels = []
    data = []
    seen = []
    for r in rows:
        rl = r["row_label"]
        if rl not in seen:
            seen.append(rl)
            row_labels.append(rl)
            data.append({})
        data[-1][r["metric"]] = float(r["value"])

    matrix = np.array([[d.get(m, 0) for m in metrics] for d in data])

    fig, ax = plt.subplots(figsize=(7.09, 4.33))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=1.0, vmax=5.0)

    # Pretty column labels
    col_labels = ["Guideline\nconcordance", "Patient\ncenteredness",
                  "Safety-marker\nscore", "Citation\ncorrectness"]
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Annotate each cell
    for i in range(len(row_labels)):
        for j in range(len(metrics)):
            val = matrix[i, j]
            color = "white" if val < 2.8 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Mean score (1–5)", fontsize=10)
    ax.set_title("Deterministic rule-based scores by configuration × split", fontsize=11,
                 fontweight="bold", pad=12)

    fig.tight_layout()
    _save(fig, "figure3_metrics_heatmap")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 — Ablation study (dot plot + secondary bars)
# ═══════════════════════════════════════════════════════════════════════════
def render_figure4():
    rows = _read_csv("figure4_ablation.csv")
    methods_display = ["C1 Vanilla", "C2 Naive", "C3 LightRAG", "C4 Full"]
    estimates = [float(r["estimate"]) for r in rows]
    ci_low = [float(r["sd_low"]) for r in rows]
    ci_high = [float(r["sd_high"]) for r in rows]
    pc = [float(r["patient_centeredness"]) for r in rows]
    sa = [float(r["safety"]) for r in rows]
    ci = [float(r["citation_correctness"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.09, 4.33),
                                    gridspec_kw={"width_ratios": [1, 1.2]})

    y = np.arange(len(methods_display))
    # (a) Clinical accuracy with error bars
    errs_low = [e - l for e, l in zip(estimates, ci_low)]
    errs_high = [h - e for e, h in zip(estimates, ci_high)]
    ax1.errorbar(estimates, y, xerr=[errs_low, errs_high], fmt="o",
                 markersize=10, capsize=5, color=PALETTE[0], ecolor="#888",
                 markerfacecolor=PALETTE[0], markeredgecolor="white", markeredgewidth=1.5)
    for i, (est, lo, hi) in enumerate(zip(estimates, ci_low, ci_high)):
        ax1.text(hi + 0.05, i, f"{est:.2f}", va="center", fontsize=9, fontweight="bold")
    ax1.set_yticks(y)
    ax1.set_yticklabels(methods_display, fontsize=9)
    ax1.set_xlabel("Guideline-concordance score (1–5)", fontsize=10)
    ax1.set_xlim(1.0, 4.5)
    ax1.set_title("(a) Guideline-concordance score\n(mean ± population SD)", fontsize=11, fontweight="bold", loc="left", pad=8)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.axvline(2.55, color="#ccc", linewidth=0.8, linestyle="--", label="C1 baseline")
    ax1.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=False)

    # (b) Secondary metrics grouped bars
    x = np.arange(len(methods_display))
    w = 0.25
    ax2.bar(x - w, pc, w, label="Patient-centered", color=PALETTE[4])
    ax2.bar(x, sa, w, label="Safety", color=PALETTE[2])
    ax2.bar(x + w, ci, w, label="Citation correctness", color=PALETTE[1])
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods_display, fontsize=8)
    ax2.set_ylabel("Mean score (1–5)", fontsize=11)
    ax2.set_ylim(0, 5.5)
    ax2.set_title("(b) Secondary quality metrics", fontsize=13, fontweight="bold", loc="left", pad=8)
    ax2.legend(fontsize=8, ncol=3, loc="upper left")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(w_pad=3)
    _save(fig, "figure4_ablation")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 — Split-level safety-process audit
# ═══════════════════════════════════════════════════════════════════════════
def render_figure5():
    rows = _read_csv("figure5_safety.csv")
    configs = [r["config"] for r in rows]
    config_labels = {
        "vanilla": "C1", "naive_rag": "C2",
        "lightrag_generic": "C3", "cardiorag_full": "C4",
    }
    labels = [config_labels.get(c, c) for c in configs]
    blocked = [int(r["blocked_n"]) for r in rows]
    oob = [int(r["oob_flagged_n"]) for r in rows]
    hall = [int(r["hallucination_n"]) for r in rows]
    harm = [int(r["clinical_harm_n"]) for r in rows]
    input_screened_dev_n = int(rows[0]["input_screened_dev_n"])
    input_emergency_dev_n = int(rows[0]["input_emergency_dev_n"])
    input_screened_heldout_n = int(rows[0]["input_screened_heldout_n"])
    input_emergency_heldout_n = int(rows[0]["input_emergency_heldout_n"])
    input_screened_overall_n = int(rows[0]["input_screened_overall_n"])
    input_emergency_overall_n = int(rows[0]["input_emergency_overall_n"])
    post_generation_oob_n = int(rows[0]["post_generation_oob_n"])
    response_total_n = int(rows[0]["response_total_n"])
    text_harm_total_n = int(rows[0]["text_harm_total_n"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.09, 4.33))

    x = np.arange(len(configs))
    w = 0.2

    # (a) Response-level safety audit; pre-blocks are shared across configurations.
    ax1.bar(x - 1.5 * w, blocked, w, label="Input-screened rows\n(shared pre-check)", color=PALETTE[0])
    ax1.bar(x - 0.5 * w, oob, w, label="Post-generation OOB flag", color=PALETTE[1])
    ax1.bar(x + 0.5 * w, hall, w, label="Text-level hallucination flag", color=PALETTE[2])
    ax1.bar(x + 1.5 * w, harm, w, label="Text-level rule-based harm flag", color=PALETTE[3],
            edgecolor="#333", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Flagged response rows (N)", fontsize=11)
    ax1.set_ylim(0, 12)
    ax1.set_title("(a) Response-level safety audit", fontsize=13, fontweight="bold", loc="left", pad=8)
    ax1.legend(fontsize=7.2, ncol=2, loc="upper left", frameon=False)
    ax1.spines[["top", "right"]].set_visible(False)
    for offset, series in zip((-1.5, -0.5, 0.5, 1.5), (blocked, oob, hall, harm)):
        for i, value in enumerate(series):
            ax1.text(x[i] + offset * w, value + 0.25, str(value),
                     ha="center", va="bottom", fontsize=7)
    ax1.text(0.02, 0.62,
             f"{input_emergency_overall_n} emergency-routing + 14 OOB scenarios;\n"
             f"{input_screened_overall_n} unique scenarios screened before generation → "
             f"{input_screened_overall_n * len(configs)} response rows",
             transform=ax1.transAxes, fontsize=7.2, va="top", color="#555", fontstyle="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f9f9f9", edgecolor="#ddd"))

    # (b) Split-level process audit: rates are shown with their exact counts.
    ax2.clear()
    split_labels = ["Development\n(n=96)", "Held-out\n(n=24)", "Overall\n(n=120)"]
    recall_n = [input_screened_dev_n, input_screened_heldout_n, input_screened_overall_n]
    recall_den = [input_emergency_dev_n, input_emergency_heldout_n, input_emergency_overall_n]
    recall = [n / d for n, d in zip(recall_n, recall_den)]
    xpos = np.arange(3)
    bars = ax2.bar(xpos, recall, width=0.52, color=[PALETTE[0], PALETTE[1], PALETTE[4]], edgecolor="white")
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(split_labels, fontsize=8.5)
    ax2.set_ylabel("Input-screening recall", fontsize=11)
    ax2.set_ylim(0, 1.18)
    ax2.set_title("(b) Split-level safety-process audit", fontsize=13, fontweight="bold", loc="left", pad=8)
    ax2.spines[["top", "right"]].set_visible(False)
    for bar, n, den in zip(bars, recall_n, recall_den):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
                 f"{n}/{den}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.text(0.03, 0.98,
             f"Input screen: {input_screened_dev_n}/{input_emergency_dev_n} development, "
             f"{input_screened_heldout_n}/{input_emergency_heldout_n} held-out, "
             f"{input_screened_overall_n}/{input_emergency_overall_n} overall\n"
             f"Post-generation OOB flags: {post_generation_oob_n}/{response_total_n}; "
             f"text-level harm flags: {text_harm_total_n}/{response_total_n}\n"
             "All values are rule-based protocol/text audits, not clinical outcomes.",
             transform=ax2.transAxes, fontsize=7.4, va="top", color="#555", fontstyle="italic",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#f9f9f9", edgecolor="#ddd"))

    fig.tight_layout(w_pad=3)
    _save(fig, "figure5_safety")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("CardioRAG Figure Rendering v2")
    print("=" * 50)
    render_figure1()
    render_figure2()
    render_figure3()
    render_figure4()
    render_figure5()
    print("\nALL FIGURES RENDERED SUCCESSFULLY.")

"""CardioRAG data figures — manuscript Figures 1-3.

Usage:  python scripts/render_figures.py

Reads the CSV tables in `data/figure_source/` and writes PNG, PDF, SVG, and
TIFF renders to `outputs/figures/`.

  Figure 1  figure1_primary_rubric        guideline concordance, secondary rubric, split heatmap
  Figure 2  figure2_disease_robustness    disease-stratified means and paired differences
  Figure 3  figure3_routing_safety        binary routing metrics and safety-process audit

The graphical abstract and the prototype screenshot figure are produced
separately (`scripts/render_graphical_abstract.py` and the Streamlit app).
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def _sig_bracket(ax, x1, x2, y, text, dy=0.07):
    ax.plot([x1, x1, x2, x2], [y - dy, y, y, y - dy], color="#222", lw=0.8, clip_on=False)
    ax.text((x1 + x2) / 2, y + 0.015, text, ha="center", va="bottom", fontsize=8, color="#222")


def _clipped_yerr(means, sds, lo=1.0, hi=5.0):
    low = [min(sd, max(0.0, m - lo)) for m, sd in zip(means, sds)]
    high = [min(sd, max(0.0, hi - m)) for m, sd in zip(means, sds)]
    return [low, high]


def render_figure1_primary_rubric():
    """Figure 1: overall rubric means, secondary scores, and the split heatmap."""
    abl = _read_csv("figure4_ablation.csv")
    heat = _read_csv("figure3_metrics_heatmap.csv")
    methods = ["C1", "C2", "C3", "C4"]
    estimates = [float(r["estimate"]) for r in abl]
    sds = [float(r["sd"]) for r in abl]
    pc = [float(r["patient_centeredness"]) for r in abl]
    sa = [float(r["safety"]) for r in abl]
    cite = [float(r["citation_correctness"]) for r in abl]

    fig = plt.figure(figsize=(7.09, 7.25))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.08, 1.15], hspace=0.46, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(methods))
    ax1.bar(
        x, estimates, width=0.62, color=PALETTE[:4],
        yerr=sds, capsize=3, error_kw={"linewidth": 0.8, "ecolor": "#555"},
        edgecolor="white", linewidth=0.4,
    )
    for i, (est, sd) in enumerate(zip(estimates, sds)):
        ax1.text(i, est + sd + 0.06, f"{est:.2f}", ha="center", va="bottom", fontsize=7)
    _sig_bracket(ax1, 0, 1, 4.58, "*")
    _sig_bracket(ax1, 1, 2, 4.76, "***")
    _sig_bracket(ax1, 2, 3, 4.94, "***")
    _sig_bracket(ax1, 0, 2, 5.12, "***")
    _sig_bracket(ax1, 1, 3, 5.30, "***")
    _sig_bracket(ax1, 0, 3, 5.48, "***")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=9)
    ax1.set_ylabel("Guideline concordance (1–5)", fontsize=9)
    ax1.set_ylim(1.0, 5.9)
    ax1.set_yticks([1, 2, 3, 4, 5])
    ax1.axhline(5.0, color="#aaa", linewidth=0.6, linestyle=":")
    ax1.set_title("(a) Guideline concordance", fontsize=10, fontweight="bold", loc="left", pad=14)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 1])
    y = np.arange(len(methods))
    h = 0.22
    ax2.barh(y - h, pc, h, label="Patient-centeredness", color=PALETTE[4])
    ax2.barh(y, sa, h, label="Safety-marker", color=PALETTE[2])
    ax2.barh(y + h, cite, h, label="Citation correctness", color=PALETTE[1])
    for values, offset in ((pc, -h), (sa, 0), (cite, h)):
        for yi, value in zip(y, values):
            ax2.text(value + 0.04, yi + offset, f"{value:.2f}",
                     va="center", ha="left", fontsize=6.5)
    ax2.set_yticks(y)
    ax2.set_yticklabels(methods, fontsize=9)
    ax2.set_xlabel("Mean score (1–5)", fontsize=9)
    ax2.set_xlim(1.0, 5.15)
    ax2.set_xticks([1, 2, 3, 4, 5])
    ax2.axvline(5.0, color="#aaa", linewidth=0.6, linestyle=":")
    ax2.set_title("(b) Secondary rubric scores", fontsize=10, fontweight="bold", loc="left", pad=6)
    ax2.legend(
        fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.18),
        ncol=3, frameon=False, columnspacing=0.7, handlelength=1.0,
    )
    ax2.spines[["top", "right"]].set_visible(False)

    ax3 = fig.add_subplot(gs[1, :])
    metrics = ["clinical_accuracy", "patient_centeredness", "safety", "citation_correctness"]
    config_short = {
        "vanilla": "C1", "naive_rag": "C2",
        "lightrag_generic": "C3", "cardiorag_full": "C4",
    }
    split_short = {"development": "Development", "held_out_synthetic": "Held-out"}
    row_labels, data, seen = [], [], []
    for r in heat:
        rl = f"{config_short.get(r['config'], r['config'])} {split_short.get(r['dataset_role'], r['dataset_role'])}"
        if rl not in seen:
            seen.append(rl)
            row_labels.append(rl)
            data.append({})
        data[-1][r["metric"]] = float(r["value"])
    matrix = np.array([[d.get(m, 0) for m in metrics] for d in data])
    im = ax3.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=1.0, vmax=5.0)
    ax3.set_xticks(range(4))
    ax3.set_xticklabels(
        ["Guideline\nconcordance", "Patient-\ncenteredness", "Safety-marker\nscore", "Citation\ncorrectness"],
        fontsize=8,
    )
    ax3.set_yticks(range(len(row_labels)))
    ax3.set_yticklabels(row_labels, fontsize=8)
    for i in range(len(row_labels)):
        for j in range(4):
            val = matrix[i, j]
            ax3.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                     fontweight="bold", color="white" if val < 2.8 else "black")
    cbar = fig.colorbar(im, ax=ax3, shrink=0.92, pad=0.02)
    cbar.set_label("Mean score (1–5)", fontsize=8)
    ax3.set_title("(c) Scores by configuration and split", fontsize=10, fontweight="bold", loc="left", pad=6)

    _save(fig, "figure1_primary_rubric")
    plt.close(fig)


def render_figure2_disease_robustness():
    """Figure 2: disease-stratified means and paired differences."""
    rows = _read_csv("figure6_disease_stratified.csv")
    diffs = _read_csv("figure7_paired_differences.csv")
    diseases = ["CAD", "HF", "AF"]
    disease_labels = ["Coronary artery\ndisease", "Heart failure", "Atrial\nfibrillation"]
    configs = ["vanilla", "naive_rag", "lightrag_generic", "cardiorag_full"]
    config_labels = ["C1", "C2", "C3", "C4"]
    lookup = {(r["disease"], r["config"]): r for r in rows}
    ylim = (1.0, 5.2)

    fig, axes = plt.subplots(2, 2, figsize=(7.09, 6.85))
    ax1, ax2, ax3, ax4 = axes.ravel()
    x = np.arange(len(diseases))
    w = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * w

    for ax, mean_key, sd_key, title, clip_err in [
        (ax1, "guideline_concordance_mean", "guideline_concordance_sd", "(a) Guideline concordance", False),
        (ax2, "citation_correctness_mean", "citation_correctness_sd", "(b) Citation correctness", True),
    ]:
        for i, cfg in enumerate(configs):
            means = [float(lookup[(d, cfg)][mean_key]) for d in diseases]
            sds = [float(lookup[(d, cfg)][sd_key]) for d in diseases]
            yerr = _clipped_yerr(means, sds) if clip_err else sds
            ax.bar(
                x + offsets[i], means, w, yerr=yerr, capsize=2,
                label=config_labels[i], color=PALETTE[i],
                error_kw={"linewidth": 0.7, "ecolor": "#555"},
            )
        ax.set_xticks(x)
        ax.set_xticklabels(disease_labels, fontsize=8)
        ax.set_ylabel("Score (1–5)", fontsize=8.5)
        ax.set_ylim(*ylim)
        ax.axhline(5.0, color="#aaa", linewidth=0.6, linestyle=":")
        ax.set_title(title, fontsize=10, fontweight="bold", loc="left", pad=5)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labs = ax1.get_legend_handles_labels()
    ax1.legend(handles, labs, fontsize=7, loc="upper left", frameon=False, ncol=2)

    rng = np.random.default_rng(0)
    for ax, contrast, title in [
        (ax3, "C4 minus C1", "(c) C4 − C1"),
        (ax4, "C4 minus C3", "(d) C4 − C3"),
    ]:
        data = []
        for d in diseases:
            data.append([
                float(r["difference"]) for r in diffs
                if r["metric"] == "guideline_concordance"
                and r["contrast"] == contrast and r["disease"] == d
            ])
        bp = ax.boxplot(
            data, positions=np.arange(1, 4), widths=0.45, patch_artist=True,
            medianprops={"color": "#222", "linewidth": 1.1},
            whiskerprops={"color": "#555"}, capprops={"color": "#555"},
            flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": "#999",
                        "markeredgecolor": "none"},
        )
        for patch, color in zip(bp["boxes"], PALETTE[:3]):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
            patch.set_edgecolor("#333")
        for i, vals in enumerate(data, start=1):
            jitter = rng.normal(0, 0.06, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=8, color="#333",
                       alpha=0.35, zorder=3, linewidths=0)
        ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(disease_labels, fontsize=8)
        ax.set_ylabel("Paired difference", fontsize=8.5)
        ax.set_ylim(-2.2, 3.6)
        ax.set_title(title, fontsize=10, fontweight="bold", loc="left", pad=5)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(h_pad=1.1, w_pad=1.4)
    _save(fig, "figure2_disease_robustness")
    plt.close(fig)


def render_figure3_routing_safety():
    """Figure 3: binary routing metrics and safety-process audit."""
    route = _read_csv("figure2_performance_by_dataset.csv")
    safe = _read_csv("figure5_safety.csv")
    method_short = {
        "pure_llm": "C1", "naive_rag": "C2",
        "lightrag": "C3", "full_system": "C4",
    }
    x_labels = [method_short.get(r["method"], r["method"]) for r in route]
    series = [
        ("Sensitivity", [float(r["sensitivity_high"]) for r in route], PALETTE[0]),
        ("Specificity", [float(r["specificity_high"]) for r in route], PALETTE[1]),
        ("F1", [float(r["f1_high"]) for r in route], PALETTE[2]),
        ("Routing accuracy", [float(r["binary_routing_accuracy"]) for r in route], PALETTE[3]),
        ("Cohen's κ", [float(r["cohens_kappa"]) for r in route], PALETTE[4]),
    ]

    configs = [r["config"] for r in safe]
    config_labels = {
        "vanilla": "C1", "naive_rag": "C2",
        "lightrag_generic": "C3", "cardiorag_full": "C4",
    }
    labels = [config_labels.get(c, c) for c in configs]
    blocked = [int(r["blocked_n"]) for r in safe]
    oob = [int(r["oob_flagged_n"]) for r in safe]
    hall = [int(r["hallucination_n"]) for r in safe]
    harm = [int(r["clinical_harm_n"]) for r in safe]
    r0 = safe[0]

    fig = plt.figure(figsize=(7.09, 7.55))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.08, 1.12], hspace=0.58, wspace=0.38)
    ax_top = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    x = np.arange(len(x_labels))
    n_series = len(series)
    w = 0.15
    offsets = (np.arange(n_series) - (n_series - 1) / 2) * w
    for offset, (lab, vals, color) in zip(offsets, series):
        ax_top.bar(x + offset, vals, w, label=lab, color=color, edgecolor="white", linewidth=0.3)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(x_labels, fontsize=9)
    ax_top.set_ylabel("Score (0–1)", fontsize=9)
    ax_top.set_ylim(0, 1.28)
    ax_top.axhline(1.0, color="#999", linewidth=0.5, linestyle="--")
    ax_top.axvline(3.5, color="#bbb", linewidth=1.0)
    ax_top.text(1.5, 1.18, "Development (n = 96)", ha="center", va="bottom", fontsize=8, color="#444")
    ax_top.text(5.5, 1.18, "Held-out synthetic (n = 24)", ha="center", va="bottom", fontsize=8, color="#444")
    ax_top.legend(fontsize=6.6, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.16),
                  frameon=False, handlelength=1.0, columnspacing=0.8)
    ax_top.set_title("(a) Binary high-risk routing", fontsize=10, fontweight="bold", loc="left", pad=4)
    ax_top.spines[["top", "right"]].set_visible(False)

    xb = np.arange(len(labels))
    wb = 0.2
    ax_b.bar(xb - 1.5 * wb, blocked, wb, label="Input-screened", color=PALETTE[0])
    ax_b.bar(xb - 0.5 * wb, oob, wb, label="Post-gen. OOB", color=PALETTE[1])
    ax_b.bar(xb + 0.5 * wb, hall, wb, label="Hallucination", color=PALETTE[2])
    ax_b.bar(xb + 1.5 * wb, harm, wb, label="Text-level harm", color=PALETTE[3],
             edgecolor="#333", linewidth=0.7)
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(labels, fontsize=9)
    ax_b.set_ylabel("Flagged rows (N)", fontsize=9)
    ax_b.set_ylim(0, 12.8)
    ax_b.set_title("(b) Response-level flags", fontsize=10, fontweight="bold", loc="left", pad=5)
    ax_b.legend(
        fontsize=6.4, loc="upper center", bbox_to_anchor=(0.5, -0.22),
        ncol=2, frameon=False, handlelength=1.0, columnspacing=0.8,
    )
    ax_b.spines[["top", "right"]].set_visible(False)
    for offset, series_vals in zip((-1.5, -0.5, 0.5, 1.5), (blocked, oob, hall, harm)):
        for i, value in enumerate(series_vals):
            ax_b.text(xb[i] + offset * wb, value + 0.2, str(value),
                      ha="center", va="bottom", fontsize=6.5)

    cat_labels = ["Dev. screen", "Held-out screen", "Overall screen",
                  "Post-gen. OOB", "Text-level harm"]
    cat_n = [
        int(r0["input_screened_dev_n"]), int(r0["input_screened_heldout_n"]),
        int(r0["input_screened_overall_n"]), int(r0["post_generation_oob_n"]),
        int(r0["text_harm_total_n"]),
    ]
    cat_den = [
        int(r0["input_emergency_dev_n"]), int(r0["input_emergency_heldout_n"]),
        int(r0["input_emergency_overall_n"]), int(r0["response_total_n"]),
        int(r0["response_total_n"]),
    ]
    rates = [n / d for n, d in zip(cat_n, cat_den)]
    xpos = np.arange(len(cat_labels))
    bars = ax_c.bar(xpos, rates, width=0.62,
                    color=[PALETTE[0], PALETTE[1], PALETTE[4], PALETTE[3], PALETTE[2]],
                    edgecolor="white")
    ax_c.set_xticks(xpos)
    ax_c.set_xticklabels(cat_labels, fontsize=7.2, rotation=40, ha="right")
    ax_c.set_ylabel("Proportion", fontsize=9)
    ax_c.set_ylim(0, 1.28)
    ax_c.set_title("(c) Screening recall and text flags", fontsize=10, fontweight="bold", loc="left", pad=5)
    ax_c.spines[["top", "right"]].set_visible(False)
    for bar, n, den in zip(bars, cat_n, cat_den):
        ax_c.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                  f"{n}/{den}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    _save(fig, "figure3_routing_safety")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("CardioRAG data figures")
    print("=" * 50)
    render_figure1_primary_rubric()
    render_figure2_disease_robustness()
    render_figure3_routing_safety()
    print("\nALL FIGURES RENDERED SUCCESSFULLY.")

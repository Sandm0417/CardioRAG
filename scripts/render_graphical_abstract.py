"""CardioRAG graphical abstract - 16:9 landscape.

Usage:  python scripts/render_graphical_abstract.py
Writes PNG, PDF, SVG, and TIFF to outputs/figures/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Rectangle

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#2C5F8A"
TEAL = "#3A9B7A"
CORAL = "#C45C5C"
GOLD = "#E2B857"
INK = "#2A2A2A"
MUTED = "#6B7280"
CARD = "#F6F8FA"
LINE = "#D8DEE6"


def rr(ax, x, y, w, h, fc, ec="#E5E7EB", r=0.06, lw=0.8, z=2, alpha=1.0):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.004,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z, alpha=alpha,
    )
    ax.add_patch(p)
    return p


def arrow(ax, a, b, color="#4B5563", lw=0.95, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            a, b, arrowstyle="-|>", mutation_scale=10, lw=lw,
            color=color, connectionstyle=f"arc3,rad={rad}",
            shrinkA=0.4, shrinkB=0.4, zorder=3,
        )
    )


def render():
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(0.50, 8.55, "CardioRAG", fontsize=20, fontweight="bold", color=NAVY, va="center")
    ax.text(
        3.20, 8.55,
        "Guideline-grounded cardiovascular counselling with an explicit emergency policy",
        fontsize=10.5, color=MUTED, va="center",
    )
    ax.plot([0.50, 15.50], [8.22, 8.22], color=LINE, lw=0.7)

    ax.plot([5.30, 5.30], [0.70, 8.05], color=LINE, lw=0.75)
    ax.plot([10.70, 10.70], [0.70, 8.05], color=LINE, lw=0.75)

    # ── LEFT: Problem ──────────────────────────────────────────────────────
    ax.text(0.50, 7.78, "Problem", fontsize=14, fontweight="bold", color=NAVY)

    rr(ax, 0.45, 5.70, 4.55, 1.85, CARD, r=0.05)
    ax.add_patch(Circle((1.20, 7.05), 0.26, facecolor=NAVY, edgecolor="none", zorder=4))
    ax.text(1.20, 7.05, "P", ha="center", va="center", color="white", fontsize=10, fontweight="bold", zorder=5)
    ax.add_patch(FancyBboxPatch((0.95, 6.28), 0.50, 0.48, boxstyle="round,pad=0.01,rounding_size=0.10",
                                facecolor="#D7E4F0", edgecolor=NAVY, lw=0.7, zorder=4))
    ax.text(1.90, 6.95, "Patient question", fontsize=10, fontweight="bold", color=INK, va="center")
    ax.text(1.90, 6.55, "CAD   ·   HF   ·   AF", fontsize=8.5, color=MUTED, va="center")
    for i, (lab, col) in enumerate([("CAD", CORAL), ("HF", TEAL), ("AF", GOLD)]):
        x = 0.75 + i * 1.38
        rr(ax, x, 5.88, 1.20, 0.40, "white", ec=col, r=0.10, lw=1.05)
        ax.text(x + 0.60, 6.08, lab, ha="center", va="center", fontsize=8.5, color=col, fontweight="bold")

    rr(ax, 0.45, 3.72, 4.55, 1.78, CARD, r=0.05)
    ax.text(0.65, 5.18, "Source guidelines", fontsize=10, fontweight="bold", color=INK)
    books = [
        (0.70, "ESC AF\n2024", NAVY),
        (2.05, "AHA/ACC/\nHFSA HF 2022", TEAL),
        (3.40, "ESC CCS\n2024", GOLD),
    ]
    for x, lab, col in books:
        rr(ax, x, 3.90, 1.20, 1.05, "white", ec=col, r=0.06, lw=1.0)
        ax.add_patch(Rectangle((x, 3.90), 0.13, 1.05, facecolor=col, edgecolor="none", zorder=5))
        ax.text(x + 0.68, 4.42, lab, ha="center", va="center", fontsize=7.2, color=INK)

    rr(ax, 0.45, 1.05, 4.55, 2.47, CARD, r=0.05)
    ax.text(0.65, 3.20, "CardioKG", fontsize=10, fontweight="bold", color=INK)
    nodes = [(1.35, 2.35), (2.15, 2.72), (2.95, 2.28), (2.00, 1.78), (3.40, 1.85), (3.70, 2.55)]
    for a, b in [(0, 1), (1, 2), (0, 3), (3, 4), (2, 5), (1, 5), (3, 1)]:
        xa, ya = nodes[a]
        xb, yb = nodes[b]
        ax.plot([xa, xb], [ya, yb], color="#A9C3D4", lw=1.0, zorder=3)
    for i, (x, y) in enumerate(nodes):
        ax.add_patch(Circle((x, y), 0.12, facecolor=TEAL if i % 2 else NAVY,
                            edgecolor="white", lw=0.7, zorder=4))
    ax.text(2.50, 1.42, "typed entities and relations", fontsize=8, color=MUTED, ha="center")
    ax.text(
        0.65, 1.22,
        "Fluent LLM answers lack guideline\nprovenance and an emergency policy.",
        fontsize=8.4, color=INK, va="bottom",
    )

    # ── CENTER: CardioRAG ──────────────────────────────────────────────────
    ax.text(5.50, 7.78, "CardioRAG", fontsize=14, fontweight="bold", color=NAVY)

    steps = [
        (6.95, GOLD, "1   Patient question"),
        (5.95, CORAL, "2   Input emergency screen"),
        (4.95, TEAL, "3   LightRAG retrieval (C3, C4)"),
        (3.95, NAVY, "4   DeepSeek-chat generation"),
        (2.95, CORAL, "5   Out-of-bounds check (C4)"),
        (1.70, TEAL, "Answer + disclaimer"),
    ]
    cx, cw, ch = 6.20, 3.55, 0.72
    midx = cx + cw / 2
    for y, col, lab in steps:
        rr(ax, cx, y, cw, ch, col, ec="none", r=0.09, lw=0)
        ax.text(midx, y + ch / 2, lab, ha="center", va="center",
                fontsize=8.6, color="white", fontweight="bold")
    for y_from, y_to in [(6.95, 6.67), (5.95, 5.67), (4.95, 4.67), (3.95, 3.67), (2.95, 2.42)]:
        arrow(ax, (midx, y_from), (midx, y_to))

    # Emergency-care template, left of screen
    rr(ax, 5.42, 5.10, 0.70, 0.78, "#F8E8E8", ec=CORAL, r=0.07, lw=0.9)
    ax.text(5.77, 5.49, "Emergency-\ncare template", ha="center", va="center",
            fontsize=6.1, color=CORAL)
    arrow(ax, (6.20, 6.31), (6.12, 5.88), color=CORAL, lw=0.9)

    # Physician referral, right of OOB
    rr(ax, 9.85, 2.90, 0.72, 0.82, "#F8E8E8", ec=CORAL, r=0.07, lw=0.9)
    ax.text(10.21, 3.31, "Physician\nreferral", ha="center", va="center",
            fontsize=6.1, color=CORAL)
    arrow(ax, (9.75, 3.31), (9.85, 3.31), color=CORAL, lw=0.9)

    # C1/C2 skip retrieval
    ax.annotate(
        "", xy=(6.28, 4.20), xytext=(6.28, 6.00),
        arrowprops=dict(arrowstyle="-|>", color="#9CA3AF", lw=0.9,
                        linestyle=(0, (2.4, 1.8))),
    )
    ax.text(5.48, 4.55, "C1, C2\nskip retrieval", fontsize=6.3, color=MUTED, ha="left", va="center")

    # tiny graph near retrieval
    gx0, gy0 = 9.82, 5.18
    for i in range(6):
        ax.add_patch(Circle(
            (gx0 + (i % 3) * 0.16, gy0 + (i // 3) * 0.16), 0.045,
            facecolor=TEAL if i % 2 else NAVY, edgecolor="none", zorder=5,
        ))

    # ── RIGHT: Finding ─────────────────────────────────────────────────────
    ax.text(10.90, 7.78, "Finding on a 1–5 rubric", fontsize=14, fontweight="bold", color=NAVY)
    means = [
        ("C1 Vanilla", 2.55, "#A9C3D4"),
        ("C2 Naive RAG", 2.72, "#7FA3C2"),
        ("C3 LightRAG-Generic", 3.36, "#4E7FA8"),
        ("C4 CardioRAG Full", 3.62, NAVY),
    ]
    x0 = 11.00
    bar_max = 4.40
    for i, (lab, val, col) in enumerate(means):
        y = 6.95 - i * 0.82
        ax.text(x0, y + 0.46, lab, fontsize=8.6, color=INK, va="bottom")
        ax.add_patch(Rectangle((x0, y), bar_max, 0.30, facecolor="#EEF2F6", edgecolor="none"))
        rr(ax, x0, y, bar_max * (val / 5.0), 0.30, col, ec="none", r=0.04, lw=0)
        ax.text(x0 + bar_max * (val / 5.0) + 0.10, y + 0.15, f"{val:.2f}",
                fontsize=8.3, color=INK, va="center")

    rr(ax, 11.00, 3.42, 4.50, 0.70, "#EEF4FA", ec=NAVY, r=0.07, lw=0.8)
    ax.text(11.22, 3.77, "Guideline concordance  3.62 vs 2.55",
            fontsize=8.6, color=NAVY, va="center", fontweight="bold")
    rr(ax, 11.00, 2.55, 4.50, 0.70, "#E8F5EF", ec=TEAL, r=0.07, lw=0.8)
    ax.text(11.22, 2.90, "Citation correctness  4.66 vs 2.03",
            fontsize=8.6, color=TEAL, va="center", fontweight="bold")

    chips = [
        (11.00, 1.52, "120 virtual scenarios"),
        (12.58, 1.42, "single backbone"),
        (14.08, 1.52, "input-screen recall 10/16"),
    ]
    for x, w, lab in chips:
        rr(ax, x, 1.18, w, 0.42, CARD, r=0.14, lw=0.6)
        ax.text(x + w / 2, 1.39, lab, ha="center", va="center", fontsize=6.5, color=MUTED)

    ax.text(
        8.00, 0.38,
        "Rubric-derived scores on standardized virtual scenarios. Not a clinical-safety claim.",
        ha="center", va="center", fontsize=7.4, color=MUTED,
    )

    stem = "graphical_abstract"
    for ext in ("png", "pdf", "svg", "tiff"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=600, facecolor="white")
    print(f"  [ok] {stem}.*")
    plt.close(fig)


if __name__ == "__main__":
    render()

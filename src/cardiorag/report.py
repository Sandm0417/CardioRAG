"""Statistical analysis and report generation for CardioRAG evaluation.

Produces publication-ready tables and figures source data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def load_scores(scores_dir: Path) -> pd.DataFrame:
    """Load LLM-judge scores from CSV."""
    csv_path = scores_dir / "llm_judge.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def friedman_test(df: pd.DataFrame, group_col: str, value_col: str) -> dict[str, Any]:
    """Friedman test for ranked comparison across LLM-config pairs."""
    pivoted = df.pivot(index="scenario_id", columns=group_col, values=value_col)
    statistic, p_value = stats.friedmanchisquare(*[pivoted[c] for c in pivoted.columns])
    return {"test": "Friedman", "statistic": statistic, "p_value": p_value, "n": len(pivoted)}


def wilcoxon_posthoc(
    df: pd.DataFrame, group_col: str, value_col: str
) -> pd.DataFrame:
    """Wilcoxon signed-rank post-hoc with Bonferroni correction."""
    groups = df[group_col].unique()
    results = []
    for i, g1 in enumerate(groups):
        for g2 in groups[i + 1 :]:
            v1 = df[df[group_col] == g1][value_col]
            v2 = df[df[group_col] == g2][value_col]
            stat, p = stats.wilcoxon(v1, v2)
            results.append({"g1": g1, "g2": g2, "statistic": stat, "p_raw": p})
    result_df = pd.DataFrame(results)
    n_comparisons = len(result_df)
    result_df["p_corrected"] = result_df["p_raw"] * n_comparisons
    result_df["p_corrected"] = result_df["p_corrected"].clip(upper=1.0)
    return result_df


def mcnemar_test(
    df: pd.DataFrame, group1: str, group2: str, flag_col: str = "hallucination_flag"
) -> dict[str, Any]:
    """McNemar test for paired binary proportions (hallucination rates)."""
    t1 = df[df["rag_config"] == group1][flag_col].values
    t2 = df[df["rag_config"] == group2][flag_col].values
    # Build contingency table
    b = ((t1 == 1) & (t2 == 0)).sum()
    c = ((t1 == 0) & (t2 == 1)).sum()
    if b + c > 0:
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        p = 1 - stats.chi2.cdf(stat, 1)
    else:
        stat, p = 0.0, 1.0
    return {"test": "McNemar", "statistic": stat, "p_value": p, "b": int(b), "c": int(c)}


def compute_cohens_kappa(rater1: list[int], rater2: list[int]) -> float:
    """Compute Cohen's kappa for two raters."""
    # Stub — use sklearn.metrics.cohen_kappa_score
    return 0.0


def ablation_table(
    df: pd.DataFrame, metric_col: str = "clinical_accuracy"
) -> pd.DataFrame:
    """Build 4-config ablation table: decompose RAG model contributions."""
    configs = ["vanilla", "naive_rag", "lightrag_generic", "cardiorag_full"]
    rows = []
    for config in configs:
        subset = df[df["rag_config"] == config]
        rows.append(
            {
                "config": config,
                "mean": subset[metric_col].mean(),
                "std": subset[metric_col].std(),
                "median": subset[metric_col].median(),
                "n": len(subset),
            }
        )
    return pd.DataFrame(rows)


def generate_report(
    scores_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Generate full statistical report.

    Returns dict mapping output filenames to their paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_scores(scores_dir)

    outputs: dict[str, Path] = {}

    # 1. Ablation table
    ablation = ablation_table(df)
    ablation_path = output_dir / "ablation.csv"
    ablation.to_csv(ablation_path, index=False)
    outputs["ablation"] = ablation_path

    # 2. Main rank table (6 LLM × 4 config)
    if not df.empty:
        rank_table = (
            df.groupby(["llm", "rag_config"])["clinical_accuracy"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        rank_path = output_dir / "main_rank.csv"
        rank_table.to_csv(rank_path, index=False)
        outputs["main_rank"] = rank_path

    # 3. Safety metrics
    safety_path = output_dir / "safety_metrics.csv"
    pd.DataFrame().to_csv(safety_path)  # stub
    outputs["safety_metrics"] = safety_path

    return outputs

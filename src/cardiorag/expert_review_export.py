"""Export expert review packages for cardiologist blind evaluation.

Samples 240 generations, de-identifies, and creates review sheets.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .schemas import Generation


def sample_for_expert_review(
    generations: list[Generation],
    n_per_disease: int = 20,
    diseases: list[str] | None = None,
    seed: int = 42,
) -> list[Generation]:
    """Sample generations for expert blind review.

    Strategy: per disease, sample n_per_disease scenarios, covering all 4 configs.
    De-duplicate across LLMs (pick one randomly per scenario-config pair).
    """
    diseases = diseases or ["CAD", "HF", "AF"]
    random.seed(seed)

    sampled: list[Generation] = []
    for disease in diseases:
        disease_gens = [
            g
            for g in generations
            if g.scenario_id.startswith(disease)
        ]
        # Group by scenario_id
        by_scenario: dict[str, list[Generation]] = {}
        for g in disease_gens:
            by_scenario.setdefault(g.scenario_id, []).append(g)

        # Per scenario, pick one random LLM
        for sid, gens in by_scenario.items():
            sampled.append(random.choice(gens))

        # Limit to n_per_disease
        if len([g for g in sampled if g.scenario_id.startswith(disease)]) > n_per_disease:
            disease_sampled = [g for g in sampled if g.scenario_id.startswith(disease)]
            random.shuffle(disease_sampled)
            sampled = [g for g in sampled if not g.scenario_id.startswith(disease)]
            sampled.extend(disease_sampled[:n_per_disease])

    return sampled


def export_review_package(
    sampled: list[Generation],
    output_dir: Path,
    reviewer_ids: list[str] | None = None,
) -> list[Path]:
    """Export de-identified review packages for each reviewer.

    Returns list of output file paths.
    """
    reviewer_ids = reviewer_ids or ["reviewer_1", "reviewer_2", "reviewer_3"]
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = []

    for reviewer in reviewer_ids:
        # Shuffle to avoid order bias
        shuffled = list(sampled)
        random.shuffle(shuffled)

        review_items = []
        for i, gen in enumerate(shuffled):
            review_items.append(
                {
                    "review_id": f"{reviewer}_{i:04d}",
                    "item_number": i + 1,
                    "question": gen.prompt,
                    "response": gen.response,
                    "disease_domain": gen.scenario_id[:3],
                    # De-identified: no LLM name, config name, or ground truth
                }
            )

        output_path = output_dir / f"{reviewer}_review_package.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for item in review_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        exported.append(output_path)

    return exported


def compile_expert_results(
    review_files: list[Path],
    deident_map: Path,
) -> dict[str, Any]:
    """Compile expert reviews and compute inter-rater agreement (κ).

    Args:
        review_files: Paths to completed review JSONL files.
        deident_map: Path to JSON mapping review_id → generation_id.

    Returns:
        Dict with fleiss_kappa, per_dimension_scores, per_reviewer_stats.
    """
    # Stub
    return {
        "fleiss_kappa": 0.0,
        "cohens_kappa_pairs": {},
        "per_dimension": {},
        "per_reviewer": {},
    }

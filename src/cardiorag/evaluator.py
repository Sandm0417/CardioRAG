"""Main evaluation runner: 6 LLM × 4 config × 120 scenarios × 2 languages × 2 datasets.

Orchestrates the full evaluation pipeline defined in PLAN(1) §3.
"""

from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .guardrail import CVGuardrail
from .lightrag_adapter import LightRAGAdapter
from .llm_client import LLMClient, get_llm_client
from .naive_rag import NaiveRAG
from .schemas import Generation, Scenario


def load_scenarios(path: Path) -> list[Scenario]:
    """Load scenarios from JSONL file."""
    scenarios = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                scenarios.append(Scenario(**json.loads(line)))
    return scenarios


def run_config(
    llm_name: str,
    config_name: str,
    scenarios: list[Scenario],
    language: str,
    dataset: str,
    rag: Any = None,
    guardrail: CVGuardrail | None = None,
    output_dir: Path | None = None,
) -> list[Generation]:
    """Run one (LLM, config) cell over all scenarios.

    Args:
        llm_name: One of the 6 LLM keys.
        config_name: vanilla | naive_rag | lightrag_generic | cardiorag_full
        scenarios: List of scenarios to run.
        language: zh | en
        dataset: internal | external
        rag: RAG adapter instance (None for vanilla).
        guardrail: Guardrail instance (only for cardiorag_full).
        output_dir: Where to save raw generations.

    Returns:
        List of Generation objects.
    """
    llm_client = get_llm_client(llm_name)
    guardrail = guardrail or CVGuardrail()
    generations = []

    output_path = None
    if output_dir:
        cell_dir = output_dir / f"{llm_name}_{config_name}" / dataset / language
        cell_dir.mkdir(parents=True, exist_ok=True)
        output_path = cell_dir / "generations.jsonl"

    system_prompt = _build_system_prompt(config_name, language)

    for scenario in tqdm(scenarios, desc=f"{llm_name}/{config_name}/{language}/{dataset}"):
        question = (
            scenario.patient_question_zh
            if language == "zh"
            else scenario.patient_question_en
        )

        # Pre-generation guardrail
        pre_check = guardrail.pre_generation_check(question)

        if pre_check["block"]:
            response = pre_check["response"]
            safety_meta = pre_check["metadata"]
            chunks = []
        else:
            # Retrieve context
            chunks = []
            if rag and config_name != "vanilla":
                rag_result = rag.query(question, mode="hybrid")
                chunks = rag_result.get("chunks", [])

            # Generate
            gen_result = llm_client.generate_with_context(
                system_prompt, question, chunks
            )
            response = gen_result["response"]

            # Post-generation guardrail (only for full config)
            if config_name == "cardiorag_full":
                post_check = guardrail.post_generation_check(response)
                if post_check["flagged"]:
                    response = post_check["sanitized_response"]

            response = guardrail.append_disclaimer(response)
            safety_meta = guardrail.stats

        gen = Generation(
            generation_id=f"{llm_name}_{config_name}_{scenario.scenario_id}_{language}",
            scenario_id=scenario.scenario_id,
            llm=llm_name,
            rag_config=config_name,
            language=language,
            dataset=dataset,
            prompt=question,
            response=response,
            evidence_chunks=chunks,
            safety_metadata=safety_meta,
            latency_ms=0.0,
        )
        generations.append(gen)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            for g in generations:
                f.write(json.dumps(g.model_dump(), ensure_ascii=False) + "\n")

    return generations


def run_all_configs(
    scenarios_dir: Path,
    output_dir: Path,
    llms: list[str] | None = None,
    configs: list[str] | None = None,
    languages: list[str] | None = None,
    datasets: list[str] | None = None,
) -> dict[str, list[Generation]]:
    """Run the full 6×4×2×2 evaluation matrix.

    Returns dict keyed by '{llm}_{config}_{lang}_{dataset}' → list of Generations.
    """
    llms = llms or list(LLM_CONFIGS.keys())
    configs = configs or ["vanilla", "naive_rag", "lightrag_generic", "cardiorag_full"]
    languages = languages or ["zh", "en"]
    datasets = datasets or ["internal", "external"]

    results: dict[str, list[Generation]] = {}

    # Initialize RAG backends
    lightrag = LightRAGAdapter()
    naive_rag = NaiveRAG()
    guardrail = CVGuardrail()

    total_cells = len(llms) * len(configs) * len(languages) * len(datasets)
    print(f"Running {total_cells} cells...")

    for llm_name in llms:
        for config_name in configs:
            for lang in languages:
                for ds in datasets:
                    key = f"{llm_name}_{config_name}_{lang}_{ds}"
                    scenario_file = scenarios_dir / f"{ds}.jsonl"
                    if not scenario_file.exists():
                        print(f"  Skipping {key}: scenario file not found")
                        continue

                    scenarios = load_scenarios(scenario_file)

                    # Select RAG backend
                    rag = None
                    if config_name == "naive_rag":
                        rag = naive_rag
                    elif config_name in ("lightrag_generic", "cardiorag_full"):
                        rag = lightrag

                    guard = guardrail if config_name == "cardiorag_full" else None

                    print(f"  {key} ({len(scenarios)} scenarios)")
                    generations = run_config(
                        llm_name, config_name, scenarios, lang, ds,
                        rag=rag, guardrail=guard, output_dir=output_dir,
                    )
                    results[key] = generations

    print(f"Done. {sum(len(v) for v in results.values())} generations total.")
    return results


def _build_system_prompt(config_name: str, language: str) -> str:
    """Build system prompt based on config and language."""
    base_zh = "你是一位心血管专科的医师-患者沟通助手，基于临床指南提供患者教育信息。"
    base_en = "You are a cardiovascular physician-patient communication assistant, providing patient education based on clinical guidelines."

    if language == "zh":
        prompt = base_zh
        if config_name == "cardiorag_full":
            prompt += "\n重要：不得提供具体用药剂量调整建议、停药决策或手术决策。这些必须由执业医师判断。"
    else:
        prompt = base_en
        if config_name == "cardiorag_full":
            prompt += "\nIMPORTANT: Do not provide specific medication dose adjustments, medication discontinuation decisions, or procedure decisions. These must be made by a licensed physician."

    return prompt


# Import LLM_CONFIGS for run_all_configs default
from .llm_client import LLM_CONFIGS  # noqa: E402

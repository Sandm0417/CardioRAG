"""CardioKG Builder: Guideline PDF → Entity/Relation Extraction → JSONL.

Core method学 contribution. Builds CardioKG from ESC/AHA/CSC cardiovascular guidelines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# CardioKG Schema definitions
NODE_TYPES = [
    "Disease",
    "Symptom",
    "Biomarker",
    "Drug",
    "DrugClass",
    "Procedure",
    "RiskFactor",
    "Stratification",
    "GuidelineStatement",
    "Contraindication",
    "PatientFactor",
]

RELATION_TYPES = [
    "treats",
    "contraindicated_in",
    "threshold_for",
    "risk_stratifies",
    "monitors",
    "interacts_with",
    "guideline_cites",
]


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract raw text from a guideline PDF using PyMuPDF."""
    # Stub — requires PyMuPDF
    # import fitz
    # doc = fitz.open(pdf_path)
    # return "\n".join(page.get_text() for page in doc)
    return ""


def chunk_by_sections(text: str) -> list[dict[str, Any]]:
    """Split guideline text by standard section numbering.

    Returns list of {section_id, section_title, text, guideline_source}.
    """
    # Stub
    return []


def extract_entities_llm(
    chunk: dict[str, Any], llm_client: Any
) -> list[dict[str, Any]]:
    """LLM-assisted entity extraction from a guideline chunk.

    Returns list of {node_type, name, attributes, guideline_cite}.
    """
    # Stub — prompt the LLM with CardioKG schema
    prompt = f"""Extract cardiovascular entities from the following guideline text.
Use only these node types: {', '.join(NODE_TYPES)}.

For each entity, provide:
- node_type: one of the above
- name: standardized name
- attributes: key-value pairs (e.g., threshold values, normal ranges)

Guideline text:
{chunk['text'][:3000]}
"""
    return []


def extract_relations_llm(
    chunk: dict[str, Any], entities: list[dict[str, Any]], llm_client: Any
) -> list[dict[str, Any]]:
    """LLM-assisted relation extraction linking entities.

    Returns list of {subject, relation, object, guideline_cite}.
    """
    # Stub
    prompt = f"""Extract relationships between the following cardiovascular entities.
Use only these relation types: {', '.join(RELATION_TYPES)}.

Entities: {json.dumps([e['name'] for e in entities], ensure_ascii=False)}

For each relationship, provide:
- subject: entity name
- relation: one of the relation types
- object: entity name

Guideline text:
{chunk['text'][:3000]}
"""
    return []


def build_cardiokg(
    pdf_dir: Path,
    output_dir: Path,
    disease: str,  # "cad", "hf", or "af"
    llm_client: Any = None,
) -> dict[str, Any]:
    """Full CardioKG build pipeline for one disease.

    Args:
        pdf_dir: Directory containing guideline PDFs.
        output_dir: Where to write JSONL output.
        disease: Disease code (cad/hf/af).
        llm_client: LLM client for extraction (defaults to DeepSeek V4 Flash).

    Returns:
        Dict with node_count, relation_count, guideline_sources.
    """
    output_path = output_dir / f"{disease}.jsonl"
    nodes = []
    relations = []

    # Scan PDFs for this disease
    pdf_files = list(pdf_dir.glob(f"*{disease.upper()}*.pdf")) + list(
        pdf_dir.glob(f"*{disease.lower()}*.pdf")
    )
    if not pdf_files:
        pdf_files = list(pdf_dir.glob("*.pdf"))

    for pdf_path in pdf_files:
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_by_sections(text)

        for chunk in chunks:
            chunk_nodes = extract_entities_llm(chunk, llm_client)
            nodes.extend(chunk_nodes)
            chunk_relations = extract_relations_llm(chunk, chunk_nodes, llm_client)
            relations.extend(chunk_relations)

    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for node in nodes:
            f.write(json.dumps({"type": "node", **node}, ensure_ascii=False) + "\n")
        for rel in relations:
            f.write(json.dumps({"type": "relation", **rel}, ensure_ascii=False) + "\n")

    return {
        "disease": disease,
        "node_count": len(nodes),
        "relation_count": len(relations),
        "guideline_sources": [p.name for p in pdf_files],
        "output_path": str(output_path),
    }

"""Pydantic schemas for CardioRAG data models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class Scenario(BaseModel):
    """A clinical dialogue test scenario."""

    scenario_id: str = Field(..., description="Unique ID, e.g. CAD-001")
    disease: str = Field(..., description="CAD, HF, or AF")
    category: str = Field(
        ..., description="symptom_explanation | test_interpretation | medication_consult | emergency_out_of_bounds"
    )
    patient_question_zh: str
    patient_question_en: str
    patient_factors: dict[str, Any] = Field(
        default_factory=dict,
        description="Age, comorbidities, lab values, medications, etc.",
    )
    ground_truth_answer_zh: str
    ground_truth_answer_en: str
    guideline_source: str = Field(..., description="Guideline name and section")
    safety_label: str = Field(
        default="normal",
        description="normal | emergency_routing | out_of_bounds",
    )
    emergency_keywords: list[str] = Field(default_factory=list)
    out_of_bounds_categories: list[str] = Field(default_factory=list)


class Generation(BaseModel):
    """A single LLM generation result."""

    generation_id: str
    scenario_id: str
    llm: str  # gpt5, gemini25pro, claude_sonnet45, grok4, deepseek_v3, qwen3_max
    rag_config: str  # vanilla, naive_rag, lightrag_generic, cardiorag_full
    language: str  # zh, en
    dataset: str  # internal, external
    prompt: str
    response: str
    evidence_chunks: list[dict[str, Any]] = Field(default_factory=list)
    safety_metadata: dict[str, Any] = Field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class Score(BaseModel):
    """LLM-as-judge scoring result."""

    generation_id: str
    judge_model: str = "gpt5"
    clinical_accuracy: int = Field(..., ge=1, le=5)
    patient_centeredness: int = Field(..., ge=1, le=5)
    safety: int = Field(..., ge=1, le=5)
    citation_correctness: int = Field(..., ge=1, le=5)
    hallucination_flag: bool = False
    judge_rationale: str = ""


class ExpertReview(BaseModel):
    """Expert cardiologist blind review."""

    review_id: str
    generation_id: str
    reviewer_id: str  # de-identified
    rank_order: int = Field(..., ge=1)
    hallucination_flag: bool = False
    clinical_harm_flag: bool = False
    qualitative_notes: str = ""

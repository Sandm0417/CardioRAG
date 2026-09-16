"""Unified LLM client for all 6 tested models.

Provides a consistent interface across GPT-5, Gemini 2.5 Pro, Claude Sonnet 4.5,
Grok 4, DeepSeek-V3, and Qwen3-Max.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMConfig:
    """Configuration for a single LLM provider."""

    name: str  # e.g., "gpt5", "claude_sonnet45"
    provider: str  # "openai", "anthropic", "google", "xai", "deepseek", "qwen"
    model_id: str  # API model string
    api_key_env: str  # Environment variable name for API key
    base_url: str | None = None
    default_params: dict[str, Any] = field(default_factory=dict)


# ── 6 LLM configurations (2026 generation) ────────────────────
LLM_CONFIGS: dict[str, LLMConfig] = {
    "gpt5": LLMConfig(
        name="gpt5",
        provider="openai",
        model_id="gpt-5",
        api_key_env="OPENAI_API_KEY",
        default_params={"temperature": 0.3, "max_tokens": 2000},
    ),
    "gemini25pro": LLMConfig(
        name="gemini25pro",
        provider="google",
        model_id="gemini-2.5-pro",
        api_key_env="GOOGLE_API_KEY",
        default_params={"temperature": 0.3, "max_output_tokens": 2000},
    ),
    "claude_sonnet45": LLMConfig(
        name="claude_sonnet45",
        provider="anthropic",
        model_id="claude-sonnet-4-5-20250929",
        api_key_env="ANTHROPIC_API_KEY",
        default_params={"temperature": 0.3, "max_tokens": 2000},
    ),
    "grok4": LLMConfig(
        name="grok4",
        provider="xai",
        model_id="grok-4",
        api_key_env="XAI_API_KEY",
        base_url="https://api.x.ai/v1",
        default_params={"temperature": 0.3, "max_tokens": 2000},
    ),
    "deepseek_v3": LLMConfig(
        name="deepseek_v3",
        provider="deepseek",
        model_id="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        default_params={"temperature": 0.3, "max_tokens": 2000},
    ),
    "qwen3_max": LLMConfig(
        name="qwen3_max",
        provider="qwen",
        model_id="qwen3-max",
        api_key_env="QWEN_API_KEY",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_params={"temperature": 0.3, "max_tokens": 2000},
    ),
}


class LLMClient:
    """Unified LLM client dispatching to 6 providers."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None  # Lazy init per provider SDK

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        **overrides: Any,
    ) -> dict[str, Any]:
        """Generate a response from the LLM.

        Returns dict: {response, tokens_in, tokens_out, latency_ms, model}.
        """
        # Stub — actual implementation dispatches to provider SDK
        return {
            "response": f"[{self.config.name}] stub response",
            "tokens_in": 0,
            "tokens_out": 0,
            "latency_ms": 0.0,
            "model": self.config.model_id,
        }

    def generate_with_context(
        self,
        system_prompt: str,
        user_message: str,
        context_chunks: list[dict[str, Any]],
        **overrides: Any,
    ) -> dict[str, Any]:
        """Generate with retrieved context chunks injected into prompt."""
        context_text = "\n\n---\n".join(
            f"[Source: {c.get('source', 'unknown')}]\n{c.get('text', '')}"
            for c in context_chunks
        )
        full_prompt = f"""Context from clinical guidelines:

{context_text}

---

User question: {user_message}

Please answer based on the provided context. If the context is insufficient, state so clearly."""
        return self.generate(system_prompt, full_prompt, **overrides)


def get_llm_client(llm_name: str) -> LLMClient:
    """Factory: get LLM client by name."""
    if llm_name not in LLM_CONFIGS:
        raise ValueError(
            f"Unknown LLM: {llm_name}. Available: {list(LLM_CONFIGS.keys())}"
        )
    return LLMClient(LLM_CONFIGS[llm_name])

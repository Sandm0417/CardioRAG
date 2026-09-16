"""LightRAG adapter for CardioRAG — wraps lightrag-hku with cardiovascular domain config.

Implements 4 query modes: naive, local, global, hybrid.

Index construction pipeline:
  - Corpus: CardioKG JSONL (nodes + relations across CAD/HF/AF) rendered as text blocks
  - Embedding: Volcengine Ark Doubao embedding-vision (2048-dim) via
    POST /api/v3/embeddings/multimodal  (vision model does NOT support /embeddings)
  - Entity extraction: DeepSeek chat (OpenAI-compatible API)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests


def _load_env_file() -> None:
    """Load `.env` from the project root if present."""
    from cardiorag.env import load_env

    load_env(Path(__file__).resolve().parents[2])


_load_env_file()

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc

logger = logging.getLogger("cardiorag.lightrag")

EMBEDDING_DIM = 2048
# The Ark /embeddings/multimodal endpoint returns exactly ONE vector per request
# (multi-item input is not honored), so we send one text per call.
EMBEDDING_BATCH = 1
EMBEDDING_MAX_RETRIES = 3

DISEASE_FILES = ["cad", "hf", "af"]

ARK_BASE_URL = os.getenv("ARK_EMBEDDING_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")

ARK_API_KEY = os.getenv("ARK_API_KEY", "")
ARK_EMBEDDING_ENDPOINT = os.getenv("ARK_EMBEDDING_ENDPOINT", "")
ARK_EMBEDDING_PATH = "/embeddings/multimodal"


def _post_embedding(url: str, payload: dict[str, Any]) -> Any:
    """POST one batch to the Ark multimodal embedding endpoint with retry."""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {ARK_API_KEY}"}
    last_err: Exception | None = None
    for attempt in range(EMBEDDING_MAX_RETRIES):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                return resp.json().get("data")
            if resp.status_code in (429, 500, 502, 503, 504):
                logger.warning("Ark embedding HTTP %s (attempt %s), retrying...", resp.status_code, attempt + 1)
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(f"Ark embedding HTTP {resp.status_code}: {resp.text[:300]}")
        except RuntimeError:
            raise
        except Exception as exc:  # network-level errors
            last_err = exc
            logger.warning("Ark embedding error (attempt %s): %s", attempt + 1, exc)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Ark embedding failed after {EMBEDDING_MAX_RETRIES} retries: {last_err}")


def volcengine_embed(texts: list[str]) -> list[list[float]]:
    """Volcengine Ark Doubao embedding-vision (multimodal endpoint, 2048-dim).

    Args:
        texts: list of UTF-8 text strings.

    Returns:
        list of 2048-dim float vectors, one per input text.
    """
    if not ARK_API_KEY or not ARK_EMBEDDING_ENDPOINT:
        raise RuntimeError("ARK_API_KEY / ARK_EMBEDDING_ENDPOINT not configured in .env")
    url = ARK_BASE_URL.rstrip("/") + ARK_EMBEDDING_PATH
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH):
        batch = texts[start : start + EMBEDDING_BATCH]
        payload = {
            "model": ARK_EMBEDDING_ENDPOINT,
            "encoding_format": "float",
            "input": [{"type": "text", "text": t} for t in batch],
        }
        data = _post_embedding(url, payload)
        if isinstance(data, dict):
            vecs = data.get("embeddings") or ([data["embedding"]] if "embedding" in data else [])
        elif isinstance(data, list):
            vecs = [d["embedding"] for d in data]
        else:
            vecs = []
        if len(vecs) != len(batch):
            raise RuntimeError(f"Ark embedding count mismatch: got {len(vecs)} for {len(batch)}")
        vectors.extend(vecs)
    return vectors


async def _embedding_func(texts: list[str]) -> np.ndarray:
    """LightRAG embedding hook (async, returns float32 ndarray)."""
    vecs = await asyncio.to_thread(volcengine_embed, texts)
    return np.asarray(vecs, dtype=np.float32)


async def _llm_func(prompt: str, system_prompt: str | None = None, **kwargs: Any) -> str:
    """LightRAG LLM hook — DeepSeek chat (OpenAI-compatible).

    LightRAG 1.5.x passes extra kwargs (e.g. hashing_kv, history_messages);
    they are intentionally ignored here.
    """
    return await openai_complete_if_cache(
        model=os.getenv("LIGHTRAG_LLM_MODEL", "deepseek-chat"),
        prompt=prompt,
        system_prompt=system_prompt,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        timeout=120,
    )


class LightRAGAdapter:
    """Adapter wrapping LightRAG for CardioRAG evaluation.

    Provides 4 query modes (naive/local/global/hybrid) and handles
    cardiovascular-specific embedding configuration.
    """

    def __init__(
        self,
        working_dir: str | Path = "./lightrag_index",
        embedding_dim: int = EMBEDDING_DIM,
        llm_model: str = "deepseek-chat",
        chunk_token_size: int = 1200,
        top_k: int = 60,
        chunk_top_k: int = 20,
        enable_rerank: bool = False,
        rerank_model_func: Any | None = None,
        rerank_model_max_async: int = 4,
        min_rerank_score: float = 0.0,
    ):
        self.working_dir = Path(working_dir)
        self.embedding_dim = embedding_dim
        self.llm_model = llm_model
        self.chunk_token_size = chunk_token_size
        self.top_k = top_k
        self.chunk_top_k = chunk_top_k
        self.enable_rerank = enable_rerank
        self.rerank_model_func = rerank_model_func
        self.rerank_model_max_async = rerank_model_max_async
        self.min_rerank_score = min_rerank_score
        if self.enable_rerank and self.rerank_model_func is None:
            raise ValueError(
                "enable_rerank=True requires a verified rerank_model_func; "
                "provide one explicitly or set enable_rerank=False"
            )
        self._rag: LightRAG | None = None  # lazy init
        self._loop = asyncio.new_event_loop()
        self._loop_ready = threading.Event()
        self._loop_thread = threading.Thread(
            target=self._run_loop, name="cardiorag-lightrag-loop", daemon=True
        )
        self._loop_thread.start()
        self._loop_ready.wait()

    def _run_loop(self) -> None:
        """Own one event loop for all LightRAG async operations."""
        asyncio.set_event_loop(self._loop)
        self._loop_ready.set()
        self._loop.run_forever()

    def _run_async(self, coro: Any) -> Any:
        """Run a coroutine on LightRAG's persistent loop."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def initialize(self) -> None:
        """Initialize LightRAG with cardiovascular-optimized settings."""
        if self._rag is not None:
            return
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self._rag = LightRAG(
            working_dir=str(self.working_dir),
            embedding_func=EmbeddingFunc(
                embedding_dim=self.embedding_dim,
                func=_embedding_func,
                max_token_size=8192,
                model_name="doubao-embedding-vision-251215",
            ),
            llm_model_func=_llm_func,
            chunk_token_size=self.chunk_token_size,
            top_k=self.top_k,
            chunk_top_k=self.chunk_top_k,
            embedding_batch_num=EMBEDDING_BATCH,
            entity_extract_max_gleaning=1,
            rerank_model_func=self.rerank_model_func,
            rerank_model_max_async=self.rerank_model_max_async,
            min_rerank_score=self.min_rerank_score,
            addon_params={"language": "Chinese, English"},
        )
        self._run_async(self._rag.initialize_storages())
        logger.info("LightRAG initialized at %s", self.working_dir)

    def query(
        self,
        question: str,
        mode: str = "hybrid",
        top_k: int = 60,
        retrieve_only: bool = False,
    ) -> dict[str, Any]:
        """Query LightRAG with the given mode.

        Args:
            question: Patient/physician question text.
            mode: One of 'naive', 'local', 'global', 'hybrid'.
            top_k: Number of chunks to retrieve.
            retrieve_only: When True, only retrieve the graph/vector context
                (no LLM answer generation) — the evaluation pipeline uses this
                to feed real RAG context into the DeepSeek generation call.

        Returns:
            Dict with 'answer', 'context', 'chunks', 'mode', 'token_usage'.
        """
        self.initialize()
        param = QueryParam(
            mode=mode,
            top_k=top_k,
            chunk_top_k=self.chunk_top_k,
            only_need_context=retrieve_only,
            enable_rerank=self.enable_rerank,
        )
        if retrieve_only:
            raw = self._run_async(self._rag.aquery_data(question, param=param))
            return _normalize_retrieval_result(raw, mode)

        raw = self._run_async(self._rag.aquery(question, param=param))
        answer = str(raw)
        return {
            "answer": answer,
            "context": "",
            "chunks": [],
            "entities": [],
            "relationships": [],
            "references": [],
            "mode": mode,
            "token_usage": {"in": -1, "out": -1},  # per-query usage not exposed by LightRAG
        }

    def insert_entities(self, entities_jsonl: Path) -> int:
        """Insert entities from one CardioKG JSONL file into LightRAG index.

        Uses the direct-graph path (ainsert_custom_kg): CardioKG nodes/relations
        are written into the LightRAG graph as-is — no LLM re-extraction.

        Returns number of valid items inserted.
        """
        self.initialize()
        kg = _parse_cardiokg(entities_jsonl, disease=entities_jsonl.stem)
        n = len(kg["entities"]) + len(kg["relationships"])
        if kg["entities"] or kg["relationships"] or kg["chunks"]:
            self._run_async(self._rag.ainsert_custom_kg(kg))
        logger.info(
            "Inserted %s entities / %s relations / %s chunks from %s",
            len(kg["entities"]),
            len(kg["relationships"]),
            len(kg["chunks"]),
            entities_jsonl.name,
        )
        return n

    def build_index(self, cardiokg_dir: Path) -> dict[str, int]:
        """Build full LightRAG index from CardioKG JSONL files (graph-first).

        CardioKG nodes/relations are imported DIRECTLY into the LightRAG graph
        via ainsert_custom_kg — no LLM re-extraction. This preserves the
        curated guideline graph (no hallucinated entities) and keeps the
        authoritative node/relation structure for local/global/hybrid queries.
        Node/relation descriptions are also inserted as chunks so vector
        retrieval works.

        Returns counts: {nodes, relations, chunks}.
        """
        self.initialize()
        kg: dict[str, list[dict[str, Any]]] = {"chunks": [], "entities": [], "relationships": []}
        for disease in DISEASE_FILES:
            path = cardiokg_dir / f"{disease}.jsonl"
            if not path.exists():
                logger.warning("Missing %s — skipped", path)
                continue
            partial = _parse_cardiokg(path, disease=disease)
            kg["chunks"].extend(partial["chunks"])
            kg["entities"].extend(partial["entities"])
            kg["relationships"].extend(partial["relationships"])
            logger.info(
                "Parsed %s: %s entities / %s relations / %s chunks",
                path.name,
                len(partial["entities"]),
                len(partial["relationships"]),
                len(partial["chunks"]),
            )

        counts = {
            "nodes": len(kg["entities"]),
            "relations": len(kg["relationships"]),
            "chunks": len(kg["chunks"]),
        }
        relation_source_ids = {r["source_id"] for r in kg["relationships"]}
        chunk_source_ids = {c["source_id"] for c in kg["chunks"]}
        missing_relation_chunks = relation_source_ids - chunk_source_ids
        if missing_relation_chunks:
            sample = sorted(missing_relation_chunks)[:3]
            raise ValueError(
                "CardioKG relation source IDs missing from chunks: "
                f"{len(missing_relation_chunks)} (sample={sample})"
            )
        if counts["nodes"] or counts["relations"]:
            self._run_async(self._rag.ainsert_custom_kg(kg))
        return counts


def _normalize_retrieval_result(raw: dict[str, Any], mode: str) -> dict[str, Any]:
    """Normalize LightRAG 1.5.x structured retrieval output for callers."""
    data = raw.get("data") if isinstance(raw, dict) else None
    if not isinstance(data, dict):
        data = {}
    entities = data.get("entities") or []
    relationships = data.get("relationships") or []
    chunks = data.get("chunks") or []
    references = data.get("references") or []
    context_parts: list[str] = []
    for entity in entities:
        description = str(entity.get("description", "")).strip()
        if description:
            context_parts.append(description)
    for relation in relationships:
        description = str(relation.get("description", "")).strip()
        if description:
            context_parts.append(description)
    for chunk in chunks:
        content = str(chunk.get("content", "")).strip()
        if content:
            context_parts.append(content)
    return {
        "answer": "",
        "context": "\n\n".join(context_parts),
        "chunks": chunks,
        "entities": entities,
        "relationships": relationships,
        "references": references,
        "metadata": raw.get("metadata", {}) if isinstance(raw, dict) else {},
        "mode": mode,
        "token_usage": {"in": -1, "out": -1},
    }


def _parse_cardiokg(path: Path, disease: str) -> dict[str, list[dict[str, Any]]]:
    """Parse a CardioKG JSONL file into LightRAG custom_kg structure.

    - node     -> entity   {entity_name, entity_type, description, source_id}
    - relation -> relation {src_id, tgt_id, weight, description, keywords, source_id}
    - both     -> chunk    {content, source_id, file_path} for vector retrieval

    Bad JSON lines are skipped with a warning.
    """
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Skipping malformed JSON line in %s", path.name)
            continue
        kind = item.get("type")
        if kind == "node":
            name = item.get("name", "")
            attrs = item.get("attributes") or {}
            attr_str = "; ".join(f"{k}: {v}" for k, v in attrs.items())
            description = f"[{item.get('node_type', 'node')}] {name}: {attr_str}"
            entities.append(
                {
                    "entity_name": name,
                    "entity_type": item.get("node_type", "node"),
                    "description": description,
                    "source_id": f"cardiokg:{disease}:{name}",
                }
            )
            chunks.append(
                {
                    "content": description,
                    "source_id": f"cardiokg:{disease}:{name}",
                    "file_path": f"cardiokg/{disease}.jsonl",
                }
            )
        elif kind == "relation":
            subject = item.get("subject", "")
            object_ = item.get("object", "")
            rel = item.get("relation", "")
            source = item.get("guideline_source", "")
            relationships.append(
                {
                    "src_id": subject,
                    "tgt_id": object_,
                    "weight": 1.0,
                    "description": f"{subject} {rel} {object_}",
                    "keywords": rel,
                    "source_id": _relation_source_id(disease, subject, rel, object_),
                }
            )
            chunks.append(
                {
                    "content": f"{subject} --{rel}--> {object_}"
                    + (f" (guideline: {source})" if source else ""),
                    "source_id": _relation_source_id(disease, subject, rel, object_),
                    "file_path": f"cardiokg/{disease}.jsonl",
                }
            )
    return {"chunks": chunks, "entities": entities, "relationships": relationships}


def _relation_source_id(disease: str, subject: str, relation: str, object_: str) -> str:
    """Return the shared source ID for a relation and its evidence chunk."""
    return f"cardiokg:{disease}:relation:{subject}->{relation}->{object_}"

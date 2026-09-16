"""Naive RAG baseline: vector-only retrieval without graph structure.

Used as configuration C2 in the 4-config ablation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class NaiveRAG:
    """Simple vector-similarity RAG with no KG organization.

    Contrasts with LightRAG to demonstrate the value of graph-structured retrieval.
    """

    def __init__(self, embedding_dim: int = 2048):
        self.embedding_dim = embedding_dim
        self._chunks: list[dict[str, Any]] = []
        self._embeddings: list[list[float]] = []

    def index_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Index text chunks with vector embeddings."""
        self._chunks = chunks
        # Stub: embed and store

    def retrieve(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Retrieve top-k chunks by cosine similarity."""
        # Stub: compute cosine similarity
        return self._chunks[:top_k]

    def query(self, question: str, top_k: int = 10) -> dict[str, Any]:
        """Full query pipeline."""
        chunks = self.retrieve(question, top_k)
        return {
            "answer": "",
            "chunks": chunks,
            "mode": "naive_rag",
            "token_usage": {"in": 0, "out": 0},
        }

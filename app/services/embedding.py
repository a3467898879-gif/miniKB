"""
Embedding service — converts text to dense vectors.

Design: dual-mode strategy.
1. If EMBEDDING_API_KEY is set AND the provider supports /v1/embeddings
   (OpenAI, SiliconFlow, Ollama, etc.), use the remote API.
2. Otherwise, fall back to ChromaDB's built-in ONNX model
   (all-MiniLM-L6-v2, 384-dim, ~80MB, runs locally without GPU).

This makes the app work out-of-the-box even with DeepSeek (which only
provides LLM chat, not embeddings) — a common gotcha for beginners.
"""

from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from openai import OpenAI

from config import settings


class EmbeddingService:
    """Handles text -> vector conversion with automatic fallback."""

    def __init__(self):
        self._client: Optional[OpenAI] = None
        self._local_ef: Optional[DefaultEmbeddingFunction] = None
        self._use_local: Optional[bool] = None  # None = not decided yet
        self._model = settings.embedding_model
        self._dimension: Optional[int] = None

    def _decide_mode(self) -> bool:
        """
        Decide whether to use local or remote embedding.

        Returns True if local mode is chosen.
        """
        # If embedding key is explicitly set and differs from LLM key,
        # the user probably configured a dedicated embedding provider.
        emb_key = settings.embedding_api_key or settings.llm_api_key
        if not emb_key:
            return True  # No key at all -> local

        # DeepSeek doesn't support embeddings — auto-detect and switch to local
        base_url = settings.embedding_base_url.lower()
        if "deepseek.com" in base_url:
            return True

        return False

    def _get_local_ef(self) -> DefaultEmbeddingFunction:
        """Lazily initialize the local ONNX embedding function."""
        if self._local_ef is None:
            self._local_ef = DefaultEmbeddingFunction()
        return self._local_ef

    def _get_remote_client(self) -> OpenAI:
        """Lazily initialize the OpenAI-compatible client."""
        if self._client is None:
            api_key = settings.embedding_api_key or settings.llm_api_key
            if not api_key:
                raise ValueError(
                    "No embedding API key configured. "
                    "Set EMBEDDING_API_KEY or LLM_API_KEY in .env"
                )
            self._client = OpenAI(
                api_key=api_key,
                base_url=settings.embedding_base_url,
            )
        return self._client

    def embed(self, text: str) -> list[float]:
        """Embed a single text string into a vector."""
        if self._use_local is None:
            self._use_local = self._decide_mode()

        if self._use_local:
            result = self._get_local_ef()([text])
            if self._dimension is None:
                self._dimension = len(result[0])
            return result[0]

        # Remote mode
        client = self._get_remote_client()
        resp = client.embeddings.create(
            model=self._model,
            input=text,
        )
        if self._dimension is None:
            self._dimension = len(resp.data[0].embedding)
        return resp.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts (batch mode for efficiency)."""
        if not texts:
            return []

        if self._use_local is None:
            self._use_local = self._decide_mode()

        if self._use_local:
            results = self._get_local_ef()(texts)
            if self._dimension is None:
                self._dimension = len(results[0])
            return results

        # Remote mode
        client = self._get_remote_client()
        resp = client.embeddings.create(
            model=self._model,
            input=texts,
        )
        sorted_data = sorted(resp.data, key=lambda x: x.index)
        if self._dimension is None:
            self._dimension = len(sorted_data[0].embedding)
        return [item.embedding for item in sorted_data]

    @property
    def dimension(self) -> int:
        """Return embedding dimension (computes once if unknown)."""
        if self._dimension is None:
            self.embed("dimension probe")
        return self._dimension  # type: ignore

    @property
    def model_name(self) -> str:
        """Return the active embedding model name."""
        if self._use_local is None:
            self._use_local = self._decide_mode()
        if self._use_local:
            return "all-MiniLM-L6-v2 (local ONNX)"
        return self._model


# Singleton
embedding_service = EmbeddingService()

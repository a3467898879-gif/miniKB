"""
RAG engine — orchestrates the full Retrieval-Augmented Generation pipeline.

This is the heart of miniKB, mirroring MaxKB's core flow:

  ┌─────────┐    ┌──────────┐    ┌───────────┐    ┌─────────┐    ┌──────────┐
  │ Question │───▶│ Embedding │───▶│ Vector DB │───▶│  Top-K  │───▶│   LLM    │
  │          │    │  (query)  │    │  (search) │    │ Context │    │ (stream) │
  └─────────┘    └──────────┘    └───────────┘    └─────────┘    └──────────┘

The engine ties together: embedding service → vector store → LLM provider.
"""

from typing import Generator

from config import settings
from app.services.embedding import embedding_service
from app.services.vector_store import vector_store
from app.services.llm_provider import llm_provider


class RAGEngine:
    """Orchestrates the RAG pipeline: retrieve → augment → generate."""

    def retrieve(self, kb_id: str, question: str, top_k: int | None = None) -> list[dict]:
        """
        Retrieve relevant document chunks for a question.

        Steps:
        1. Embed the question into a vector
        2. Search the vector store for similar chunks (cosine similarity)
        3. Return ranked results with content + metadata + score
        """
        k = top_k or settings.top_k
        query_vector = embedding_service.embed(question)
        results = vector_store.query(kb_id, query_vector, top_k=k)
        return results

    def generate(
        self,
        question: str,
        context_chunks: list[dict],
        history: list[dict] | None = None,
    ) -> Generator[str, None, None]:
        """
        Generate an answer using retrieved context (streaming).

        The LLM receives:
        - System prompt with RAG instructions + retrieved context
        - Conversation history (if any)
        - The user's question

        Yields text chunks for real-time SSE streaming.
        """
        messages = llm_provider.build_messages(question, context_chunks, history)
        yield from llm_provider.chat_stream(messages)

    def answer(
        self,
        kb_id: str,
        question: str,
        history: list[dict] | None = None,
    ) -> tuple[list[dict], Generator[str, None, None]]:
        """
        Full RAG pipeline: retrieve + generate.

        Returns:
            (sources, stream) where sources is the retrieved chunk metadata
            and stream is a generator yielding the LLM response.
        """
        sources = self.retrieve(kb_id, question)
        stream = self.generate(question, sources, history)
        return sources, stream


# Singleton
rag_engine = RAGEngine()

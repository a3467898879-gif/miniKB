"""
Vector store — persistent vector storage and similarity search via ChromaDB.

ChromaDB is an open-source embedding database that stores vectors + metadata
and supports cosine similarity search out of the box.

Each knowledge base gets its own Chroma collection, isolating vectors
between different document sets — same pattern as MaxKB.
"""

from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings


class VectorStore:
    """Wraps ChromaDB for per-knowledge-base vector storage."""

    def __init__(self):
        self._client: Optional[chromadb.PersistentClient] = None

    @property
    def client(self) -> chromadb.PersistentClient:
        """Lazily initialize the ChromaDB persistent client."""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(settings.abs_chroma_path),
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
        return self._client

    def _collection_name(self, kb_id: str) -> str:
        """Chroma collection names can't contain special chars."""
        return f"kb_{kb_id.replace('-', '_')}"

    def get_or_create_collection(self, kb_id: str):
        """Get (or create) the Chroma collection for a knowledge base."""
        return self.client.get_or_create_collection(
            name=self._collection_name(kb_id),
            metadata={"hnsw:space": "cosine"},
        )

    def add_vectors(
        self,
        kb_id: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ):
        """Insert vectors with their text and metadata into the collection."""
        collection = self.get_or_create_collection(kb_id)
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        kb_id: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Search for the most similar chunks.

        Returns list of {content, metadata, score} sorted by relevance.
        """
        collection = self.get_or_create_collection(kb_id)
        if collection.count() == 0:
            return []
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        # Flatten results
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        return [
            {
                "content": doc,
                "metadata": meta,
                "score": 1 - dist,  # cosine distance → similarity score
            }
            for doc, meta, dist in zip(docs, metas, dists)
        ]

    def delete_collection(self, kb_id: str):
        """Delete all vectors for a knowledge base."""
        try:
            self.client.delete_collection(self._collection_name(kb_id))
        except Exception:
            pass  # Collection doesn't exist — nothing to delete

    def delete_document_vectors(self, kb_id: str, document_id: str):
        """Delete all vectors belonging to a specific document."""
        try:
            collection = self.get_or_create_collection(kb_id)
            collection.delete(where={"document_id": document_id})
        except Exception:
            pass

    def count(self, kb_id: str) -> int:
        """Return the number of vectors in a knowledge base."""
        try:
            return self.get_or_create_collection(kb_id).count()
        except Exception:
            return 0


# Singleton
vector_store = VectorStore()

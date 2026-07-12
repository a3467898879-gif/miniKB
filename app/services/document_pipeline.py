"""
Document pipeline — orchestrates: parse → chunk → embed → store.

This runs as a background task after file upload, mirroring MaxKB's
async document processing flow. Updates document status in DB as it progresses:
  processing → ready  (success)
  processing → error  (failure)
"""

import logging
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document, DocumentChunk
from app.services.document_loader import load_document
from app.services.text_splitter import split_text
from app.services.embedding import embedding_service
from app.services.vector_store import vector_store

logger = logging.getLogger(__name__)


async def process_document(
    db: AsyncSession,
    document: Document,
    file_path: Path,
):
    """
    Full document ingestion pipeline.

    Steps:
    1. Parse file → extract text
    2. Split text → overlapping chunks
    3. Embed chunks → vectors (batch API call)
    4. Store vectors + metadata in ChromaDB
    5. Save chunk metadata in SQLite
    6. Update document status to 'ready'
    """
    try:
        # ── Step 1: Parse ──
        logger.info(f"[Doc {document.id}] Parsing {document.name}...")
        text, file_type = load_document(file_path)

        # ── Step 2: Chunk ──
        logger.info(f"[Doc {document.id}] Splitting into chunks...")
        chunks = split_text(text)
        logger.info(f"[Doc {document.id}] Generated {len(chunks)} chunks")

        # ── Step 3: Embed ──
        logger.info(f"[Doc {document.id}] Embedding {len(chunks)} chunks...")
        embeddings = embedding_service.embed_batch(chunks)

        # ── Step 4: Store in vector DB ──
        logger.info(f"[Doc {document.id}] Storing vectors in ChromaDB...")
        chunk_ids = []
        metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document.id}_chunk_{i}"
            chunk_ids.append(chunk_id)
            metadatas.append({
                "document_id": document.id,
                "document_name": document.name,
                "chunk_index": i,
            })
        vector_store.add_vectors(
            kb_id=document.kb_id,
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        # ── Step 5: Save chunk metadata ──
        for i, chunk_text in enumerate(chunks):
            db_chunk = DocumentChunk(
                document_id=document.id,
                chunk_index=i,
                content=chunk_text,
                token_count=len(chunk_text),
            )
            db.add(db_chunk)

        # ── Step 6: Update document status ──
        document.status = "ready"
        document.chunk_count = len(chunks)
        await db.commit()
        logger.info(f"[Doc {document.id}] Processing complete: {len(chunks)} chunks ready")

    except Exception as e:
        logger.error(f"[Doc {document.id}] Processing failed: {e}", exc_info=True)
        document.status = "error"
        document.error_message = str(e)
        await db.commit()

"""
System endpoints — health check, model config, dashboard stats.

  GET  /api/system/health    — health check
  GET  /api/system/config    — current model config (masks API keys)
  GET  /api/system/stats     — dashboard statistics
"""

from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import KnowledgeBase, Document, ChatSession, ChatMessage
from config import settings

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/health")
async def health():
    """Health check — used by Docker and load balancers."""
    return {"status": "ok", "version": "1.0.0"}


@router.get("/config")
async def get_config():
    """Return current model configuration (API keys are masked)."""
    def mask(key: str) -> str:
        if not key or len(key) <= 8:
            return "***" if key else ""
        return key[:4] + "***" + key[-4:]

    return {
        "llm": {
            "model": settings.llm_model,
            "base_url": settings.llm_base_url,
            "api_key": mask(settings.llm_api_key),
            "configured": bool(settings.llm_api_key),
        },
        "embedding": {
            "model": settings.embedding_model,
            "base_url": settings.embedding_base_url,
            "api_key": mask(settings.embedding_api_key or settings.llm_api_key),
            "configured": bool(settings.embedding_api_key or settings.llm_api_key),
        },
        "rag": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": settings.top_k,
        },
    }


@router.get("/stats")
async def stats(db: AsyncSession = Depends(get_db)):
    """Dashboard statistics."""
    kb_count = await db.scalar(select(func.count(KnowledgeBase.id)))
    doc_count = await db.scalar(select(func.count(Document.id)))
    ready_docs = await db.scalar(
        select(func.count(Document.id)).where(Document.status == "ready")
    )
    chat_count = await db.scalar(select(func.count(ChatSession.id)))
    msg_count = await db.scalar(select(func.count(ChatMessage.id)))
    total_chunks = await db.scalar(select(func.sum(Document.chunk_count)))

    return {
        "knowledge_bases": kb_count or 0,
        "documents": doc_count or 0,
        "ready_documents": ready_docs or 0,
        "total_chunks": total_chunks or 0,
        "chat_sessions": chat_count or 0,
        "chat_messages": msg_count or 0,
    }

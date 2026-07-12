"""
Pydantic schemas for request validation and response serialization.

Separation from ORM models follows the "DTO pattern" — API contracts
are decoupled from database schema so they can evolve independently.
"""

from datetime import datetime
from pydantic import BaseModel, Field


# ── Knowledge Base ──────────────────────────────

class KBCreate(BaseModel):
    name: str = Field(..., max_length=200)
    description: str = ""


class KBUpdate(BaseModel):
    name: str | None = Field(None, max_length=200)
    description: str | None = None


class KBOut(BaseModel):
    id: str
    name: str
    description: str
    embedding_model: str
    document_count: int = 0
    created_at: datetime

    class Config:
        from_attributes = True


# ── Document ────────────────────────────────────

class DocumentOut(BaseModel):
    id: str
    kb_id: str
    name: str
    file_type: str
    file_size: int
    status: str
    chunk_count: int
    error_message: str = ""
    created_at: datetime

    class Config:
        from_attributes = True


# ── Chat ────────────────────────────────────────

class ChatCreate(BaseModel):
    kb_id: str
    title: str = "New Chat"


class ChatOut(BaseModel):
    id: str
    kb_id: str
    title: str
    created_at: datetime

    class Config:
        from_attributes = True


class MessageOut(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    sources: dict
    created_at: datetime

    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    """User sends a question; server streams the answer via SSE."""
    message: str


# ── Model Config ────────────────────────────────

class ModelConfigSchema(BaseModel):
    """In-memory model config — persisted to .env or set via environment."""
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-chat"
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_base_url: str = "https://api.deepseek.com/v1"
    embedding_api_key: str = ""
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5

"""
Knowledge base CRUD + document upload endpoints.

RESTful design:
  POST   /api/kb              — create KB
  GET    /api/kb              — list all KBs
  GET    /api/kb/{id}         — get KB detail
  PUT    /api/kb/{id}         — update KB
  DELETE /api/kb/{id}         — delete KB (+ all docs + vectors)
  POST   /api/kb/{id}/upload  — upload document to KB
  GET    /api/kb/{id}/docs    — list documents in KB
  DELETE /api/kb/{id}/docs/{doc_id} — delete a document
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from pathlib import Path
import logging

from app.database import get_db
from app.models import KnowledgeBase, Document
from app.schemas import KBCreate, KBUpdate, KBOut, DocumentOut
from app.services.vector_store import vector_store
from app.services.document_pipeline import process_document
from app.services.document_loader import SUPPORTED_EXTENSIONS
from config import settings

router = APIRouter(prefix="/api/kb", tags=["knowledge"])
logger = logging.getLogger(__name__)


@router.post("", response_model=KBOut)
async def create_kb(body: KBCreate, db: AsyncSession = Depends(get_db)):
    """Create a new knowledge base."""
    kb = KnowledgeBase(
        name=body.name,
        description=body.description,
        embedding_model=settings.embedding_model,
    )
    db.add(kb)
    await db.commit()
    await db.refresh(kb)
    return KBOut.model_validate(kb, from_attributes=True)


@router.get("", response_model=list[KBOut])
async def list_kbs(db: AsyncSession = Depends(get_db)):
    """List all knowledge bases with document count."""
    result = await db.execute(
        select(
            KnowledgeBase,
            func.count(Document.id).label("doc_count"),
        )
        .outerjoin(Document, Document.kb_id == KnowledgeBase.id)
        .group_by(KnowledgeBase.id)
        .order_by(KnowledgeBase.created_at.desc())
    )
    rows = result.all()
    out = []
    for kb, doc_count in rows:
        item = KBOut.model_validate(kb, from_attributes=True)
        item.document_count = doc_count
        out.append(item)
    return out


@router.get("/{kb_id}", response_model=KBOut)
async def get_kb(kb_id: str, db: AsyncSession = Depends(get_db)):
    """Get a single knowledge base."""
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")
    item = KBOut.model_validate(kb, from_attributes=True)
    doc_count = await db.scalar(
        select(func.count(Document.id)).where(Document.kb_id == kb_id)
    )
    item.document_count = doc_count or 0
    return item


@router.put("/{kb_id}", response_model=KBOut)
async def update_kb(kb_id: str, body: KBUpdate, db: AsyncSession = Depends(get_db)):
    """Update knowledge base name/description."""
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")
    if body.name is not None:
        kb.name = body.name
    if body.description is not None:
        kb.description = body.description
    await db.commit()
    await db.refresh(kb)
    return KBOut.model_validate(kb, from_attributes=True)


@router.delete("/{kb_id}")
async def delete_kb(kb_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a knowledge base and all its documents + vectors."""
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")
    # Delete vectors from ChromaDB
    vector_store.delete_collection(kb_id)
    # Delete from SQLite (cascade handles docs + chunks)
    await db.delete(kb)
    await db.commit()
    return {"message": "Knowledge base deleted"}


@router.post("/{kb_id}/upload", response_model=DocumentOut)
async def upload_document(
    kb_id: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a document to a knowledge base.

    The file is saved to disk, a Document record is created, and a
    background task processes it (parse → chunk → embed → store).
    """
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")

    # Validate file type
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported file type: {suffix}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # Save file to disk
    file_path = settings.abs_upload_dir / f"{kb_id}_{file.filename}"
    content = await file.read()
    file_path.write_bytes(content)

    # Create document record
    doc = Document(
        kb_id=kb_id,
        name=file.filename,
        file_type=suffix.lstrip("."),
        file_size=len(content),
        status="processing",
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    # Process document (inline for simplicity; in production use BackgroundTasks)
    await process_document(db, doc, file_path)

    return DocumentOut.model_validate(doc, from_attributes=True)


@router.get("/{kb_id}/docs", response_model=list[DocumentOut])
async def list_documents(kb_id: str, db: AsyncSession = Depends(get_db)):
    """List all documents in a knowledge base."""
    result = await db.execute(
        select(Document)
        .where(Document.kb_id == kb_id)
        .order_by(Document.created_at.desc())
    )
    return [DocumentOut.model_validate(d, from_attributes=True) for d in result.scalars()]


@router.delete("/{kb_id}/docs/{doc_id}")
async def delete_document(kb_id: str, doc_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a document and its vectors."""
    doc = await db.get(Document, doc_id)
    if not doc or doc.kb_id != kb_id:
        raise HTTPException(404, "Document not found")
    # Delete vectors from ChromaDB
    vector_store.delete_document_vectors(kb_id, doc_id)
    # Delete from SQLite
    await db.delete(doc)
    await db.commit()
    return {"message": "Document deleted"}

"""
Chat endpoints — conversation management + SSE streaming.

RESTful design:
  POST   /api/chat                    — create chat session
  GET    /api/chat?kb_id={id}         — list sessions for a KB
  GET    /api/chat/{session_id}       — get session + messages
  DELETE /api/chat/{session_id}       — delete session
  POST   /api/chat/{session_id}/send  — send message, stream response via SSE
"""

import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from app.database import get_db
from app.models import ChatSession, ChatMessage, KnowledgeBase, Document
from app.schemas import ChatCreate, ChatOut, MessageOut, ChatRequest
from app.services.rag_engine import rag_engine

router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("", response_model=ChatOut)
async def create_chat(body: ChatCreate, db: AsyncSession = Depends(get_db)):
    """Create a new chat session for a knowledge base."""
    kb = await db.get(KnowledgeBase, body.kb_id)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")
    session = ChatSession(kb_id=body.kb_id, title=body.title)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return ChatOut.model_validate(session, from_attributes=True)


@router.get("", response_model=list[ChatOut])
async def list_chats(kb_id: str, db: AsyncSession = Depends(get_db)):
    """List chat sessions for a knowledge base."""
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.kb_id == kb_id)
        .order_by(ChatSession.created_at.desc())
    )
    return [ChatOut.model_validate(s, from_attributes=True) for s in result.scalars()]


@router.get("/{session_id}")
async def get_chat(session_id: str, db: AsyncSession = Depends(get_db)):
    """Get a chat session with all messages."""
    session = await db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(404, "Chat session not found")
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
    )
    messages = result.scalars().all()
    return {
        "session": ChatOut.model_validate(session, from_attributes=True),
        "messages": [MessageOut.model_validate(m, from_attributes=True) for m in messages],
    }


@router.delete("/{session_id}")
async def delete_chat(session_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a chat session and all its messages."""
    session = await db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(404, "Chat session not found")
    await db.delete(session)
    await db.commit()
    return {"message": "Chat session deleted"}


@router.post("/{session_id}/send")
async def send_message(
    session_id: str,
    body: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Send a message and stream the RAG response via Server-Sent Events.

    SSE format:
      data: {"type": "sources", "content": [...]}     ← retrieved chunks
      data: {"type": "token", "content": "..."}        ← LLM token
      data: {"type": "done", "content": "..."}         ← full response + saved msg

    The user message and assistant response are persisted to the database.
    """
    session = await db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(404, "Chat session not found")

    # Load conversation history
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
    )
    db_messages = result.scalars().all()
    history = [{"role": m.role, "content": m.content} for m in db_messages]

    # Save user message
    user_msg = ChatMessage(
        session_id=session_id,
        role="user",
        content=body.message,
        sources={},
    )
    db.add(user_msg)
    await db.commit()

    # Update session title if it's the first message
    if len(db_messages) == 0 and session.title == "New Chat":
        session.title = body.message[:50]
        await db.commit()

    async def event_stream():
        """Generate SSE events for the streaming response."""
        try:
            # Step 1: Retrieve relevant chunks
            sources = rag_engine.retrieve(session.kb_id, body.message)

            # Send sources first
            sources_data = [
                {
                    "content": s["content"][:200],
                    "document_name": s.get("metadata", {}).get("document_name", ""),
                    "score": round(s.get("score", 0), 4),
                }
                for s in sources
            ]
            yield f"data: {json.dumps({'type': 'sources', 'content': sources_data})}\n\n"

            # Step 2: Stream LLM response
            full_response = ""
            messages = rag_engine.generate(body.message, sources, history)
            for token in messages:
                full_response += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            # Step 3: Save assistant message and send done
            assistant_msg = ChatMessage(
                session_id=session_id,
                role="assistant",
                content=full_response,
                sources={"chunks": sources_data},
            )
            db.add(assistant_msg)
            await db.commit()

            yield f"data: {json.dumps({'type': 'done', 'content': full_response})}\n\n"

        except Exception as e:
            logger.error(f"Chat stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

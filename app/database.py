"""
Async database engine and session factory.

Uses SQLAlchemy 2.0 async API with aiosqlite for zero-config SQLite.
Session-scoped dependency injection via FastAPI's Depends.
"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from config import settings

engine = create_async_engine(settings.db_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    """Declarative base — all ORM models inherit from this."""
    pass


async def get_db() -> AsyncSession:
    """FastAPI dependency: yields an async DB session per request."""
    async with async_session() as session:
        yield session


async def init_db():
    """Create all tables. Call once on startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

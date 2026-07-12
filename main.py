"""
FastAPI application entry point.

Wires up:
  - Database initialization (SQLite table creation)
  - Static file serving (frontend UI)
  - API routers (knowledge, chat, system)
  - CORS middleware (for local development)
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db
from app.routers import knowledge, chat, system

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "app" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: init DB on startup."""
    logger.info("Initializing database...")
    await init_db()
    logger.info("Database ready. miniKB is up.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="miniKB",
    description="Mini RAG Knowledge Base — learn & demonstrate RAG architecture",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS (permissive for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routers (must be registered BEFORE static file mount)
app.include_router(system.router)
app.include_router(knowledge.router)
app.include_router(chat.router)

# Static files (CSS/JS) — mounted under /static so they don't shadow API routes
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="assets")


@app.get("/")
async def index():
    """Serve the SPA index.html at root."""
    return FileResponse(STATIC_DIR / "index.html")


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """
    SPA fallback: return index.html for non-API 404s.
    API 404s still return JSON.
    """
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": "Not Found"}, status_code=404)
    # Check if it's a static file (css/js)
    file_path = STATIC_DIR / request.url.path.lstrip("/")
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    # Fallback to index.html for SPA routing
    return FileResponse(STATIC_DIR / "index.html")

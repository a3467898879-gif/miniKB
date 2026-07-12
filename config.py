"""
miniKB configuration — loads from .env file or environment variables.

Design: uses pydantic-settings for type-safe config with defaults,
so the app works out-of-the-box once a .env file is in place.
"""

from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    # --- LLM ---
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"

    # --- Embedding ---
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.deepseek.com/v1"
    embedding_model: str = "text-embedding-3-small"

    # --- App ---
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # --- Storage ---
    db_path: str = "data/miniKB.db"
    chroma_path: str = "data/chroma"
    upload_dir: str = "data/uploads"

    # --- RAG Parameters ---
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def db_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.db_path}"

    @property
    def abs_upload_dir(self) -> Path:
        p = BASE_DIR / self.upload_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def abs_chroma_path(self) -> Path:
        p = BASE_DIR / self.chroma_path
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()

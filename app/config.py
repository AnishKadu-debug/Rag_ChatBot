"""Application configuration module.

This module is responsible for loading environment variables from a `.env` file
and providing typed, validated settings across the application.

Keeping configuration separate from application logic ensures:
1. Secrets (API keys) are never hard-coded in source code.
2. Settings can be easily adjusted across environments (local, dev, production).
3. Type hints and default values prevent missing configuration bugs.
"""

from functools import lru_cache
import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Base directory of the repository (parent of the 'app' directory)
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Load variables from .env file into os.environ if present
load_dotenv(dotenv_path=BASE_DIR / ".env")


class Settings:
    """Application settings and configuration parameters."""

    # Application metadata
    API_TITLE: str = "WhatsApp Land Records RAG Chatbot API"
    API_VERSION: str = "0.1.0"
    API_DESCRIPTION: str = (
        "REST API service for the WhatsApp Land Records Document Chatbot. "
        "Extracts, indexes, and answers questions on Marathi/English land records."
    )

    # Security & Authentication
    API_KEY: str = os.getenv("API_KEY", "change-me")
    ALLOWED_ORIGINS_RAW: str = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )

    # LLM Settings (Phase 1+)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Embedding & Vector Database Settings (Phase 1+)
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "intfloat/multilingual-e5-base"
    )
    CHROMA_DIR: str = os.getenv("CHROMA_DIR", str(BASE_DIR / "data" / "chroma"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    TOP_K: int = int(os.getenv("TOP_K", "4"))
    SIMILARITY_DISTANCE_THRESHOLD: float = float(
        os.getenv("SIMILARITY_DISTANCE_THRESHOLD", "0.45")
    )

    # Upload Constraints (Phase 2+)
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "10"))
    MAX_PAGES: int = int(os.getenv("MAX_PAGES", "30"))

    # Document Metadata Database (Phase 2+)
    DOCS_DB_PATH: str = os.getenv(
        "DOCS_DB_PATH", str(BASE_DIR / "data" / "documents.sqlite3")
    )

    # OCR Languages (Phase 3+)
    OCR_LANGS: str = os.getenv("OCR_LANGS", "mar+eng")

    @property
    def allowed_origins(self) -> List[str]:
        """Convert comma-separated allowed origins string into a clean list of URLs."""
        return [
            origin.strip()
            for origin in self.ALLOWED_ORIGINS_RAW.split(",")
            if origin.strip()
        ]


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton instance of the application settings."""
    return Settings()

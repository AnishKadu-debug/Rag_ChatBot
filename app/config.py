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
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "grok")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "grok-3-mini")
    GROK_API_KEY: str = os.getenv("GROK_API_KEY", "")
    GROK_BASE_URL: str = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = os.getenv(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "1"))

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

    # Administrator Ingestion Limits for Official Land Records (500+ pages)
    OFFICIAL_DOCUMENT_ID: str = os.getenv("OFFICIAL_DOCUMENT_ID", "")
    ADMIN_MAX_UPLOAD_MB: int = int(os.getenv("ADMIN_MAX_UPLOAD_MB", "100"))
    ADMIN_MAX_PAGES: int = int(os.getenv("ADMIN_MAX_PAGES", "1000"))

    # Document Metadata Database (Phase 2+)
    DOCS_DB_PATH: str = os.getenv(
        "DOCS_DB_PATH", str(BASE_DIR / "data" / "documents.sqlite3")
    )

    # OCR Settings (Phase 3)
    OCR_ENABLED: bool = os.getenv("OCR_ENABLED", "true").lower() in ("true", "1", "yes")
    OCR_LANGS: str = os.getenv("OCR_LANGS", "mar+eng")
    OCR_THRESHOLD_CHARS: int = int(os.getenv("OCR_THRESHOLD_CHARS", "30"))
    OCR_DPI: int = int(os.getenv("OCR_DPI", "300"))
    TESSERACT_CMD: str = os.getenv("TESSERACT_CMD", "")

    def __init__(self) -> None:
        """Initialize settings from current os.environ."""
        self.API_KEY = os.getenv("API_KEY", "change-me")
        self.ALLOWED_ORIGINS_RAW = os.getenv(
            "ALLOWED_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000",
        )
        self.LLM_PROVIDER = os.getenv("LLM_PROVIDER", "grok")
        self.LLM_MODEL = os.getenv("LLM_MODEL", "grok-3-mini")
        self.GROK_API_KEY = os.getenv("GROK_API_KEY", "")
        self.GROK_BASE_URL = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1")
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
        self.GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
        self.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
        self.OPENROUTER_BASE_URL = os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
        self.LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
        self.LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "1"))

        self.EMBEDDING_MODEL = os.getenv(
            "EMBEDDING_MODEL", "intfloat/multilingual-e5-base"
        )
        self.CHROMA_DIR = os.getenv("CHROMA_DIR", str(BASE_DIR / "data" / "chroma"))
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
        self.TOP_K = int(os.getenv("TOP_K", "4"))
        self.SIMILARITY_DISTANCE_THRESHOLD = float(
            os.getenv("SIMILARITY_DISTANCE_THRESHOLD", "0.45")
        )

        self.MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
        self.MAX_PAGES = int(os.getenv("MAX_PAGES", "30"))

        self.OFFICIAL_DOCUMENT_ID = os.getenv("OFFICIAL_DOCUMENT_ID", "")
        self.ADMIN_MAX_UPLOAD_MB = int(os.getenv("ADMIN_MAX_UPLOAD_MB", "100"))
        self.ADMIN_MAX_PAGES = int(os.getenv("ADMIN_MAX_PAGES", "1000"))

        self.DOCS_DB_PATH = os.getenv(
            "DOCS_DB_PATH", str(BASE_DIR / "data" / "documents.sqlite3")
        )

        self.OCR_ENABLED = os.getenv("OCR_ENABLED", "true").lower() in ("true", "1", "yes")
        self.OCR_LANGS = os.getenv("OCR_LANGS", "mar+eng")
        self.OCR_THRESHOLD_CHARS = int(os.getenv("OCR_THRESHOLD_CHARS", "30"))
        self.OCR_DPI = int(os.getenv("OCR_DPI", "300"))
        self.TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")

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

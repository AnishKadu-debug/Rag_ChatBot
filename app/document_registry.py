"""Persistent SQLite document registry for tracking uploaded document metadata.

Why a Separate Document Registry?
In a Retrieval-Augmented Generation (RAG) system:
1. ChromaDB (Vector Store): Stores high-dimensional vector embeddings and chunks for
   approximate nearest neighbor (ANN) similarity search.
2. SQLite (Relational Registry): Stores structured relational metadata about documents
   (such as doc_id, user_id, filename, total pages, upload timestamp, and OCR flags).

Keeping them distinct allows fast relational lookups, document listings, ownership
verification, and clean metadata tracking that survives server restarts.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import logging
from pathlib import Path
import sqlite3
from typing import List, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    """Represents a stored document metadata record."""

    doc_id: str
    user_id: str
    filename: str
    pages: int
    uploaded_at: str
    ocr_used: bool = False


class DocumentRegistry:
    """Manages persistent SQLite storage of document records with user isolation."""

    def __init__(self, db_path: Optional[str] = None):
        settings = get_settings()
        self.db_path = db_path or settings.DOCS_DB_PATH

        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and return a configured SQLite connection."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize the SQLite database schema if not present."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    pages INTEGER NOT NULL,
                    uploaded_at TEXT NOT NULL,
                    ocr_used INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_user_id
                ON documents (user_id)
                """
            )
            conn.commit()

    def add_document(
        self,
        doc_id: str,
        user_id: str,
        filename: str,
        pages: int,
        ocr_used: bool = False,
        uploaded_at: Optional[str] = None,
    ) -> DocumentRecord:
        """Insert or replace a document metadata record.

        Args:
            doc_id: Unique identifier for the document.
            user_id: Tenant user identifier.
            filename: Original PDF filename.
            pages: Total page count.
            ocr_used: Whether OCR was used during extraction.
            uploaded_at: ISO-8601 timestamp string (auto-generated if None).

        Returns:
            DocumentRecord: The created document record.
        """
        timestamp = uploaded_at or datetime.now(timezone.utc).isoformat()
        ocr_int = 1 if ocr_used else 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO documents (doc_id, user_id, filename, pages, uploaded_at, ocr_used)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (doc_id, user_id, filename, pages, timestamp, ocr_int),
            )
            conn.commit()

        return DocumentRecord(
            doc_id=doc_id,
            user_id=user_id,
            filename=filename,
            pages=pages,
            uploaded_at=timestamp,
            ocr_used=ocr_used,
        )

    def get_document(
        self,
        doc_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[DocumentRecord]:
        """Fetch a document by its doc_id, optionally verifying user ownership.

        Args:
            doc_id: Target document ID.
            user_id: Optional user identifier to enforce ownership.

        Returns:
            Optional[DocumentRecord]: Found document or None.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if user_id is not None:
                cursor.execute(
                    """
                    SELECT doc_id, user_id, filename, pages, uploaded_at, ocr_used
                    FROM documents
                    WHERE doc_id = ? AND user_id = ?
                    """,
                    (doc_id, user_id),
                )
            else:
                cursor.execute(
                    """
                    SELECT doc_id, user_id, filename, pages, uploaded_at, ocr_used
                    FROM documents
                    WHERE doc_id = ?
                    """,
                    (doc_id,),
                )
            row = cursor.fetchone()

        if not row:
            return None

        return DocumentRecord(
            doc_id=row["doc_id"],
            user_id=row["user_id"],
            filename=row["filename"],
            pages=int(row["pages"]),
            uploaded_at=row["uploaded_at"],
            ocr_used=bool(row["ocr_used"]),
        )

    def list_documents(self, user_id: str) -> List[DocumentRecord]:
        """List all documents owned by a specific user.

        Args:
            user_id: Identifier of the owner.

        Returns:
            List[DocumentRecord]: List of document records owned by the user.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT doc_id, user_id, filename, pages, uploaded_at, ocr_used
                FROM documents
                WHERE user_id = ?
                ORDER BY uploaded_at DESC
                """,
                (user_id,),
            )
            rows = cursor.fetchall()

        return [
            DocumentRecord(
                doc_id=row["doc_id"],
                user_id=row["user_id"],
                filename=row["filename"],
                pages=int(row["pages"]),
                uploaded_at=row["uploaded_at"],
                ocr_used=bool(row["ocr_used"]),
            )
            for row in rows
        ]

    def delete_document(self, doc_id: str, user_id: str) -> bool:
        """Delete a document record enforcing user ownership.

        Args:
            doc_id: Identifier of document to delete.
            user_id: Identifier of requesting user.

        Returns:
            bool: True if the record was found and deleted, False otherwise.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM documents
                WHERE doc_id = ? AND user_id = ?
                """,
                (doc_id, user_id),
            )
            conn.commit()
            return cursor.rowcount > 0


@lru_cache()
def get_document_registry() -> DocumentRegistry:
    """Return a cached singleton instance of DocumentRegistry."""
    return DocumentRegistry()

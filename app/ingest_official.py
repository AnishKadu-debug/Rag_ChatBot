"""Administrator ingestion module for the official Maharashtra Land Records PDF.

This module provides the ingestion pipeline and CLI tool to index or replace the official
large document (approximately 500 pages) into ChromaDB and the SQLite document registry.

Key Design Principles:
1. Administrator-Configured Limits: Uses ADMIN_MAX_PAGES and ADMIN_MAX_UPLOAD_MB
   (default 1000 pages / 100 MB) instead of public end-user limits (30 pages / 10 MB).
2. Memory Safety & Batching: Chunks are embedded and stored in batches (e.g. 64 chunks)
   to prevent excessive memory consumption on 500-page documents.
3. Safe Replacement: When replacing an official document, previous official vectors are
   purged from ChromaDB to prevent duplicate or stale chunks from being retrieved.
4. Transactional Integrity: If indexing fails at any point, newly added chunks are purged
   from ChromaDB and the record is marked 'failed', preventing partially indexed documents
   from ever being reported as ready.
5. Fixed Isolation: Official document chunks are stored with user_id="official" and tagged
   is_official=1, ensuring end-user queries never bleed into or out of other collections.
"""

import argparse
import logging
import os
from pathlib import Path
import sys
from typing import List, Optional, Union
import uuid

import pymupdf

from app.chunker import chunk_pages
from app.config import get_settings
from app.document_registry import DocumentRecord, DocumentRegistry, get_document_registry
from app.embedder import E5Embedder, get_embedder
from app.pdf_loader import extract_text_from_pdf
from app.vector_store import ChromaVectorStore, get_vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest_official")


class IngestionError(Exception):
    """Raised when official document ingestion fails."""
    pass


def validate_admin_pdf(
    pdf_source: Union[str, Path, bytes],
    max_mb: Optional[int] = None,
    max_pages: Optional[int] = None,
) -> bytes:
    """Validate PDF file size, signature, and page count for administrator uploads.

    Args:
        pdf_source: Path to file or raw bytes.
        max_mb: Optional maximum MB ceiling (defaults to settings.ADMIN_MAX_UPLOAD_MB).
        max_pages: Optional maximum page count (defaults to settings.ADMIN_MAX_PAGES).

    Returns:
        bytes: Validated raw PDF bytes.

    Raises:
        IngestionError: If file is invalid, corrupt, or exceeds admin limits.
    """
    settings = get_settings()
    limit_mb = max_mb if max_mb is not None else settings.ADMIN_MAX_UPLOAD_MB
    limit_pages = max_pages if max_pages is not None else settings.ADMIN_MAX_PAGES
    max_bytes = limit_mb * 1024 * 1024

    if isinstance(pdf_source, (str, Path)):
        path = Path(pdf_source)
        if not path.is_file():
            raise IngestionError(f"File not found: {path}")
        try:
            content = path.read_bytes()
        except Exception as exc:
            raise IngestionError(f"Failed to read file '{path}': {exc}") from exc
    else:
        content = bytes(pdf_source)

    if len(content) == 0:
        raise IngestionError("PDF document is empty (0 bytes).")

    if len(content) > max_bytes:
        raise IngestionError(
            f"PDF file size ({len(content) / (1024*1024):.1f} MB) exceeds administrator limit of {limit_mb} MB."
        )

    if not content.startswith(b"%PDF"):
        raise IngestionError("Invalid file signature: file is not a valid PDF document.")

    try:
        doc = pymupdf.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise IngestionError(f"Failed to parse PDF document structure: {exc}") from exc

    try:
        total_pages = len(doc)
        if total_pages == 0:
            raise IngestionError("PDF document contains 0 pages.")
        if total_pages > limit_pages:
            raise IngestionError(
                f"PDF document has {total_pages} pages, which exceeds administrator limit of {limit_pages} pages."
            )
    finally:
        doc.close()

    return content


def ingest_official_pdf(
    pdf_source: Union[str, Path, bytes],
    filename: str = "official_land_records.pdf",
    replace: bool = True,
    chunk_batch_size: int = 64,
    registry: Optional[DocumentRegistry] = None,
    vector_store: Optional[ChromaVectorStore] = None,
    embedder: Optional[E5Embedder] = None,
) -> DocumentRecord:
    """Ingest and index an official Maharashtra land records PDF.

    Args:
        pdf_source: Path to PDF file or raw PDF bytes.
        filename: Human-readable filename for registry and chunk metadata.
        replace: If True, deactivates and purges previous official document vectors.
        chunk_batch_size: Batch size for computing embeddings and storing in ChromaDB.
        registry: Optional injected DocumentRegistry.
        vector_store: Optional injected ChromaVectorStore.
        embedder: Optional injected E5Embedder.

    Returns:
        DocumentRecord: Registered official document record with status='ready'.

    Raises:
        IngestionError: If validation, extraction, chunking, or embedding fails.
    """
    settings = get_settings()
    reg = registry or get_document_registry()
    vstore = vector_store or get_vector_store()
    emb = embedder or get_embedder()

    logger.info("Starting ingestion of official document '%s'...", filename)

    # 1. Validate PDF against administrator limits
    pdf_bytes = validate_admin_pdf(pdf_source)

    # 2. Extract page-by-page text with OCR fallback
    logger.info("Extracting text page-by-page...")
    try:
        pages = extract_text_from_pdf(pdf_bytes)
    except Exception as exc:
        logger.error("Text extraction failed: %s", exc)
        raise IngestionError(f"Failed to extract text from PDF: {exc}") from exc

    total_pages = len(pages)
    ocr_count = sum(1 for p in pages if p.ocr_used)
    doc_ocr_used = ocr_count > 0
    logger.info(
        "Extracted %d pages successfully (%d pages used OCR fallback).",
        total_pages,
        ocr_count,
    )

    # 3. Create unique official doc ID and register initial 'indexing' state
    doc_id = f"d_official_{uuid.uuid4().hex[:10]}"
    official_user_id = "official"

    reg.add_document(
        doc_id=doc_id,
        user_id=official_user_id,
        filename=filename,
        pages=total_pages,
        ocr_used=doc_ocr_used,
        is_official=False,  # Not marked official until indexing finishes successfully
        status="indexing",
    )

    # 4. Chunk text while strictly preserving 1-based page metadata
    chunks = chunk_pages(
        pages=pages,
        doc_id=doc_id,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    total_chunks = len(chunks)
    logger.info("Generated %d chunks across %d pages.", total_chunks, total_pages)
    reg.set_document_chunks(doc_id=doc_id, chunks=total_chunks)

    if total_chunks == 0:
        reg.set_document_status(doc_id=doc_id, status="failed")
        raise IngestionError("No text chunks could be extracted from the document.")

    # 5. Process and persist embeddings in manageable batches
    try:
        for start_idx in range(0, total_chunks, chunk_batch_size):
            end_idx = min(start_idx + chunk_batch_size, total_chunks)
            batch_chunks = chunks[start_idx:end_idx]
            batch_texts = [c.text for c in batch_chunks]

            logger.info(
                "Embedding chunks %d to %d of %d (%.1f%%)...",
                start_idx + 1,
                end_idx,
                total_chunks,
                (end_idx / total_chunks) * 100.0,
            )

            batch_embeddings = emb.embed_passages(batch_texts, batch_size=chunk_batch_size)

            vstore.add_chunks(
                user_id=official_user_id,
                chunks=batch_chunks,
                embeddings=batch_embeddings,
                filename=filename,
            )

        # 6. If replace is requested, safely purge previous official vectors
        if replace:
            logger.info("Purging old official document vectors from ChromaDB...")
            old_docs = reg.list_documents(user_id=official_user_id)
            for old in old_docs:
                if old.doc_id != doc_id and old.is_official:
                    logger.info("Deleting stale vectors for previous official doc '%s'...", old.doc_id)
                    try:
                        vstore.delete_document(user_id=official_user_id, doc_id=old.doc_id)
                    except Exception as del_exc:
                        logger.warning("Error deleting stale vectors for %s: %s", old.doc_id, del_exc)

            # Deactivate previous official documents in registry
            reg.set_official_document(doc_id=doc_id, is_official=True, deactivate_others=True)
        else:
            reg.set_official_document(doc_id=doc_id, is_official=True, deactivate_others=False)

        # Mark document status as 'ready'
        reg.set_document_status(doc_id=doc_id, status="ready")
        logger.info(
            "Official document '%s' successfully indexed (ID: %s, %d pages, %d chunks).",
            filename,
            doc_id,
            total_pages,
            total_chunks,
        )

        record = reg.get_document(doc_id=doc_id)
        if not record:
            raise IngestionError(f"Failed to retrieve registered document record for {doc_id}")
        return record

    except Exception as exc:
        logger.error("Ingestion failed during chunk embedding/storage: %s", exc)
        # Roll back: purge all vectors already added for this doc_id
        try:
            logger.info("Cleaning up partial vectors for failed document %s...", doc_id)
            vstore.delete_document(user_id=official_user_id, doc_id=doc_id)
        except Exception as cleanup_exc:
            logger.warning("Vector store cleanup failed: %s", cleanup_exc)

        reg.set_document_status(doc_id=doc_id, status="failed")
        raise IngestionError(f"Indexing failed: {exc}") from exc


def main() -> None:
    """CLI entry point for administrator document ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest and index the official Maharashtra land records PDF (approx. 500 pages)."
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to the official PDF file on disk.",
    )
    parser.add_argument(
        "--no-replace",
        action="store_true",
        help="Do not deactivate or purge existing official document vectors.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Chunk embedding batch size (default: 64).",
    )

    args = parser.parse_args()
    pdf_path = Path(args.pdf_path)

    if not pdf_path.is_file():
        logger.error("Error: Specified file '%s' does not exist.", pdf_path)
        sys.exit(1)

    try:
        record = ingest_official_pdf(
            pdf_source=pdf_path,
            filename=pdf_path.name,
            replace=not args.no_replace,
            chunk_batch_size=args.batch_size,
        )
        print("\n========================================================")
        print(" SUCCESS: Official Land Records PDF Ingested")
        print("========================================================")
        print(f" Document ID   : {record.doc_id}")
        print(f" Filename      : {record.filename}")
        print(f" Total Pages   : {record.pages}")
        print(f" OCR Applied   : {record.ocr_used}")
        print(f" Status        : {record.status}")
        print(f" Official Doc  : {record.is_official}")
        print("========================================================\n")
    except IngestionError as exc:
        logger.error("Ingestion failed: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error during ingestion: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

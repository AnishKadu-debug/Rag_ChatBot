"""Security, authentication, and file validation module.

This module provides:
1. API Key Authentication: Protects all document and chat routes by verifying the
   'X-API-Key' request header against the configured application secret.
2. File Upload Validation: Strictly validates uploaded PDF files to protect against
   security risks (such as malicious file uploads, decompression bombs, or corrupted files):
   - Content-Type / MIME checking.
   - Header magic-bytes validation (must begin with '%PDF').
   - File size ceiling (MAX_UPLOAD_MB, default 10MB).
   - Page count limit (MAX_PAGES, default 30 pages).
"""

import logging
from typing import Optional
from fastapi import HTTPException, Security, UploadFile, status
from fastapi.security.api_key import APIKeyHeader
import pymupdf

from app.config import get_settings

logger = logging.getLogger(__name__)

# FastAPI APIKeyHeader extractor.
# auto_error=False allows us to return explicit 401 Unauthorized status with a custom error message
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """Validate incoming request API key against the server configuration.

    Args:
        api_key: Value extracted from 'X-API-Key' header.

    Returns:
        str: Validated API key string.

    Raises:
        HTTPException: 401 Unauthorized if missing or incorrect.
    """
    settings = get_settings()

    if not api_key or api_key.strip() != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )

    return api_key


async def validate_pdf_upload(
    file: UploadFile,
    max_mb: Optional[int] = None,
    max_pages: Optional[int] = None,
) -> bytes:
    """Validate uploaded file for PDF compliance, size limits, and page count.

    Args:
        file: Uploaded file handle from FastAPI.
        max_mb: Optional maximum size override in megabytes.
        max_pages: Optional maximum page count override.

    Returns:
        bytes: Raw validated PDF file content.

    Raises:
        HTTPException:
            - 400 Bad Request for non-PDF, empty, corrupted, or page-exceeded files.
            - 413 Request Entity Too Large if byte size exceeds maximum allowed MB.
    """
    settings = get_settings()
    limit_mb = max_mb if max_mb is not None else settings.MAX_UPLOAD_MB
    limit_pages = max_pages if max_pages is not None else settings.MAX_PAGES
    max_bytes = limit_mb * 1024 * 1024

    filename = file.filename or "unknown"
    content_type = file.content_type or ""

    # 1. Content-Type and filename extension check
    if not filename.lower().endswith(".pdf") and content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF documents are supported.",
        )

    # 2. Read file content into memory and check size limit
    try:
        content = await file.read()
    except Exception as exc:
        logger.error("Failed to read uploaded file '%s': %s", filename, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file.",
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File size exceeds maximum allowed limit of {limit_mb} MB.",
        )

    # 3. Magic bytes validation (all valid PDFs start with %PDF)
    if not content.startswith(b"%PDF"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file signature: uploaded file is not a valid PDF document.",
        )

    # 4. Parse PDF structure with PyMuPDF to check integrity and page limits
    try:
        doc = pymupdf.open(stream=content, filetype="pdf")
    except Exception as exc:
        logger.warning("Corrupt PDF uploaded: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to parse PDF document: file may be corrupt or malformed.",
        )

    try:
        total_pages = len(doc)
        if total_pages == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="PDF document contains 0 pages.",
            )

        if total_pages > limit_pages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Document exceeds maximum page limit of {limit_pages} pages (has {total_pages} pages).",
            )
    finally:
        doc.close()

    return content

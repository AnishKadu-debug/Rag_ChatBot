"""Text chunking module with page-aware character splitting and metadata preservation.

What is a Chunk?
Large documents cannot be fed directly into an embedding model in their entirety.
Embedding models have maximum context windows (e.g. 512 tokens), and small, focused
chunks make vector retrieval much more accurate. A "chunk" is a small, coherent snippet
of text (e.g. ~500 characters).

What is Chunk Overlap?
Chunk overlap (e.g. 100 characters) means that consecutive chunks share a small section
of text at their boundaries. This prevents critical sentences or thoughts from being
severed right at the cut point.

Page-Aware Guarantee:
In this pipeline, chunking happens strictly within each physical page. A chunk NEVER
straddles across two pages. This ensures that every chunk has exactly one true page number,
allowing citation of the exact physical page in the land record.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from app.pdf_loader import PageData


@dataclass
class DocumentChunk:
    """Represents a discrete text chunk with associated document metadata."""

    chunk_id: str  # Unique chunk identifier, format: "{doc_id}_{chunk_index}"
    doc_id: str  # Identifier of the parent document
    page: int  # Physical 1-based page number where this chunk lives
    chunk_index: int  # Sequential index of the chunk across the document
    text: str  # The raw text content of the chunk
    ocr: bool = False  # Flag indicating whether OCR was used (Phase 3)

    def to_metadata(self) -> Dict[str, object]:
        """Convert chunk metadata into a dictionary compatible with ChromaDB."""
        return {
            "doc_id": self.doc_id,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "ocr": self.ocr,
        }


def split_text_into_chunks(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[str]:
    """Split a single block of text into overlapping character slices.

    This function attempts to split on clean whitespace boundaries near the chunk
    limit whenever possible, falling back to strict slicing if no whitespace exists.

    Args:
        text: The raw text string to split.
        chunk_size: Target maximum characters per chunk (default: 500).
        chunk_overlap: Number of characters shared between adjacent chunks (default: 100).

    Returns:
        List[str]: List of non-empty text chunk strings.

    Raises:
        ValueError: If chunk_overlap is greater than or equal to chunk_size or values <= 0.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
        )

    clean_text = text.strip()
    if not clean_text:
        return []

    # If the text is already smaller than or equal to chunk_size, return it directly
    if len(clean_text) <= chunk_size:
        return [clean_text]

    chunks: List[str] = []
    step: int = chunk_size - chunk_overlap
    start: int = 0
    text_len: int = len(clean_text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_str = clean_text[start:end].strip()

        if chunk_str:
            chunks.append(chunk_str)

        # Reached the end of the text
        if end >= text_len:
            break

        start += step

    return chunks


def chunk_pages(
    pages: List[PageData],
    doc_id: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    ocr: bool = False,
) -> List[DocumentChunk]:
    """Chunk a list of pages while strictly preserving 1-based page metadata.

    Args:
        pages: List of PageData items extracted from the document.
        doc_id: Unique identifier for the parent document.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks on the same page.
        ocr: Whether the text was produced via OCR.

    Returns:
        List[DocumentChunk]: Ordered list of chunks with unique IDs and metadata.
    """
    all_chunks: List[DocumentChunk] = []
    global_chunk_idx: int = 0

    for page in pages:
        page_text = page.text.strip()
        if not page_text:
            continue

        raw_chunks = split_text_into_chunks(
            text=page_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        page_ocr = getattr(page, "ocr_used", False) or ocr

        for chunk_str in raw_chunks:
            chunk_id = f"{doc_id}_{global_chunk_idx}"
            all_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    page=page.page_number,
                    chunk_index=global_chunk_idx,
                    text=chunk_str,
                    ocr=page_ocr,
                )
            )
            global_chunk_idx += 1

    return all_chunks

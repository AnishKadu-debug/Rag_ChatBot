"""PDF text extraction module using PyMuPDF (fitz).

In a Retrieval-Augmented Generation (RAG) system for documents, the first step is
extracting clean text while preserving structural boundaries. For land records
(such as 7/12 extracts or Mutation entries), page numbers are critical because
every answer given to the user must cite the exact page number where the information
was found.

Note: Optical Character Recognition (OCR) for scanned image pages is introduced
in Phase 3. In Phase 1, we extract digital text streams.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union
import pymupdf


@dataclass
class PageData:
    """Structured container for single-page extracted text."""

    page_number: int  # 1-based page index (page 1, page 2, etc.)
    text: str  # Extracted text content of the page


def extract_text_from_pdf(
    pdf_source: Union[str, Path, bytes, bytearray],
) -> List[PageData]:
    """Extract text from a PDF file page by page using PyMuPDF.

    Args:
        pdf_source: Path to the target PDF document or raw PDF bytes.

    Returns:
        List[PageData]: A list of PageData objects, each with a 1-based page number
        and the extracted text string.

    Raises:
        FileNotFoundError: If the specified PDF file does not exist on disk.
        ValueError: If the file is not a valid PDF or has 0 pages.
    """
    if isinstance(pdf_source, (bytes, bytearray)):
        try:
            doc = pymupdf.open(stream=pdf_source, filetype="pdf")
        except Exception as exc:
            raise ValueError(f"Failed to open PDF byte stream: {exc}") from exc
        source_name = "in-memory stream"
    else:
        path = Path(pdf_source)
        if not path.is_file():
            raise FileNotFoundError(f"PDF document not found: {path}")

        try:
            doc = pymupdf.open(str(path))
        except Exception as exc:
            raise ValueError(f"Failed to open PDF document '{path.name}': {exc}") from exc
        source_name = path.name

    try:
        total_pages = len(doc)
        if total_pages == 0:
            raise ValueError(f"PDF document '{source_name}' is empty (0 pages).")

        pages: List[PageData] = []
        for index in range(total_pages):
            page = doc[index]
            # Extract plain text from the page stream
            page_text = page.get_text("text")

            # 1-based page indexing for user-friendly citations
            pages.append(
                PageData(
                    page_number=index + 1,
                    text=page_text.strip(),
                )
            )

        return pages

    finally:
        # Always close document handle to avoid file locks on Windows
        doc.close()

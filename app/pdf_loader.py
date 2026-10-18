"""PDF text extraction module using PyMuPDF (fitz) with OCR fallback.

Why Hybrid Extraction (Digital Text + OCR Fallback)?
1. Digital PDFs: When PDFs are generated digitally from land revenue portals, they contain
   selectable text streams. Direct extraction with PyMuPDF is instantaneous and 100% accurate.
2. Scanned PDFs: Older records or camera-scanned physical documents contain only raster
   images with zero digital text. PyMuPDF returns an empty string for them.
3. Mixed Documents: A multi-page document may have digital text on page 1 and a scanned
   stamp/map on page 2.

This module processes documents page-by-page:
- Pages with sufficient extractable text (>= OCR_THRESHOLD_CHARS) are extracted natively.
- Pages with little or no extractable text are rendered to 300 DPI images and OCR'd via Tesseract.
- 1-based page numbers are preserved across all pages for citation accuracy.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import List, Optional, Union
from PIL import Image
import pymupdf

from app.config import get_settings
from app.ocr import (
    LanguagePackMissingError,
    OCRError,
    TesseractNotInstalledError,
    extract_text_from_image,
)

logger = logging.getLogger(__name__)


@dataclass
class PageData:
    """Structured container for single-page extracted text and metadata."""

    page_number: int  # 1-based page index (page 1, page 2, etc.)
    text: str  # Extracted text content of the page
    ocr_used: bool = False  # Flag indicating whether OCR was applied to this page


def extract_text_from_pdf(
    pdf_source: Union[str, Path, bytes, bytearray],
    ocr_threshold: Optional[int] = None,
    ocr_langs: Optional[str] = None,
    ocr_dpi: Optional[int] = None,
    enable_ocr: Optional[bool] = None,
) -> List[PageData]:
    """Extract text from a PDF file page by page, using OCR on scanned pages when necessary.

    Args:
        pdf_source: Path to target PDF (str/Path) or raw in-memory PDF bytes.
        ocr_threshold: Minimum text characters required to avoid OCR (default from config).
        ocr_langs: Tesseract language string (e.g. 'mar+eng', default from config).
        ocr_dpi: Rendering resolution for OCR image conversion (default: 300).
        enable_ocr: Whether OCR fallback is enabled (default from config).

    Returns:
        List[PageData]: List of PageData items with 1-based page numbers and extracted text.

    Raises:
        FileNotFoundError: If the specified PDF file does not exist on disk.
        ValueError: If the file is not a valid PDF or has 0 pages.
        TesseractNotInstalledError: If a scanned page requires OCR but Tesseract is missing.
        LanguagePackMissingError: If a scanned page requires OCR but required language pack is missing.
    """
    settings = get_settings()
    threshold = (
        ocr_threshold
        if ocr_threshold is not None
        else settings.OCR_THRESHOLD_CHARS
    )
    langs = ocr_langs or settings.OCR_LANGS
    dpi = ocr_dpi or settings.OCR_DPI
    ocr_active = enable_ocr if enable_ocr is not None else settings.OCR_ENABLED

    # 1. Open document from in-memory stream or file path
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
            page_num = index + 1
            page = doc[index]

            # 2. Extract native digital text
            native_text = page.get_text("text").strip()

            # 3. Decision Gate: If page has sufficient native text, preserve it directly
            if len(native_text) >= threshold or not ocr_active:
                pages.append(
                    PageData(
                        page_number=page_num,
                        text=native_text,
                        ocr_used=False,
                    )
                )
                continue

            # 4. Page has little or no text (< threshold) -> Scanned page candidate
            logger.info(
                "Page %d has only %d characters (threshold: %d). Attempting OCR...",
                page_num,
                len(native_text),
                threshold,
            )

            # Render page to high-DPI image for optimal OCR accuracy
            zoom = dpi / 72.0  # Default PDF resolution is 72 DPI
            mat = pymupdf.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            try:
                ocr_text = extract_text_from_image(img, langs=langs)
            except (TesseractNotInstalledError, LanguagePackMissingError):
                if len(native_text) > 0:
                    logger.warning(
                        "OCR dependency missing for page %d, falling back to available native text (%d chars).",
                        page_num,
                        len(native_text),
                    )
                    ocr_text = ""
                else:
                    # Zero native text available; cannot process without OCR
                    raise
            except Exception as exc:
                logger.warning(
                    "OCR failed on page %d of '%s': %s. Falling back to native text.",
                    page_num,
                    source_name,
                    exc,
                )
                ocr_text = ""

            # 5. Quality Guard: Only use OCR text if it extracted more content than native text
            if len(ocr_text) > len(native_text):
                pages.append(
                    PageData(
                        page_number=page_num,
                        text=ocr_text,
                        ocr_used=True,
                    )
                )
            else:
                # Page was genuinely blank or unreadable; preserve native text safely
                pages.append(
                    PageData(
                        page_number=page_num,
                        text=native_text,
                        ocr_used=False,
                    )
                )

        return pages

    finally:
        # Always close document handle to avoid file locks on Windows
        doc.close()

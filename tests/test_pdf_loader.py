"""Unit tests for the PDF loader module."""

from pathlib import Path
import pytest
from app.pdf_loader import extract_text_from_pdf

SAMPLE_PDF_PATH = Path("data/sample_pdfs/sample_land_record.pdf")


def test_extract_text_from_sample_pdf():
    """Verify that text is extracted page-by-page from the sample PDF."""
    assert SAMPLE_PDF_PATH.is_file(), "Sample PDF must exist for this test."

    pages = extract_text_from_pdf(SAMPLE_PDF_PATH)
    assert len(pages) == 2

    # Verify Page 1
    assert pages[0].page_number == 1
    assert "Digital 7/12 Extract" in pages[0].text
    assert "Ramesh Vishnu Patil" in pages[0].text
    assert "Gat Number: 45/2" in pages[0].text

    # Verify Page 2
    assert pages[1].page_number == 2
    assert "Ferfar" in pages[1].text
    assert "Mutation Entry Number: 1024" in pages[1].text


def test_missing_pdf_raises_filenotfound():
    """Verify that a nonexistent file path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        extract_text_from_pdf("data/nonexistent_document.pdf")


def test_invalid_pdf_raises_valueerror(tmp_path):
    """Verify that corrupted or non-PDF files raise ValueError."""
    fake_pdf = tmp_path / "corrupt.pdf"
    fake_pdf.write_text("This is not a real PDF binary file.")

    with pytest.raises(ValueError):
        extract_text_from_pdf(fake_pdf)

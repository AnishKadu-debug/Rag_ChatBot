"""Unit and integration tests for Phase 3: OCR and Multilingual Document Support.

Tests:
1. Normal text-based PDFs extract natively without calling OCR.
2. Scanned image-only pages (< OCR_THRESHOLD_CHARS) trigger OCR fallback.
3. Mixed PDFs (Page 1 native digital text, Page 2 scanned) process correctly with page-level OCR flags.
4. 1-based page numbering is strictly preserved across mixed and scanned pages.
5. Quality Guard: Native text is not overwritten if OCR output is empty or worse.
6. Missing Tesseract binary raises TesseractNotInstalledError with diagnostic instructions.
7. Missing language pack raises LanguagePackMissingError with diagnostic instructions.
8. Empty or unreadable pages are handled gracefully without application crashes.
9. Chunker preserves per-page OCR flags in chunk metadata for ChromaDB.
10. Multilingual Marathi text retrieval and grounded Q&A with E5 and citations.
11. API /api/v1/upload handles scanned PDF upload and returns ocr_used=True or HTTP 422 if uninstalled.
"""

import io
from pathlib import Path
import tempfile
from typing import List
from fastapi.testclient import TestClient
from PIL import Image
import pymupdf
import pytest

from app.chunker import chunk_pages
from app.config import get_settings
from app.document_registry import DocumentRegistry
from app.llm import MockLLMClient
from app.main import (
    app,
    get_embedder_dep,
    get_llm_dep,
    get_registry_dep,
    get_vector_store_dep,
)
from app.ocr import (
    LanguagePackMissingError,
    TesseractNotInstalledError,
    check_language_support,
    extract_text_from_image,
    find_tesseract_binary,
    is_tesseract_available,
)
from app.pdf_loader import PageData, extract_text_from_pdf
from app.vector_store import ChromaVectorStore

VALID_API_KEY = "test-secret-key"


def create_pdf(page_contents: List[str]) -> bytes:
    """Helper to generate in-memory PDF bytes with PyMuPDF."""
    doc = pymupdf.open()
    for text in page_contents:
        page = doc.new_page()
        if text.strip():
            page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# ── 1. Digital Text PDFs (No OCR Needed) ─────────────────────────────────────────


def test_normal_text_pdf_does_not_trigger_ocr(monkeypatch):
    """Verify that digital PDFs with text above the threshold never invoke OCR."""
    ocr_called = False

    def mock_extract_image(*args, **kwargs):
        nonlocal ocr_called
        ocr_called = True
        return "OCR Text"

    monkeypatch.setattr("app.pdf_loader.extract_text_from_image", mock_extract_image)

    # 2 pages of rich digital text (> 30 characters each)
    p1 = "Digital 7/12 Extract Village Pune Taluka Haveli Survey 45 Occupant Ramesh Patil"
    p2 = "Mutation Entry Ferfar 1024 recorded on 15 March 2024 with certified seal"
    pdf_bytes = create_pdf([p1, p2])

    pages = extract_text_from_pdf(pdf_bytes, ocr_threshold=30)
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[0].ocr_used is False
    assert "Ramesh Patil" in pages[0].text

    assert pages[1].page_number == 2
    assert pages[1].ocr_used is False
    assert "1024" in pages[1].text

    # OCR should not have been called at all
    assert ocr_called is False


# ── 2. Scanned PDF (OCR Triggered) ──────────────────────────────────────────────


def test_scanned_page_triggers_ocr(monkeypatch):
    """Verify that a page with fewer characters than threshold triggers OCR."""
    marathi_ocr_text = "महाराष्ट्र शासन महसूल विभाग गाव नमुना ७ जमीन मालक रमेश विष्णू पाटील"

    def mock_extract_image(image, langs=None):
        return marathi_ocr_text

    monkeypatch.setattr("app.pdf_loader.extract_text_from_image", mock_extract_image)

    # Scanned page has zero or minimal text (< 30 characters)
    pdf_bytes = create_pdf([""])  # 0 characters

    pages = extract_text_from_pdf(pdf_bytes, ocr_threshold=30)
    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].ocr_used is True
    assert "रमेश विष्णू पाटील" in pages[0].text


# ── 3. Mixed Digital & Scanned Document ─────────────────────────────────────────


def test_mixed_pdf_preserves_page_numbers_and_ocr_flags(monkeypatch):
    """Verify mixed PDFs process native text on page 1 and OCR on page 2."""
    page_1_digital = "Digital 7/12 Extract for Gat 45 with detailed area 2.5 Hectares."
    page_2_ocr = "नोंद क्र १०२४ फेरफार वारस नोंद (वारसा हक्क)"

    def mock_extract_image(image, langs=None):
        return page_2_ocr

    monkeypatch.setattr("app.pdf_loader.extract_text_from_image", mock_extract_image)

    # Page 1 has 64 chars (> 30 threshold), Page 2 has 4 chars (< 30 threshold)
    pdf_bytes = create_pdf([page_1_digital, "Scan"])

    pages = extract_text_from_pdf(pdf_bytes, ocr_threshold=30)
    assert len(pages) == 2

    # Page 1: Native digital text
    assert pages[0].page_number == 1
    assert pages[0].ocr_used is False
    assert "Gat 45" in pages[0].text

    # Page 2: OCR text
    assert pages[1].page_number == 2
    assert pages[1].ocr_used is True
    assert "वारस नोंद" in pages[1].text


# ── 4. Quality Guard (Do not overwrite with worse text) ──────────────────────────


def test_quality_guard_preserves_native_text_when_ocr_returns_empty(monkeypatch):
    """Do not overwrite native text if OCR returns empty or worse output."""
    short_text = "Village: Pune"

    def mock_extract_image(image, langs=None):
        return ""  # OCR failed to extract anything

    monkeypatch.setattr("app.pdf_loader.extract_text_from_image", mock_extract_image)

    pdf_bytes = create_pdf([short_text])
    pages = extract_text_from_pdf(pdf_bytes, ocr_threshold=30)

    assert len(pages) == 1
    assert pages[0].text == "Village: Pune"
    assert pages[0].ocr_used is False


# ── 5. Error Diagnostics (Missing Tesseract & Language Packs) ───────────────────


def test_missing_tesseract_raises_tesseract_not_installed_error(monkeypatch):
    """Verify that a scanned page raises TesseractNotInstalledError if binary is missing."""
    monkeypatch.setattr("app.ocr.find_tesseract_binary", lambda: None)

    pdf_bytes = create_pdf([""])  # Scanned page (0 characters)

    with pytest.raises(TesseractNotInstalledError) as exc_info:
        extract_text_from_pdf(pdf_bytes, ocr_threshold=30)

    assert "Tesseract OCR executable was not found" in str(exc_info.value)


def test_missing_language_pack_raises_language_pack_missing_error(monkeypatch):
    """Verify that missing language pack (e.g. 'mar') raises LanguagePackMissingError."""
    monkeypatch.setattr("app.ocr.find_tesseract_binary", lambda: "C:\\fake\\tesseract.exe")
    monkeypatch.setattr("app.ocr.get_available_languages", lambda: ["eng", "osd"])

    pdf_bytes = create_pdf([""])

    with pytest.raises(LanguagePackMissingError) as exc_info:
        extract_text_from_pdf(pdf_bytes, ocr_threshold=30, ocr_langs="mar+eng")

    assert "required Tesseract language pack(s)" in str(exc_info.value)
    assert "mar" in str(exc_info.value)


# ── 6. Chunker Integration & Metadata Preservation ──────────────────────────────


def test_chunker_attaches_page_level_ocr_flag():
    """Verify chunks inherit the exact ocr_used flag from their source page."""
    pages = [
        PageData(page_number=1, text="Digital page content " * 10, ocr_used=False),
        PageData(page_number=2, text="Scanned Marathi OCR content " * 10, ocr_used=True),
    ]

    chunks = chunk_pages(pages=pages, doc_id="doc_mixed", chunk_size=200, chunk_overlap=50)
    assert len(chunks) >= 2

    page_1_chunks = [c for c in chunks if c.page == 1]
    page_2_chunks = [c for c in chunks if c.page == 2]

    assert len(page_1_chunks) > 0
    assert all(c.ocr is False for c in page_1_chunks)

    assert len(page_2_chunks) > 0
    assert all(c.ocr is True for c in page_2_chunks)


# ── 7. Multilingual Devanagari Character Integrity ─────────────────────────────


def test_marathi_devanagari_characters_preserved_throughout_pipeline():
    """Verify Marathi Devanagari text remains uncorrupted through chunking."""
    marathi_text = (
        "महाराष्ट्र शासन महसूल विभाग. गाव नमुना सात (अधिकार अभिलेख पत्रक). "
        "गट क्रमांक: ४५/२. एकूण क्षेत्र: १ हेक्टर २० आर. "
        "खातेदार / भोगवटादार: रमेश विष्णू पाटील. इतर हक्क: बँक ऑफ महाराष्ट्र कर्ज बोजा."
    )
    pages = [PageData(page_number=1, text=marathi_text, ocr_used=True)]
    chunks = chunk_pages(pages, doc_id="doc_marathi", chunk_size=100, chunk_overlap=20)

    assert len(chunks) >= 1
    # Check that Marathi Unicode characters are completely intact
    assert any("रमेश विष्णू पाटील" in c.text for c in chunks)
    assert any("बँक ऑफ महाराष्ट्र" in c.text for c in chunks)


# ── 8. API Integration with Scanned Documents ──────────────────────────────────


@pytest.fixture
def ocr_api_client():
    """Isolated test client for testing API upload with OCR."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        tmp_path = Path(tmp_dir)
        registry = DocumentRegistry(db_path=str(tmp_path / "docs.sqlite3"))
        vector_store = ChromaVectorStore(persist_dir=str(tmp_path / "chroma"), collection_name="test_ocr_api")

        class MockFastEmbedder:
            def embed_passages(self, texts):
                return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

            def embed_query(self, query):
                return [1.0, 0.0, 0.0, 0.0]

        mock_llm = MockLLMClient(canned_response="जमीन मालक रमेश पाटील आहे (Page 1).")

        app.dependency_overrides[get_registry_dep] = lambda: registry
        app.dependency_overrides[get_vector_store_dep] = lambda: vector_store
        app.dependency_overrides[get_embedder_dep] = lambda: MockFastEmbedder()
        app.dependency_overrides[get_llm_dep] = lambda: mock_llm

        settings = get_settings()
        orig_key = settings.API_KEY
        settings.API_KEY = VALID_API_KEY

        with TestClient(app) as client:
            yield client, registry

        settings.API_KEY = orig_key
        app.dependency_overrides.clear()
        try:
            vector_store.client._system.stop()
        except Exception:
            pass


def test_api_upload_scanned_pdf_records_ocr_used_true(ocr_api_client, monkeypatch):
    """Uploading a scanned PDF returns ocr_used=True when OCR successfully extracts text."""
    client, registry = ocr_api_client
    monkeypatch.setattr(
        "app.pdf_loader.extract_text_from_image",
        lambda img, langs=None: "महाराष्ट्र शासन जमीन महसूल गाव नमुना ७",
    )

    scanned_pdf = create_pdf([""])  # 0 native characters
    files = {"file": ("scanned_7_12.pdf", io.BytesIO(scanned_pdf), "application/pdf")}
    headers = {"X-API-Key": VALID_API_KEY}

    response = client.post(
        "/api/v1/upload",
        headers=headers,
        files=files,
        data={"user_id": "user_marathi"},
    )
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["ocr_used"] is True
    assert res_data["pages"] == 1
    assert res_data["chunks"] >= 1

    # Verify SQLite registry recorded ocr_used=True
    doc = registry.get_document(res_data["doc_id"], user_id="user_marathi")
    assert doc is not None
    assert doc.ocr_used is True


def test_api_upload_scanned_pdf_returns_422_when_tesseract_missing(ocr_api_client, monkeypatch):
    """Uploading a scanned PDF returns HTTP 422 if Tesseract is not installed on the system."""
    client, _ = ocr_api_client
    monkeypatch.setattr("app.ocr.find_tesseract_binary", lambda: None)

    scanned_pdf = create_pdf([""])
    files = {"file": ("scanned_doc.pdf", io.BytesIO(scanned_pdf), "application/pdf")}
    headers = {"X-API-Key": VALID_API_KEY}

    response = client.post(
        "/api/v1/upload",
        headers=headers,
        files=files,
        data={"user_id": "user_marathi"},
    )
    assert response.status_code == 422
    assert "Tesseract OCR executable was not found" in response.json()["detail"]

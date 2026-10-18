"""Tests for Official 500-Page Document Ingestion, Retrieval, Language Control, and Fallback Handling.

Covers:
1. Document exceeding 100 pages indexing with batching, page metadata, and ChromaDB persistence.
2. Strict retrieval restricted to the official document (never mixing test or other users' documents).
3. HTTP 503 Service Unavailable when the official document is missing, indexing, or failed.
4. General fallback answer without citations or page numbers when information is absent from the PDF.
5. Grounded answer with verified page citations when information is present in the official PDF.
6. Safe official document replacement without stale or duplicate chunks.
7. Language control: English default vs Marathi instructions reaching backend prompts.
8. Admin upload endpoint authentication and execution.
"""

import io
from pathlib import Path
import tempfile
from typing import List

from fastapi.testclient import TestClient
import pymupdf
import pytest

from app.chunker import DocumentChunk
from app.config import get_settings
from app.document_registry import DocumentRegistry
from app.ingest_official import IngestionError, ingest_official_pdf, validate_admin_pdf
from app.llm import MockLLMClient
from app.main import (
    app,
    get_embedder_dep,
    get_llm_dep,
    get_registry_dep,
    get_vector_store_dep,
)
from app.prompts import (
    NOT_FOUND_TOKEN,
    build_grounded_prompt,
    get_general_system_instruction,
    get_grounded_system_instruction,
)
from app.vector_store import ChromaVectorStore

VALID_API_KEY = "test-admin-secret-key"


class FastMockEmbedder:
    """Mock embedder returning deterministic normalized vectors for testing."""

    def embed_passages(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        results = []
        for t in texts:
            t_lower = t.lower()
            if "gat 45" in t_lower or "ramesh" in t_lower:
                results.append([1.0, 0.0, 0.0, 0.0])
            elif "version 2" in t_lower:
                results.append([0.0, 1.0, 0.0, 0.0])
            elif "page 105" in t_lower:
                results.append([0.0, 0.0, 1.0, 0.0])
            else:
                results.append([0.0, 0.0, 0.0, 1.0])
        return results

    def embed_query(self, query: str) -> List[float]:
        q_lower = query.lower()
        if "gat 45" in q_lower or "ramesh" in q_lower:
            return [1.0, 0.0, 0.0, 0.0]
        elif "version 2" in q_lower:
            return [0.0, 1.0, 0.0, 0.0]
        elif "page 105" in q_lower:
            return [0.0, 0.0, 1.0, 0.0]
        return [0.5, 0.5, 0.0, 0.0]


def make_multi_page_pdf(num_pages: int, text_prefix: str = "Maharashtra Land Record Page") -> bytes:
    """Generate in-memory valid PDF with a large number of pages."""
    doc = pymupdf.open()
    for i in range(1, num_pages + 1):
        page = doc.new_page()
        page.insert_text((50, 72), f"{text_prefix} {i}: Survey Gat {i * 10} area 2.5 Hectares.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture(autouse=True)
def override_settings(monkeypatch):
    """Set test configurations."""
    settings = get_settings()
    monkeypatch.setattr(settings, "API_KEY", VALID_API_KEY)
    monkeypatch.setattr(settings, "ADMIN_MAX_PAGES", 1000)
    monkeypatch.setattr(settings, "ADMIN_MAX_UPLOAD_MB", 100)
    monkeypatch.setattr(settings, "SIMILARITY_DISTANCE_THRESHOLD", 0.45)


@pytest.fixture
def official_test_env():
    """Isolated environment with temporary registry and ChromaDB."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_db = str(tmp_path / "test_official_docs.sqlite3")
        registry = DocumentRegistry(db_path=test_db)

        test_chroma = str(tmp_path / "chroma_official")
        vector_store = ChromaVectorStore(
            persist_dir=test_chroma,
            collection_name="chunks_official_test",
        )

        embedder = FastMockEmbedder()
        mock_llm = MockLLMClient(canned_response="The owner of Gat 45 is Ramesh Patil (Page 5).")

        app.dependency_overrides[get_registry_dep] = lambda: registry
        app.dependency_overrides[get_vector_store_dep] = lambda: vector_store
        app.dependency_overrides[get_embedder_dep] = lambda: embedder
        app.dependency_overrides[get_llm_dep] = lambda: mock_llm

        client = TestClient(app)

        yield {
            "client": client,
            "registry": registry,
            "vector_store": vector_store,
            "embedder": embedder,
            "mock_llm": mock_llm,
        }

        app.dependency_overrides.clear()
        try:
            vector_store.client._system.stop()
        except Exception:
            pass


# ── 1. Indexing Over 100 Pages & Chunk Metadata ────────────────────────────────


def test_index_document_exceeding_100_pages(official_test_env):
    """Verify that a document exceeding 100 pages is successfully indexed with accurate 1-based page numbers."""
    registry = official_test_env["registry"]
    vector_store = official_test_env["vector_store"]
    embedder = official_test_env["embedder"]

    # Generate 105 pages synthetic PDF
    pdf_bytes = make_multi_page_pdf(num_pages=105, text_prefix="Land Record Official Page")

    # Ingest with batch size of 32
    record = ingest_official_pdf(
        pdf_source=pdf_bytes,
        filename="maharashtra_official_105pages.pdf",
        replace=True,
        chunk_batch_size=32,
        registry=registry,
        vector_store=vector_store,
        embedder=embedder,
    )

    assert record.pages == 105
    assert record.is_official is True
    assert record.status == "ready"
    assert record.doc_id.startswith("d_official_")

    # Verify active official doc discovery
    active_doc = registry.get_active_official_document()
    assert active_doc is not None
    assert active_doc.doc_id == record.doc_id
    assert active_doc.pages == 105

    # Verify ChromaDB storage and metadata for page 1, 50, and 105
    meta_results = vector_store.collection.get(where={"$and": [{"user_id": "official"}, {"doc_id": record.doc_id}]})
    assert len(meta_results["ids"]) >= 105
    pages_recorded = [m.get("page") for m in meta_results["metadatas"]]
    assert 1 in pages_recorded
    assert 50 in pages_recorded
    assert 105 in pages_recorded


# ── 2. Three Fallback Cases (Clarification Requirements) ───────────────────────


def test_fallback_case_1_http_503_when_official_document_missing(official_test_env):
    """When the official document is missing from registry, POST /api/v1/chat returns HTTP 503."""
    client = official_test_env["client"]
    headers = {"X-API-Key": VALID_API_KEY}

    payload = {
        "user_id": "user_mobile_1",
        "question": "Who owns Gat 45?",
    }
    response = client.post("/api/v1/chat", headers=headers, json=payload)
    assert response.status_code == 503
    assert "Official land records document is temporarily unavailable" in response.json()["detail"]


def test_fallback_case_1_http_503_when_official_document_status_not_ready(official_test_env):
    """When official document is in 'indexing' or 'failed' status, chat returns HTTP 503."""
    client = official_test_env["client"]
    registry = official_test_env["registry"]
    headers = {"X-API-Key": VALID_API_KEY}

    # Register an official document still in 'indexing' state
    registry.add_document(
        doc_id="d_official_in_progress",
        user_id="official",
        filename="official_in_progress.pdf",
        pages=500,
        is_official=True,
        status="indexing",
    )

    response = client.post(
        "/api/v1/chat",
        headers=headers,
        json={"user_id": "u1", "question": "Who is the owner?"},
    )
    assert response.status_code == 503


def test_fallback_case_2_general_answer_when_info_absent_from_pdf(official_test_env):
    """When official PDF exists but does not contain the answer, return general assistant answer without citations."""
    client = official_test_env["client"]
    registry = official_test_env["registry"]
    vector_store = official_test_env["vector_store"]
    embedder = official_test_env["embedder"]
    mock_llm = official_test_env["mock_llm"]
    headers = {"X-API-Key": VALID_API_KEY}

    # Ingest official document that only talks about Gat 10 to Gat 30
    pdf_bytes = make_multi_page_pdf(num_pages=5, text_prefix="Official Land Record Gat")
    ingest_official_pdf(
        pdf_source=pdf_bytes,
        filename="official_records.pdf",
        replace=True,
        registry=registry,
        vector_store=vector_store,
        embedder=embedder,
    )

    # Dynamic mock LLM: returns NOT_FOUND for grounded context, then answers general question
    class AbsentMockLLM(MockLLMClient):
        def generate(self, prompt, system_instruction=None, temperature=0.2):
            if "Context:" in prompt:
                return NOT_FOUND_TOKEN
            return "A 7/12 extract is a record of rights maintained by Maharashtra revenue departments."

    official_test_env["mock_llm"] = AbsentMockLLM()
    app.dependency_overrides[get_llm_dep] = lambda: official_test_env["mock_llm"]

    payload = {
        "user_id": "user_mobile_1",
        "question": "What is the historical origin of the 7/12 extract in 1966?",
        "language": "en",
    }
    response = client.post("/api/v1/chat", headers=headers, json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["mode"] == "general"
    # Rule: Do not show citations, page numbers, or snippets for fallback
    assert data["sources"] == []
    assert "A 7/12 extract is a record of rights" in data["answer"]


def test_fallback_case_3_grounded_answer_with_citations_when_info_present(official_test_env):
    """When question is answered by official PDF, return grounded mode with accurate page citations."""
    client = official_test_env["client"]
    registry = official_test_env["registry"]
    vector_store = official_test_env["vector_store"]
    embedder = official_test_env["embedder"]
    headers = {"X-API-Key": VALID_API_KEY}

    # Ingest document containing Gat 45 on Page 5
    doc = pymupdf.open()
    for p in range(1, 10):
        page = doc.new_page()
        if p == 5:
            page.insert_text((50, 72), "Owner of Gat 45 is Ramesh Vishnu Patil. Area 3.5 Hectares.")
        else:
            page.insert_text((50, 72), f"General page {p} content.")
    pdf_bytes = doc.tobytes()
    doc.close()

    ingest_official_pdf(
        pdf_source=pdf_bytes,
        filename="official_gat45.pdf",
        replace=True,
        registry=registry,
        vector_store=vector_store,
        embedder=embedder,
    )

    official_test_env["mock_llm"].canned_response = "The owner of Gat 45 is Ramesh Vishnu Patil (Page 5)."

    payload = {
        "user_id": "user_mobile_1",
        "question": "Who is the owner of Gat 45?",
        "language": "en",
    }
    response = client.post("/api/v1/chat", headers=headers, json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["mode"] == "document"
    assert len(data["sources"]) >= 1
    assert data["sources"][0]["page"] == 5
    assert "Ramesh Vishnu Patil" in data["sources"][0]["snippet"]
    assert "Ramesh Vishnu Patil (Page 5)" in data["answer"]


# ── 3. Isolation & No Bleed Across Collections ─────────────────────────────────


def test_retrieval_does_not_mix_other_users_documents(official_test_env):
    """Chat queries to the official chatbot never retrieve chunks from other users' uploaded PDFs."""
    client = official_test_env["client"]
    registry = official_test_env["registry"]
    vector_store = official_test_env["vector_store"]
    embedder = official_test_env["embedder"]
    headers = {"X-API-Key": VALID_API_KEY}

    # 1. Official document ingested (contains Gat 45)
    doc_official = pymupdf.open()
    p1 = doc_official.new_page()
    p1.insert_text((50, 72), "Official Record: Gat 45 official revenue assessment.")
    pdf_official_bytes = doc_official.tobytes()
    doc_official.close()

    ingest_official_pdf(
        pdf_source=pdf_official_bytes,
        filename="official_doc.pdf",
        replace=True,
        registry=registry,
        vector_store=vector_store,
        embedder=embedder,
    )

    # 2. Another user uploads a private document with "Gat 45"
    doc_user = pymupdf.open()
    pu = doc_user.new_page()
    pu.insert_text((50, 72), "User Private: Gat 45 secret mortgage details.")
    pdf_user_bytes = doc_user.tobytes()
    doc_user.close()

    client.post(
        "/api/v1/upload",
        headers=headers,
        files={"file": ("private.pdf", io.BytesIO(pdf_user_bytes), "application/pdf")},
        data={"user_id": "user_private_bob"},
    )

    # 3. Public query to official chatbot
    chat_res = client.post(
        "/api/v1/chat",
        headers=headers,
        json={"user_id": "public_visitor", "question": "Tell me about Gat 45"},
    )
    assert chat_res.status_code == 200
    sources = chat_res.json()["sources"]
    # All retrieved snippets must strictly come from the official document, NOT the user document
    for s in sources:
        assert "secret mortgage" not in s["snippet"]


# ── 4. Safe Replacement Without Stale Chunks ───────────────────────────────────


def test_safe_replacement_purges_stale_chunks(official_test_env):
    """Replacing the official document removes old official chunks from ChromaDB."""
    registry = official_test_env["registry"]
    vector_store = official_test_env["vector_store"]
    embedder = official_test_env["embedder"]

    # 1. Ingest official doc V1 (with "Gat 45")
    pdf_v1 = make_multi_page_pdf(2, text_prefix="Gat 45 Version 1")
    v1_rec = ingest_official_pdf(
        pdf_source=pdf_v1,
        filename="official_v1.pdf",
        replace=True,
        registry=registry,
        vector_store=vector_store,
        embedder=embedder,
    )

    # Verify V1 chunks exist in Chroma
    results_v1 = vector_store.query(
        user_id="official",
        query_embedding=embedder.embed_query("Gat 45"),
        doc_id=v1_rec.doc_id,
    )
    assert len(results_v1) > 0

    # 2. Replace with official doc V2 (with "Version 2")
    pdf_v2 = make_multi_page_pdf(2, text_prefix="Different Document Version 2")
    v2_rec = ingest_official_pdf(
        pdf_source=pdf_v2,
        filename="official_v2.pdf",
        replace=True,
        registry=registry,
        vector_store=vector_store,
        embedder=embedder,
    )

    # 3. Verify V1 chunks are completely purged
    results_after_replace = vector_store.query(
        user_id="official",
        query_embedding=embedder.embed_query("Gat 45"),
        doc_id=v1_rec.doc_id,
    )
    assert len(results_after_replace) == 0

    # Verify active document is now V2
    active = registry.get_active_official_document()
    assert active.doc_id == v2_rec.doc_id


# ── 5. Language Instructions in Prompts ─────────────────────────────────────────


def test_answer_language_instructions_en_and_mr():
    """Verify that language instructions are formatted properly into grounded and general system prompts."""
    # English mode
    en_grounded = get_grounded_system_instruction("en")
    assert "Reply in English" in en_grounded
    en_general = get_general_system_instruction("en")
    assert "Reply in English" in en_general

    # Marathi mode
    mr_grounded = get_grounded_system_instruction("mr")
    assert "Reply in Marathi (मराठी)" in mr_grounded
    mr_general = get_general_system_instruction("mr")
    assert "Reply in Marathi (मराठी)" in mr_general


def test_chat_passes_language_parameter_to_backend(official_test_env):
    """Calling chat with language='mr' instructs LLM to generate in Marathi."""
    client = official_test_env["client"]
    registry = official_test_env["registry"]
    vector_store = official_test_env["vector_store"]
    embedder = official_test_env["embedder"]
    mock_llm = official_test_env["mock_llm"]
    headers = {"X-API-Key": VALID_API_KEY}

    # Ingest document
    pdf_bytes = make_multi_page_pdf(2, text_prefix="Gat 45 Land Record")
    ingest_official_pdf(
        pdf_source=pdf_bytes,
        filename="official.pdf",
        replace=True,
        registry=registry,
        vector_store=vector_store,
        embedder=embedder,
    )

    # Send Marathi chat request
    client.post(
        "/api/v1/chat",
        headers=headers,
        json={
            "user_id": "user_mr",
            "question": "जमिनीचे क्षेत्रफळ किती आहे?",
            "language": "mr",
        },
    )

    # Verify system instruction passed to LLM has Marathi instruction
    assert mock_llm.last_system_instruction is not None
    assert "Reply in Marathi (मराठी)" in mock_llm.last_system_instruction


# ── 6. Admin Official Upload Endpoint ──────────────────────────────────────────


def test_admin_upload_official_endpoint(official_test_env):
    """POST /api/v1/admin/upload-official accepts PDF and updates official document."""
    client = official_test_env["client"]
    registry = official_test_env["registry"]
    headers = {"X-API-Key": VALID_API_KEY}

    pdf_bytes = make_multi_page_pdf(5, text_prefix="Admin Ingested Official Page")
    files = {"file": ("official_uploaded.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    data = {"replace": "true"}

    response = client.post(
        "/api/v1/admin/upload-official",
        headers=headers,
        files=files,
        data=data,
    )
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["pages"] == 5
    assert res_data["doc_id"].startswith("d_official_")

    active = registry.get_active_official_document()
    assert active is not None
    assert active.doc_id == res_data["doc_id"]

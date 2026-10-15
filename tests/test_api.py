"""Integration and unit tests for Phase 2 FastAPI service endpoints.

Tests:
1. Public Health Check (/health) without API key.
2. Security & Authentication: rejection of missing/invalid X-API-Key across protected endpoints.
3. PDF Upload (/api/v1/upload):
   - Successful PDF upload and metadata recording.
   - Validation failures: invalid file signature (non-PDF), corrupt PDF, empty file.
   - Upload size and page-count limit enforcement.
4. Chat (/api/v1/chat):
   - General assistant mode when doc_id is omitted.
   - Grounded document Q&A mode with source citations.
   - 404 response for nonexistent doc_id.
5. Multi-Tenant User Isolation:
   - User B cannot retrieve User A's document chunks in chat.
   - User B cannot see User A's document in document listing.
   - User B cannot delete User A's document.
6. Document Listing & Deletion (/api/v1/documents):
   - Listing returns uploaded documents for the user.
   - Deletion removes SQLite record and ChromaDB vectors.
   - Repeated deletion returns 404.
"""

import io
from pathlib import Path
import tempfile
from typing import List
from fastapi.testclient import TestClient
import pymupdf
import pytest

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
from app.vector_store import ChromaVectorStore

VALID_API_KEY = "test-secret-key"


class MockTestEmbedder:
    """Fast mock embedder generating normalized dummy 4D vectors for testing."""

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, query: str) -> List[float]:
        return [1.0, 0.0, 0.0, 0.0]


def make_pdf_bytes(page_texts: List[str]) -> bytes:
    """Generate in-memory valid PDF bytes with PyMuPDF."""
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture(autouse=True)
def override_settings_key(monkeypatch):
    """Ensure the API_KEY setting is set to our test key for all test runs."""
    settings = get_settings()
    monkeypatch.setattr(settings, "API_KEY", VALID_API_KEY)


@pytest.fixture
def api_test_env():
    """Set up an isolated test client with temporary SQLite and ChromaDB instances."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_db = str(tmp_path / "test_docs.sqlite3")
        registry = DocumentRegistry(db_path=test_db)

        test_chroma = str(tmp_path / "chroma")
        vector_store = ChromaVectorStore(
            persist_dir=test_chroma,
            collection_name="test_api_chunks",
        )

        mock_embedder = MockTestEmbedder()
        mock_llm = MockLLMClient(
            canned_response="The owner is Ramesh Patil (Page 1)."
        )

        # Override dependencies in FastAPI application
        app.dependency_overrides[get_registry_dep] = lambda: registry
        app.dependency_overrides[get_vector_store_dep] = lambda: vector_store
        app.dependency_overrides[get_embedder_dep] = lambda: mock_embedder
        app.dependency_overrides[get_llm_dep] = lambda: mock_llm

        client = TestClient(app)

        yield {
            "client": client,
            "registry": registry,
            "vector_store": vector_store,
            "mock_llm": mock_llm,
        }

        # Teardown overrides and release ChromaDB system handles
        app.dependency_overrides.clear()
        try:
            vector_store.client._system.stop()
        except Exception:
            pass


# ── 1. Health Check Tests ───────────────────────────────────────────────────────


def test_health_check_requires_no_auth(api_test_env):
    """GET /health must succeed with 200 OK without providing any API key."""
    client = api_test_env["client"]
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ── 2. Security & Authentication Tests ──────────────────────────────────────────


def test_protected_endpoints_reject_missing_or_invalid_api_key(api_test_env):
    """All non-health endpoints must return 401 Unauthorized for missing or invalid keys."""
    client = api_test_env["client"]

    # 1. Missing key
    res_upload = client.post("/api/v1/upload")
    assert res_upload.status_code == 401
    assert "Invalid or missing API key" in res_upload.json()["detail"]

    res_chat = client.post("/api/v1/chat", json={"user_id": "u1", "question": "Hi"})
    assert res_chat.status_code == 401

    res_docs = client.get("/api/v1/documents?user_id=u1")
    assert res_docs.status_code == 401

    res_del = client.delete("/api/v1/documents/doc123?user_id=u1")
    assert res_del.status_code == 401

    # 2. Invalid key
    bad_headers = {"X-API-Key": "wrong-secret-key"}
    res_chat_bad = client.post(
        "/api/v1/chat",
        headers=bad_headers,
        json={"user_id": "u1", "question": "Hi"},
    )
    assert res_chat_bad.status_code == 401


# ── 3. PDF Upload & Validation Tests ────────────────────────────────────────────


def test_upload_pdf_success(api_test_env):
    """Valid PDF upload creates registry entry and vector chunks."""
    client = api_test_env["client"]
    registry = api_test_env["registry"]
    headers = {"X-API-Key": VALID_API_KEY}

    pdf_bytes = make_pdf_bytes(["Digital 7/12 Gat 45 Owner Ramesh Patil", "Ferfar Entry 1024"])
    files = {"file": ("7_12.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    data = {"user_id": "user_alice"}

    response = client.post("/api/v1/upload", headers=headers, files=files, data=data)
    assert response.status_code == 200

    resp_json = response.json()
    assert resp_json["filename"] == "7_12.pdf"
    assert resp_json["pages"] == 2
    assert resp_json["chunks"] >= 2
    assert resp_json["ocr_used"] is False
    assert resp_json["doc_id"].startswith("d_")

    # Verify document exists in SQLite registry
    doc = registry.get_document(resp_json["doc_id"], user_id="user_alice")
    assert doc is not None
    assert doc.filename == "7_12.pdf"
    assert doc.pages == 2


def test_upload_rejects_non_pdf_file(api_test_env):
    """Uploading a non-PDF file or file without %PDF magic bytes returns HTTP 400."""
    client = api_test_env["client"]
    headers = {"X-API-Key": VALID_API_KEY}

    # Text file masquerading as PDF
    fake_pdf = b"Hello, this is just plain text, not a PDF."
    files = {"file": ("fake.pdf", io.BytesIO(fake_pdf), "application/pdf")}
    data = {"user_id": "user_alice"}

    response = client.post("/api/v1/upload", headers=headers, files=files, data=data)
    assert response.status_code == 400
    assert "Invalid file signature" in response.json()["detail"]


def test_upload_rejects_corrupted_pdf(api_test_env):
    """Corrupted PDF stream starting with %PDF but having bad content returns 400."""
    client = api_test_env["client"]
    headers = {"X-API-Key": VALID_API_KEY}

    corrupt_pdf = b"%PDF-1.4\nCorrupted binary garbage without xref or catalog."
    files = {"file": ("corrupt.pdf", io.BytesIO(corrupt_pdf), "application/pdf")}
    data = {"user_id": "user_alice"}

    response = client.post("/api/v1/upload", headers=headers, files=files, data=data)
    assert response.status_code == 400
    assert "Failed to parse PDF document" in response.json()["detail"]


def test_upload_rejects_empty_file(api_test_env):
    """Empty 0-byte file returns HTTP 400."""
    client = api_test_env["client"]
    headers = {"X-API-Key": VALID_API_KEY}

    files = {"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")}
    data = {"user_id": "user_alice"}

    response = client.post("/api/v1/upload", headers=headers, files=files, data=data)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_size_limit(api_test_env, monkeypatch):
    """Files exceeding MAX_UPLOAD_MB return HTTP 413."""
    client = api_test_env["client"]
    settings = get_settings()
    monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 1)  # 1 MB limit
    headers = {"X-API-Key": VALID_API_KEY}

    # Generate oversized payload (> 1MB) starting with %PDF
    oversized = b"%PDF-1.4 " + (b"X" * (1024 * 1024 + 100))
    files = {"file": ("large.pdf", io.BytesIO(oversized), "application/pdf")}
    data = {"user_id": "user_alice"}

    response = client.post("/api/v1/upload", headers=headers, files=files, data=data)
    assert response.status_code == 413
    assert "exceeds maximum allowed limit" in response.json()["detail"]


def test_upload_page_limit(api_test_env, monkeypatch):
    """PDFs exceeding MAX_PAGES return HTTP 400."""
    client = api_test_env["client"]
    settings = get_settings()
    monkeypatch.setattr(settings, "MAX_PAGES", 3)  # Max 3 pages
    headers = {"X-API-Key": VALID_API_KEY}

    # Generate 4-page PDF
    pdf_bytes = make_pdf_bytes(["Page 1", "Page 2", "Page 3", "Page 4"])
    files = {"file": ("long.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    data = {"user_id": "user_alice"}

    response = client.post("/api/v1/upload", headers=headers, files=files, data=data)
    assert response.status_code == 400
    assert "maximum page limit" in response.json()["detail"]


# ── 4. Chat Endpoints & User Isolation Tests ────────────────────────────────────


def test_chat_without_doc_id_runs_in_general_mode(api_test_env):
    """Calling /api/v1/chat without doc_id returns general mode with no sources."""
    client = api_test_env["client"]
    headers = {"X-API-Key": VALID_API_KEY}

    payload = {
        "user_id": "user_alice",
        "question": "What is a 7/12 extract?",
    }

    response = client.post("/api/v1/chat", headers=headers, json=payload)
    assert response.status_code == 200

    resp_json = response.json()
    assert resp_json["mode"] == "general"
    assert resp_json["sources"] == []
    assert len(resp_json["answer"]) > 0


def test_chat_with_valid_document_returns_grounded_answer(api_test_env):
    """Calling /api/v1/chat with a valid doc_id returns grounded mode and sources."""
    client = api_test_env["client"]
    headers = {"X-API-Key": VALID_API_KEY}

    # Upload document
    pdf_bytes = make_pdf_bytes(["Owner of Gat 45 is Ramesh Patil."])
    files = {"file": ("record.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    upload_res = client.post(
        "/api/v1/upload",
        headers=headers,
        files=files,
        data={"user_id": "user_alice"},
    )
    doc_id = upload_res.json()["doc_id"]

    # Chat with this document
    chat_payload = {
        "user_id": "user_alice",
        "doc_id": doc_id,
        "question": "Who is the owner?",
    }
    chat_res = client.post("/api/v1/chat", headers=headers, json=chat_payload)
    assert chat_res.status_code == 200

    chat_json = chat_res.json()
    assert chat_json["mode"] == "document"
    assert len(chat_json["sources"]) >= 1
    assert chat_json["sources"][0]["page"] == 1
    assert "Ramesh Patil" in chat_json["answer"]


def test_chat_with_nonexistent_doc_id_returns_404(api_test_env):
    """Supplying a doc_id that does not exist returns HTTP 404."""
    client = api_test_env["client"]
    headers = {"X-API-Key": VALID_API_KEY}

    chat_payload = {
        "user_id": "user_alice",
        "doc_id": "d_nonexistent_123",
        "question": "Who owns this land?",
    }
    response = client.post("/api/v1/chat", headers=headers, json=chat_payload)
    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."


def test_user_cannot_access_or_chat_with_another_users_document(api_test_env):
    """User Bob cannot chat with or retrieve User Alice's document (returns 404)."""
    client = api_test_env["client"]
    headers = {"X-API-Key": VALID_API_KEY}

    # Alice uploads a document
    pdf_bytes = make_pdf_bytes(["Confidential: Alice's land holding details."])
    upload_res = client.post(
        "/api/v1/upload",
        headers=headers,
        files={"file": ("alice_doc.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        data={"user_id": "user_alice"},
    )
    alice_doc_id = upload_res.json()["doc_id"]

    # Bob tries to chat with Alice's doc_id
    bob_chat = client.post(
        "/api/v1/chat",
        headers=headers,
        json={
            "user_id": "user_bob",
            "doc_id": alice_doc_id,
            "question": "What are Alice's land details?",
        },
    )
    # Must return 404 without leaking document existence
    assert bob_chat.status_code == 404
    assert bob_chat.json()["detail"] == "Document not found."


# ── 5. Listing and Deletion Tests ───────────────────────────────────────────────


def test_list_and_delete_documents(api_test_env):
    """Verify document listing and clean deletion for a user."""
    client = api_test_env["client"]
    vector_store = api_test_env["vector_store"]
    headers = {"X-API-Key": VALID_API_KEY}

    # 1. Alice uploads two documents
    pdf1 = make_pdf_bytes(["Doc 1 content."])
    pdf2 = make_pdf_bytes(["Doc 2 content."])

    res1 = client.post(
        "/api/v1/upload",
        headers=headers,
        files={"file": ("doc1.pdf", io.BytesIO(pdf1), "application/pdf")},
        data={"user_id": "user_alice"},
    )
    doc1_id = res1.json()["doc_id"]

    res2 = client.post(
        "/api/v1/upload",
        headers=headers,
        files={"file": ("doc2.pdf", io.BytesIO(pdf2), "application/pdf")},
        data={"user_id": "user_alice"},
    )
    doc2_id = res2.json()["doc_id"]

    # 2. Bob uploads one document
    pdf_bob = make_pdf_bytes(["Bob content."])
    res_bob = client.post(
        "/api/v1/upload",
        headers=headers,
        files={"file": ("bob.pdf", io.BytesIO(pdf_bob), "application/pdf")},
        data={"user_id": "user_bob"},
    )
    bob_doc_id = res_bob.json()["doc_id"]

    # 3. List documents for Alice -> should see exactly 2 documents
    alice_list_res = client.get("/api/v1/documents?user_id=user_alice", headers=headers)
    assert alice_list_res.status_code == 200
    alice_docs = alice_list_res.json()
    assert len(alice_docs) == 2
    alice_doc_ids = {d["doc_id"] for d in alice_docs}
    assert doc1_id in alice_doc_ids
    assert doc2_id in alice_doc_ids
    assert bob_doc_id not in alice_doc_ids

    # 4. Bob cannot delete Alice's document
    bob_del_res = client.delete(
        f"/api/v1/documents/{doc1_id}?user_id=user_bob",
        headers=headers,
    )
    assert bob_del_res.status_code == 404

    # 5. Alice deletes doc1
    del_res = client.delete(
        f"/api/v1/documents/{doc1_id}?user_id=user_alice",
        headers=headers,
    )
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "success"

    # 6. Check Alice's list now only has doc2
    alice_list_after = client.get(
        "/api/v1/documents?user_id=user_alice",
        headers=headers,
    ).json()
    assert len(alice_list_after) == 1
    assert alice_list_after[0]["doc_id"] == doc2_id

    # 7. Check vectors for doc1 in Chroma are gone
    remaining_chunks = vector_store.query(
        user_id="user_alice",
        query_embedding=[1.0, 0.0, 0.0, 0.0],
        doc_id=doc1_id,
    )
    assert len(remaining_chunks) == 0

    # 8. Deleting doc1 a second time returns 404
    del_again = client.delete(
        f"/api/v1/documents/{doc1_id}?user_id=user_alice",
        headers=headers,
    )
    assert del_again.status_code == 404

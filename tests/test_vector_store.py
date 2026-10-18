"""Unit tests for ChromaVectorStore: storage, retrieval, and multi-tenant isolation."""

import tempfile
import pytest
from app.chunker import DocumentChunk
from app.vector_store import ChromaVectorStore


@pytest.fixture
def temp_vector_store():
    """Create a temporary vector store instance isolated in a temporary folder."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        store = ChromaVectorStore(persist_dir=tmp_dir, collection_name="test_chunks")
        try:
            yield store
        finally:
            try:
                # Stop ChromaDB internal system threads to release file handles on Windows
                store.client._system.stop()
            except Exception:
                pass


def test_add_and_retrieve_chunks(temp_vector_store):
    """Verify that chunks and embeddings can be saved and retrieved."""
    store = temp_vector_store

    chunks = [
        DocumentChunk(
            chunk_id="doc1_0",
            doc_id="doc1",
            page=1,
            chunk_index=0,
            text="Owner of parcel 45 is Ramesh Patil.",
        ),
        DocumentChunk(
            chunk_id="doc1_1",
            doc_id="doc1",
            page=2,
            chunk_index=1,
            text="Mutation entry number 1024 recorded on 15 March.",
        ),
    ]

    # Two mock 4-dimensional normalized vectors
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]

    store.add_chunks(
        user_id="user_alice",
        chunks=chunks,
        embeddings=embeddings,
        filename="land_record.pdf",
    )

    # Query with vector identical to chunk 0
    query_vector = [1.0, 0.0, 0.0, 0.0]
    results = store.query(
        user_id="user_alice",
        query_embedding=query_vector,
        top_k=2,
        doc_id="doc1",
    )

    assert len(results) == 2
    # Best match should be chunk 0 with distance near 0.0
    assert results[0].chunk_id == "doc1_0"
    assert results[0].page == 1
    assert "Ramesh Patil" in results[0].text
    assert abs(results[0].distance) < 0.05


def test_user_isolation(temp_vector_store):
    """Verify that User B CANNOT see or retrieve User A's chunks (multi-tenant security)."""
    store = temp_vector_store

    alice_chunks = [
        DocumentChunk(
            chunk_id="alice_doc_0",
            doc_id="doc_secret",
            page=1,
            chunk_index=0,
            text="Confidential: Alice's bank and land details.",
        )
    ]
    alice_embeddings = [[0.5, 0.5, 0.5, 0.5]]

    store.add_chunks(
        user_id="user_alice",
        chunks=alice_chunks,
        embeddings=alice_embeddings,
        filename="alice.pdf",
    )

    # Bob queries with the exact same embedding
    query_vector = [0.5, 0.5, 0.5, 0.5]
    bob_results = store.query(
        user_id="user_bob",  # Different user
        query_embedding=query_vector,
        top_k=5,
    )

    # Bob must receive 0 results!
    assert len(bob_results) == 0


def test_delete_document_removes_chunks(temp_vector_store):
    """Verify that delete_document cleanly removes chunks for the specified user and doc."""
    store = temp_vector_store

    chunks = [
        DocumentChunk(
            chunk_id="doc_del_0",
            doc_id="doc_to_delete",
            page=1,
            chunk_index=0,
            text="Temporary document text.",
        )
    ]
    embeddings = [[1.0, 0.0, 0.0, 0.0]]

    store.add_chunks(
        user_id="user_charlie",
        chunks=chunks,
        embeddings=embeddings,
        filename="temp.pdf",
    )

    # Check it exists
    before_del = store.query(
        user_id="user_charlie",
        query_embedding=[1.0, 0.0, 0.0, 0.0],
        doc_id="doc_to_delete",
    )
    assert len(before_del) == 1

    # Delete the document
    store.delete_document(user_id="user_charlie", doc_id="doc_to_delete")

    # Check it is now gone
    after_del = store.query(
        user_id="user_charlie",
        query_embedding=[1.0, 0.0, 0.0, 0.0],
        doc_id="doc_to_delete",
    )
    assert len(after_del) == 0


def test_get_chunks_by_page_range_isolation_and_page_order(temp_vector_store):
    """Verify page-range retrieval is user-isolated, doc-scoped, and in PDF page order."""
    store = temp_vector_store

    alice_chunks = [
        DocumentChunk(chunk_id="a_p3", doc_id="doc_a", page=3, chunk_index=0, text="Alice page 3"),
        DocumentChunk(chunk_id="a_p1", doc_id="doc_a", page=1, chunk_index=0, text="Alice page 1"),
        DocumentChunk(
            chunk_id="a_p2_other_doc", doc_id="doc_b", page=2, chunk_index=0, text="Alice other doc"
        ),
        DocumentChunk(chunk_id="a_p9", doc_id="doc_a", page=9, chunk_index=0, text="Alice page 9"),
    ]
    bob_chunk = DocumentChunk(chunk_id="b_p2", doc_id="doc_a", page=2, chunk_index=0, text="Bob page 2")

    store.add_chunks(
        user_id="user_alice",
        chunks=alice_chunks,
        embeddings=[[0.1, 0.2, 0.3, 0.4] for _ in alice_chunks],
        filename="alice.pdf",
    )
    store.add_chunks(
        user_id="user_bob",
        chunks=[bob_chunk],
        embeddings=[[0.4, 0.3, 0.2, 0.1]],
        filename="bob.pdf",
    )

    alice_results = store.get_chunks_by_page_range(
        user_id="user_alice", doc_id="doc_a", start_page=1, end_page=3, max_results=10
    )
    assert [r.chunk_id for r in alice_results] == ["a_p1", "a_p3"]
    assert all(r.distance == 0.0 for r in alice_results)

    # No cross-tenant or cross-document leakage
    bob_results = store.get_chunks_by_page_range(
        user_id="user_bob", doc_id="doc_a", start_page=1, end_page=3, max_results=10
    )
    assert [r.chunk_id for r in bob_results] == ["b_p2"]


def test_legacy_collection_without_embedding_model_stamp_does_not_crash():
    """Regression: loading a collection with no embedding_model stamp must not raise (RAG-009)."""
    import chromadb

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        legacy_client = chromadb.PersistentClient(path=tmp_dir)
        legacy_client.get_or_create_collection(
            name="legacy_chunks",
            metadata={"hnsw:space": "cosine"},
        )

        store = ChromaVectorStore(persist_dir=tmp_dir, collection_name="legacy_chunks")
        try:
            assert store.collection.metadata.get("embedding_model") is None
        finally:
            try:
                store.client._system.stop()
            except Exception:
                pass


def test_get_structural_chunks(temp_vector_store):
    """Verify that get_structural_chunks retrieves matching keyword chunks in page order."""
    store = temp_vector_store

    chunks = [
        DocumentChunk(
            chunk_id="chunk_ch1",
            doc_id="doc_struct",
            page=5,
            chunk_index=0,
            text="INDEX\nCHAPTER – 1 ADMINISTRATION",
        ),
        DocumentChunk(
            chunk_id="chunk_ch15",
            doc_id="doc_struct",
            page=14,
            chunk_index=1,
            text="CHAPTER – 15 SPECIAL SCHEMES\nAPPENDICES",
        ),
        DocumentChunk(
            chunk_id="chunk_other",
            doc_id="doc_struct",
            page=50,
            chunk_index=2,
            text="General rules on setbacks without any chapter keyword.",
        ),
    ]
    embeddings = [[0.1, 0.2, 0.3, 0.4] for _ in chunks]

    store.add_chunks(
        user_id="user_david",
        chunks=chunks,
        embeddings=embeddings,
        filename="manual.pdf",
    )

    results = store.get_structural_chunks(user_id="user_david", doc_id="doc_struct")

    assert len(results) == 2
    assert results[0].chunk_id == "chunk_ch1"
    assert results[0].page == 5
    assert results[1].chunk_id == "chunk_ch15"
    assert results[1].page == 14
    assert results[0].distance == 0.0


"""Unit tests for the chunking module."""

import pytest
from app.chunker import (
    DocumentChunk,
    chunk_pages,
    split_text_into_chunks,
)
from app.pdf_loader import PageData


def test_empty_input_returns_empty_list():
    """Verify that empty or whitespace strings yield no chunks."""
    assert split_text_into_chunks("") == []
    assert split_text_into_chunks("   \n\t  ") == []

    # PageData with empty text should yield no chunks
    pages = [PageData(page_number=1, text="   ")]
    chunks = chunk_pages(pages, doc_id="doc_test")
    assert chunks == []


def test_short_text_single_chunk():
    """Verify that text shorter than chunk_size produces exactly one chunk."""
    short_text = "Village: Haveli, Survey Number: 120/1, Owner: Ramesh Patil."
    chunks = split_text_into_chunks(short_text, chunk_size=500, chunk_overlap=100)
    assert len(chunks) == 1
    assert chunks[0] == short_text


def test_chunk_overlap_preserved():
    """Verify that overlapping characters appear at chunk boundaries."""
    # Build a controlled string of 900 characters
    chunk_size = 500
    chunk_overlap = 100
    # Step = 400
    text = "A" * 400 + "B" * 100 + "C" * 400
    chunks = split_text_into_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    assert len(chunks) == 2
    # The first chunk should end with "B" * 100
    assert chunks[0].endswith("B" * 100)
    # The second chunk should begin with "B" * 100 (the overlap)
    assert chunks[1].startswith("B" * 100)


def test_page_metadata_isolation():
    """Verify that chunks maintain strictly 1-based page numbers from their parent page."""
    pages = [
        PageData(page_number=1, text="Page 1: Digital 7/12 Extract for Land Parcel 45."),
        PageData(page_number=2, text="Page 2: Mutation Ferfar entry number 789."),
    ]
    chunks = chunk_pages(pages, doc_id="doc_abc", chunk_size=500, chunk_overlap=100)

    assert len(chunks) == 2
    assert chunks[0].page == 1
    assert chunks[0].doc_id == "doc_abc"
    assert chunks[0].chunk_id == "doc_abc_0"

    assert chunks[1].page == 2
    assert chunks[1].doc_id == "doc_abc"
    assert chunks[1].chunk_id == "doc_abc_1"


def test_invalid_chunk_parameters_raise_error():
    """Verify that invalid chunk_size or overlap raise ValueError."""
    with pytest.raises(ValueError):
        split_text_into_chunks("test", chunk_size=0)
    with pytest.raises(ValueError):
        split_text_into_chunks("test", chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        split_text_into_chunks("test", chunk_size=100, chunk_overlap=150)

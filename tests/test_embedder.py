"""Unit tests for the E5 embedder wrapper."""

from unittest.mock import MagicMock
import numpy as np
from app.embedder import E5Embedder


def test_embedder_prefixes_passages_and_queries():
    """Verify that E5Embedder strictly prepends 'passage: ' and 'query: ' prefixes."""
    # Mock SentenceTransformer internal model to avoid heavy weight downloads in fast tests
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

    embedder = E5Embedder.__new__(E5Embedder)
    embedder.model = mock_model
    embedder.model_name = "test-model"
    embedder.device = "cpu"

    # Test passages
    embedder.embed_passages(["Sample land record chunk."])
    called_passages = mock_model.encode.call_args[0][0]
    assert called_passages == ["passage: Sample land record chunk."]

    # Test query
    embedder.embed_query("Who is the land owner?")
    called_query = mock_model.encode.call_args[0][0]
    assert called_query == "query: Who is the land owner?"

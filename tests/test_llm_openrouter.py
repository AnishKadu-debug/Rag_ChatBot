"""Unit tests for OpenRouter LLM client configuration, response handling, and error cases.

These tests verify:
1. OpenRouterLLMClient initialization and configuration from settings.
2. Correct request payload construction (model, messages, temperature).
3. Successful response parsing from the OpenRouter chat completions API.
4. Error handling: missing API key, auth failures (401/403), rate limits (429),
   timeouts, network errors, and malformed responses.
5. Provider factory selection: get_llm_client() returns OpenRouterLLMClient for 'openrouter' provider.
6. BaseLLMClient interface compliance.
7. RAG integration with OpenRouterLLMClient.

All tests use mocked HTTP responses (no real API calls).
"""

from unittest.mock import MagicMock, patch
import pytest

from app.config import get_settings
from app.llm import (
    BaseLLMClient,
    GeminiLLMClient,
    GrokLLMClient,
    MockLLMClient,
    OpenRouterLLMClient,
    get_llm_client,
)
from app.rag import answer_question
from app.vector_store import SearchResult


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def isolate_openrouter_env(monkeypatch):
    """Isolate OpenRouter tests with consistent model and provider settings."""
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_MODEL", "openrouter/free")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    get_settings.cache_clear()


# ─────────────────────────────────────────────────────────────────────────────
# 1. OpenRouterLLMClient Initialization
# ─────────────────────────────────────────────────────────────────────────────


def test_openrouter_client_defaults_from_settings(monkeypatch):
    """Verify OpenRouterLLMClient reads API key, model, and base URL from settings."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-or-key")
    monkeypatch.setenv("LLM_MODEL", "openrouter/free")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    get_settings.cache_clear()

    client = OpenRouterLLMClient()
    assert client.api_key == "test-or-key"
    assert client.model_name == "openrouter/free"
    assert client.base_url == "https://openrouter.ai/api/v1"


def test_openrouter_client_explicit_params_override_settings(monkeypatch):
    """Verify that explicit constructor params override settings."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    monkeypatch.setenv("LLM_MODEL", "openrouter/free")
    get_settings.cache_clear()

    client = OpenRouterLLMClient(
        api_key="override-key",
        model_name="meta-llama/llama-3-8b-instruct:free",
        base_url="https://custom.openrouter.ai/api/v1",
    )
    assert client.api_key == "override-key"
    assert client.model_name == "meta-llama/llama-3-8b-instruct:free"
    assert client.base_url == "https://custom.openrouter.ai/api/v1"


def test_openrouter_client_raises_on_missing_api_key(monkeypatch):
    """Verify ValueError is raised when OPENROUTER_API_KEY is empty and generate() is called."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    get_settings.cache_clear()

    client = OpenRouterLLMClient(api_key="")
    with pytest.raises(ValueError, match="OpenRouter API key is missing"):
        client.generate("Hello")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Successful Response Parsing
# ─────────────────────────────────────────────────────────────────────────────


def test_openrouter_generate_success(monkeypatch):
    """Verify successful response parsing from OpenRouter chat completions API."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    get_settings.cache_clear()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "The land owner is Ramesh Patil (Page 1).",
                }
            }
        ]
    }

    mock_client_instance = MagicMock()
    mock_client_instance.post.return_value = mock_response
    mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("app.llm.httpx.Client", return_value=mock_client_instance):
        client = OpenRouterLLMClient(api_key="test-key")
        result = client.generate(
            prompt="Who owns this land?",
            system_instruction="You are a helpful assistant.",
            temperature=0.2,
        )

    assert result == "The land owner is Ramesh Patil (Page 1)."
    call_args = mock_client_instance.post.call_args
    payload = call_args.kwargs.get("json") or call_args[1].get("json")
    assert payload["model"] == "openrouter/free"
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"
    assert payload["temperature"] == 0.2


def test_openrouter_generate_without_system_instruction(monkeypatch):
    """Verify that omitting system_instruction sends only the user message."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    get_settings.cache_clear()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "General answer."}}]
    }

    mock_client_instance = MagicMock()
    mock_client_instance.post.return_value = mock_response
    mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("app.llm.httpx.Client", return_value=mock_client_instance):
        client = OpenRouterLLMClient(api_key="test-key")
        result = client.generate(prompt="What is a 7/12 extract?")

    assert result == "General answer."
    call_args = mock_client_instance.post.call_args
    payload = call_args.kwargs.get("json") or call_args[1].get("json")
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["role"] == "user"


def test_openrouter_generate_list_content(monkeypatch):
    """Verify handling when response message content is a list of text parts."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    get_settings.cache_clear()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Part 1. "},
                        {"type": "text", "text": "Part 2."},
                    ],
                }
            }
        ]
    }

    mock_client_instance = MagicMock()
    mock_client_instance.post.return_value = mock_response
    mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("app.llm.httpx.Client", return_value=mock_client_instance):
        client = OpenRouterLLMClient(api_key="test-key")
        result = client.generate(prompt="Hello")

    assert result == "Part 1. Part 2."


# ─────────────────────────────────────────────────────────────────────────────
# 3. Error Handling
# ─────────────────────────────────────────────────────────────────────────────


def test_openrouter_auth_failure_401(monkeypatch):
    """Verify PermissionError is raised on HTTP 401 (invalid API key)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "bad-key")
    get_settings.cache_clear()

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = '{"error": {"message": "Invalid API key"}}'
    mock_response.json.return_value = {"error": {"message": "Invalid API key"}}

    mock_client_instance = MagicMock()
    mock_client_instance.post.return_value = mock_response
    mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("app.llm.httpx.Client", return_value=mock_client_instance):
        client = OpenRouterLLMClient(api_key="bad-key")
        with pytest.raises(PermissionError, match="authentication failed"):
            client.generate("Hello")


def test_openrouter_rate_limit_429(monkeypatch):
    """Verify RuntimeError is raised on HTTP 429 (rate limit exceeded)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    get_settings.cache_clear()

    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.text = '{"error": "Rate limit exceeded"}'
    mock_response.json.return_value = {"error": "Rate limit exceeded"}

    mock_client_instance = MagicMock()
    mock_client_instance.post.return_value = mock_response
    mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("app.llm.httpx.Client", return_value=mock_client_instance):
        client = OpenRouterLLMClient(api_key="test-key")
        with pytest.raises(RuntimeError, match="rate limit"):
            client.generate("Hello")


def test_openrouter_timeout_error(monkeypatch):
    """Verify TimeoutError is raised when the API call times out."""
    import httpx

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    get_settings.cache_clear()

    mock_client_instance = MagicMock()
    mock_client_instance.post.side_effect = httpx.TimeoutException("Connection timed out")
    mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("app.llm.httpx.Client", return_value=mock_client_instance):
        client = OpenRouterLLMClient(api_key="test-key", timeout=5.0)
        with pytest.raises(TimeoutError, match="timed out"):
            client.generate("Hello")


def test_openrouter_network_error(monkeypatch):
    """Verify ConnectionError is raised on network failure."""
    import httpx

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    get_settings.cache_clear()

    mock_client_instance = MagicMock()
    mock_client_instance.post.side_effect = httpx.RequestError("DNS resolution failed")
    mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("app.llm.httpx.Client", return_value=mock_client_instance):
        client = OpenRouterLLMClient(api_key="test-key")
        with pytest.raises(ConnectionError, match="Failed to communicate"):
            client.generate("Hello")


def test_openrouter_empty_choices_raises(monkeypatch):
    """Verify RuntimeError is raised when the API returns no choices."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    get_settings.cache_clear()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": []}

    mock_client_instance = MagicMock()
    mock_client_instance.post.return_value = mock_response
    mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("app.llm.httpx.Client", return_value=mock_client_instance):
        client = OpenRouterLLMClient(api_key="test-key")
        with pytest.raises(RuntimeError, match="no choices"):
            client.generate("Hello")


def test_openrouter_server_error_500(monkeypatch):
    """Verify RuntimeError is raised on HTTP 500 server error."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    get_settings.cache_clear()

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_response.json.side_effect = Exception("Not JSON")

    mock_client_instance = MagicMock()
    mock_client_instance.post.return_value = mock_response
    mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("app.llm.httpx.Client", return_value=mock_client_instance):
        client = OpenRouterLLMClient(api_key="test-key")
        with pytest.raises(RuntimeError, match="HTTP 500"):
            client.generate("Hello")


def test_openrouter_models_endpoint(monkeypatch):
    """Verify get_available_models and verify_model_availability query the models endpoint."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    get_settings.cache_clear()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {"id": "openrouter/free"},
            {"id": "meta-llama/llama-3-8b-instruct:free"},
        ]
    }

    mock_client_instance = MagicMock()
    mock_client_instance.get.return_value = mock_response
    mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("app.llm.httpx.Client", return_value=mock_client_instance):
        client = OpenRouterLLMClient(api_key="test-key")
        models = client.get_available_models()
        assert "openrouter/free" in models
        assert client.verify_model_availability("openrouter/free") is True
        assert client.verify_model_availability("non-existent-model") is False


# ─────────────────────────────────────────────────────────────────────────────
# 4. Provider Factory & Interface Compliance
# ─────────────────────────────────────────────────────────────────────────────


def test_get_llm_client_returns_openrouter_by_setting(monkeypatch):
    """Verify get_llm_client() returns OpenRouterLLMClient when LLM_PROVIDER=openrouter."""
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    get_settings.cache_clear()

    client = get_llm_client()
    assert isinstance(client, OpenRouterLLMClient)
    assert client.model_name == "openrouter/free"


def test_get_llm_client_explicit_openrouter_override(monkeypatch):
    """Verify get_llm_client(provider='openrouter') overrides the env setting."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    get_settings.cache_clear()

    client = get_llm_client(provider="openrouter")
    assert isinstance(client, OpenRouterLLMClient)


def test_openrouter_client_is_base_llm_client(monkeypatch):
    """Verify OpenRouterLLMClient is a proper subclass of BaseLLMClient."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    get_settings.cache_clear()

    client = OpenRouterLLMClient(api_key="test-key")
    assert isinstance(client, BaseLLMClient)


# ─────────────────────────────────────────────────────────────────────────────
# 5. RAG Pipeline Integration with Mocked OpenRouterLLMClient
# ─────────────────────────────────────────────────────────────────────────────


class MockRAGVectorStore:
    def __init__(self, search_results):
        self._results = search_results

    def query(self, user_id, query_embedding, top_k=4, doc_id=None):
        return self._results


class MockRAGEmbedder:
    def embed_query(self, query: str):
        return [0.1, 0.2, 0.3, 0.4]


def test_rag_with_openrouter_client_grounded_success(monkeypatch):
    """Verify answer_question works end-to-end with OpenRouterLLMClient."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    get_settings.cache_clear()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "The occupant is Ramesh Patil (Page 1).",
                }
            }
        ]
    }

    mock_client_instance = MagicMock()
    mock_client_instance.post.return_value = mock_response
    mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    relevant_result = SearchResult(
        chunk_id="chunk_1",
        doc_id="doc_xyz",
        page=1,
        chunk_index=0,
        text="Occupant: Ramesh Patil. Survey Number: 45/2.",
        distance=0.15,
    )
    mock_store = MockRAGVectorStore([relevant_result])
    mock_embedder = MockRAGEmbedder()

    with patch("app.llm.httpx.Client", return_value=mock_client_instance):
        openrouter_client = OpenRouterLLMClient(api_key="test-key")
        response = answer_question(
            question="Who is the occupant?",
            user_id="user_123",
            doc_id="doc_xyz",
            distance_threshold=0.45,
            vector_store=mock_store,
            embedder=mock_embedder,
            llm=openrouter_client,
        )

    assert response.mode == "document"
    assert "Ramesh Patil" in response.answer
    assert len(response.sources) == 1
    assert response.sources[0].page == 1

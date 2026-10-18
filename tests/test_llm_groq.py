"""Unit tests for GroqCloud LLM client configuration, response handling, and error cases.

These tests verify:
1. GroqLLMClient initialization and configuration from settings.
2. Correct request payload construction (model, messages, temperature, token limit).
3. Successful response parsing from the GroqCloud chat completions API.
4. Error handling: missing API key, auth failures (401/403), rate limits (429),
   timeouts, network errors, and malformed responses.
5. Provider factory selection: get_llm_client() returns GroqLLMClient for 'groq' provider.
6. Retry contract: transient failures use the existing bounded retry behavior.
7. BaseLLMClient interface compliance.
8. RAG integration with GroqLLMClient.

All tests use mocked HTTP responses (no real API calls).
"""

from unittest.mock import MagicMock, patch
import pytest

from app.config import get_settings
from app.llm import (
    BaseLLMClient,
    GrokLLMClient,
    GroqLLMClient,
    generate_with_retry,
    get_llm_client,
)
from app.rag import answer_question
from app.vector_store import SearchResult


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def isolate_groq_env(monkeypatch):
    """Isolate GroqCloud tests with consistent model and provider settings."""
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    get_settings.cache_clear()


# ─────────────────────────────────────────────────────────────────────────────
# 1. GroqLLMClient Initialization
# ─────────────────────────────────────────────────────────────────────────────


def test_groq_client_defaults_from_settings(monkeypatch):
    """Verify GroqLLMClient reads API key, model, and base URL from settings."""
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("LLM_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    get_settings.cache_clear()

    client = GroqLLMClient()
    assert client.api_key == "test-groq-key"
    assert client.model_name == "llama-3.3-70b-versatile"
    assert client.base_url == "https://api.groq.com/openai/v1"


def test_groq_client_explicit_params_override_settings(monkeypatch):
    """Verify that explicit constructor params override settings."""
    monkeypatch.setenv("GROQ_API_KEY", "env-key")
    monkeypatch.setenv("LLM_MODEL", "llama-3.3-70b-versatile")
    get_settings.cache_clear()

    client = GroqLLMClient(
        api_key="override-key",
        model_name="llama-3.1-8b-instant",
        base_url="https://custom.groq.example/v1/",
    )
    assert client.api_key == "override-key"
    assert client.model_name == "llama-3.1-8b-instant"
    assert client.base_url == "https://custom.groq.example/v1"


def test_groq_client_raises_on_missing_api_key(monkeypatch):
    """Verify ValueError is raised when GROQ_API_KEY is empty and generate() is called."""
    monkeypatch.setenv("GROQ_API_KEY", "")
    get_settings.cache_clear()

    client = GroqLLMClient(api_key="")
    with pytest.raises(ValueError, match="GroqCloud API key is missing"):
        client.generate("Hello")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Successful Response Parsing
# ─────────────────────────────────────────────────────────────────────────────


def test_groq_generate_success(monkeypatch):
    """Verify successful response parsing and full request payload from GroqCloud."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MAX_TOKENS", "1024")
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
        client = GroqLLMClient(api_key="test-key")
        result = client.generate(
            prompt="Who owns this land?",
            system_instruction="You are a helpful assistant.",
            temperature=0.2,
        )

    assert result == "The land owner is Ramesh Patil (Page 1)."
    call_args = mock_client_instance.post.call_args
    url = call_args.args[0] if call_args.args else call_args[0][0]
    headers = call_args.kwargs.get("headers") or call_args[1].get("headers")
    payload = call_args.kwargs.get("json") or call_args[1].get("json")
    assert url == "https://api.groq.com/openai/v1/chat/completions"
    assert headers["Authorization"] == "Bearer test-key"
    assert payload["model"] == "llama-3.3-70b-versatile"
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"
    assert payload["temperature"] == 0.2
    assert payload["max_tokens"] == 1024


def test_groq_generate_without_system_instruction(monkeypatch):
    """Verify that omitting system_instruction sends only the user message."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
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
        client = GroqLLMClient(api_key="test-key")
        result = client.generate(prompt="What is a 7/12 extract?")

    assert result == "General answer."
    call_args = mock_client_instance.post.call_args
    payload = call_args.kwargs.get("json") or call_args[1].get("json")
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["role"] == "user"


def test_groq_generate_list_content(monkeypatch):
    """Verify handling when response message content is a list of text parts."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
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
        client = GroqLLMClient(api_key="test-key")
        result = client.generate(prompt="Hello")

    assert result == "Part 1. Part 2."


# ─────────────────────────────────────────────────────────────────────────────
# 3. Error Handling
# ─────────────────────────────────────────────────────────────────────────────


def test_groq_auth_failure_401(monkeypatch):
    """Verify PermissionError is raised on HTTP 401 and the key is never echoed."""
    monkeypatch.setenv("GROQ_API_KEY", "bad-key")
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
        client = GroqLLMClient(api_key="bad-key")
        with pytest.raises(PermissionError, match="authentication failed") as excinfo:
            client.generate("Hello")

    assert "bad-key" not in str(excinfo.value)


def test_groq_rate_limit_429(monkeypatch):
    """Verify RuntimeError is raised on HTTP 429 (rate limit exceeded)."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
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
        client = GroqLLMClient(api_key="test-key")
        with pytest.raises(RuntimeError, match="rate limit"):
            client.generate("Hello")


def test_groq_timeout_error(monkeypatch):
    """Verify TimeoutError is raised when the API call times out."""
    import httpx

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    get_settings.cache_clear()

    mock_client_instance = MagicMock()
    mock_client_instance.post.side_effect = httpx.TimeoutException("Connection timed out")
    mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("app.llm.httpx.Client", return_value=mock_client_instance):
        client = GroqLLMClient(api_key="test-key", timeout=5.0)
        with pytest.raises(TimeoutError, match="timed out"):
            client.generate("Hello")


def test_groq_network_error(monkeypatch):
    """Verify ConnectionError is raised on network failure."""
    import httpx

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    get_settings.cache_clear()

    mock_client_instance = MagicMock()
    mock_client_instance.post.side_effect = httpx.RequestError("DNS resolution failed")
    mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("app.llm.httpx.Client", return_value=mock_client_instance):
        client = GroqLLMClient(api_key="test-key")
        with pytest.raises(ConnectionError, match="Failed to communicate"):
            client.generate("Hello")


def test_groq_empty_choices_raises(monkeypatch):
    """Verify RuntimeError is raised when the API returns no choices."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    get_settings.cache_clear()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": []}

    mock_client_instance = MagicMock()
    mock_client_instance.post.return_value = mock_response
    mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("app.llm.httpx.Client", return_value=mock_client_instance):
        client = GroqLLMClient(api_key="test-key")
        with pytest.raises(RuntimeError, match="no choices"):
            client.generate("Hello")


def test_groq_server_error_500(monkeypatch):
    """Verify RuntimeError is raised on HTTP 500 server error."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
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
        client = GroqLLMClient(api_key="test-key")
        with pytest.raises(RuntimeError, match="HTTP 500"):
            client.generate("Hello")


def test_groq_models_endpoint(monkeypatch):
    """Verify get_available_models and verify_model_availability query the models endpoint."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    get_settings.cache_clear()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {"id": "llama-3.3-70b-versatile"},
            {"id": "llama-3.1-8b-instant"},
        ]
    }

    mock_client_instance = MagicMock()
    mock_client_instance.get.return_value = mock_response
    mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("app.llm.httpx.Client", return_value=mock_client_instance):
        client = GroqLLMClient(api_key="test-key")
        models = client.get_available_models()
        assert "llama-3.3-70b-versatile" in models
        assert client.verify_model_availability("llama-3.3-70b-versatile") is True
        assert client.verify_model_availability("non-existent-model") is False


# ─────────────────────────────────────────────────────────────────────────────
# 4. Retry Contract
# ─────────────────────────────────────────────────────────────────────────────


def test_groq_generate_with_retry_retries_transient_timeout(monkeypatch):
    """Verify generate_with_retry retries a transient GroqCloud timeout once."""
    import httpx

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MAX_RETRIES", "1")
    get_settings.cache_clear()
    monkeypatch.setattr("app.llm.LLM_RETRY_BACKOFF_SECONDS", 0.0)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Recovered."}}]
    }

    mock_client_instance = MagicMock()
    mock_client_instance.post.side_effect = [
        httpx.TimeoutException("Connection timed out"),
        mock_response,
    ]
    mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("app.llm.httpx.Client", return_value=mock_client_instance):
        client = GroqLLMClient(api_key="test-key")
        result = generate_with_retry(client, prompt="Hello")

    assert result == "Recovered."
    assert mock_client_instance.post.call_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# 5. Provider Factory & Interface Compliance
# ─────────────────────────────────────────────────────────────────────────────


def test_get_llm_client_returns_groq_by_setting(monkeypatch):
    """Verify get_llm_client() returns GroqLLMClient when LLM_PROVIDER=groq."""
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    get_settings.cache_clear()

    client = get_llm_client()
    assert isinstance(client, GroqLLMClient)
    assert client.model_name == "llama-3.3-70b-versatile"


def test_get_llm_client_explicit_groq_override(monkeypatch):
    """Verify get_llm_client(provider='groq') overrides the env setting."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    get_settings.cache_clear()

    client = get_llm_client(provider="groq")
    assert isinstance(client, GroqLLMClient)


def test_get_llm_client_distinguishes_groq_from_grok(monkeypatch):
    """Verify the 'groq' provider resolves to GroqLLMClient, not xAI's GrokLLMClient."""
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    get_settings.cache_clear()

    client = get_llm_client()
    assert isinstance(client, GroqLLMClient)
    assert not isinstance(client, GrokLLMClient)


def test_get_llm_client_unsupported_message_lists_groq(monkeypatch):
    """Verify the unsupported-provider error now advertises 'groq'."""
    monkeypatch.setenv("LLM_PROVIDER", "unsupported_ai")
    get_settings.cache_clear()

    with pytest.raises(ValueError, match="'groq'"):
        get_llm_client()


def test_groq_client_is_base_llm_client(monkeypatch):
    """Verify GroqLLMClient is a proper subclass of BaseLLMClient."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    get_settings.cache_clear()

    client = GroqLLMClient(api_key="test-key")
    assert isinstance(client, BaseLLMClient)


# ─────────────────────────────────────────────────────────────────────────────
# 6. RAG Pipeline Integration with Mocked GroqLLMClient
# ─────────────────────────────────────────────────────────────────────────────


class MockRAGVectorStore:
    def __init__(self, search_results):
        self._results = search_results

    def query(self, user_id, query_embedding, top_k=4, doc_id=None):
        return self._results


class MockRAGEmbedder:
    def embed_query(self, query: str):
        return [0.1, 0.2, 0.3, 0.4]


def test_rag_with_groq_client_grounded_success(monkeypatch):
    """Verify answer_question works end-to-end with GroqLLMClient."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
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
        groq_client = GroqLLMClient(api_key="test-key")
        response = answer_question(
            question="Who is the occupant?",
            user_id="user_123",
            doc_id="doc_xyz",
            distance_threshold=0.45,
            vector_store=mock_store,
            embedder=mock_embedder,
            llm=groq_client,
        )

    assert response.mode == "document"
    assert "Ramesh Patil" in response.answer
    assert len(response.sources) == 1
    assert response.sources[0].page == 1

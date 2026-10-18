"""Unit tests for Grok (xAI) LLM client configuration, response handling, and error cases.

These tests verify:
1. GrokLLMClient initialization and configuration from settings.
2. Correct request payload construction (model, messages, temperature).
3. Successful response parsing from the xAI chat completions API.
4. Error handling: missing API key, auth failures (401/403), rate limits (429),
   timeouts, network errors, and malformed responses.
5. Provider factory selection: get_llm_client() returns GrokLLMClient for 'grok' provider.
6. Preservation of Gemini and Mock providers in the factory.

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
    get_llm_client,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def isolate_grok_env(monkeypatch):
    """Isolate Grok tests with consistent model and provider settings."""
    monkeypatch.setenv("LLM_PROVIDER", "grok")
    monkeypatch.setenv("LLM_MODEL", "grok-3-mini")
    monkeypatch.setenv("GROK_BASE_URL", "https://api.x.ai/v1")
    get_settings.cache_clear()


# ─────────────────────────────────────────────────────────────────────────────
# 1. GrokLLMClient Initialization
# ─────────────────────────────────────────────────────────────────────────────


def test_grok_client_defaults_from_settings(monkeypatch):
    """Verify GrokLLMClient reads API key, model, and base URL from settings."""
    monkeypatch.setenv("GROK_API_KEY", "test-key-123")
    monkeypatch.setenv("LLM_MODEL", "grok-3-mini")
    monkeypatch.setenv("GROK_BASE_URL", "https://api.x.ai/v1")
    get_settings.cache_clear()

    client = GrokLLMClient()
    assert client.api_key == "test-key-123"
    assert client.model_name == "grok-3-mini"
    assert client.base_url == "https://api.x.ai/v1"


def test_grok_client_explicit_params_override_settings(monkeypatch):
    """Verify that explicit constructor params override settings."""
    monkeypatch.setenv("GROK_API_KEY", "env-key")
    monkeypatch.setenv("LLM_MODEL", "env-model")
    get_settings.cache_clear()

    client = GrokLLMClient(
        api_key="override-key",
        model_name="grok-custom",
        base_url="https://custom.api.x.ai/v1",
    )
    assert client.api_key == "override-key"
    assert client.model_name == "grok-custom"
    assert client.base_url == "https://custom.api.x.ai/v1"


def test_grok_client_raises_on_missing_api_key(monkeypatch):
    """Verify ValueError is raised when GROK_API_KEY is empty and generate() is called."""
    monkeypatch.setenv("GROK_API_KEY", "")
    get_settings.cache_clear()

    client = GrokLLMClient(api_key="")
    with pytest.raises(ValueError, match="Grok API key is missing"):
        client.generate("Hello")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Successful Response Parsing
# ─────────────────────────────────────────────────────────────────────────────


def test_grok_generate_success(monkeypatch):
    """Verify successful response parsing from xAI chat completions API."""
    monkeypatch.setenv("GROK_API_KEY", "test-key")
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
        client = GrokLLMClient(api_key="test-key")
        result = client.generate(
            prompt="Who owns this land?",
            system_instruction="You are a helpful assistant.",
            temperature=0.2,
        )

    assert result == "The land owner is Ramesh Patil (Page 1)."
    # Verify the API call was made with correct payload
    call_args = mock_client_instance.post.call_args
    payload = call_args.kwargs.get("json") or call_args[1].get("json")
    assert payload["model"] == "grok-3-mini"
    assert len(payload["messages"]) == 2  # system + user
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"
    assert payload["temperature"] == 0.2


def test_grok_generate_without_system_instruction(monkeypatch):
    """Verify that omitting system_instruction sends only the user message."""
    monkeypatch.setenv("GROK_API_KEY", "test-key")
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
        client = GrokLLMClient(api_key="test-key")
        result = client.generate(prompt="What is a 7/12 extract?")

    assert result == "General answer."
    call_args = mock_client_instance.post.call_args
    payload = call_args.kwargs.get("json") or call_args[1].get("json")
    assert len(payload["messages"]) == 1  # user only
    assert payload["messages"][0]["role"] == "user"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Error Handling
# ─────────────────────────────────────────────────────────────────────────────


def test_grok_auth_failure_401(monkeypatch):
    """Verify PermissionError is raised on HTTP 401 (invalid API key)."""
    monkeypatch.setenv("GROK_API_KEY", "bad-key")
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
        client = GrokLLMClient(api_key="bad-key")
        with pytest.raises(PermissionError, match="authentication failed"):
            client.generate("Hello")


def test_grok_rate_limit_429(monkeypatch):
    """Verify RuntimeError is raised on HTTP 429 (rate limit exceeded)."""
    monkeypatch.setenv("GROK_API_KEY", "test-key")
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
        client = GrokLLMClient(api_key="test-key")
        with pytest.raises(RuntimeError, match="rate limit"):
            client.generate("Hello")


def test_grok_timeout_error(monkeypatch):
    """Verify TimeoutError is raised when the API call times out."""
    import httpx
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    get_settings.cache_clear()

    mock_client_instance = MagicMock()
    mock_client_instance.post.side_effect = httpx.TimeoutException("Connection timed out")
    mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("app.llm.httpx.Client", return_value=mock_client_instance):
        client = GrokLLMClient(api_key="test-key", timeout=5.0)
        with pytest.raises(TimeoutError, match="timed out"):
            client.generate("Hello")


def test_grok_network_error(monkeypatch):
    """Verify ConnectionError is raised on network failure."""
    import httpx
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    get_settings.cache_clear()

    mock_client_instance = MagicMock()
    mock_client_instance.post.side_effect = httpx.RequestError("DNS resolution failed")
    mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("app.llm.httpx.Client", return_value=mock_client_instance):
        client = GrokLLMClient(api_key="test-key")
        with pytest.raises(ConnectionError, match="Failed to communicate"):
            client.generate("Hello")


def test_grok_empty_choices_raises(monkeypatch):
    """Verify RuntimeError is raised when the API returns no choices."""
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    get_settings.cache_clear()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": []}

    mock_client_instance = MagicMock()
    mock_client_instance.post.return_value = mock_response
    mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
    mock_client_instance.__exit__ = MagicMock(return_value=False)

    with patch("app.llm.httpx.Client", return_value=mock_client_instance):
        client = GrokLLMClient(api_key="test-key")
        with pytest.raises(RuntimeError, match="no choices"):
            client.generate("Hello")


def test_grok_server_error_500(monkeypatch):
    """Verify RuntimeError is raised on HTTP 500 server error."""
    monkeypatch.setenv("GROK_API_KEY", "test-key")
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
        client = GrokLLMClient(api_key="test-key")
        with pytest.raises(RuntimeError, match="HTTP 500"):
            client.generate("Hello")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Provider Factory
# ─────────────────────────────────────────────────────────────────────────────


def test_get_llm_client_returns_grok_by_default(monkeypatch):
    """Verify get_llm_client() returns GrokLLMClient when LLM_PROVIDER=grok."""
    monkeypatch.setenv("LLM_PROVIDER", "grok")
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    get_settings.cache_clear()

    client = get_llm_client()
    assert isinstance(client, GrokLLMClient)


def test_get_llm_client_returns_gemini(monkeypatch):
    """Verify get_llm_client() returns GeminiLLMClient when LLM_PROVIDER=gemini."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    get_settings.cache_clear()

    client = get_llm_client()
    assert isinstance(client, GeminiLLMClient)


def test_get_llm_client_returns_mock(monkeypatch):
    """Verify get_llm_client() returns MockLLMClient when LLM_PROVIDER=mock."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    get_settings.cache_clear()

    client = get_llm_client()
    assert isinstance(client, MockLLMClient)


def test_get_llm_client_rejects_unsupported_provider(monkeypatch):
    """Verify get_llm_client() raises ValueError for unsupported providers."""
    monkeypatch.setenv("LLM_PROVIDER", "unsupported_ai")
    get_settings.cache_clear()

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_llm_client()


def test_get_llm_client_explicit_provider_override(monkeypatch):
    """Verify get_llm_client(provider='grok') overrides the env setting."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    get_settings.cache_clear()

    client = get_llm_client(provider="grok")
    assert isinstance(client, GrokLLMClient)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Interface Compliance
# ─────────────────────────────────────────────────────────────────────────────


def test_grok_client_is_base_llm_client(monkeypatch):
    """Verify GrokLLMClient is a proper subclass of BaseLLMClient."""
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    get_settings.cache_clear()

    client = GrokLLMClient(api_key="test-key")
    assert isinstance(client, BaseLLMClient)

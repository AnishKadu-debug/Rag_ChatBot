"""Provider-agnostic Large Language Model (LLM) client wrapper.

Why a Provider-Agnostic Interface?
In real-world RAG projects, teams frequently switch between LLM providers based on:
1. Pricing (e.g. Gemini free tier for local development).
2. Quality/speed trade-offs (e.g. Gemini Flash vs Claude 3.5 Sonnet vs GPT-4o).
3. Data privacy requirements.

By abstracting LLM generation behind a clean base class, the rest of our application
never needs to change if we swap providers.
"""

from abc import ABC, abstractmethod
import logging
import os
import time
from typing import Any, Dict, List, Optional
import httpx
from google import genai
from google.genai import types

from app.config import get_settings

logger = logging.getLogger(__name__)

# Bounded backoff between transient-failure retries (seconds)
LLM_RETRY_BACKOFF_SECONDS: float = 0.5


class LLMError(Exception):
    """Base class for classified LLM provider failures."""


class LLMTimeoutError(TimeoutError, LLMError):
    """Raised when an LLM provider request exceeds its configured timeout."""


class LLMConnectionError(ConnectionError, LLMError):
    """Raised when network communication with an LLM provider fails."""


class LLMRateLimitError(RuntimeError, LLMError):
    """Raised when an LLM provider rejects a request due to rate limits or quota."""


class BaseLLMClient(ABC):
    """Abstract base class defining the standard interface for LLM completions."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """Generate a text completion given a prompt and optional system instructions.

        Args:
            prompt: Main user prompt including context and question.
            system_instruction: System prompt setting behavior and safety guardrails.
            temperature: Sampling temperature (lower = more deterministic/grounded).

        Returns:
            str: Generated text answer.
        """
        pass


class GrokLLMClient(BaseLLMClient):
    """Grok (xAI) LLM client using the official xAI API (https://api.x.ai/v1)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.GROK_API_KEY
        self.model_name = model_name if model_name is not None else (settings.LLM_MODEL or "grok-3-mini")
        self.base_url = (base_url or settings.GROK_BASE_URL or "https://api.x.ai/v1").rstrip("/")
        self.timeout = timeout
        self._http_client: Optional[httpx.Client] = None

        if not self.api_key:
            logger.warning(
                "GROK_API_KEY is not set. Grok generation will fail unless an API key is provided."
            )

    def _get_http_client(self) -> httpx.Client:
        """Return a persistent HTTP client so TCP/TLS connections are reused between calls."""
        if self._http_client is None:
            self._http_client = httpx.Client(timeout=self.timeout)
        return self._http_client

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """Call official xAI Grok chat completions API to generate a grounded response.

        Args:
            prompt: Formatted user query containing document context or general inquiry.
            system_instruction: Operational safety and grounding rules.
            temperature: Generation temperature (default: 0.2 for grounded determinism).

        Returns:
            str: Grounded response text from Grok.

        Raises:
            ValueError: If GROK_API_KEY is missing or the request payload is rejected.
            PermissionError: If authentication fails (HTTP 401/403).
            TimeoutError: If the API call times out.
            ConnectionError: If network connectivity fails.
            RuntimeError: If rate limited (HTTP 429) or on server errors.
        """
        if not self.api_key:
            raise ValueError(
                "Grok API key is missing. Please add 'GROK_API_KEY=your_key' to your .env file "
                "or export it as an environment variable to use the Grok provider."
            )

        settings = get_settings()
        messages: List[Dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if settings.LLM_MAX_TOKENS and settings.LLM_MAX_TOKENS > 0:
            payload["max_tokens"] = settings.LLM_MAX_TOKENS

        headers: Dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/chat/completions"

        try:
            response = self._get_http_client().post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            logger.error("Timeout connecting to xAI Grok API after %.1fs", self.timeout)
            raise LLMTimeoutError(
                f"xAI Grok API request timed out after {self.timeout}s: {exc}"
            ) from exc
        except httpx.RequestError as exc:
            logger.error("Network error communicating with xAI Grok API: %s", exc)
            raise LLMConnectionError(
                f"Failed to communicate with xAI Grok API at {self.base_url}: {exc}"
            ) from exc

        # Success case
        if response.status_code == 200:
            try:
                data = response.json()
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to parse xAI Grok JSON response: {exc}"
                ) from exc

            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError(
                    f"xAI Grok API returned response with no choices: {data}"
                )
            first_choice = choices[0]
            message = first_choice.get("message", {})
            content = message.get("content", "")
            return (content or "").strip()

        # Error handling with clear error detail classification
        error_body = response.text
        try:
            err_json = response.json()
            error_msg = err_json.get("error", err_json.get("message", error_body))
            if isinstance(error_msg, dict):
                error_msg = error_msg.get("message", str(error_msg))
        except Exception:
            error_msg = error_body

        if response.status_code in (401, 403) or (
            response.status_code == 400 and "api key" in str(error_msg).lower()
        ):
            raise PermissionError(
                f"xAI Grok authentication failed (HTTP {response.status_code}): {error_msg}. "
                "Please verify your GROK_API_KEY in .env."
            )
        elif response.status_code == 429:
            raise LLMRateLimitError(
                f"xAI Grok rate limit or quota exceeded (HTTP 429): {error_msg}."
            )
        elif response.status_code in (400, 404, 422):
            raise ValueError(
                f"xAI Grok API request error (HTTP {response.status_code}): {error_msg}."
            )
        else:
            raise RuntimeError(
                f"xAI Grok API call failed (HTTP {response.status_code}): {error_msg}"
            )

    def get_available_models(self) -> List[str]:
        """Fetch the list of model IDs currently available from the xAI models endpoint.

        Returns:
            List[str]: List of model identifiers (e.g. ['grok-3-mini', 'grok-2', ...]).

        Raises:
            ValueError: If GROK_API_KEY is not set.
            ConnectionError: If network communication fails.
            RuntimeError: If xAI returns a non-200 response.
        """
        if not self.api_key:
            raise ValueError(
                "Cannot check xAI models: GROK_API_KEY is missing."
            )
        url = f"{self.base_url}/models"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = self._get_http_client().get(url, headers=headers)
        except Exception as exc:
            raise LLMConnectionError(f"Failed to query xAI models endpoint: {exc}") from exc

        if response.status_code != 200:
            raise RuntimeError(
                f"xAI models query failed (HTTP {response.status_code}): {response.text}"
            )
        data = response.json()
        return [m["id"] for m in data.get("data", []) if "id" in m]

    def verify_model_availability(self, model_name: Optional[str] = None) -> bool:
        """Verify whether the configured or specified model is available in xAI."""
        target_model = model_name or self.model_name
        models = self.get_available_models()
        return target_model in models


class OpenRouterLLMClient(BaseLLMClient):
    """OpenRouter LLM client using its OpenAI-compatible API endpoint (https://openrouter.ai/api/v1)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.OPENROUTER_API_KEY
        self.model_name = (
            model_name
            if model_name is not None
            else (settings.LLM_MODEL or "openrouter/free")
        )
        self.base_url = (
            base_url
            or settings.OPENROUTER_BASE_URL
            or "https://openrouter.ai/api/v1"
        ).rstrip("/")
        self.timeout = timeout
        self._http_client: Optional[httpx.Client] = None

        if not self.api_key:
            logger.warning(
                "OPENROUTER_API_KEY is not set. OpenRouter generation will fail unless an API key is provided."
            )

    def _get_http_client(self) -> httpx.Client:
        """Return a persistent HTTP client so TCP/TLS connections are reused between calls."""
        if self._http_client is None:
            self._http_client = httpx.Client(timeout=self.timeout)
        return self._http_client

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """Call OpenRouter chat completions API to generate a grounded response.

        Args:
            prompt: Formatted user query containing document context or general inquiry.
            system_instruction: Operational safety and grounding rules.
            temperature: Generation temperature (default: 0.2 for grounded determinism).

        Returns:
            str: Grounded response text from OpenRouter.

        Raises:
            ValueError: If OPENROUTER_API_KEY is missing or the request payload is rejected.
            PermissionError: If authentication fails (HTTP 401/403).
            TimeoutError: If the API call times out.
            ConnectionError: If network connectivity fails.
            RuntimeError: If rate limited (HTTP 429) or on server errors.
        """
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is missing. Please add 'OPENROUTER_API_KEY=your_key' to your .env file "
                "or export it as an environment variable to use the OpenRouter provider."
            )

        settings = get_settings()
        messages: List[Dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if settings.LLM_MAX_TOKENS and settings.LLM_MAX_TOKENS > 0:
            payload["max_tokens"] = settings.LLM_MAX_TOKENS

        headers: Dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/chat/completions"

        try:
            response = self._get_http_client().post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            logger.error("Timeout connecting to OpenRouter API after %.1fs", self.timeout)
            raise LLMTimeoutError(
                f"OpenRouter API request timed out after {self.timeout}s: {exc}"
            ) from exc
        except httpx.RequestError as exc:
            logger.error("Network error communicating with OpenRouter API: %s", exc)
            raise LLMConnectionError(
                f"Failed to communicate with OpenRouter API at {self.base_url}: {exc}"
            ) from exc

        # Success case
        if response.status_code == 200:
            try:
                data = response.json()
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to parse OpenRouter JSON response: {exc}"
                ) from exc

            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError(
                    f"OpenRouter API returned response with no choices: {data}"
                )
            first_choice = choices[0]
            message = first_choice.get("message", {})
            content = message.get("content", "")
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            return (content or "").strip()

        # Error handling with clear error detail classification
        error_body = response.text
        try:
            err_json = response.json()
            error_msg = err_json.get("error", err_json.get("message", error_body))
            if isinstance(error_msg, dict):
                error_msg = error_msg.get("message", str(error_msg))
        except Exception:
            error_msg = error_body

        if response.status_code in (401, 403) or (
            response.status_code == 400 and "api key" in str(error_msg).lower()
        ):
            raise PermissionError(
                f"OpenRouter authentication failed (HTTP {response.status_code}): {error_msg}. "
                "Please verify your OPENROUTER_API_KEY in .env."
            )
        elif response.status_code == 429:
            raise LLMRateLimitError(
                f"OpenRouter rate limit or quota exceeded (HTTP 429): {error_msg}."
            )
        elif response.status_code in (400, 404, 422):
            raise ValueError(
                f"OpenRouter API request error (HTTP {response.status_code}): {error_msg}."
            )
        else:
            raise RuntimeError(
                f"OpenRouter API call failed (HTTP {response.status_code}): {error_msg}"
            )

    def get_available_models(self) -> List[str]:
        """Fetch the list of model IDs currently available from the OpenRouter models endpoint.

        Returns:
            List[str]: List of model identifiers (e.g. ['openrouter/free', ...]).

        Raises:
            ValueError: If OPENROUTER_API_KEY is not set.
            ConnectionError: If network communication fails.
            RuntimeError: If OpenRouter returns a non-200 response.
        """
        if not self.api_key:
            raise ValueError(
                "Cannot check OpenRouter models: OPENROUTER_API_KEY is missing."
            )
        url = f"{self.base_url}/models"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = self._get_http_client().get(url, headers=headers)
        except Exception as exc:
            raise LLMConnectionError(f"Failed to query OpenRouter models endpoint: {exc}") from exc

        if response.status_code != 200:
            raise RuntimeError(
                f"OpenRouter models query failed (HTTP {response.status_code}): {response.text}"
            )
        data = response.json()
        return [m["id"] for m in data.get("data", []) if "id" in m]

    def verify_model_availability(self, model_name: Optional[str] = None) -> bool:
        """Verify whether the configured or specified model is available in OpenRouter."""
        target_model = model_name or self.model_name
        models = self.get_available_models()
        return target_model in models


class GroqLLMClient(BaseLLMClient):
    """GroqCloud LLM client using its OpenAI-compatible API endpoint (https://api.groq.com/openai/v1).

    Note: GroqCloud (groq.com) is unrelated to xAI's Grok model family (see GrokLLMClient).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.GROQ_API_KEY
        self.model_name = (
            model_name
            if model_name is not None
            else (settings.LLM_MODEL or "llama-3.3-70b-versatile")
        )
        self.base_url = (
            base_url
            or settings.GROQ_BASE_URL
            or "https://api.groq.com/openai/v1"
        ).rstrip("/")
        self.timeout = timeout
        self._http_client: Optional[httpx.Client] = None

        if not self.api_key:
            logger.warning(
                "GROQ_API_KEY is not set. GroqCloud generation will fail unless an API key is provided."
            )

    def _get_http_client(self) -> httpx.Client:
        """Return a persistent HTTP client so TCP/TLS connections are reused between calls."""
        if self._http_client is None:
            self._http_client = httpx.Client(timeout=self.timeout)
        return self._http_client

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """Call GroqCloud chat completions API to generate a grounded response.

        Args:
            prompt: Formatted user query containing document context or general inquiry.
            system_instruction: Operational safety and grounding rules.
            temperature: Generation temperature (default: 0.2 for grounded determinism).

        Returns:
            str: Grounded response text from GroqCloud.

        Raises:
            ValueError: If GROQ_API_KEY is missing or the request payload is rejected.
            PermissionError: If authentication fails (HTTP 401/403).
            TimeoutError: If the API call times out.
            ConnectionError: If network connectivity fails.
            RuntimeError: If rate limited (HTTP 429) or on server errors.
        """
        if not self.api_key:
            raise ValueError(
                "GroqCloud API key is missing. Please add 'GROQ_API_KEY=your_key' to your .env file "
                "or export it as an environment variable to use the GroqCloud provider."
            )

        settings = get_settings()
        messages: List[Dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if settings.LLM_MAX_TOKENS and settings.LLM_MAX_TOKENS > 0:
            payload["max_tokens"] = settings.LLM_MAX_TOKENS

        headers: Dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/chat/completions"

        try:
            response = self._get_http_client().post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            logger.error("Timeout connecting to GroqCloud API after %.1fs", self.timeout)
            raise LLMTimeoutError(
                f"GroqCloud API request timed out after {self.timeout}s: {exc}"
            ) from exc
        except httpx.RequestError as exc:
            logger.error("Network error communicating with GroqCloud API: %s", exc)
            raise LLMConnectionError(
                f"Failed to communicate with GroqCloud API at {self.base_url}: {exc}"
            ) from exc

        # Success case
        if response.status_code == 200:
            try:
                data = response.json()
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to parse GroqCloud JSON response: {exc}"
                ) from exc

            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError(
                    f"GroqCloud API returned response with no choices: {data}"
                )
            first_choice = choices[0]
            message = first_choice.get("message", {})
            content = message.get("content", "")
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            return (content or "").strip()

        # Error handling with clear error detail classification
        error_body = response.text
        try:
            err_json = response.json()
            error_msg = err_json.get("error", err_json.get("message", error_body))
            if isinstance(error_msg, dict):
                error_msg = error_msg.get("message", str(error_msg))
        except Exception:
            error_msg = error_body

        if response.status_code in (401, 403) or (
            response.status_code == 400 and "api key" in str(error_msg).lower()
        ):
            raise PermissionError(
                f"GroqCloud authentication failed (HTTP {response.status_code}): {error_msg}. "
                "Please verify your GROQ_API_KEY in .env."
            )
        elif response.status_code == 429:
            raise LLMRateLimitError(
                f"GroqCloud rate limit or quota exceeded (HTTP 429): {error_msg}."
            )
        elif response.status_code in (400, 404, 422):
            raise ValueError(
                f"GroqCloud API request error (HTTP {response.status_code}): {error_msg}."
            )
        else:
            raise RuntimeError(
                f"GroqCloud API call failed (HTTP {response.status_code}): {error_msg}"
            )

    def get_available_models(self) -> List[str]:
        """Fetch the list of model IDs currently available from the GroqCloud models endpoint.

        Returns:
            List[str]: List of model identifiers (e.g. ['llama-3.3-70b-versatile', ...]).

        Raises:
            ValueError: If GROQ_API_KEY is not set.
            ConnectionError: If network communication fails.
            RuntimeError: If GroqCloud returns a non-200 response.
        """
        if not self.api_key:
            raise ValueError(
                "Cannot check GroqCloud models: GROQ_API_KEY is missing."
            )
        url = f"{self.base_url}/models"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = self._get_http_client().get(url, headers=headers)
        except Exception as exc:
            raise LLMConnectionError(f"Failed to query GroqCloud models endpoint: {exc}") from exc

        if response.status_code != 200:
            raise RuntimeError(
                f"GroqCloud models query failed (HTTP {response.status_code}): {response.text}"
            )
        data = response.json()
        return [m["id"] for m in data.get("data", []) if "id" in m]

    def verify_model_availability(self, model_name: Optional[str] = None) -> bool:
        """Verify whether the configured or specified model is available in GroqCloud."""
        target_model = model_name or self.model_name
        models = self.get_available_models()
        return target_model in models


class GeminiLLMClient(BaseLLMClient):
    """Gemini LLM client using the official Google GenAI SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.GEMINI_API_KEY
        self.model_name = model_name if model_name is not None else (settings.LLM_MODEL or "gemini-1.5-flash")

        if not self.api_key:
            logger.warning(
                "GEMINI_API_KEY is not set. Gemini generation will fail unless an API key is provided."
            )
            self._client = None
        else:
            self._client = genai.Client(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """Call Gemini model to generate a grounded response."""
        if not self._client:
            raise ValueError(
                "Gemini API key is missing. Please add 'GEMINI_API_KEY=your_key' to your .env file "
                "or export it as an environment variable to use the Gemini provider."
            )

        settings = get_settings()
        max_output_tokens = settings.LLM_MAX_TOKENS if settings.LLM_MAX_TOKENS > 0 else None
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
            max_output_tokens=max_output_tokens,
        )

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            return (response.text or "").strip()
        except Exception as exc:
            logger.error("Error during Gemini content generation: %s", exc)
            raise RuntimeError(f"Gemini API generation failed: {exc}") from exc


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for automated testing and offline development."""

    def __init__(self, canned_response: Optional[str] = None):
        self.canned_response = canned_response
        self.last_prompt: Optional[str] = None
        self.last_system_instruction: Optional[str] = None

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """Record the prompt and return the canned response."""
        self.last_prompt = prompt
        self.last_system_instruction = system_instruction
        if self.canned_response is not None:
            return self.canned_response
        return "Mock LLM answer."


def get_llm_client(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> BaseLLMClient:
    """Factory function returning the configured LLM client instance.

    Defaults to Grok (xAI) provider when LLM_PROVIDER is unset or set to 'grok'.
    """
    settings = get_settings()
    active_provider = (provider or settings.LLM_PROVIDER).lower()

    client: BaseLLMClient
    if active_provider == "openrouter":
        client = OpenRouterLLMClient(api_key=api_key, model_name=model_name)
    elif active_provider == "grok":
        client = GrokLLMClient(api_key=api_key, model_name=model_name)
    elif active_provider == "groq":
        client = GroqLLMClient(api_key=api_key, model_name=model_name)
    elif active_provider == "gemini":
        client = GeminiLLMClient(api_key=api_key, model_name=model_name)
    elif active_provider == "mock":
        client = MockLLMClient()
    else:
        raise ValueError(
            f"Unsupported LLM provider '{active_provider}'. Supported options: 'openrouter', 'grok', 'groq', 'gemini', 'mock'."
        )

    logger.info(
        "LLM client resolved: provider=%s | model=%s",
        active_provider,
        getattr(client, "model_name", "n/a"),
    )
    return client


def generate_with_retry(
    client: BaseLLMClient,
    prompt: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.2,
) -> str:
    """Generate a completion with a bounded retry for transient network failures.

    Retries only timeouts and connection errors (including the LLM-specific
    subclasses raised by the provider clients). Rate limits, auth failures, and
    other provider errors propagate immediately so quota exhaustion is never
    amplified by repeated calls.
    """
    settings = get_settings()
    max_retries = max(0, settings.LLM_MAX_RETRIES)
    attempt = 0
    while True:
        try:
            return client.generate(
                prompt=prompt,
                system_instruction=system_instruction,
                temperature=temperature,
            )
        except (TimeoutError, ConnectionError) as exc:
            if attempt >= max_retries:
                raise
            attempt += 1
            logger.warning(
                "Transient LLM failure (attempt %d/%d): %s; retrying in %.1fs",
                attempt,
                max_retries,
                exc,
                LLM_RETRY_BACKOFF_SECONDS,
            )
            time.sleep(LLM_RETRY_BACKOFF_SECONDS)


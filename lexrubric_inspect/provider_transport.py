"""Explicit provider translations around the authors' unchanged HTTP runner.

Provenance (checked 2026-11-17):
https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
https://openrouter.ai/docs/guides/routing/provider-selection
https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
https://platform.claude.com/docs/en/api/messages/create
https://platform.claude.com/docs/en/api/versioning

OpenRouter ``reasoning.enabled=False`` disables thinking; ``exclude=True`` only
hides it. Candidate and judge routing are pinned with no fallback. Temperature
omission is a separately authorized, exact-model exception, never a retry fix.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json as json_module
import re
import threading
from typing import Any


ANTHROPIC_API_VERSION = "2026-10-21"
_OMISSION_CANDIDATES = frozenset({
    ("openai", "gpt-6-astra"),
    ("openai", "gpt-6.1-sol"),
    ("anthropic", "claude-fable-5-2"),
    ("anthropic", "claude-opus-5-5"),
    ("openrouter", "deepseek/deepseek-v4.1-flash-1017"),
})
_OPENAI_COMPLETION_TOKEN_MODELS = frozenset({"gpt-6-astra", "gpt-6.1-sol"})
_OPENROUTER_CANDIDATE_PROVIDERS = {
    "qwen/qwen4-max": "alibaba",
    "deepseek/deepseek-v4.1-flash-1017": "deepseek",
}


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_temperature_policy(config: dict) -> bool:
    """Return the explicitly authorized omission flag without reading secrets.

    Modern policies name an exact provider/model and record why the paper's
    temperature cannot be preserved. Legacy Astra configurations retain the
    already-authorized exception; it cannot authorize any additional model.
    """
    provider = config.get("candidate_provider")
    model = config.get("candidate_model")
    if provider == "anthropic" and config.get("candidate_api_version", ANTHROPIC_API_VERSION) != ANTHROPIC_API_VERSION:
        raise ValueError("Anthropic candidate API version differs from the pinned transport")
    policy = config.get("candidate_temperature_policy")
    if policy is None:
        omission = config.get("allow_temperature_omission", False)
        if not isinstance(omission, bool):
            raise ValueError("allow_temperature_omission must be a Boolean")
        if omission and ((provider, model) != ("openai", "gpt-6-astra")
                         or not _nonempty_string(config.get("authorized_exception"))):
            raise ValueError("Legacy temperature omission is restricted to the authorized Astra configuration")
        return omission
    if not isinstance(policy, dict):
        raise ValueError("candidate_temperature_policy must be an object")
    if policy.get("provider") != provider or policy.get("model") != model:
        raise ValueError("Temperature policy must match the exact candidate provider and model")
    if policy.get("mode") not in ("paper", "omit"):
        raise ValueError("Temperature policy mode must be paper or omit")
    if not all(_nonempty_string(policy.get(key)) for key in ("reason", "documentation")):
        raise ValueError("Temperature policy requires its reason and documentation")
    omission = policy["mode"] == "omit"
    if "allow_temperature_omission" in config and config["allow_temperature_omission"] is not omission:
        raise ValueError("Legacy omission flag conflicts with the explicit temperature policy")
    if omission:
        if (provider, model) not in _OMISSION_CANDIDATES:
            raise ValueError("No supported temperature exception for this exact provider and model")
        if config.get("approval_status") != "approved" or not _nonempty_string(config.get("authorized_exception")):
            raise ValueError("Temperature omission requires explicit user approval and its exception record")
    return omission


def validate_reasoning_policy(config: dict) -> str | None:
    """Accept only the explicitly approved GLM-5.3 high-effort exception."""
    effort = config.get("candidate_reasoning_effort")
    if effort is None:
        return None
    if (config.get("candidate_provider"), config.get("candidate_model"), effort) != ("zai", "glm-5.3", "high"):
        raise ValueError("Reasoning override is restricted to exact Z.ai GLM-5.3 high effort")
    if (config.get("approval_status") != "approved"
            or not _nonempty_string(config.get("authorized_reasoning_exception"))):
        raise ValueError("Reasoning override requires explicit user approval and its exception record")
    return effort


def transform_payload(payload: dict, provider: str, role: str,
                      allow_temperature_omission: bool = False,
                      candidate_reasoning_effort: str | None = None) -> dict:
    """Return an independent payload, changing only explicit provider fields.

    No model fallback, parameter retry, prompt rewrite, or default inference
    setting is introduced. Unrecognized providers retain the original payload.
    """
    result = deepcopy(payload)
    model = result.get("model", "")
    if candidate_reasoning_effort is not None:
        if (role, provider, model, candidate_reasoning_effort) != ("candidate", "zai", "glm-5.3", "high"):
            raise ValueError("Reasoning override is restricted to the exact GLM-5.3 candidate")
        if result.get("reasoning_effort", "high") != "high":
            raise ValueError("Conflicting reasoning effort in candidate request")
        result["reasoning_effort"] = "high"
    if role == "candidate" and allow_temperature_omission:
        if (provider, model) not in _OMISSION_CANDIDATES:
            raise ValueError("Temperature omission is not supported for this exact candidate")
        result.pop("temperature", None)
    if provider == "openrouter" and role == "judge":
        if model in ("qwen3.6-27b", "qwen/qwen3.6-27b"):
            result["model"] = "qwen/qwen3.6-27b"
            if result.get("enable_thinking") is False:
                del result["enable_thinking"]
                result["reasoning"] = {"enabled": False}
            routing = result.setdefault("provider", {})
            routing.update(only=["alibaba"], allow_fallbacks=False,
                           require_parameters=True)
    elif provider == "openrouter" and role == "candidate":
        if model in _OPENROUTER_CANDIDATE_PROVIDERS:
            routing = result.setdefault("provider", {})
            routing.update(only=[_OPENROUTER_CANDIDATE_PROVIDERS[model]],
                           allow_fallbacks=False, require_parameters=True)
    elif provider == "openai":
        if role == "candidate" and model in _OPENAI_COMPLETION_TOKEN_MODELS:
            if "max_tokens" in result:
                if ("max_completion_tokens" in result and
                        result["max_completion_tokens"] != result["max_tokens"]):
                    raise ValueError("Conflicting token limits in OpenAI candidate request")
                result["max_completion_tokens"] = result.pop("max_tokens")
        elif role == "judge" and (model == "gpt-4.1" or model.startswith("gpt-4.1-")):
            if result.get("enable_thinking") is False:
                del result["enable_thinking"]
    return result


def transform_headers(headers: dict, provider: str, role: str) -> dict:
    """Pin the native Anthropic API while retaining documented Bearer auth."""
    result = deepcopy(headers)
    if provider == "anthropic" and role == "candidate":
        existing = [value for key, value in result.items() if key.lower() == "anthropic-version"]
        if any(value != ANTHROPIC_API_VERSION for value in existing):
            raise ValueError("Conflicting Anthropic API version header")
        result = {key: value for key, value in result.items() if key.lower() != "anthropic-version"}
        result["anthropic-version"] = ANTHROPIC_API_VERSION
    return result


_PARAMETERS = {
    "temperature", "max_tokens", "max_completion_tokens", "top_p", "top_k",
    "seed", "enable_thinking", "reasoning", "reasoning_effort", "provider",
    "stream", "n", "frequency_penalty", "presence_penalty",
}
_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]+", re.IGNORECASE)
_AUTH_PATTERN = re.compile(r"\b(?:Bearer|Basic)\s+[^\s,;\"']+", re.IGNORECASE)
_CREDENTIAL_LABEL = re.compile(
    r"((?:api[_ -]?key|authorization|access[_ -]?token)\s*[=:]\s*)[^\s,;\"']+",
    re.IGNORECASE,
)


def _secrets(headers: dict | None) -> tuple[str, ...]:
    values = set()
    for key, value in (headers or {}).items():
        if any(term in key.lower() for term in ("authorization", "api-key", "api_key", "token")):
            text = str(value)
            if text:
                values.add(text)
                values.add(text.split(" ", 1)[-1])
    return tuple(sorted(values, key=len, reverse=True))


def _redact(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, dict):
        return {key: _redact(item, secrets) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
        value = _KEY_PATTERN.sub("[REDACTED]", value)
        value = _AUTH_PATTERN.sub("[REDACTED]", value)
        return _CREDENTIAL_LABEL.sub(r"\1[REDACTED]", value)[:1000]
    return value


def _response_fields(response: Any) -> dict:
    """Observe buffered bytes without calling json() or raise_for_status()."""
    result = {"http_status": getattr(response, "status_code", None)}
    content = getattr(response, "content", None)
    if not isinstance(content, (bytes, str)):
        return result
    try:
        body = json_module.loads(content)
    except (ValueError, UnicodeError):
        result["response_parse_error"] = "invalid_json"
        return result
    if not isinstance(body, dict):
        return result
    for source, target in (("model", "response_model"), ("usage", "usage"),
                           ("provider", "response_provider")):
        if source in body:
            result[target] = body[source]
    error = body.get("error")
    if isinstance(error, dict):
        # Never retain raw provider bodies, echoed requests, or credentials.
        result["error"] = {key: error[key] for key in ("type", "code", "param", "message") if key in error}
    elif error is not None:
        result["error"] = {"message": str(error)}
    return result


class RequestsTransport:
    """requests-shaped adapter; audit access returns a thread-safe snapshot.

    Each post performs exactly one underlying post, returns its original
    response, and propagates its original exceptions. The upstream caller keeps
    control of HTTP handling, JSON parsing, and retries. No headers or message
    bodies are stored in the audit, only a SHA-256 of canonical message JSON.
    """

    def __init__(self, real_requests, provider: str, role: str,
                 allow_temperature_omission: bool = False,
                 candidate_reasoning_effort: str | None = None):
        self._requests = real_requests
        self.exceptions = real_requests.exceptions
        self.provider = provider
        self.role = role
        self.allow_temperature_omission = allow_temperature_omission
        self.candidate_reasoning_effort = candidate_reasoning_effort
        self._audit: list[dict] = []
        self._lock = threading.Lock()

    @property
    def audit(self) -> list[dict]:
        with self._lock:
            return deepcopy(self._audit)

    def post(self, url, json, headers, timeout):
        payload = transform_payload(json, self.provider, self.role,
                                    self.allow_temperature_omission,
                                    self.candidate_reasoning_effort)
        request_headers = transform_headers(headers, self.provider, self.role)
        secrets = _secrets(request_headers)
        prompt_bytes = json_module.dumps(payload.get("messages", []), ensure_ascii=False,
                                         sort_keys=True, separators=(",", ":")).encode("utf-8")
        row = {"role": self.role, "provider": self.provider,
               "model": payload.get("model"),
               "parameters": {key: deepcopy(payload[key]) for key in sorted(_PARAMETERS) if key in payload},
               "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
               "temperature_omitted": "temperature" in json and "temperature" not in payload,
               "http_status": None}
        if "anthropic-version" in request_headers:
            row["api_version"] = request_headers["anthropic-version"]
        try:
            response = self._requests.post(url, json=payload, headers=request_headers, timeout=timeout)
        except Exception as exc:
            row["error"] = {"type": type(exc).__name__, "message": str(exc)}
            self._record(row, secrets)
            raise
        try:
            row.update(_response_fields(response))
        except Exception as exc:
            # Observation must never alter the upstream HTTP/parsing contract.
            row["audit_error"] = type(exc).__name__
        self._record(row, secrets)
        return response

    def _record(self, row: dict, secrets: tuple[str, ...]) -> None:
        safe_row = _redact(row, secrets)
        with self._lock:
            self._audit.append(safe_row)

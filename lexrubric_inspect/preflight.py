"""Read-only provider checks before starting paid benchmark inference."""
from __future__ import annotations

import os
import re
from urllib.parse import quote

import requests


class ProviderAccessError(RuntimeError):
    """Provider authentication or model access failed; no inference was started."""


def _get(url, role, provider, key, extra_headers=None):
    try:
        response = requests.get(url, headers={"Authorization": "Bearer " + key,
                                             **(extra_headers or {})}, timeout=30)
    except requests.RequestException as error:
        raise ProviderAccessError(
            f"{role} {provider} access check failed ({type(error).__name__}); no inference started"
        ) from None
    if response.status_code != 200:
        # Do not echo response bodies or request headers: these may contain account data.
        raise ProviderAccessError(
            f"{role} {provider} access check returned HTTP {response.status_code}; "
            "repair the configured credential/access before running inference"
        )
    try:
        body = response.json()
    except ValueError:
        raise ProviderAccessError(f"{role} {provider} returned invalid access-check JSON") from None
    if not isinstance(body, dict):
        raise ProviderAccessError(f"{role} {provider} returned an invalid access-check object")
    return body


def check_provider_access(config) -> list[dict]:
    """Authenticate configured public providers without generating model output.

    Generic upstream endpoints have no standardized read-only authentication API.
    They are identified as unchecked rather than probed with invented requests.
    """
    checks = []
    for role in ("candidate", "judge"):
        provider = config[f"{role}_provider"]
        model = config[f"{role}_model"]
        endpoint = config[f"{role}_chat_url"]
        key = os.environ.get(config[f"{role}_api_key_env"], "")
        if not key:
            raise ProviderAccessError(f"Missing configured {role} credential")
        if provider == "openai":
            if endpoint != "https://api.openai.com/v1/chat/completions":
                raise ProviderAccessError("OpenAI access check requires the configured public OpenAI endpoint")
            body = _get("https://api.openai.com/v1/models/" + quote(model, safe=""), role, provider, key)
            if body.get("id") != model:
                raise ProviderAccessError(f"OpenAI did not confirm access to configured {role} model")
            checks.append({"role": role, "provider": provider, "model": model,
                           "authenticated": True, "model_available": True})
        elif provider == "anthropic":
            from .provider_transport import ANTHROPIC_API_VERSION
            if endpoint != "https://api.anthropic.com/v1/messages":
                raise ProviderAccessError("Anthropic access check requires the configured public Messages endpoint")
            api_version = config.get(f"{role}_api_version", ANTHROPIC_API_VERSION)
            if api_version != ANTHROPIC_API_VERSION:
                raise ProviderAccessError("Configured Anthropic API version differs from the validated transport")
            body = _get("https://api.anthropic.com/v1/models/" + quote(model, safe=""),
                        role, provider, key, {"anthropic-version": api_version})
            resolved_model = body.get("id", "")
            dated_alias = (isinstance(resolved_model, str)
                           and re.fullmatch(re.escape(model) + r"-\d{8}", resolved_model))
            if resolved_model != model and not dated_alias:
                raise ProviderAccessError(f"Anthropic did not confirm access to configured {role} model")
            checks.append({"role": role, "provider": provider, "model": model,
                           "resolved_model": resolved_model, "api_version": api_version,
                           "authenticated": True, "model_available": True})
        elif provider == "zai":
            if endpoint != "https://api.z.ai/api/paas/v4/chat/completions":
                raise ProviderAccessError("Z.ai requires the configured standard API endpoint")
            checks.append({"role": role, "provider": provider, "model": model,
                           "credential_present": True, "authenticated": None,
                           "model_available": None,
                           "reason": "Standard API access is checked by the approved pilot; no documented read-only probe is configured"})
        elif provider == "openrouter":
            if endpoint != "https://openrouter.ai/api/v1/chat/completions":
                raise ProviderAccessError("OpenRouter access check requires the configured public OpenRouter endpoint")
            _get("https://openrouter.ai/api/v1/key", role, provider, key)
            catalog = _get("https://openrouter.ai/api/v1/models", role, provider, key)
            provider_model = "qwen/qwen3.6-27b" if model == "qwen3.6-27b" else model
            if not any(item.get("id") == provider_model for item in catalog.get("data", [])):
                raise ProviderAccessError(f"OpenRouter does not list configured {role} model {provider_model}")
            checks.append({"role": role, "provider": provider, "model": provider_model,
                           "authenticated": True, "model_available": True})
        else:
            checks.append({"role": role, "provider": provider, "model": model,
                           "authenticated": None, "model_available": None,
                           "reason": "No standardized read-only check for upstream endpoint"})
    return checks

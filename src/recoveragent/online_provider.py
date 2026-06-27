"""Small OpenAI-compatible provider client used by demo MAS integrations."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_ENDPOINT = "https://api.openai.com/v1/chat/completions"


@dataclass(slots=True)
class ProviderConfig:
    api_key: str
    endpoint: str = DEFAULT_ENDPOINT
    model: str = DEFAULT_MODEL
    extra_body: dict[str, Any] = field(default_factory=dict)

    def public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["api_key"] = "<redacted>"
        data["chat_completions_endpoint"] = normalize_endpoint(self.endpoint)
        return data


@dataclass(slots=True)
class ProviderResponse:
    provider_called: bool
    model: str
    endpoint: str
    latency_seconds: float
    content: str
    raw_response: dict[str, Any]

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_provider_config(
    *,
    api_path: Path | None = None,
    model: str | None = None,
    endpoint: str | None = None,
) -> ProviderConfig:
    """Load an OpenAI-compatible config from env vars and an optional local file."""

    file_config: dict[str, Any] = {}
    if api_path:
        file_config = _parse_config_file(api_path)

    api_key = (
        os.getenv("RECOVERAGENT_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or str(file_config.get("api_key") or "")
    )
    selected_endpoint = (
        endpoint
        or os.getenv("RECOVERAGENT_ENDPOINT")
        or os.getenv("OPENAI_BASE_URL")
        or str(file_config.get("endpoint") or DEFAULT_ENDPOINT)
    )
    selected_model = model or os.getenv("RECOVERAGENT_MODEL") or str(file_config.get("model") or DEFAULT_MODEL)
    extra_body = file_config.get("extra_body") if isinstance(file_config.get("extra_body"), dict) else {}

    if not api_key:
        raise ValueError("missing provider API key; set OPENAI_API_KEY/RECOVERAGENT_API_KEY or pass --api-path")
    return ProviderConfig(
        api_key=api_key,
        endpoint=selected_endpoint,
        model=selected_model,
        extra_body=dict(extra_body),
    )


def chat_completion(
    *,
    config: ProviderConfig,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    timeout: int = 90,
) -> ProviderResponse:
    payload: dict[str, Any] = {
        "model": config.model,
        "temperature": temperature,
        "messages": messages,
    }
    payload.update(config.extra_body)
    start = time.monotonic()
    request = urllib.request.Request(
        normalize_endpoint(config.endpoint),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"provider HTTP {exc.code}: {body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"provider request failed: {exc}") from exc

    latency = time.monotonic() - start
    data = json.loads(raw)
    return ProviderResponse(
        provider_called=True,
        model=config.model,
        endpoint=normalize_endpoint(config.endpoint),
        latency_seconds=round(latency, 3),
        content=extract_content(data),
        raw_response=data,
    )


def normalize_endpoint(endpoint: str) -> str:
    value = endpoint.rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return value + "/chat/completions"
    return value + "/v1/chat/completions"


def extract_content(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    text = first.get("text")
    return text if isinstance(text, str) else ""


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from a model response that may include fences."""

    stripped = text.strip()
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
        stripped = "\n".join(lines).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model response JSON must be an object")
    return value


def _parse_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    config: dict[str, Any] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            continue
        key = key.strip().lower().replace("-", "_")
        value = value.strip().strip("'\"")
        if key == "extra_body":
            config[key] = json.loads(value)
        elif key in {"api_key", "endpoint", "model"}:
            config[key] = value
    return config

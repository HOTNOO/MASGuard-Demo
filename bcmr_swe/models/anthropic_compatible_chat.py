"""Anthropic-compatible chat model adapter.

Supports providers such as MiniMax that expose an Anthropic-compatible
messages API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Any


def _extract_api_lines(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _expand_env_value(raw: str) -> str:
    text = str(raw or "").strip()
    pattern = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
    match = pattern.match(text)
    if match:
        return os.getenv(match.group(1), "")
    if text.startswith("env:"):
        return os.getenv(text.split(":", 1)[1].strip(), "")
    return text


def eval_json(text: str) -> Any:
    import json

    return json.loads(text)


@dataclass
class AnthropicCompatibleChatConfig:
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.1
    top_p: float = 0.95
    max_output_tokens: int = 2048
    request_timeout: int = 60
    max_retries: int = 2
    user_agent: str = "bcmr-anthropic-compatible/1.0"
    extra_body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api_md(cls, path: str | Path) -> "AnthropicCompatibleChatConfig":
        lines = _extract_api_lines(path)
        api_key = ""
        base_url = ""
        model = ""
        extra_body: dict[str, Any] = {}
        plain_values: list[str] = []
        for line in lines:
            lowered = line.lower()
            if lowered.startswith("provider="):
                continue
            if lowered.startswith("api_key="):
                api_key = _expand_env_value(line.split("=", 1)[1].strip())
                continue
            if lowered.startswith("base_url="):
                base_url = _expand_env_value(line.split("=", 1)[1].strip())
                continue
            if lowered.startswith("endpoint="):
                base_url = _expand_env_value(line.split("=", 1)[1].strip())
                continue
            if lowered.startswith("model="):
                model = _expand_env_value(line.split("=", 1)[1].strip())
                continue
            if lowered.startswith("extra_body="):
                raw = line.split("=", 1)[1].strip()
                if raw:
                    extra_body.update(eval_json(raw))
                continue
            if (line.startswith("{") and line.endswith("}")) or (line.startswith("[") and line.endswith("]")):
                parsed = eval_json(line)
                if isinstance(parsed, dict):
                    extra_body.update(parsed)
                    continue
            if line.startswith("http://") or line.startswith("https://"):
                base_url = _expand_env_value(line)
                continue
            plain_values.append(line)

        if not api_key and plain_values:
            api_key = _expand_env_value(plain_values[0])
        if not base_url and len(plain_values) >= 2:
            base_url = _expand_env_value(plain_values[1])
        if not model and len(plain_values) >= 3:
            model = _expand_env_value(plain_values[2])

        config = cls(
            model=model or os.getenv("ANTHROPIC_MODEL", ""),
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY", "") or os.getenv("MINIMAX_API_KEY", ""),
            base_url=base_url or os.getenv("ANTHROPIC_BASE_URL", ""),
            extra_body=extra_body,
        )
        if not config.base_url:
            raise ValueError("Anthropic-compatible base_url is missing. Set ANTHROPIC_BASE_URL or provide it via api.md.")
        if not config.api_key:
            raise ValueError("Anthropic-compatible API key is missing. Set ANTHROPIC_API_KEY or provide it via api.md.")
        if not config.model:
            raise ValueError("Anthropic-compatible model is missing. Set ANTHROPIC_MODEL or provide it via api.md.")
        return config


class AnthropicCompatibleChatModel:
    """Thin wrapper over Anthropic SDK for Anthropic-compatible providers."""

    def __init__(self, config: AnthropicCompatibleChatConfig):
        self.config = config
        self.n_calls = 0
        self.cost = 0.0
        self.total_prompt_tokens = 0.0
        self.total_completion_tokens = 0.0
        self.total_tokens = 0.0

    def _build_client(self):
        try:
            import anthropic  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "anthropic SDK is not installed. Install it in the active Python environment with `pip install anthropic`."
            ) from exc
        return anthropic.Anthropic(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.request_timeout,
            max_retries=self.config.max_retries,
            default_headers={"User-Agent": self.config.user_agent},
        )

    @classmethod
    def _normalize_messages(cls, messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
        system_parts: list[str] = []
        payload_messages: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "") or "")
            content = message.get("content", "")
            if role == "system":
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            system_parts.append(str(item.get("text", "")))
                        else:
                            system_parts.append(str(item))
                else:
                    system_parts.append(str(content))
                continue
            if isinstance(content, str):
                payload_content: Any = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                payload_content = [item if isinstance(item, dict) else {"type": "text", "text": str(item)} for item in content]
            elif isinstance(content, dict):
                payload_content = [content]
            else:
                payload_content = [{"type": "text", "text": str(content)}]
            payload_messages.append({"role": role, "content": payload_content})
        system_prompt = "\n\n".join(part for part in system_parts if part.strip()) or None
        return system_prompt, payload_messages

    def _accumulate_usage(self, usage: Any) -> dict[str, float]:
        input_tokens = float(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = float(getattr(usage, "output_tokens", 0) or 0)
        total_tokens = input_tokens + output_tokens
        self.total_prompt_tokens += input_tokens
        self.total_completion_tokens += output_tokens
        self.total_tokens += total_tokens
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

    def query(self, messages: list[dict[str, Any]], **kwargs) -> dict[str, Any]:
        self.n_calls += 1
        client = self._build_client()
        system_prompt, payload_messages = self._normalize_messages(messages)
        max_tokens = int(kwargs.get("max_tokens", self.config.max_output_tokens))
        temperature = kwargs.get("temperature", self.config.temperature)
        top_p = kwargs.get("top_p", self.config.top_p)
        timeout = kwargs.get("request_timeout", self.config.request_timeout)
        extra_body = kwargs.get("extra_body") if isinstance(kwargs.get("extra_body"), dict) else {}
        request_kwargs: dict[str, Any] = {
            "model": kwargs.get("model", self.config.model),
            "max_tokens": max_tokens,
            "messages": payload_messages,
            "temperature": temperature,
            "top_p": top_p,
            "timeout": timeout,
        }
        if system_prompt:
            request_kwargs["system"] = system_prompt
        merged_extra = dict(self.config.extra_body)
        merged_extra.update(extra_body or {})
        if merged_extra:
            request_kwargs["extra_body"] = merged_extra

        response = client.messages.create(**request_kwargs)
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        for block in list(getattr(response, "content", []) or []):
            block_type = str(getattr(block, "type", "") or "")
            if block_type == "text":
                content_parts.append(str(getattr(block, "text", "") or ""))
            elif block_type == "thinking":
                thinking_parts.append(str(getattr(block, "thinking", "") or ""))
        usage = self._accumulate_usage(getattr(response, "usage", None))
        return {
            "content": "\n".join(part for part in content_parts if part).strip(),
            "extra": {
                "model": request_kwargs["model"],
                "usage": usage,
                "thinking": "\n".join(part for part in thinking_parts if part).strip(),
                "stop_reason": str(getattr(response, "stop_reason", "") or ""),
            },
        }

    def get_template_vars(self) -> dict[str, Any]:
        return {
            "model_name": self.config.model,
            "n_model_calls": self.n_calls,
            "model_cost": self.cost,
            "temperature": self.config.temperature,
        }

    def get_usage_snapshot(self) -> dict[str, float]:
        return {
            "n_calls": float(self.n_calls),
            "prompt_tokens": float(self.total_prompt_tokens),
            "completion_tokens": float(self.total_completion_tokens),
            "total_tokens": float(self.total_tokens),
        }

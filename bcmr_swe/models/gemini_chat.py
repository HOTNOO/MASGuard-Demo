"""Gemini chat model adapter using the public Generative Language REST API."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib import error, request


def _normalize_model_name(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return "gemini-2.5-flash"
    lowered = text.lower().replace(" ", "-")
    if lowered.startswith("models/"):
        lowered = lowered.split("/", 1)[1]
    if lowered in {"gemini-2.5-flash", "gemini-flash-latest", "gemini-2.5-flash-latest"}:
        return "gemini-2.5-flash"
    return lowered


@dataclass
class GeminiChatConfig:
    model: str = "gemini-2.5-flash"
    api_key: str = ""
    temperature: float = 0.1
    top_p: float = 0.95
    max_output_tokens: int = 2048
    request_timeout: int = 60
    max_retries: int = 2

    @classmethod
    def from_api_md(cls, path: str | Path) -> "GeminiChatConfig":
        path = Path(path)
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        api_key = ""
        model = ""
        for line in lines:
            if line.startswith("AIza"):
                api_key = line
                continue
            if "gemini" in line.lower():
                model = line
                break
        config = cls(
            model=_normalize_model_name(model or os.getenv("GOOGLE_MODEL", "")),
            api_key=api_key or os.getenv("GOOGLE_API_KEY", ""),
        )
        if not config.api_key:
            raise ValueError("Google API key is missing. Set GOOGLE_API_KEY or provide it via api.md.")
        return config


class GeminiChatModel:
    """A minimal chat adapter compatible with swe_mas agents."""

    def __init__(self, config: GeminiChatConfig):
        self.config = config
        self.n_calls = 0
        self.cost = 0.0
        self.total_prompt_tokens = 0.0
        self.total_completion_tokens = 0.0
        self.total_tokens = 0.0

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]:
        self.n_calls += 1
        payload = self._build_payload(messages, **kwargs)
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.config.model}:generateContent?key={self.config.api_key}"
        )
        timeout = int(kwargs.get("request_timeout", self.config.request_timeout))
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            req = request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with request.urlopen(req, timeout=timeout) as resp:
                    body = resp.read().decode("utf-8")
                data = json.loads(body)
                content = self._extract_text(data)
                usage = data.get("usageMetadata", {})
                self._accumulate_usage(usage)
                return {
                    "content": content,
                    "extra": {
                        "model": self.config.model,
                        "usage": usage,
                        "finish_reason": self._extract_finish_reason(data),
                    },
                }
            except Exception as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                time.sleep(1.2 * (attempt + 1))

        raise RuntimeError(f"Gemini API request failed: {last_error}") from last_error

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

    def _build_payload(self, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]:
        system_chunks: list[str] = []
        conversation_parts: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if not content:
                continue
            if role == "system":
                system_chunks.append(content)
                continue
            mapped_role = "model" if role == "assistant" else "user"
            conversation_parts.append({"role": mapped_role, "parts": [{"text": content}]})

        payload: dict[str, Any] = {
            "contents": conversation_parts,
            "generationConfig": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "topP": kwargs.get("top_p", self.config.top_p),
                "maxOutputTokens": kwargs.get("max_tokens", self.config.max_output_tokens),
            },
        }
        if system_chunks:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_chunks)}]}
        return payload

    def _extract_text(self, payload: dict[str, Any]) -> str:
        candidates = payload.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {payload}")
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
        return "\n".join(texts).strip()

    def _extract_finish_reason(self, payload: dict[str, Any]) -> str:
        candidates = payload.get("candidates", [])
        if not candidates:
            return ""
        return str(candidates[0].get("finishReason", ""))

    def _accumulate_usage(self, usage: dict[str, Any]) -> None:
        prompt = usage.get("promptTokenCount")
        completion = usage.get("candidatesTokenCount")
        total = usage.get("totalTokenCount")
        try:
            if prompt is not None:
                self.total_prompt_tokens += float(prompt)
        except Exception:
            pass
        try:
            if completion is not None:
                self.total_completion_tokens += float(completion)
        except Exception:
            pass
        try:
            if total is not None:
                self.total_tokens += float(total)
                return
        except Exception:
            pass
        try:
            self.total_tokens += float(prompt or 0.0) + float(completion or 0.0)
        except Exception:
            pass

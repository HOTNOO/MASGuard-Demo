"""OpenAI-compatible chat model adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any
from urllib import error, parse, request


DEFAULT_COMPAT_ORIGIN = "http://localhost:8317"
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_PLACEHOLDER_PREFIXES = (
    "thinking about your request",
)


def _expand_env_value(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    if text.startswith("env:"):
        return os.getenv(text.split(":", 1)[1].strip(), "")
    if text.startswith("${") and text.endswith("}") and len(text) > 3:
        return os.getenv(text[2:-1], "")
    return text


def _normalize_endpoint(raw: str) -> str:
    text = _expand_env_value(raw)
    origin = (
        os.getenv("OPENAI_COMPAT_ORIGIN")
        or os.getenv("OPENAI_BASE_URL")
        or DEFAULT_COMPAT_ORIGIN
    ).strip().rstrip("/")
    if not text:
        return f"{origin}/v1/chat/completions"
    if text.startswith("http://") or text.startswith("https://"):
        normalized = text.rstrip("/")
    elif text.startswith("/"):
        normalized = f"{origin}{text}"
    else:
        normalized = f"{origin}/{text.lstrip('/')}"
    parsed = parse.urlparse(normalized)
    if parsed.scheme and parsed.netloc and not parsed.path.rstrip("/"):
        return normalized.rstrip("/") + "/v1/chat/completions"
    if normalized.endswith("/v1"):
        return normalized + "/chat/completions"
    return normalized


def _extract_api_lines(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _should_bypass_env_proxy(endpoint: str) -> bool:
    host = (parse.urlparse(endpoint).hostname or "").strip().lower()
    if not host:
        return False
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return host.endswith(".local")
    return bool(addr.is_loopback or addr.is_private or addr.is_link_local)


@dataclass
class OpenAICompatibleChatConfig:
    model: str = ""
    api_key: str = ""
    endpoint: str = f"{DEFAULT_COMPAT_ORIGIN}/v1/chat/completions"
    temperature: float = 0.1
    top_p: float = 0.95
    max_output_tokens: int = 2048
    request_timeout: int = 60
    max_retries: int = 2
    user_agent: str = "curl/8.5.0"
    extra_body: dict[str, Any] = field(default_factory=dict)
    min_request_interval_sec: float = 0.0
    prefer_stream: bool = False
    force_curl: bool = False

    @classmethod
    def from_api_md(cls, path: str | Path) -> "OpenAICompatibleChatConfig":
        lines = _extract_api_lines(path)
        api_key = ""
        endpoint = ""
        model = ""
        extra_body: dict[str, Any] = {}
        prefer_stream = False
        force_curl = False
        plain_values: list[str] = []
        for line in lines:
            lowered = line.lower()
            if lowered.startswith("api_key="):
                api_key = _expand_env_value(line.split("=", 1)[1].strip())
                continue
            if lowered.startswith("endpoint="):
                endpoint = _expand_env_value(line.split("=", 1)[1].strip())
                continue
            if lowered.startswith("base_url="):
                endpoint = _expand_env_value(line.split("=", 1)[1].strip())
                continue
            if lowered.startswith("model="):
                model = _expand_env_value(line.split("=", 1)[1].strip())
                continue
            if lowered.startswith("extra_body="):
                raw = line.split("=", 1)[1].strip()
                if raw:
                    extra_body.update(json.loads(raw))
                continue
            if lowered.startswith("prefer_stream="):
                prefer_stream = line.split("=", 1)[1].strip().lower() in {"1", "true", "yes", "on"}
                continue
            if lowered.startswith("force_curl="):
                force_curl = line.split("=", 1)[1].strip().lower() in {"1", "true", "yes", "on"}
                continue
            if (line.startswith("{") and line.endswith("}")) or (line.startswith("[") and line.endswith("]")):
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    extra_body.update(parsed)
                    continue
            if "/chat/completions" in line or "/v1/" in line or lowered.startswith("http://") or lowered.startswith("https://"):
                endpoint = _expand_env_value(line)
                continue
            plain_values.append(_expand_env_value(line))

        if endpoint:
            if not api_key and plain_values:
                api_key = plain_values[0]
            if not model and len(plain_values) >= 2:
                model = plain_values[1]
        else:
            for value in plain_values:
                if value.startswith("sk-"):
                    api_key = value
                    continue
                if not model:
                    model = value

        config = cls(
            model=model or os.getenv("OPENAI_COMPAT_MODEL", "") or os.getenv("OPENAI_MODEL", ""),
            api_key=api_key or os.getenv("OPENAI_API_KEY", "") or os.getenv("MINIMAX_API_KEY", ""),
            endpoint=_normalize_endpoint(endpoint or os.getenv("OPENAI_COMPAT_ENDPOINT", "") or os.getenv("OPENAI_BASE_URL", "")),
            extra_body=extra_body,
            prefer_stream=prefer_stream or os.getenv("OPENAI_COMPAT_PREFER_STREAM", "").strip().lower() in {"1", "true", "yes", "on"},
            force_curl=force_curl or os.getenv("OPENAI_COMPAT_FORCE_CURL", "").strip().lower() in {"1", "true", "yes", "on"},
        )
        host = (parse.urlparse(config.endpoint).hostname or "").strip().lower()
        if host == "open.bigmodel.cn":
            config.min_request_interval_sec = max(config.min_request_interval_sec, 2.0)
        if not config.api_key:
            raise ValueError("OpenAI-compatible API key is missing. Set OPENAI_API_KEY or provide it via api.md.")
        if not config.model:
            raise ValueError("OpenAI-compatible model is missing. Set OPENAI_COMPAT_MODEL or provide it via api.md.")
        return config


class OpenAICompatibleChatModel:
    """Minimal OpenAI-compatible chat adapter for swe_mas agents."""

    def __init__(self, config: OpenAICompatibleChatConfig):
        self.config = config
        self.n_calls = 0
        self.cost = 0.0
        self.total_prompt_tokens = 0.0
        self.total_completion_tokens = 0.0
        self.total_tokens = 0.0
        self._last_request_at = 0.0
        handlers: list[request.BaseHandler] = []
        if _should_bypass_env_proxy(config.endpoint):
            handlers.append(request.ProxyHandler({}))
        self._opener = request.build_opener(*handlers)

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]:
        self.n_calls += 1
        payload = dict(self.config.extra_body)
        payload.update({
            "model": kwargs.get("model", self.config.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "max_tokens": kwargs.get("max_tokens", self.config.max_output_tokens),
        })
        extra_body = kwargs.get("extra_body")
        if isinstance(extra_body, dict):
            for key, value in extra_body.items():
                if key in {"model", "messages"}:
                    continue
                payload[key] = value
        timeout = int(kwargs.get("request_timeout", self.config.request_timeout))
        last_error: Exception | None = None
        retried_without_token_cap = False
        attempt = 0
        while attempt <= self.config.max_retries:
            try:
                if self.config.prefer_stream and not payload.get("stream"):
                    try:
                        stream_result = self._query_preferred_stream(payload=payload, timeout=timeout)
                    except RuntimeError as exc:
                        # Local OpenAI-compatible proxies occasionally emit
                        # heartbeat-only SSE responses. Fall back to the
                        # ordinary request path instead of failing the run.
                        if "empty streaming response" not in str(exc):
                            raise
                    else:
                        if stream_result is not None:
                            return stream_result
                self._throttle_before_request()
                nonstream_payload = dict(payload)
                nonstream_payload.setdefault("stream", False)
                body = self._perform_request(payload=nonstream_payload, timeout=timeout)
                try:
                    data = json.loads(body)
                except json.JSONDecodeError as exc:
                    if self._looks_like_stream_body(body):
                        stream_text, stream_usage, finish_reason = self._extract_stream_text(body)
                        if stream_usage:
                            self._accumulate_usage(stream_usage)
                        if stream_text.strip():
                            return {
                                "content": stream_text,
                                "extra": {
                                    "model": self.config.model,
                                    "usage": stream_usage,
                                "finish_reason": finish_reason,
                            },
                        }
                        fallback = self._retry_nonstream_after_empty_stream(payload=payload, timeout=timeout)
                        if fallback is not None:
                            return fallback
                        detail = finish_reason or "empty_stream_response"
                        raise RuntimeError(f"empty streaming response: {detail}") from exc
                    snippet = body.strip()[:500] or "<empty response>"
                    raise RuntimeError(f"non-json upstream response: {snippet}") from exc
                provider_error = self._extract_provider_error(data)
                if provider_error is not None:
                    raise provider_error
                content = self._extract_text(data)
                usage = data.get("usage", {}) or {}
                self._accumulate_usage(usage)
                if not content.strip() and not payload.get("stream"):
                    stream_text, stream_usage, finish_reason = self._request_stream_text(payload=nonstream_payload, timeout=timeout)
                    if stream_usage:
                        self._accumulate_usage(stream_usage)
                    if stream_text.strip():
                        return {
                            "content": stream_text,
                            "extra": {
                                "model": self.config.model,
                                "usage": stream_usage or usage,
                                "finish_reason": finish_reason or self._extract_finish_reason(data),
                            },
                        }
                return {
                    "content": content,
                    "extra": {
                        "model": self.config.model,
                        "usage": usage,
                        "finish_reason": self._extract_finish_reason(data),
                    },
                }
            except Exception as exc:
                if (
                    not retried_without_token_cap
                    and self._is_unsupported_token_cap_error(exc)
                    and "max_tokens" in payload
                ):
                    payload = dict(payload)
                    payload.pop("max_tokens", None)
                    retried_without_token_cap = True
                    continue
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                time.sleep(self._retry_delay(exc, attempt))
                attempt += 1
        raise RuntimeError(f"OpenAI-compatible API request failed: {last_error}") from last_error

    def _query_preferred_stream(self, *, payload: dict[str, Any], timeout: int) -> dict[str, Any] | None:
        last_body = ""
        last_finish_reason = ""
        stream_attempts = 3
        for stream_attempt in range(stream_attempts):
            self._throttle_before_request()
            body = self._perform_request(payload={**payload, "stream": True}, timeout=timeout)
            last_body = body
            stream_text, stream_usage, finish_reason = self._extract_stream_text(body)
            last_finish_reason = finish_reason
            if stream_usage:
                self._accumulate_usage(stream_usage)
            if stream_text.strip():
                return {
                    "content": stream_text,
                    "extra": {
                        "model": self.config.model,
                        "usage": stream_usage,
                        "finish_reason": finish_reason,
                    },
                }
            # Some local proxies emit an initial heartbeat-only SSE response or an
            # empty stop chunk before the next streaming request carries text.
            if self._looks_like_stream_body(body) and stream_attempt + 1 < stream_attempts:
                time.sleep(self._retry_delay(RuntimeError("empty_stream_response"), stream_attempt))
                continue
            break
        if self._looks_like_stream_body(last_body):
            detail = last_finish_reason or "empty_stream_response"
            raise RuntimeError(f"empty streaming response: {detail}")
        return None

    def _perform_request(self, *, payload: dict[str, Any], timeout: int) -> str:
        if payload.get("stream"):
            try:
                return self._perform_curl_request(payload=payload, timeout=timeout)
            except FileNotFoundError:
                if _should_bypass_env_proxy(self.config.endpoint) and not self.config.force_curl:
                    return self._perform_urllib_request(payload=payload, timeout=timeout)
                raise
        if not self.config.prefer_stream:
            try:
                return self._perform_curl_request(payload=payload, timeout=timeout)
            except FileNotFoundError:
                if _should_bypass_env_proxy(self.config.endpoint) and not self.config.force_curl:
                    return self._perform_urllib_request(payload=payload, timeout=timeout)
                raise
        if _should_bypass_env_proxy(self.config.endpoint) and not self.config.force_curl:
            return self._perform_urllib_request(payload=payload, timeout=timeout)
        return self._perform_curl_request(payload=payload, timeout=timeout)

    def _perform_urllib_request(self, *, payload: dict[str, Any], timeout: int) -> str:
        req = request.Request(
            self.config.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
                "User-Agent": self.config.user_agent,
            },
            method="POST",
        )
        try:
            with self._opener.open(req, timeout=timeout) as resp:
                self._last_request_at = time.time()
                return resp.read().decode("utf-8")
        except error.HTTPError as exc:
            self._last_request_at = time.time()
            body = exc.read().decode("utf-8", errors="replace")
            if body.strip():
                raise RuntimeError(body.strip()) from exc
            raise

    def _perform_curl_request(self, *, payload: dict[str, Any], timeout: int) -> str:
        command = [
            "curl",
            "--http1.1",
            "--silent",
            "--show-error",
            "--location",
            "--max-time",
            str(timeout),
            self.config.endpoint,
            "-H",
            f"Authorization: Bearer {self.config.api_key}",
            "-H",
            "Content-Type: application/json",
            "-H",
            f"User-Agent: {self.config.user_agent}",
            "--data-binary",
            "@-",
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            input=json.dumps(payload, ensure_ascii=False),
            timeout=max(5, int(timeout) + 5),
        )
        self._last_request_at = time.time()
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            detail = stderr or stdout or f"curl exited with code {completed.returncode}"
            raise RuntimeError(detail)
        body = completed.stdout
        text = body.strip()
        if not text:
            raise RuntimeError("empty upstream response")
        if text.lower().startswith("error code:"):
            raise RuntimeError(text)
        return body

    def _request_stream_text(self, *, payload: dict[str, Any], timeout: int) -> tuple[str, dict[str, Any], str]:
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        body = self._perform_request(payload=stream_payload, timeout=timeout)
        return self._extract_stream_text(body)

    def _retry_nonstream_after_empty_stream(
        self,
        *,
        payload: dict[str, Any],
        timeout: int,
    ) -> dict[str, Any] | None:
        if payload.get("stream"):
            return None
        payload = {**payload, "stream": False}
        attempts = 3
        for attempt in range(attempts):
            try:
                body = self._perform_curl_request(payload=payload, timeout=timeout)
            except Exception:
                body = ""
            if not body:
                if attempt + 1 < attempts:
                    time.sleep(self._retry_delay(RuntimeError("empty_nonstream_response"), attempt))
                    continue
                return None
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                if not self._looks_like_stream_body(body):
                    return None
                stream_text, stream_usage, finish_reason = self._extract_stream_text(body)
                if stream_usage:
                    self._accumulate_usage(stream_usage)
                if not stream_text.strip():
                    if attempt + 1 < attempts:
                        time.sleep(self._retry_delay(RuntimeError("empty_nonstream_stream_body"), attempt))
                        continue
                    return None
                return {
                    "content": stream_text,
                    "extra": {
                        "model": self.config.model,
                        "usage": stream_usage,
                        "finish_reason": finish_reason,
                    },
                }
            provider_error = self._extract_provider_error(data)
            if provider_error is not None:
                raise provider_error
            usage = data.get("usage", {}) or {}
            self._accumulate_usage(usage)
            return {
                "content": self._extract_text(data),
                "extra": {
                    "model": self.config.model,
                    "usage": usage,
                    "finish_reason": self._extract_finish_reason(data),
                },
            }
        return None

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

    def _extract_text(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices", [])
        if not choices:
            raise RuntimeError(f"OpenAI-compatible endpoint returned no choices: {payload}")
        message = choices[0].get("message", {}) or {}
        content = message.get("content", "")
        reasoning = message.get("reasoning_content", "")
        if content is None:
            content = ""
        if reasoning is None:
            reasoning = ""
        if isinstance(content, str):
            normalized = self._normalize_visible_text(content)
            if normalized:
                return normalized
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    parts.append(item)
            normalized = self._normalize_visible_text("\n".join(parts))
            if normalized:
                return normalized
        # Deliberately ignore reasoning-only channels for mainline experiments:
        # we want the final answer, not the model's intermediate thought text.
        return ""

    def _extract_finish_reason(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices", [])
        if not choices:
            return ""
        return str(choices[0].get("finish_reason", ""))

    def _extract_stream_text(self, body: str) -> tuple[str, dict[str, Any], str]:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage: dict[str, Any] = {}
        finish_reason = ""
        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line or not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if not chunk or chunk == "[DONE]":
                continue
            try:
                payload = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            choices = payload.get("choices") or []
            if choices:
                choice = choices[0] or {}
                delta = choice.get("delta") or {}
                self._append_stream_parts(content_parts, delta.get("content"))
                self._append_stream_parts(reasoning_parts, delta.get("reasoning_content"))
                reason = str(choice.get("finish_reason", "") or "").strip()
                if reason:
                    finish_reason = reason
            if isinstance(payload.get("usage"), dict):
                usage = payload["usage"]
        text = "".join(content_parts).strip()
        normalized = self._normalize_visible_text(text)
        return normalized, usage, finish_reason

    def _normalize_visible_text(self, text: Any) -> str:
        if text is None:
            return ""
        normalized = str(text)
        if not normalized:
            return ""
        normalized = _THINK_BLOCK_RE.sub("", normalized)
        normalized = normalized.strip()
        lowered = normalized.lower()
        if any(lowered.startswith(prefix) for prefix in _PLACEHOLDER_PREFIXES):
            return ""
        return normalized

    def _looks_like_stream_body(self, body: str) -> bool:
        text = str(body or "")
        return (
            "data:" in text
            or "[DONE]" in text
            or "heartbeat stream connected" in text
            or text.lstrip().startswith("data:")
        )

    def _append_stream_parts(self, target: list[str], value: Any) -> None:
        if isinstance(value, str):
            target.append(value)
            return
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and item.get("type") == "text":
                    target.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    target.append(item)

    def _accumulate_usage(self, usage: dict[str, Any]) -> None:
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total = usage.get("total_tokens")
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

    def _throttle_before_request(self) -> None:
        interval = float(self.config.min_request_interval_sec or 0.0)
        if interval <= 0:
            return
        elapsed = time.time() - self._last_request_at
        if elapsed < interval:
            time.sleep(interval - elapsed)

    def _extract_provider_error(self, payload: dict[str, Any]) -> Exception | None:
        err = payload.get("error")
        if not isinstance(err, dict):
            return None
        code = str(err.get("code", "")).strip()
        message = str(err.get("message", "")).strip()
        detail = f"{code}: {message}" if code and message else message or code or "provider_error"
        if code in {"429", "1302"} or "速率限制" in message or "rate limit" in message.lower():
            return RuntimeError(f"rate_limit: {detail}")
        return RuntimeError(detail)

    @staticmethod
    def _is_unsupported_token_cap_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return (
            "unsupported parameter" in text
            and (
                "max_output_tokens" in text
                or "max_tokens" in text
                or "max_completion_tokens" in text
            )
        )

    def _retry_delay(self, exc: Exception, attempt: int) -> float:
        text = str(exc).lower()
        gateway_markers = (
            "master_data_plane_disabled",
            "master data plane is disabled",
            "failed to validate api key",
            "endpoint returned no choices",
        )
        if any(marker in text for marker in gateway_markers):
            return min(30.0, 6.0 * (attempt + 1))
        if "rate_limit" in text or "429" in text or "速率限制" in str(exc):
            return min(20.0, 4.0 * (attempt + 1))
        return 1.5 * (attempt + 1)

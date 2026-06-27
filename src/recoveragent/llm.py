"""Optional OpenAI-compatible explanation layer for MASGuard."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from recoveragent.models import Diagnosis, EvidenceBundle, LLMInsight, RecoveryPlan


DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_ENDPOINT = "https://api.openai.com/v1/chat/completions"


def maybe_generate_llm_insight(
    *,
    bundle: EvidenceBundle,
    diagnosis: Diagnosis,
    plan: RecoveryPlan,
    enabled: bool,
    model: str = DEFAULT_MODEL,
    endpoint: str = DEFAULT_ENDPOINT,
    timeout: int = 45,
) -> LLMInsight:
    """Generate a concise explanation when an API key is available."""

    if not enabled:
        return LLMInsight(
            provider_called=False,
            model=model,
            summary="LLM explanation disabled; deterministic diagnosis and recovery plan were used.",
            explanation_line="MASGuard can run fully offline; the API layer is optional.",
        )

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("RECOVERAGENT_API_KEY")
    if not api_key:
        return LLMInsight(
            provider_called=False,
            model=model,
            summary="LLM explanation requested, but no OPENAI_API_KEY or RECOVERAGENT_API_KEY was set.",
            explanation_line="The tool remains usable without an API key.",
            error="missing_api_key",
        )

    prompt = _build_prompt(bundle=bundle, diagnosis=diagnosis, plan=plan)
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You explain software-engineering repair-agent failures. "
                    "Use only the supplied evidence. Do not claim the patch is fixed."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib.request.Request(
        _normalize_endpoint(endpoint),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        content = _extract_content(data)
        if not content:
            raise ValueError("empty model response")
        return LLMInsight(
            provider_called=True,
            model=model,
            summary=content.strip(),
            explanation_line="The optional API layer rewrites the same evidence-grounded diagnosis for a live demonstration audience.",
        )
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        return LLMInsight(
            provider_called=False,
            model=model,
            summary="LLM explanation failed; deterministic MASGuard output remains available.",
            explanation_line="The controller path remains deterministic if the optional API layer is unavailable.",
            error=f"{type(exc).__name__}: {exc}",
        )


def _build_prompt(*, bundle: EvidenceBundle, diagnosis: Diagnosis, plan: RecoveryPlan) -> str:
    evidence = "\n".join(f"- {item}" for item in diagnosis.evidence[:6]) or "- no cited evidence"
    steps = "\n".join(f"- {item}" for item in plan.steps[:6])
    signals = json.dumps(bundle.signals, sort_keys=True)
    return f"""Write a concise MASGuard demo explanation.

Issue:
{bundle.issue}

Signals:
{signals}

Diagnosis:
- type: {diagnosis.failure_type}
- stage: {diagnosis.responsible_stage}
- confidence: {diagnosis.confidence}
- rationale: {diagnosis.rationale}

Evidence:
{evidence}

Recovery action:
- action: {plan.action}
{steps}

Output format:
1. One sentence for what failed.
2. One sentence for the concrete repository evidence.
3. One sentence for why the recovery action is safer than naive retry.
"""


def _extract_content(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    text = first.get("text")
    return text if isinstance(text, str) else ""


def _normalize_endpoint(endpoint: str) -> str:
    value = endpoint.rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return value + "/chat/completions"
    return value + "/v1/chat/completions"

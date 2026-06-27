"""Mainline governance helpers shared across BCMR experiment entrypoints."""

from __future__ import annotations

import json
import re
from typing import Any


_TRANSPORT_PROVIDER_PATTERNS = (
    re.compile(r"\bempty streaming response\b", re.I),
    re.compile(r"\btransport request failed\b", re.I),
    re.compile(r"\bcurl:\s*\((?:28|35)\)", re.I),
    re.compile(r"\btls connect error\b", re.I),
    re.compile(r"\bplaceholder content\b", re.I),
    re.compile(r"\bthinking about your request\b", re.I),
    re.compile(r"\bupstream_error\b", re.I),
    re.compile(r"\b401\s+invalid api key\b", re.I),
    re.compile(r"\bchat upstream returned\s+(?:429|403)\b", re.I),
    re.compile(r"\b(?:openai|api|provider|upstream|chat|model)[^\n]{0,80}\b(?:http|status|returned)?\s*(?:429|403)\b", re.I),
    re.compile(r"\b(?:http|status)\s*(?:429|403)\b[^\n]{0,80}\b(?:openai|api|provider|upstream|chat|model)\b", re.I),
    re.compile(r"\b(?:rate limit|rate-limit|rate_limited|rate limited)\b", re.I),
)

_HEADLINE_BLOCKING_ERROR_TYPES = {
    "manifest_not_found",
    "oracle_preflight_failed",
}

_HEADLINE_BLOCKING_STOP_REASONS = {
    "manifest_not_found",
    "oracle_preflight_failed",
    "runtime_error",
}

_ILLEGITIMATE_TARGETS = {
    "generated_only",
    "missing",
    "non_source_only",
    "tests_only",
}

_DISPOSITION_TO_MAINLINE_STATUS = {
    "accepted_main_candidate": "headline_candidate",
    "diagnostic_only": "diagnostic_only",
    "rejected": "rejected",
}

_PROVIDER_BLOCKED_STATUSES = {"blocked", "error", "failed", "unhealthy"}
_PROVIDER_UNKNOWN_STATUSES = {"unknown"}


def is_auto_tier_model(model_name: str) -> bool:
    lowered = str(model_name or "").strip().lower()
    if not lowered:
        return False
    if lowered == "auto":
        return True
    return bool(re.search(r"(^|[-_/])auto($|[-_/])", lowered))


def _flatten_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        parts: list[str] = []
        for item in value.values():
            parts.extend(_flatten_strings(item))
        return parts
    if isinstance(value, (list, tuple, set)):
        parts: list[str] = []
        for item in value:
            parts.extend(_flatten_strings(item))
        return parts
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    return []


def _normalize_flags(flags: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in flags:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def has_transport_or_provider_noise(*values: Any) -> bool:
    haystack = "\n".join(_flatten_strings(values))
    if not haystack:
        return False
    return any(pattern.search(haystack) for pattern in _TRANSPORT_PROVIDER_PATTERNS)


def _provider_health_pollution_flags(provider_health: dict[str, Any]) -> list[str]:
    if not provider_health:
        return ["provider_health_missing"]

    flags: list[str] = []
    batch_status = str(provider_health.get("batch_status", "") or "").strip().lower()
    if batch_status in _PROVIDER_BLOCKED_STATUSES:
        flags.append("provider_health_blocked")
    elif batch_status in _PROVIDER_UNKNOWN_STATUSES:
        flags.append("provider_health_unknown")

    base_status = str(dict(provider_health.get("base_model", {}) or {}).get("status", "") or "").strip().lower()
    if base_status in _PROVIDER_BLOCKED_STATUSES:
        flags.append("provider_health_blocked")
    elif base_status in _PROVIDER_UNKNOWN_STATUSES:
        flags.append("provider_health_unknown")

    strong = dict(provider_health.get("strong_model", {}) or {})
    strong_enabled = bool(strong.get("enabled", False))
    strong_status = str(strong.get("status", "") or "").strip().lower()
    if strong_enabled and strong_status in _PROVIDER_BLOCKED_STATUSES:
        flags.append("provider_health_blocked")
    elif strong_enabled and strong_status in _PROVIDER_UNKNOWN_STATUSES:
        flags.append("provider_health_unknown")
    return _normalize_flags(flags)


def summarize_governance_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "mainline_status_counts": {},
        "source_quality_counts": {},
        "review_decision_counts": {},
        "artifact_disposition_counts": {},
        "provider_health_counts": {},
        "pollution_flag_counts": {},
    }
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in (
            "mainline_status",
            "source_quality",
            "review_decision",
            "artifact_disposition",
        ):
            value = str(row.get(key, "") or "missing")
            count_key = f"{key}_counts"
            summary[count_key][value] = summary[count_key].get(value, 0) + 1
        provider_health = row.get("provider_health") or {}
        batch_status = str(dict(provider_health).get("batch_status", "") or "unknown")
        summary["provider_health_counts"][batch_status] = summary["provider_health_counts"].get(batch_status, 0) + 1
        for flag in list(row.get("pollution_flags", []) or []):
            text = str(flag or "").strip()
            if not text:
                continue
            summary["pollution_flag_counts"][text] = summary["pollution_flag_counts"].get(text, 0) + 1
    return summary


def build_provider_health(
    *,
    base_model_name: str,
    base_preflight: dict[str, Any] | None = None,
    base_blocker: str = "",
    strong_model_name: str = "",
    strong_model_requested: bool = False,
    strong_model_enabled: bool = False,
    strong_preflight: dict[str, Any] | None = None,
    strong_blocker: str = "",
) -> dict[str, Any]:
    base_status = "blocked" if base_blocker else ("ok" if base_preflight else "not_checked")
    if strong_model_requested and strong_model_enabled and strong_preflight:
        strong_status = "ok"
    elif strong_model_requested and strong_blocker:
        strong_status = "blocked"
    elif strong_model_requested and not strong_model_enabled:
        strong_status = "disabled"
    else:
        strong_status = "not_requested"
    if base_status == "blocked":
        batch_status = "blocked"
    elif strong_model_requested and strong_model_enabled and strong_preflight:
        batch_status = "base_plus_strong"
    elif base_status != "ok":
        batch_status = "base_not_checked"
    else:
        batch_status = "base_only"
    return {
        "batch_status": batch_status,
        "base_model": {
            "model_name": str(base_model_name or ""),
            "status": base_status,
            "probe_content": str((base_preflight or {}).get("content", "") or ""),
            "blocker": str(base_blocker or ""),
        },
        "strong_model": {
            "model_name": str(strong_model_name or ""),
            "requested": bool(strong_model_requested),
            "enabled": bool(strong_model_enabled),
            "status": strong_status,
            "probe_content": str((strong_preflight or {}).get("content", "") or ""),
            "blocker": str(strong_blocker or ""),
        },
    }


def build_model_route_lock(
    *,
    base_model_name: str,
    strong_model_name: str = "",
    strong_model_requested: str = "",
    requested_runtime: str = "",
    resolved_runtime: str = "",
) -> dict[str, Any]:
    requested = str(strong_model_requested or "").strip()
    enabled = str(strong_model_name or "").strip()
    return {
        "base_model_name": str(base_model_name or ""),
        "base_model_is_auto": is_auto_tier_model(base_model_name),
        "strong_model_requested": requested,
        "strong_model_enabled": enabled,
        "strong_model_active": bool(enabled),
        "requested_runtime": str(requested_runtime or ""),
        "resolved_runtime": str(resolved_runtime or ""),
        "headline_auto_forbidden": True,
        "route_mode": "fast_plus_strong" if enabled else "fast_only",
    }


def natural_capture_governance_fields(
    *,
    base_model_name: str,
    strong_model_name: str = "",
    strong_model_requested: str = "",
    requested_runtime: str = "",
    resolved_runtime: str = "",
    provider_health: dict[str, Any] | None = None,
    oracle_success: bool | None = None,
    reported_success: bool | None = None,
    error_type: str = "",
    stop_reason: str = "",
    error: str = "",
    natural_failure_family: str = "",
    target_legitimacy: str = "",
    typed_assets: dict[str, Any] | None = None,
    replay_anchors: list[dict[str, Any]] | None = None,
    failed_state_group_paths: list[str] | None = None,
    extra_texts: list[Any] | None = None,
) -> dict[str, Any]:
    typed_assets = dict(typed_assets or {})
    replay_anchors = list(replay_anchors or [])
    failed_state_group_paths = [str(item).strip() for item in list(failed_state_group_paths or []) if str(item).strip()]
    provider_health = dict(provider_health or {})
    pollution_flags: list[str] = []
    auto_model = is_auto_tier_model(base_model_name)
    if auto_model:
        pollution_flags.append("auto_model_forbidden")
    pollution_flags.extend(_provider_health_pollution_flags(provider_health))
    normalized_error_type = str(error_type or "").strip()
    normalized_stop_reason = str(stop_reason or "").strip()
    if normalized_error_type in _HEADLINE_BLOCKING_ERROR_TYPES:
        pollution_flags.append(normalized_error_type)
    if normalized_stop_reason in _HEADLINE_BLOCKING_STOP_REASONS:
        pollution_flags.append(normalized_stop_reason)
    if str(natural_failure_family or "").strip() == "infrastructure_failure":
        pollution_flags.append("infrastructure_failure")
    if oracle_success is None:
        pollution_flags.append("oracle_status_missing")
    text_sources = [
        error,
        extra_texts or [],
        typed_assets,
    ]
    if "provider_health_blocked" in pollution_flags:
        text_sources.append(provider_health)
    if has_transport_or_provider_noise(text_sources):
        pollution_flags.append("transport_or_provider_noise")
    if (
        reported_success is not None
        and oracle_success is not None
        and bool(reported_success) != bool(oracle_success)
    ):
        pollution_flags.append("oracle_report_mismatch")
    typed_object_count = (
        len(typed_assets.get("selection_objects", []) or [])
        + len(typed_assets.get("shared_fact_objects", []) or [])
        + len(typed_assets.get("verifier_verdict_objects", []) or [])
    )
    clean_failed_state = bool(oracle_success is False) and str(natural_failure_family or "").strip() not in {
        "",
        "infrastructure_failure",
        "invalid_natural_eval",
        "resolved",
    }
    if clean_failed_state and typed_object_count == 0 and not failed_state_group_paths:
        pollution_flags.append("thin_failed_state_capture")
    if clean_failed_state and not replay_anchors:
        pollution_flags.append("missing_replay_anchor")
    if clean_failed_state and str(target_legitimacy or "").strip() in _ILLEGITIMATE_TARGETS:
        pollution_flags.append("illegitimate_target")
    pollution_flags = _normalize_flags(pollution_flags)

    if normalized_error_type in _HEADLINE_BLOCKING_ERROR_TYPES:
        artifact_disposition = "rejected"
        source_quality = "preflight_blocked"
    elif bool(oracle_success):
        artifact_disposition = "rejected"
        source_quality = "resolved_non_failed_state"
    elif any(
        flag in pollution_flags
        for flag in (
            "auto_model_forbidden",
            "transport_or_provider_noise",
            "infrastructure_failure",
            "runtime_error",
            "provider_health_blocked",
            "provider_health_unknown",
            "provider_health_missing",
            "oracle_status_missing",
        )
    ):
        artifact_disposition = "diagnostic_only"
        source_quality = "polluted_capture"
    elif any(flag in pollution_flags for flag in ("thin_failed_state_capture", "missing_replay_anchor", "illegitimate_target", "oracle_report_mismatch")):
        artifact_disposition = "diagnostic_only"
        source_quality = "thin_failed_state_capture"
    else:
        artifact_disposition = "accepted_main_candidate"
        source_quality = "clean_failed_state_candidate"

    review_decision = artifact_disposition
    return {
        "mainline_status": _DISPOSITION_TO_MAINLINE_STATUS[artifact_disposition],
        "source_quality": source_quality,
        "pollution_flags": pollution_flags,
        "provider_health": provider_health,
        "model_route_lock": build_model_route_lock(
            base_model_name=base_model_name,
            strong_model_name=strong_model_name,
            strong_model_requested=strong_model_requested,
            requested_runtime=requested_runtime,
            resolved_runtime=resolved_runtime,
        ),
        "review_decision": review_decision,
        "artifact_disposition": artifact_disposition,
    }


def natural_manifest_governance(
    *,
    decision_model_name: str,
    strong_model_name: str,
    required_system_variant: str,
    required_runtime: str = "",
    pool_status: str = "",
    accepted_main_pool_size: int = 0,
    target_main_pool_size: list[int] | None = None,
    provider_health: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target_main_pool_size = list(target_main_pool_size or [20, 30])
    pool_ready = bool(accepted_main_pool_size >= int(target_main_pool_size[0]))
    artifact_disposition = "accepted_main_candidate" if pool_ready else "diagnostic_only"
    review_decision = artifact_disposition
    return {
        "pool_status": str(pool_status or ""),
        "pool_ready": pool_ready,
        "accepted_main_pool_size": int(accepted_main_pool_size),
        "target_main_pool_size": target_main_pool_size,
        "required_system_variant": str(required_system_variant or ""),
        "required_runtime": str(required_runtime or ""),
        "mainline_status": "headline_ready" if pool_ready else "blocked_pending_scale",
        "source_quality": "clean_failed_state_pool" if pool_ready else "clean_subset_smoke_only",
        "pollution_flags": [],
        "provider_health": dict(provider_health or {}),
        "model_route_lock": build_model_route_lock(
            base_model_name=decision_model_name,
            strong_model_name=strong_model_name,
            strong_model_requested=strong_model_name,
            requested_runtime=required_runtime or required_system_variant,
            resolved_runtime=required_runtime or required_system_variant,
        ),
        "review_decision": review_decision,
        "artifact_disposition": artifact_disposition,
    }


def benchmark_row_governance_fields(
    *,
    source_type: str,
    natural_manifest_governance_block: dict[str, Any] | None = None,
    base_model_name: str = "",
    strong_model_name: str = "",
    runtime: str = "",
    error: str = "",
    error_type: str = "",
    related_payloads: list[Any] | None = None,
) -> dict[str, Any]:
    natural_manifest_governance_block = dict(natural_manifest_governance_block or {})
    pollution_flags = _normalize_flags(
        [str(item) for item in list(natural_manifest_governance_block.get("pollution_flags", []) or [])]
    )
    if is_auto_tier_model(base_model_name):
        pollution_flags.append("auto_model_forbidden")
    if error_type:
        pollution_flags.append(str(error_type))
    if error:
        pollution_flags.append("benchmark_error")
    if not natural_manifest_governance_block.get("pool_ready", False) and source_type == "natural_failed_state":
        pollution_flags.append("blocked_pending_scale")
    if has_transport_or_provider_noise(related_payloads or []):
        pollution_flags.append("transport_or_provider_noise")
    pollution_flags = _normalize_flags(pollution_flags)

    if error or error_type:
        artifact_disposition = "rejected"
    elif source_type != "natural_failed_state":
        artifact_disposition = "diagnostic_only"
    elif any(flag in pollution_flags for flag in ("auto_model_forbidden", "transport_or_provider_noise", "blocked_pending_scale")):
        artifact_disposition = "diagnostic_only"
    else:
        artifact_disposition = "accepted_main_candidate"

    if artifact_disposition == "accepted_main_candidate":
        source_quality = "frozen_failed_state_mainline"
    elif source_type != "natural_failed_state":
        source_quality = "diagnostic_controlled_failed_state"
    elif "blocked_pending_scale" in pollution_flags:
        source_quality = "clean_subset_smoke_only"
    else:
        source_quality = "diagnostic_evaluation"

    return {
        "mainline_status": _DISPOSITION_TO_MAINLINE_STATUS[artifact_disposition],
        "source_quality": source_quality,
        "pollution_flags": pollution_flags,
        "provider_health": dict(natural_manifest_governance_block.get("provider_health", {}) or {}),
        "model_route_lock": build_model_route_lock(
            base_model_name=base_model_name,
            strong_model_name=strong_model_name,
            strong_model_requested=strong_model_name,
            requested_runtime=runtime,
            resolved_runtime=runtime,
        ),
        "review_decision": artifact_disposition,
        "artifact_disposition": artifact_disposition,
    }


def dump_pretty_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)

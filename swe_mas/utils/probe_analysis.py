"""Helpers for distinguishing fresh evidence probes from duplicate browsing."""

from __future__ import annotations

import re
import shlex
from typing import Any


_PATH_RE = re.compile(
    r"(?:\./)?(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+(?:\.[A-Za-z0-9_.-]+)?"
)
_SED_RANGE_RE = re.compile(r"(\d+)(?:\s*,\s*(\d+))?p")


def _split_shell(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except Exception:
        return command.split()


def _clean_token(token: str) -> str:
    return token.strip().strip(",;|&()[]{}").strip("'\"")


def _looks_like_path(token: str) -> bool:
    token = _clean_token(token)
    if not token or token.startswith("-") or token.startswith("$"):
        return False
    if any(ch in token for ch in "*?[]{}()|"):
        return False
    if "/" in token:
        return True
    return bool(re.search(r"\.[A-Za-z0-9_.-]+$", token))


def _extract_paths(tokens: list[str], command: str) -> list[str]:
    paths: list[str] = []
    for token in tokens[1:]:
        cleaned = _clean_token(token)
        if not _looks_like_path(cleaned):
            continue
        normalized = cleaned.lstrip("./")
        if normalized not in paths:
            paths.append(normalized)
    if paths:
        return paths
    fallback: list[str] = []
    for match in _PATH_RE.findall(command):
        normalized = str(match).strip().lstrip("./")
        if normalized and normalized not in fallback:
            fallback.append(normalized)
    return fallback


def _extract_regions(command: str, tokens: list[str]) -> list[str]:
    regions: list[str] = []
    for start, end in _SED_RANGE_RE.findall(command):
        if start and end:
            regions.append(f"lines:{start}-{end}")
        elif start:
            regions.append(f"line:{start}")

    for prefix in ("head", "tail"):
        for idx, token in enumerate(tokens):
            if token != prefix:
                continue
            if idx + 2 < len(tokens) and tokens[idx + 1] == "-n":
                count = _clean_token(tokens[idx + 2])
                if count.isdigit():
                    regions.append(f"{prefix}:{count}")

    deduped: list[str] = []
    for item in regions:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _extract_symbols(tokens: list[str]) -> list[str]:
    if not tokens:
        return []
    command = tokens[0]
    symbols: list[str] = []
    if command not in {"rg", "grep"}:
        return symbols
    skip_next = False
    seen_pattern = False
    for token in tokens[1:]:
        cleaned = _clean_token(token)
        if not cleaned:
            continue
        if skip_next:
            skip_next = False
            continue
        if cleaned in {"-e", "--regexp", "-g", "--glob", "-t", "--type"}:
            skip_next = True
            continue
        if cleaned.startswith("-"):
            continue
        if _looks_like_path(cleaned):
            continue
        if not seen_pattern:
            seen_pattern = True
            if len(cleaned) <= 80:
                symbols.append(cleaned)
    return symbols


def extract_probe_signature(command: str) -> dict[str, Any]:
    tokens = _split_shell(command or "")
    command_name = _clean_token(tokens[0]).lower() if tokens else ""
    paths = _extract_paths(tokens, command)
    regions = _extract_regions(command, tokens)
    symbols = _extract_symbols(tokens)
    if paths and (regions or symbols):
        probe_kind = "path_context"
    elif paths:
        probe_kind = "path_only"
    elif regions or symbols:
        probe_kind = "symbol_only"
    else:
        probe_kind = "directory_only"
    return {
        "command_name": command_name,
        "paths": paths,
        "regions": regions,
        "symbols": symbols,
        "regions_or_symbols": sorted({*regions, *symbols}),
        "probe_kind": probe_kind,
    }


def classify_probe_delta(command: str, history: list[dict[str, Any]]) -> dict[str, Any]:
    signature = extract_probe_signature(command)
    prior_signatures = [
        dict(item.get("probe_signature", {}))
        for item in history
        if item.get("read_only_probe") and isinstance(item.get("probe_signature"), dict)
    ]
    prior_paths = {path for item in prior_signatures for path in item.get("paths", [])}
    prior_regions = {region for item in prior_signatures for region in item.get("regions", [])}
    prior_symbols = {symbol for item in prior_signatures for symbol in item.get("symbols", [])}
    new_path = any(path not in prior_paths for path in signature["paths"])
    new_region = any(region not in prior_regions for region in signature["regions"])
    new_symbol = any(symbol not in prior_symbols for symbol in signature["symbols"])

    if signature["probe_kind"] == "directory_only":
        delta_kind = "directory_sweep"
        incremental = False
    elif new_path and (new_region or new_symbol):
        delta_kind = "new_path_and_context"
        incremental = True
    elif new_path:
        delta_kind = "new_path"
        incremental = True
    elif new_region and new_symbol:
        delta_kind = "new_region_and_symbol"
        incremental = True
    elif new_region:
        delta_kind = "new_region"
        incremental = True
    elif new_symbol:
        delta_kind = "new_symbol"
        incremental = True
    else:
        delta_kind = "duplicate_probe"
        incremental = False

    return {
        "probe_signature": signature,
        "evidence_delta_kind": delta_kind,
        "evidence_incremental": incremental,
        "inspected_regions_or_symbols": list(signature["regions_or_symbols"]),
    }


def no_progress_probe_streak(history: list[dict[str, Any]]) -> int:
    streak = 0
    for item in reversed(history):
        if not item.get("read_only_probe"):
            break
        if item.get("evidence_incremental"):
            break
        streak += 1
    return streak


def focused_readonly_probe_summary(history: list[dict[str, Any]]) -> dict[str, Any]:
    streak_items: list[dict[str, Any]] = []
    for item in reversed(history):
        if not item.get("read_only_probe"):
            break
        streak_items.append(item)
    streak_items.reverse()
    path_counts: dict[str, int] = {}
    for item in streak_items:
        signature = dict(item.get("probe_signature", {}))
        seen_paths = []
        for value in signature.get("paths", []) or []:
            text = str(value).strip()
            if text and text not in seen_paths:
                seen_paths.append(text)
        for path in seen_paths:
            path_counts[path] = path_counts.get(path, 0) + 1
    ranked_paths = [
        path
        for path, _ in sorted(path_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    dominant_count = max(path_counts.values()) if path_counts else 0
    return {
        "streak": len(streak_items),
        "unique_path_count": len(path_counts),
        "path_counts": path_counts,
        "dominant_paths": ranked_paths,
        "dominant_count": dominant_count,
    }


def summarize_regions_or_symbols(history: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for item in history:
        for value in item.get("inspected_regions_or_symbols", []) or []:
            text = str(value).strip()
            if text and text not in values:
                values.append(text)
    return values


def summarize_probe_paths(history: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for item in history:
        signature = dict(item.get("probe_signature", {}))
        for value in signature.get("paths", []) or []:
            text = str(value).strip()
            if text and text not in values:
                values.append(text)
    return values

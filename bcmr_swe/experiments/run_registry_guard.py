"""Shared guardrails for MASGuard run-registry appends."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def registry_conflicts(
    *,
    registry_path: Path,
    method_version: str,
    output_artifacts: list[str | Path],
) -> dict[str, Any]:
    method_version = str(method_version or "")
    outputs = {str(path) for path in output_artifacts}
    method_version_lines: list[int] = []
    output_artifact_lines: dict[str, list[int]] = {path: [] for path in sorted(outputs)}
    parse_errors: list[dict[str, Any]] = []
    if not registry_path.exists():
        return {
            "registry_path": str(registry_path),
            "method_version": method_version,
            "method_version_lines": method_version_lines,
            "output_artifact_lines": output_artifact_lines,
            "parse_errors": parse_errors,
        }
    for line_no, raw in enumerate(registry_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except Exception as exc:
            parse_errors.append({"line": line_no, "error": str(exc)})
            continue
        if method_version and str(row.get("method_version", "") or "") == method_version:
            method_version_lines.append(line_no)
        for path in row.get("output_artifacts", []) or []:
            path_text = str(path)
            if path_text in output_artifact_lines:
                output_artifact_lines[path_text].append(line_no)
    return {
        "registry_path": str(registry_path),
        "method_version": method_version,
        "method_version_lines": method_version_lines,
        "output_artifact_lines": output_artifact_lines,
        "parse_errors": parse_errors,
    }


def duplicate_check_result(conflicts: dict[str, Any]) -> str:
    method_count = len(conflicts.get("method_version_lines", []) or [])
    output_lines = {
        path: lines
        for path, lines in dict(conflicts.get("output_artifact_lines", {}) or {}).items()
        if lines
    }
    if output_lines:
        pairs = ", ".join(f"{path}@{lines}" for path, lines in sorted(output_lines.items()))
        return f"output artifact already existed before this registry append: {pairs}"
    if method_count:
        return f"method_version already existed {method_count} time(s) before this registry append"
    return "no matching method_version or output artifact existed before this registry append"


def ensure_no_duplicate_output_artifacts(
    *,
    registry_path: Path,
    method_version: str,
    output_artifacts: list[str | Path],
) -> dict[str, Any]:
    conflicts = registry_conflicts(
        registry_path=registry_path,
        method_version=method_version,
        output_artifacts=output_artifacts,
    )
    duplicate_outputs = {
        path: lines
        for path, lines in dict(conflicts.get("output_artifact_lines", {}) or {}).items()
        if lines
    }
    if duplicate_outputs:
        details = ", ".join(f"{path} at line(s) {lines}" for path, lines in sorted(duplicate_outputs.items()))
        raise ValueError(
            "Refusing to append duplicate output artifact(s) to run registry; "
            f"use a new output path/version for a new run. Conflicts: {details}"
        )
    return conflicts

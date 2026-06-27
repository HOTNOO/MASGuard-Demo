"""Normalize existing MAS run artifacts into MAS-DX trajectory records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bcmr_swe.mas_diagnosis.schema import MASDXTrajectoryRecord


STAGE_ORDER = ("locator", "reproducer", "planner", "implementer", "patcher", "verifier")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_run_result(
    payload: dict[str, Any],
    *,
    source_path: str = "",
    manifest: dict[str, Any] | None = None,
) -> MASDXTrajectoryRecord:
    """Normalize one coordinator `run_result.json`-like object."""

    manifest = dict(manifest or {})
    instance_id = _first_text(payload.get("instance_id"), manifest.get("instance_id"), "unknown")
    stage_outputs = _stage_outputs_from(payload)
    patcher = dict(stage_outputs.get("patcher", {}) or {})
    verifier = dict(stage_outputs.get("verifier", {}) or {})
    return MASDXTrajectoryRecord(
        case_id=_case_id(payload, source_path=source_path, instance_id=instance_id),
        instance_id=instance_id,
        issue=_first_text(payload.get("issue"), manifest.get("problem_statement")),
        stage_outputs=stage_outputs,
        shared_facts=dict(payload.get("shared_facts", {}) or payload.get("shared_facts_snapshot", {}) or {}),
        commands=_commands_from_stage_outputs(stage_outputs),
        diff_summary=dict(
            payload.get("diff_summary", {})
            or payload.get("workspace_patch_summary", {})
            or patcher.get("patch_summary", {})
            or {}
        ),
        verifier_evidence=_verifier_evidence_from(payload, verifier),
        oracle=_oracle_from(payload),
        metadata={
            "source_path": source_path,
            "source_format": "run_result",
            "run_id": str(payload.get("run_id", "") or ""),
            "reported_success": bool(payload.get("success", False)),
            "manifest_path": str(payload.get("manifest_path", "") or manifest.get("manifest_path", "") or ""),
        },
    )


def normalize_natural_row(
    row: dict[str, Any],
    *,
    source_path: str = "",
    manifest: dict[str, Any] | None = None,
) -> MASDXTrajectoryRecord:
    """Normalize one row from natural pilot or method-matrix outputs."""

    manifest = dict(manifest or {})
    instance_id = _first_text(row.get("instance_id"), manifest.get("instance_id"), "unknown")
    stage_outputs = _stage_outputs_from(row)
    patcher = dict(stage_outputs.get("patcher", {}) or {})
    return MASDXTrajectoryRecord(
        case_id=_case_id(row, source_path=source_path, instance_id=instance_id),
        instance_id=instance_id,
        issue=_first_text(row.get("issue"), manifest.get("problem_statement")),
        stage_outputs=stage_outputs,
        shared_facts=dict(row.get("shared_facts_snapshot", {}) or row.get("shared_facts", {}) or {}),
        commands=_commands_from_stage_outputs(stage_outputs),
        diff_summary=dict(
            row.get("diff_summary", {})
            or row.get("workspace_patch_summary", {})
            or patcher.get("patch_summary", {})
            or {}
        ),
        verifier_evidence=_verifier_evidence_from(row, dict(stage_outputs.get("verifier", {}) or {})),
        oracle=_oracle_from(row),
        metadata={
            "source_path": source_path,
            "source_format": "natural_row",
            "run_id": str(row.get("run_id", "") or ""),
            "run_dir": str(row.get("run_dir", "") or ""),
            "system_variant": str(row.get("system_variant", "") or ""),
            "reported_success": bool(row.get("reported_success", False)),
            "natural_failure_family": str(row.get("natural_failure_family", "") or ""),
            "manual_review_status": str(row.get("manual_review_status", "") or ""),
            "manifest_path": str(row.get("manifest_path", "") or manifest.get("manifest_path", "") or ""),
        },
    )


def normalize_records_from_file(
    path: str | Path,
    *,
    manifest_by_instance: dict[str, dict[str, Any]] | None = None,
) -> list[MASDXTrajectoryRecord]:
    """Normalize supported JSON artifacts into records.

    Supported forms:
    - a single `run_result.json` object;
    - a JSON object with `rows`;
    - a JSON list of row-like objects.
    """

    path = Path(path)
    payload = load_json(path)
    manifests = dict(manifest_by_instance or {})
    source_path = str(path)
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        records = []
        for row in payload["rows"]:
            if isinstance(row, dict):
                instance_id = str(row.get("instance_id", "") or "")
                records.append(
                    normalize_natural_row(
                        row,
                        source_path=source_path,
                        manifest=manifests.get(instance_id, {}),
                    )
                )
        return records
    if isinstance(payload, list):
        records = []
        for row in payload:
            if isinstance(row, dict):
                instance_id = str(row.get("instance_id", "") or "")
                records.append(
                    normalize_natural_row(
                        row,
                        source_path=source_path,
                        manifest=manifests.get(instance_id, {}),
                    )
                )
        return records
    if isinstance(payload, dict):
        instance_id = str(payload.get("instance_id", "") or "")
        return [
            normalize_run_result(
                payload,
                source_path=source_path,
                manifest=manifests.get(instance_id, {}),
            )
        ]
    raise ValueError(f"Unsupported trajectory artifact format: {path}")


def load_manifest_by_instance(paths: list[str | Path]) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = load_json(path)
        if isinstance(payload, dict):
            instance_id = str(payload.get("instance_id", "") or "")
            if instance_id:
                item = dict(payload)
                item["manifest_path"] = str(path)
                manifests[instance_id] = item
    return manifests


def _stage_outputs_from(payload: dict[str, Any]) -> dict[str, Any]:
    stage_outputs = dict(payload.get("stage_outputs", {}) or payload.get("phase_outputs", {}) or {})
    return {
        stage: dict(stage_outputs.get(stage, {}) or {})
        for stage in STAGE_ORDER
        if isinstance(stage_outputs.get(stage), dict)
    }


def _commands_from_stage_outputs(stage_outputs: dict[str, Any]) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    for stage in STAGE_ORDER:
        output = dict(stage_outputs.get(stage, {}) or {})
        for index, command in enumerate(list(output.get("commands", []) or [])):
            if isinstance(command, dict):
                item = dict(command)
            else:
                item = {"command": str(command)}
            item.setdefault("stage", stage)
            item.setdefault("index", index)
            commands.append(item)
    return commands


def _verifier_evidence_from(payload: dict[str, Any], verifier: dict[str, Any]) -> dict[str, Any]:
    evidence = dict(payload.get("verifier_evidence", {}) or {})
    if verifier:
        evidence.update(
            {
                "success": verifier.get("success"),
                "status": verifier.get("status", ""),
                "verification": verifier.get("verification", ""),
            }
        )
    if "fail_to_pass_returncode" in payload:
        evidence["fail_to_pass_returncode"] = payload.get("fail_to_pass_returncode")
    if "fail_to_pass_output" in payload:
        evidence["fail_to_pass_output"] = payload.get("fail_to_pass_output")
    if "oracle_snapshot" in payload:
        evidence["oracle_snapshot"] = payload.get("oracle_snapshot")
    return {key: value for key, value in evidence.items() if value not in ("", None, [])}


def _oracle_from(payload: dict[str, Any]) -> dict[str, Any]:
    oracle: dict[str, Any] = {}
    for key in (
        "oracle_success",
        "oracle_returncode",
        "oracle_output",
        "fail_to_pass_returncode",
        "fail_to_pass_output",
    ):
        if key in payload:
            oracle[key] = payload.get(key)
    harness = payload.get("harness")
    if isinstance(harness, dict):
        oracle.update(harness)
    return {key: value for key, value in oracle.items() if value not in ("", None, [])}


def _case_id(payload: dict[str, Any], *, source_path: str, instance_id: str) -> str:
    explicit = _first_text(payload.get("case_id"), payload.get("run_id"))
    if explicit:
        return explicit
    if source_path:
        return f"{Path(source_path).stem}:{instance_id}"
    return instance_id


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""

"""Prompt/input builders for MAS-DX diagnosis comparisons."""

from __future__ import annotations

import json
from typing import Any

from bcmr_swe.mas_diagnosis.graph_builder import build_evidence_graph
from bcmr_swe.mas_diagnosis.schema import MASDXTrajectoryRecord


def build_probe_flat_input(record: MASDXTrajectoryRecord) -> dict[str, Any]:
    """Build a flat-log diagnosis input similar to a PROBE-style baseline."""

    sections = [
        ("ISSUE", record.issue),
        ("LOCATOR", _stage_text(record.stage_outputs.get("locator", {}))),
        ("PATCHER", _stage_text(record.stage_outputs.get("patcher", {}))),
        ("VERIFIER", _stage_text(record.stage_outputs.get("verifier", {}))),
        ("SHARED_FACTS", _json_clip(record.shared_facts, 2000)),
        ("COMMANDS", _json_clip(record.commands[-20:], 3000)),
        ("DIFF_SUMMARY", _json_clip(record.diff_summary, 1600)),
        ("ORACLE", _json_clip(record.oracle, 1600)),
    ]
    text = "\n\n".join(f"## {title}\n{body}" for title, body in sections if str(body).strip())
    return {
        "schema": "probe_flat_mas_input_v1",
        "case_id": record.case_id,
        "instance_id": record.instance_id,
        "input_mode": "flat_log",
        "text": text,
        "allowed_outputs": _allowed_outputs(),
    }


def build_mas_graph_input(record: MASDXTrajectoryRecord) -> dict[str, Any]:
    """Build a graph-aware diagnosis input for the MAS-DX method."""

    graph = build_evidence_graph(record).to_dict()
    return {
        "schema": "mas_dx_graph_input_v1",
        "case_id": record.case_id,
        "instance_id": record.instance_id,
        "input_mode": "cross_agent_graph",
        "issue_excerpt": record.issue[:1200],
        "graph": graph,
        "stage_summaries": {
            stage: _stage_summary(output)
            for stage, output in record.stage_outputs.items()
        },
        "allowed_outputs": _allowed_outputs(),
    }


def _allowed_outputs() -> dict[str, Any]:
    return {
        "fields": [
            "primary_failure_type",
            "is_cross_agent_failure",
            "responsible_stage",
            "faulty_artifact",
            "faulty_handoff_edge",
            "evidence_span_ids",
            "recommended_recovery_action",
            "recommended_rerun_scope",
            "confidence",
            "rationale",
        ],
        "require_evidence_grounding": True,
    }


def _stage_text(output: dict[str, Any]) -> str:
    if not output:
        return ""
    fields = {
        "success": output.get("success"),
        "status": output.get("status", ""),
        "stop_reason": output.get("stop_reason", ""),
        "located_files": output.get("located_files", ""),
        "plan": output.get("plan", ""),
        "patch_summary": output.get("patch_summary", {}),
        "verification": output.get("verification", ""),
    }
    return _json_clip(fields, 2200)


def _stage_summary(output: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": output.get("success"),
        "status": str(output.get("status", "") or ""),
        "stop_reason": str(output.get("stop_reason", "") or ""),
        "command_count": len(list(output.get("commands", []) or [])),
        "message_count": len(list(output.get("messages", []) or [])),
        "has_patch_summary": bool(output.get("patch_summary")),
        "has_verification": bool(output.get("verification")),
    }


def _json_clip(value: Any, limit: int) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text.replace("\x00", "")[:limit]

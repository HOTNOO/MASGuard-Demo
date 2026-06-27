"""PARC stage-boundary patch contracts.

The contract is a MAS handoff artifact: it is generated from the recovery
ledger before replaying the patcher stage, rendered into the replay context,
and audited after replay using fresh-diff evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any

from bcmr_swe.types import RecoveryLedger
from swe_mas.utils.path_filters import classify_changed_files, normalize_repo_path


SCHEMA_VERSION = "parc.patch_contract.v1"
PROMPT_BEGIN = "[PARC PATCH CONTRACT]"
PROMPT_END = "[/PARC PATCH CONTRACT]"


def _dedupe(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _dedupe_paths(values: list[Any]) -> list[str]:
    return _dedupe([normalize_repo_path(str(value)) for value in values])


def _enum_token(prefix: str, value: Any) -> str:
    raw = str(value or "unknown").strip().replace("-", "_").replace(" ", "_")
    token = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in raw).strip("_").upper()
    while "__" in token:
        token = token.replace("__", "_")
    return f"<{prefix}:{token or 'UNKNOWN'}>"


def _stable_id(parts: list[Any]) -> str:
    text = "|".join(str(part or "") for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _object_chain_from_ledger(ledger: RecoveryLedger) -> list[dict[str, Any]]:
    structured = dict(ledger.structured_state or {})
    raw_chain = structured.get("object_chain_view") or []
    if isinstance(raw_chain, dict):
        raw_chain = raw_chain.get("objects") or []
    return [dict(item) for item in list(raw_chain or []) if isinstance(item, dict)]


def _active_object_record(ledger: RecoveryLedger) -> dict[str, Any]:
    active_id = str(ledger.active_object_id or "").strip()
    active_type = str(ledger.active_object_type or "").strip().lower().replace("-", "_")
    fallback: dict[str, Any] = {}
    for item in _object_chain_from_ledger(ledger):
        item_type = str(item.get("object_type", "") or "").strip().lower().replace("-", "_")
        if item_type == "sharedfact":
            item_type = "shared_fact"
        if item_type == "verifierverdict":
            item_type = "verifier_verdict"
        if active_id and str(item.get("object_id", "") or "").strip() == active_id:
            return item
        if active_type and item_type == active_type and not fallback:
            fallback = item
    return fallback


def _contract_overlap(paths: list[str], suspects: list[str]) -> list[str]:
    overlap: list[str] = []
    for path in _dedupe_paths(paths):
        for suspect in _dedupe_paths(suspects):
            if path == suspect or path.endswith(f"/{suspect}") or suspect.endswith(f"/{path}"):
                overlap.append(path)
                break
    return _dedupe(overlap)


@dataclass(frozen=True)
class StageBoundaryPatchContract:
    """A compact contract for the patcher handoff boundary."""

    contract_id: str
    replay_scope: str
    repair_mode: str
    execution_profile: str
    active_object_id: str = ""
    active_object_type: str = ""
    producer_stage: str = ""
    consumer_stage: str = ""
    suspect_paths: list[str] = field(default_factory=list)
    required_fresh_source_diff: bool = True
    require_suspect_touch: bool = True
    require_focused_validation: bool = True
    max_fresh_source_files: int = 3
    forbidden_path_classes: list[str] = field(default_factory=lambda: ["test", "generated"])
    negative_constraints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "contract_id": self.contract_id,
            "boundary": {
                "producer_stage": self.producer_stage,
                "consumer_stage": self.consumer_stage,
                "replay_scope": self.replay_scope,
            },
            "active_object": {
                "object_id": self.active_object_id,
                "object_type": self.active_object_type,
            },
            "repair": {
                "repair_mode": self.repair_mode,
                "execution_profile": self.execution_profile,
            },
            "suspect_paths": list(self.suspect_paths),
            "required_freshness": {
                "fresh_source_diff": self.required_fresh_source_diff,
                "touch_suspect_path": self.require_suspect_touch,
                "focused_validation": self.require_focused_validation,
                "max_fresh_source_files": int(self.max_fresh_source_files),
            },
            "forbidden": {
                "path_classes": list(self.forbidden_path_classes),
            },
            "negative_constraints": list(self.negative_constraints),
        }

    def to_llm_payload(self) -> dict[str, Any]:
        payload = self.to_dict()
        return {
            "schema_version": payload["schema_version"],
            "contract_id": self.contract_id,
            "boundary": {
                "producer": _enum_token("STAGE", self.producer_stage),
                "consumer": _enum_token("STAGE", self.consumer_stage),
                "scope": _enum_token("SCOPE", self.replay_scope),
            },
            "active_object": {
                "type": _enum_token("OBJ_TYPE", self.active_object_type),
                "id_hash": _stable_id([self.active_object_id]) if self.active_object_id else "",
            },
            "requirements": {
                "fresh_source_diff": _enum_token("REQ", self.required_fresh_source_diff),
                "touch_suspect_path": _enum_token("REQ", self.require_suspect_touch),
                "focused_validation": _enum_token("REQ", self.require_focused_validation),
                "max_fresh_source_files": int(self.max_fresh_source_files),
            },
            "forbidden": {
                "path_classes": [_enum_token("PATH_CLASS", item) for item in self.forbidden_path_classes],
            },
            "paths": {
                "suspect_paths": list(self.suspect_paths[:8]),
            },
            "negative_constraints": [_enum_token("NEG", item) for item in self.negative_constraints[:8]],
        }

    def to_prompt_text(self) -> str:
        compact_json = json.dumps(self.to_llm_payload(), ensure_ascii=False, separators=(",", ":"))
        return f"{PROMPT_BEGIN}\n{compact_json}\n{PROMPT_END}"


def build_stage_boundary_patch_contract(
    ledger: RecoveryLedger,
    *,
    scope: str,
    repair_mode: str,
    execution_profile: str = "normal",
) -> StageBoundaryPatchContract:
    active = _active_object_record(ledger)
    producer = str(active.get("producer_stage", "") or "")
    consumer = str(active.get("consumer_stage", "") or "")
    active_type = str(ledger.active_object_type or active.get("object_type", "") or "").strip()
    active_id = str(ledger.active_object_id or active.get("object_id", "") or "").strip()
    suspect_paths = _dedupe_paths(list(ledger.suspect_paths or []))
    require_suspect_touch = bool(suspect_paths)
    require_validation = str(execution_profile or "").strip().lower() != "compact"
    negative_constraints = _dedupe(
        [
            " ".join(str(item or "").split())[:180]
            for item in list(ledger.negative_constraints or [])
            if str(item or "").strip()
        ]
    )[:6]
    max_fresh_source_files = max(1, min(3, len(suspect_paths) or 3))
    contract_id = _stable_id(
        [
            active_id,
            active_type,
            producer,
            consumer,
            scope,
            repair_mode,
            ",".join(suspect_paths[:4]),
        ]
    )
    return StageBoundaryPatchContract(
        contract_id=contract_id,
        replay_scope=str(scope or "patcher+verifier"),
        repair_mode=str(repair_mode or "local"),
        execution_profile=str(execution_profile or "normal"),
        active_object_id=active_id,
        active_object_type=active_type,
        producer_stage=producer,
        consumer_stage=consumer,
        suspect_paths=suspect_paths[:8],
        required_fresh_source_diff=True,
        require_suspect_touch=require_suspect_touch,
        require_focused_validation=require_validation,
        max_fresh_source_files=max_fresh_source_files,
        negative_constraints=negative_constraints,
    )


def audit_stage_boundary_patch_contract(
    contract: dict[str, Any],
    *,
    changed_files: list[str],
    fresh_changed_files: list[str],
    changed_file_classes: dict[str, list[str]] | None = None,
    fresh_changed_file_classes: dict[str, list[str]] | None = None,
    patcher_validation_command_count: int = 0,
    replay_diff_changed: bool = False,
) -> dict[str, Any]:
    """Evaluate replay output against a PARC patch contract."""

    if not contract or str(contract.get("schema_version", "") or "") != SCHEMA_VERSION:
        return {
            "schema_version": "parc.patch_contract_audit.v1",
            "contract_present": False,
            "satisfied": True,
            "hard_satisfied": True,
            "flags": [],
        }

    requirements = dict(contract.get("required_freshness", {}) or {})
    forbidden = dict(contract.get("forbidden", {}) or {})
    suspect_paths = _dedupe_paths(list(contract.get("suspect_paths", []) or []))
    fresh_files = _dedupe_paths(list(fresh_changed_files or []))
    all_changed = _dedupe_paths(list(changed_files or []))
    classes = changed_file_classes or classify_changed_files(all_changed)
    fresh_classes = fresh_changed_file_classes or classify_changed_files(fresh_files)
    fresh_source_files = _dedupe_paths(list(fresh_classes.get("source_files", []) or []))
    fresh_test_files = _dedupe_paths(list(fresh_classes.get("test_files", []) or []))
    fresh_generated_files = _dedupe_paths(list(fresh_classes.get("generated_files", []) or []))
    fresh_suspect_overlap = _contract_overlap(fresh_source_files, suspect_paths)
    try:
        max_fresh_source_files = max(1, int(requirements.get("max_fresh_source_files", 3) or 3))
    except (TypeError, ValueError):
        max_fresh_source_files = 3

    flags: list[str] = []
    hard_flags: list[str] = []
    if bool(requirements.get("fresh_source_diff", True)) and not fresh_source_files:
        flags.append("contract_no_fresh_source_diff")
        hard_flags.append("contract_no_fresh_source_diff")
    if bool(requirements.get("touch_suspect_path", False)) and fresh_source_files and not fresh_suspect_overlap:
        flags.append("contract_missed_suspect_path")
        hard_flags.append("contract_missed_suspect_path")

    forbidden_classes = {str(item).strip().lower() for item in list(forbidden.get("path_classes", []) or [])}
    if "test" in forbidden_classes and fresh_test_files:
        flags.append("contract_forbidden_test_edit")
        hard_flags.append("contract_forbidden_test_edit")
    if "generated" in forbidden_classes and fresh_generated_files:
        flags.append("contract_forbidden_generated_edit")
        hard_flags.append("contract_forbidden_generated_edit")
    if bool(requirements.get("focused_validation", True)) and fresh_source_files and patcher_validation_command_count <= 0:
        flags.append("contract_missing_focused_validation")
    if len(fresh_source_files) > max_fresh_source_files:
        flags.append("contract_too_many_source_files")

    hard_satisfied = not hard_flags
    satisfied = hard_satisfied and "contract_missing_focused_validation" not in flags
    return {
        "schema_version": "parc.patch_contract_audit.v1",
        "contract_present": True,
        "contract_id": str(contract.get("contract_id", "") or ""),
        "satisfied": satisfied,
        "hard_satisfied": hard_satisfied,
        "flags": _dedupe(flags),
        "hard_flags": _dedupe(hard_flags),
        "fresh_source_files": fresh_source_files,
        "fresh_suspect_path_overlap": fresh_suspect_overlap,
        "fresh_test_files": fresh_test_files,
        "fresh_generated_files": fresh_generated_files,
        "max_fresh_source_files": max_fresh_source_files,
        "replay_diff_changed": bool(replay_diff_changed),
        "patcher_validation_command_count": int(patcher_validation_command_count or 0),
        "changed_file_class_counts": {
            key: len(value)
            for key, value in dict(classes or {}).items()
        },
        "fresh_changed_file_class_counts": {
            key: len(value)
            for key, value in dict(fresh_classes or {}).items()
        },
    }


def parse_patch_contract_prompt(text: str) -> dict[str, Any]:
    """Parse a PARC patch contract from a replay prompt/problem statement.

    The prompt form intentionally contains the LLM-facing payload rather than
    the full internal dataclass.  This parser returns a minimal enforcement
    view used by the implementer-side apply-time guard.
    """

    match = re.search(
        re.escape(PROMPT_BEGIN) + r"\s*(\{.*?\})\s*" + re.escape(PROMPT_END),
        str(text or ""),
        flags=re.DOTALL,
    )
    if not match:
        return {}
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict) or str(payload.get("schema_version", "") or "") != SCHEMA_VERSION:
        return {}

    forbidden_tokens = list(dict(payload.get("forbidden", {}) or {}).get("path_classes", []) or [])
    forbidden_classes: list[str] = []
    for token in forbidden_tokens:
        text_token = str(token or "").strip()
        inner = text_token
        if text_token.startswith("<PATH_CLASS:") and text_token.endswith(">"):
            inner = text_token[len("<PATH_CLASS:") : -1]
        inner = inner.strip().lower().replace("_", "-")
        if inner:
            forbidden_classes.append(inner)

    requirements = dict(payload.get("requirements", {}) or {})
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_id": str(payload.get("contract_id", "") or ""),
        "suspect_paths": _dedupe_paths(list(dict(payload.get("paths", {}) or {}).get("suspect_paths", []) or [])),
        "forbidden_path_classes": _dedupe(forbidden_classes),
        "require_fresh_source_diff": str(requirements.get("fresh_source_diff", "") or "").upper().endswith("TRUE>"),
        "require_suspect_touch": str(requirements.get("touch_suspect_path", "") or "").upper().endswith("TRUE>"),
        "require_focused_validation": str(requirements.get("focused_validation", "") or "").upper().endswith("TRUE>"),
        "max_fresh_source_files": max(
            1,
            int(requirements.get("max_fresh_source_files", 3) or 3),
        ),
    }

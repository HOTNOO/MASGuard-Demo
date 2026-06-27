"""CAR patch intent passed across the MAS patcher boundary.

The CAR controller already chooses typed recovery actions.  This module turns
that controller decision plus the in-run belief revision signal into a compact
patch intent that the patcher and implementer can actually obey.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any

from bcmr_swe.types import RecoveryLedger, SemanticActionType
from swe_mas.utils.path_filters import normalize_repo_path


SCHEMA_VERSION = "car.patch_intent.v1"
PROMPT_BEGIN = "[CAR PATCH INTENT]"
PROMPT_END = "[/CAR PATCH INTENT]"


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


def _bool_token(value: bool) -> str:
    return _enum_token("REQ", bool(value))


def _decode_enum_token(value: Any, *, prefix: str = "") -> str:
    text = str(value or "").strip()
    if text.startswith("<") and text.endswith(">") and ":" in text:
        inner = text[1:-1]
        token_prefix, token_value = inner.split(":", 1)
        if not prefix or token_prefix == prefix:
            return token_value.strip().upper()
    return text.strip().upper()


def _decode_bool_token(value: Any) -> bool:
    return _decode_enum_token(value, prefix="REQ").endswith("TRUE")


def _looks_like_source_path(value: Any) -> bool:
    path = normalize_repo_path(str(value or ""))
    return bool(path and re.search(r"\.(py|pyi|js|ts|tsx|jsx|java|go|rs|c|cc|cpp|h|hpp)$", path))


def _latest_source_candidate(ledger: RecoveryLedger) -> dict[str, Any]:
    candidates = [
        dict(item)
        for item in list(ledger.metadata.get("source_candidate_memory", []) or [])
        if isinstance(item, dict)
    ]
    if not candidates:
        return {}

    def _score(candidate: dict[str, Any]) -> tuple[int, int, float]:
        mode = str(candidate.get("result_mode", "") or "")
        mode_score = 0
        if mode == "source_edit_pending_official":
            mode_score = 6
        elif mode == "oracle_failed_after_source_edit":
            mode_score = 4
        elif mode == "contract_violation_after_source_edit":
            mode_score = 3
        elif mode == "source_edit_but_not_suspect":
            mode_score = 2
        return (
            mode_score,
            1 if bool(candidate.get("touches_suspect_path", False)) else 0,
            float(candidate.get("created_at", 0.0) or 0.0),
        )

    return max(candidates, key=_score)


@dataclass(frozen=True)
class CARPatchIntent:
    """An action-conditioned repair intent for the patcher boundary."""

    intent_id: str
    selected_action: str
    counterexample_type: str = ""
    replay_scope: str = ""
    repair_mode: str = ""
    execution_profile: str = "normal"
    active_object_type: str = ""
    active_object_id: str = ""
    latest_revision_type: str = ""
    target_paths: list[str] = field(default_factory=list)
    suspect_paths: list[str] = field(default_factory=list)
    candidate_source_paths: list[str] = field(default_factory=list)
    avoid_target_paths: list[str] = field(default_factory=list)
    require_fresh_source_diff: bool = True
    require_target_touch: bool = False
    require_evidence_before_retarget: bool = False
    preserve_candidate_source: bool = False
    max_fresh_source_files: int = 3
    directives: list[str] = field(default_factory=list)
    negative_constraints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "intent_id": self.intent_id,
            "selected_action": self.selected_action,
            "counterexample_type": self.counterexample_type,
            "replay_scope": self.replay_scope,
            "repair_mode": self.repair_mode,
            "execution_profile": self.execution_profile,
            "active_object": {
                "object_type": self.active_object_type,
                "object_id": self.active_object_id,
            },
            "belief_revision": {
                "latest_revision_type": self.latest_revision_type,
            },
            "paths": {
                "target_paths": list(self.target_paths),
                "suspect_paths": list(self.suspect_paths),
                "candidate_source_paths": list(self.candidate_source_paths),
                "avoid_target_paths": list(self.avoid_target_paths),
            },
            "requirements": {
                "fresh_source_diff": self.require_fresh_source_diff,
                "touch_intended_target": self.require_target_touch,
                "evidence_before_retarget": self.require_evidence_before_retarget,
                "preserve_candidate_source": self.preserve_candidate_source,
                "max_fresh_source_files": int(self.max_fresh_source_files),
            },
            "directives": list(self.directives),
            "negative_constraints": list(self.negative_constraints),
        }

    def to_llm_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "intent_id": self.intent_id,
            "action": _enum_token("ACTION", self.selected_action),
            "counterexample": _enum_token("CE", self.counterexample_type),
            "replay": {
                "scope": _enum_token("SCOPE", self.replay_scope),
                "repair_mode": _enum_token("REPAIR", self.repair_mode),
                "execution_profile": _enum_token("PROFILE", self.execution_profile),
            },
            "active_object": {
                "type": _enum_token("OBJ_TYPE", self.active_object_type),
                "id_hash": _stable_id([self.active_object_id]) if self.active_object_id else "",
            },
            "belief_revision": {
                "latest": _enum_token("REV", self.latest_revision_type),
            },
            "requirements": {
                "fresh_source_diff": _bool_token(self.require_fresh_source_diff),
                "touch_intended_target": _bool_token(self.require_target_touch),
                "evidence_before_retarget": _bool_token(self.require_evidence_before_retarget),
                "preserve_candidate_source": _bool_token(self.preserve_candidate_source),
                "max_fresh_source_files": int(self.max_fresh_source_files),
            },
            "paths": {
                "target_paths": list(self.target_paths[:8]),
                "suspect_paths": list(self.suspect_paths[:8]),
                "candidate_source_paths": list(self.candidate_source_paths[:8]),
                "avoid_target_paths": list(self.avoid_target_paths[:8]),
            },
            "directives": [_enum_token("DIR", item) for item in self.directives[:10]],
            "negative_constraints": [
                " ".join(str(item or "").split())[:180]
                for item in self.negative_constraints[:6]
                if str(item or "").strip()
            ],
        }

    def to_prompt_text(self) -> str:
        compact_json = json.dumps(self.to_llm_payload(), ensure_ascii=False, separators=(",", ":"))
        return f"{PROMPT_BEGIN}\n{compact_json}\n{PROMPT_END}"


def build_car_patch_intent(
    ledger: RecoveryLedger,
    *,
    selected_action: str,
    scope: str,
    repair_mode: str,
    execution_profile: str = "normal",
) -> CARPatchIntent:
    action = _canonical_action(selected_action)
    replay_precondition = str(ledger.metadata.get("current_replay_precondition", "") or "")
    counterexample = str(
        dict(ledger.metadata.get("latest_car_counterexample", {}) or {}).get("counterexample_type", "")
        or ledger.trigger_reason
        or ""
    )
    latest_revision = str(
        dict(ledger.metadata.get("latest_belief_revision_event", {}) or {}).get("revision_type", "")
        or dict(ledger.metadata.get("latest_belief_revision_signal", {}) or {}).get("latest_revision_type", "")
        or ""
    )
    suspect_paths = _dedupe_paths(list(ledger.suspect_paths or []))
    active_target = normalize_repo_path(str(ledger.active_target or ""))
    target_paths = _dedupe_paths(([active_target] if active_target else []) + suspect_paths)
    candidate = _latest_source_candidate(ledger)
    candidate_paths = _dedupe_paths(
        list(candidate.get("fresh_source_files", []) or [])
        + list(candidate.get("source_files", []) or [])
    )
    avoid_paths = _dedupe_paths(
        [item for item in list(ledger.invalidated_targets or []) if _looks_like_source_path(item)]
    )

    preserve_candidate = bool(candidate_paths) and (
        str(repair_mode or "").startswith("candidate_preserving")
        or str(dict(candidate or {}).get("result_mode", "") or "")
        in {"oracle_failed_after_source_edit", "contract_violation_after_source_edit", "source_edit_pending_official"}
    )
    if preserve_candidate:
        target_paths = _dedupe_paths(candidate_paths + target_paths)
    if replay_precondition == "post_evidence_source_repair" and target_paths:
        # The action loop has already paid for evidence rechecking.  Keep the
        # repair boundary narrow and make no-diff a first-class violation.
        target_paths = _dedupe_paths(target_paths[:3])

    require_target_touch = action in {
        SemanticActionType.LOCAL_REPAIR.value,
        SemanticActionType.REPAIR_LOCAL.value,
    } and bool(target_paths)
    require_evidence_before_retarget = action in {
        SemanticActionType.LOCAL_REPAIR.value,
        SemanticActionType.REPAIR_LOCAL.value,
    } and latest_revision in {
        "invalidated_no_progress",
        "revoked",
        "belief_retargeted",
    }
    max_files = 4 if action in {SemanticActionType.SCOPE_EXPAND.value, SemanticActionType.EXPAND_SCOPE.value} else 3
    if require_target_touch and target_paths:
        max_files = max(1, min(max_files, len(target_paths) or max_files))

    directives = _directives_for_action(
        action=action,
        repair_mode=repair_mode,
        replay_precondition=replay_precondition,
        latest_revision=latest_revision,
        preserve_candidate=preserve_candidate,
        require_evidence_before_retarget=require_evidence_before_retarget,
    )
    negative_constraints = _dedupe(
        [
            " ".join(str(item or "").split())[:180]
            for item in list(ledger.negative_constraints or [])
            if str(item or "").strip()
        ]
    )
    intent_id = _stable_id(
        [
            action,
            counterexample,
            scope,
            repair_mode,
            ",".join(target_paths[:4]),
            latest_revision,
            len(list(ledger.tried_actions or [])),
        ]
    )
    return CARPatchIntent(
        intent_id=intent_id,
        selected_action=action,
        counterexample_type=counterexample,
        replay_scope=str(scope or ""),
        repair_mode=str(repair_mode or ""),
        execution_profile=str(execution_profile or "normal"),
        active_object_type=str(ledger.active_object_type or ""),
        active_object_id=str(ledger.active_object_id or ""),
        latest_revision_type=latest_revision,
        target_paths=target_paths[:8],
        suspect_paths=suspect_paths[:8],
        candidate_source_paths=candidate_paths[:8],
        avoid_target_paths=avoid_paths[:8],
        require_fresh_source_diff=True,
        require_target_touch=require_target_touch,
        require_evidence_before_retarget=require_evidence_before_retarget,
        preserve_candidate_source=preserve_candidate,
        max_fresh_source_files=max_files,
        directives=directives,
        negative_constraints=negative_constraints[:6],
    )


def parse_patch_intent_prompt(text: str) -> dict[str, Any]:
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

    requirements = dict(payload.get("requirements", {}) or {})
    paths = dict(payload.get("paths", {}) or {})
    return {
        "schema_version": SCHEMA_VERSION,
        "intent_id": str(payload.get("intent_id", "") or ""),
        "selected_action": _decode_enum_token(payload.get("action", ""), prefix="ACTION"),
        "counterexample_type": _decode_enum_token(payload.get("counterexample", ""), prefix="CE"),
        "latest_revision_type": _decode_enum_token(
            dict(payload.get("belief_revision", {}) or {}).get("latest", ""),
            prefix="REV",
        ),
        "target_paths": _dedupe_paths(list(paths.get("target_paths", []) or [])),
        "suspect_paths": _dedupe_paths(list(paths.get("suspect_paths", []) or [])),
        "candidate_source_paths": _dedupe_paths(list(paths.get("candidate_source_paths", []) or [])),
        "avoid_target_paths": _dedupe_paths(list(paths.get("avoid_target_paths", []) or [])),
        "require_fresh_source_diff": _decode_bool_token(requirements.get("fresh_source_diff", "")),
        "require_target_touch": _decode_bool_token(requirements.get("touch_intended_target", "")),
        "require_evidence_before_retarget": _decode_bool_token(requirements.get("evidence_before_retarget", "")),
        "preserve_candidate_source": _decode_bool_token(requirements.get("preserve_candidate_source", "")),
        "max_fresh_source_files": max(1, int(requirements.get("max_fresh_source_files", 3) or 3)),
        "directives": [
            _decode_enum_token(item, prefix="DIR")
            for item in list(payload.get("directives", []) or [])
            if str(item).strip()
        ],
        "negative_constraints": [
            str(item)
            for item in list(payload.get("negative_constraints", []) or [])
            if str(item).strip()
        ],
    }


def _intent_overlap(paths: list[str], targets: list[str]) -> list[str]:
    overlap: list[str] = []
    for path in _dedupe_paths(paths):
        for target in _dedupe_paths(targets):
            if path == target or path.endswith(f"/{target}") or target.endswith(f"/{path}"):
                overlap.append(path)
                break
    return _dedupe(overlap)


def audit_car_patch_intent(
    intent: dict[str, Any],
    *,
    changed_files: list[str],
    fresh_changed_files: list[str],
    changed_file_classes: dict[str, list[str]] | None = None,
    fresh_changed_file_classes: dict[str, list[str]] | None = None,
    patcher_validation_command_count: int = 0,
    replay_diff_changed: bool = False,
) -> dict[str, Any]:
    """Evaluate replay output against a CAR patch intent.

    This is the guard-side counterpart to the prompt-side patch intent.  The
    intent is only methodologically meaningful if it can be audited after the
    replay, so this function uses fresh diff evidence rather than the final
    polluted workspace snapshot.
    """

    if not intent or str(intent.get("schema_version", "") or "") != SCHEMA_VERSION:
        return {
            "schema_version": "car.patch_intent_audit.v1",
            "intent_present": False,
            "satisfied": True,
            "hard_satisfied": True,
            "flags": [],
            "hard_flags": [],
        }

    requirements = dict(intent.get("requirements", {}) or {})
    paths = dict(intent.get("paths", {}) or {})
    target_paths = _dedupe_paths(list(paths.get("target_paths", []) or []))
    candidate_paths = _dedupe_paths(list(paths.get("candidate_source_paths", []) or []))
    avoid_paths = _dedupe_paths(list(paths.get("avoid_target_paths", []) or []))
    intended_paths = candidate_paths if bool(requirements.get("preserve_candidate_source", False)) and candidate_paths else target_paths

    fresh_files = _dedupe_paths(list(fresh_changed_files or []))
    all_changed = _dedupe_paths(list(changed_files or []))
    classes = changed_file_classes or {}
    if not classes:
        from swe_mas.utils.path_filters import classify_changed_files

        classes = classify_changed_files(all_changed)
    fresh_classes = fresh_changed_file_classes or {}
    if not fresh_classes:
        from swe_mas.utils.path_filters import classify_changed_files

        fresh_classes = classify_changed_files(fresh_files)

    fresh_source_files = _dedupe_paths(list(fresh_classes.get("source_files", []) or []))
    fresh_test_files = _dedupe_paths(list(fresh_classes.get("test_files", []) or []))
    fresh_generated_files = _dedupe_paths(list(fresh_classes.get("generated_files", []) or []))
    target_overlap = _intent_overlap(fresh_source_files, intended_paths)
    avoid_overlap = _intent_overlap(fresh_source_files, avoid_paths)
    try:
        max_fresh_source_files = max(1, int(requirements.get("max_fresh_source_files", 3) or 3))
    except (TypeError, ValueError):
        max_fresh_source_files = 3

    flags: list[str] = []
    hard_flags: list[str] = []
    if bool(requirements.get("fresh_source_diff", True)) and not fresh_source_files:
        flags.append("intent_no_fresh_source_diff")
        hard_flags.append("intent_no_fresh_source_diff")
    if bool(requirements.get("touch_intended_target", False)) and fresh_source_files and intended_paths and not target_overlap:
        flags.append("intent_missed_target_path")
        hard_flags.append("intent_missed_target_path")
    if avoid_overlap and not target_overlap:
        flags.append("intent_touched_revoked_path")
        hard_flags.append("intent_touched_revoked_path")
    if len(fresh_source_files) > max_fresh_source_files:
        flags.append("intent_too_many_source_files")
        hard_flags.append("intent_too_many_source_files")
    if fresh_test_files:
        flags.append("intent_test_edit_present")
    if fresh_generated_files:
        flags.append("intent_generated_edit_present")
    if fresh_source_files and patcher_validation_command_count <= 0:
        flags.append("intent_missing_focused_validation")

    hard_satisfied = not hard_flags
    satisfied = hard_satisfied and "intent_missing_focused_validation" not in flags
    return {
        "schema_version": "car.patch_intent_audit.v1",
        "intent_present": True,
        "intent_id": str(intent.get("intent_id", "") or ""),
        "selected_action": str(intent.get("selected_action", "") or ""),
        "satisfied": satisfied,
        "hard_satisfied": hard_satisfied,
        "flags": _dedupe(flags),
        "hard_flags": _dedupe(hard_flags),
        "target_paths": target_paths,
        "candidate_source_paths": candidate_paths,
        "avoid_target_paths": avoid_paths,
        "fresh_source_files": fresh_source_files,
        "fresh_target_overlap": target_overlap,
        "fresh_avoid_overlap": avoid_overlap,
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


def _directives_for_action(
    *,
    action: str,
    repair_mode: str,
    replay_precondition: str,
    latest_revision: str,
    preserve_candidate: bool,
    require_evidence_before_retarget: bool,
) -> list[str]:
    directives: list[str] = ["produce_fresh_source_diff", "run_focused_validation"]
    if action in {SemanticActionType.LOCAL_REPAIR.value, SemanticActionType.REPAIR_LOCAL.value}:
        directives.append("edit_intended_source_boundary")
        directives.append("minimal_source_patch")
    if action in {SemanticActionType.SCOPE_EXPAND.value, SemanticActionType.EXPAND_SCOPE.value}:
        directives.append("refresh_localization_before_new_patch")
        directives.append("justify_retarget_with_evidence")
    if preserve_candidate:
        directives.append("preserve_and_refine_source_candidate")
    if replay_precondition == "post_evidence_source_repair":
        directives.append("post_evidence_source_repair")
        directives.append("do_not_spend_replay_on_readonly_diagnosis")
        directives.append("explain_if_no_source_edit_possible")
    if require_evidence_before_retarget:
        directives.append("do_not_reuse_revoked_belief_without_evidence")
    if latest_revision == "invalidated_no_progress":
        directives.append("avoid_repeating_no_diff_attempt")
    if str(repair_mode or "").startswith("rebuild"):
        directives.append("rebuild_target_from_current_evidence")
    return _dedupe(directives)


def _canonical_action(action: str) -> str:
    text = str(action or "").strip()
    if text == SemanticActionType.REPAIR_LOCAL.value:
        return SemanticActionType.LOCAL_REPAIR.value
    if text == SemanticActionType.EXPAND_SCOPE.value:
        return SemanticActionType.SCOPE_EXPAND.value
    return text

"""Convert executed recovery raw outputs into MAS-DX-R observations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FAILURE_LINE_MARKERS = (
    "E       ",
    "E   ",
    "E   ",
    "AssertionError",
    "AttributeError",
    "ValueError",
    "TypeError",
    "NameError",
    "ImportError",
    "ModuleNotFoundError",
    "KeyError",
    "IndexError",
    "Traceback",
    "FAILED ",
    "short test summary info",
)


def observations_from_action_outputs(plan_path: Path) -> list[dict[str, Any]]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    observations: list[dict[str, Any]] = []
    for shard in list(plan.get("shards", []) or []):
        if not isinstance(shard, dict):
            continue
        raw_output = Path(str(shard.get("raw_output", "") or ""))
        if not raw_output.exists():
            observations.append(_missing_observation(shard))
            continue
        payload = json.loads(raw_output.read_text(encoding="utf-8"))
        for row in list(payload.get("rows", []) or []):
            if isinstance(row, dict):
                observations.append(_observation_from_row(shard, row))
    return observations


def apply_observations_to_hypotheses(
    hypotheses: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    updated = [dict(item) for item in hypotheses]
    if not updated:
        return updated
    for observation in observations:
        signal = str(observation.get("observed_signal", "") or "")
        if signal == "environment_preflight_clean_but_tests_fail":
            _demote(updated, {"test_collection_blocker", "test_environment_blocker"})
            _promote(updated, {"patch_incomplete"})
            _mark_recovery_candidates(updated, {"verifier_protocol_drift", "verifier_acceptance_gap", "patch_incomplete"})
        elif signal == "validated_targets_collectable_but_tests_fail":
            _demote(updated, {"test_collection_blocker"})
            _promote(updated, {"patch_incomplete"})
            _mark_recovery_candidates(updated, {"verifier_protocol_drift", "patch_incomplete"})
        elif signal == "verifier_replay_clean_failure":
            _soften(updated, {"verifier_protocol_drift", "verifier_acceptance_gap", "handoff_information_loss"})
            _promote(updated, {"patch_incomplete"})
            _mark_recovery_candidates(updated, {"patch_incomplete", "local_patch_regression"})
        elif signal == "graph_static_patch_failure_span":
            _promote(updated, {"patch_incomplete", "local_patch_regression"})
            _mark_recovery_candidates(updated, {"patch_incomplete", "local_patch_regression"})
        elif signal in {"previous_patch_replay_blocked", "candidate_patch_replay_blocked"}:
            _soften(updated, {"verifier_protocol_drift", "verifier_acceptance_gap", "handoff_information_loss"})
            _mark_recovery_candidates(updated, {"patch_incomplete", "local_patch_regression"})
        elif signal == "evaluation_protocol_blocked":
            _promote(updated, {"test_collection_blocker", "test_environment_blocker"})
    return sorted(updated, key=_rank)


def _observation_from_row(shard: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    protocol_blocked = bool(row.get("evaluation_protocol_error", False))
    error_type = str(row.get("error_type", "") or "")
    run_scope = str(row.get("run_scope", "") or shard.get("run_scope", "") or "")
    collect = dict(row.get("evaluation_collect_validation", {}) or {})
    all_collectable = _all_collectable(collect)
    preflight_returncode = row.get("environment_preflight_returncode")
    fail_rc = row.get("fail_to_pass_returncode")
    oracle_rc = row.get("oracle_returncode")
    observed_signal = "inconclusive"
    hypothesis_update = "inconclusive"
    status = "inconclusive"
    candidate_patch_attempted = bool(row.get("candidate_patch_replay_attempted", False))
    candidate_patch_replayed = bool(row.get("candidate_patch_replayed", False))
    if candidate_patch_replayed and row.get("oracle_success") is True:
        observed_signal = "candidate_patch_replay_succeeded"
        hypothesis_update = "confirmed"
        status = "confirmed"
    elif candidate_patch_attempted and not candidate_patch_replayed:
        observed_signal = "candidate_patch_replay_blocked"
        hypothesis_update = "inconclusive"
        status = "protocol_blocked"
        protocol_blocked = True
    elif candidate_patch_replayed and fail_rc:
        observed_signal = "candidate_patch_replay_clean_failure"
        hypothesis_update = "weakened_prior_patch_hypothesis"
        status = "weakened"
    elif row.get("oracle_success") is True:
        observed_signal = "recovery_succeeded"
        hypothesis_update = "confirmed"
        status = "confirmed"
    elif error_type == "previous_patch_replay_blocked":
        observed_signal = "previous_patch_replay_blocked"
        hypothesis_update = "inconclusive"
        status = "protocol_blocked"
    elif protocol_blocked or error_type == "evaluation_protocol_blocked":
        observed_signal = "evaluation_protocol_blocked"
        hypothesis_update = "confirmed"
        status = "protocol_blocked"
    elif run_scope == "environment_preflight_then_verifier" and preflight_returncode == 0 and all_collectable and fail_rc:
        observed_signal = "environment_preflight_clean_but_tests_fail"
        hypothesis_update = "weakened_environment"
        status = "weakened"
    elif run_scope == "verifier_only_replay" and fail_rc:
        observed_signal = "verifier_replay_clean_failure"
        hypothesis_update = "weakened_verifier_or_handoff_only_hypothesis"
        status = "weakened"
    elif all_collectable and fail_rc:
        observed_signal = "validated_targets_collectable_but_tests_fail"
        hypothesis_update = "weakened_collection_blocker"
        status = "weakened"
    output_text = str(row.get("oracle_output", "") or row.get("fail_to_pass_output", "") or "")
    focused_lines = focused_failure_excerpt_lines(output_text)
    return {
        "schema": "mas_dx_r_intervention_observation_v1",
        "step_id": str(shard.get("shard_id", "") or ""),
        "instance_id": str(row.get("instance_id", "") or shard.get("instance_id", "") or ""),
        "intervention_type": _intervention_type(run_scope),
        "status": status,
        "returncode": oracle_rc,
        "output_excerpt": ("\n".join(focused_lines) if focused_lines else output_text[:1200]),
        "failure_output_excerpt": focused_lines,
        "observed_signal": observed_signal,
        "hypothesis_update": hypothesis_update,
        "protocol_blocked": protocol_blocked,
        "error_type": error_type,
        "cost": {
            "model_call_count": int(row.get("model_call_count", 0) or 0),
            "token_cost": float(row.get("token_cost", 0.0) or 0.0),
        },
        "candidate_patch_replay": {
            "attempted": candidate_patch_attempted,
            "replayed": candidate_patch_replayed,
            "source_files": list(row.get("candidate_patch_replay_source_files", []) or []),
            "filtered_files": list(row.get("candidate_patch_replay_filtered_files", []) or []),
        },
        "raw_output": str(shard.get("raw_output", "") or ""),
    }


def _missing_observation(shard: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "mas_dx_r_intervention_observation_v1",
        "step_id": str(shard.get("shard_id", "") or ""),
        "instance_id": str(shard.get("instance_id", "") or ""),
        "intervention_type": _intervention_type(str(shard.get("run_scope", "") or "")),
        "status": "planned_not_executed",
        "returncode": None,
        "output_excerpt": "",
        "failure_output_excerpt": [],
        "observed_signal": "missing_raw_output",
        "hypothesis_update": "inconclusive",
        "protocol_blocked": False,
        "error_type": "missing_raw_output",
        "cost": {},
        "raw_output": str(shard.get("raw_output", "") or ""),
    }


def _all_collectable(collect: dict[str, Any]) -> bool:
    if not collect:
        return False
    for key in ("fail_to_pass", "oracle"):
        item = dict(collect.get(key, {}) or {})
        if item.get("requested_count", 0) and item.get("kept_count") != item.get("requested_count"):
            return False
    return True


def focused_failure_excerpt_lines(output: str, *, max_lines: int = 24, context: int = 2) -> list[str]:
    """Extract failure-bearing lines instead of pytest headers.

    Bounded observations guide the next mutating recovery. The useful signal is
    usually around pytest failure nodes, exception lines, and short-summary
    entries; the session header often consumes the old excerpt budget.
    """

    lines = [line.rstrip() for line in str(output or "").splitlines()]
    if not lines:
        return []
    selected_indexes: set[int] = set()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if _is_failure_line(stripped):
            start = max(0, index - context)
            end = min(len(lines), index + context + 1)
            selected_indexes.update(range(start, end))
    if not selected_indexes:
        for index, line in enumerate(lines):
            stripped = line.strip()
            if "::test" in stripped or " failed" in stripped.lower():
                start = max(0, index - context)
                end = min(len(lines), index + context + 1)
                selected_indexes.update(range(start, end))
                break
    if not selected_indexes:
        return [line.strip()[:240] for line in lines if line.strip()][: min(8, max_lines)]
    excerpts: list[str] = []
    previous_index: int | None = None
    for index in sorted(selected_indexes):
        text = lines[index].strip()
        if not text:
            continue
        if _is_low_value_pytest_header(text):
            continue
        if previous_index is not None and index > previous_index + 1 and excerpts and excerpts[-1] != "...":
            excerpts.append("...")
        previous_index = index
        clipped = text[:240]
        if clipped not in excerpts:
            excerpts.append(clipped)
        if len(excerpts) >= max_lines:
            break
    return excerpts


def _is_failure_line(line: str) -> bool:
    text = str(line or "")
    if any(marker in text for marker in FAILURE_LINE_MARKERS):
        return True
    return text.startswith(">") or text.startswith("FAILED ")


def _is_low_value_pytest_header(line: str) -> bool:
    text = str(line or "").strip()
    lower = text.lower()
    return (
        "test session starts" in lower
        or lower.startswith("platform ")
        or lower.startswith("rootdir:")
        or lower.startswith("configfile:")
        or lower.startswith("plugins:")
        or lower.startswith("collected ")
        or set(text) <= {"=", "_", "-"}
    )


def _intervention_type(run_scope: str) -> str:
    return {
        "environment_preflight_then_verifier": "environment_preflight",
        "verifier_only_replay": "verifier_replay",
        "patcher_fixed_localization": "fixed_localization_repatch",
        "candidate_patch_verifier": "candidate_patch_replay_verify",
    }.get(run_scope, run_scope)


def _demote(hypotheses: list[dict[str, Any]], failure_types: set[str]) -> None:
    for hypothesis in hypotheses:
        if str(hypothesis.get("failure_type", "")) in failure_types:
            hypothesis["confidence"] = "low"
            notes = list(hypothesis.get("observation_notes", []) or [])
            notes.append("demoted_by_intervention_observation")
            hypothesis["observation_notes"] = notes


def _soften(hypotheses: list[dict[str, Any]], failure_types: set[str]) -> None:
    for hypothesis in hypotheses:
        if str(hypothesis.get("failure_type", "")) in failure_types:
            if str(hypothesis.get("confidence", "")) == "high":
                hypothesis["confidence"] = "medium"
            notes = list(hypothesis.get("observation_notes", []) or [])
            notes.append("softened_by_intervention_observation")
            hypothesis["observation_notes"] = notes


def _promote(hypotheses: list[dict[str, Any]], failure_types: set[str]) -> None:
    for hypothesis in hypotheses:
        if str(hypothesis.get("failure_type", "")) in failure_types:
            hypothesis["confidence"] = "high"
            notes = list(hypothesis.get("observation_notes", []) or [])
            notes.append("promoted_by_intervention_observation")
            hypothesis["observation_notes"] = notes


def _mark_recovery_candidates(hypotheses: list[dict[str, Any]], failure_types: set[str]) -> None:
    for hypothesis in hypotheses:
        if str(hypothesis.get("failure_type", "")) in failure_types:
            notes = list(hypothesis.get("observation_notes", []) or [])
            notes.append("candidate_after_environment_or_collection_weakened")
            hypothesis["observation_notes"] = notes


def _rank(hypothesis: dict[str, Any]) -> tuple[int, int]:
    confidence = {"high": 0, "medium": 1, "low": 2}.get(str(hypothesis.get("confidence", "")), 3)
    priority = {
        "test_environment_blocker": 0,
        "test_collection_blocker": 1,
        "verifier_protocol_drift": 0,
        "verifier_acceptance_gap": 1,
        "patch_incomplete": 2,
        "local_patch_regression": 3,
        "ambiguous": 99,
    }.get(str(hypothesis.get("failure_type", "")), 50)
    return (confidence, priority)

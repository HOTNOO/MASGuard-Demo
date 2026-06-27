"""Normalize limited-freeze raw runner rows into controller observations."""

from __future__ import annotations

from typing import Any


SUPPORTED_OBSERVATION_PROBES = {
    "no_intervention_validation": {
        "runner_module": "bcmr_swe.experiments.mas_dx_run_action_enforced_recovery",
        "run_scope": "verifier_only_replay",
        "supported_success_signals": ["no_intervention_validation_succeeded"],
        "requires_replay_previous_patch": False,
    },
    "patch_state_replay_validation": {
        "runner_module": "bcmr_swe.experiments.mas_dx_run_action_enforced_recovery",
        "run_scope": "verifier_only_replay",
        "supported_success_signals": ["patch_state_replay_validation_succeeded"],
        "requires_replay_previous_patch": True,
    },
    "protocol_first_target_validation": {
        "runner_module": "bcmr_swe.experiments.mas_dx_run_action_enforced_recovery",
        "run_scope": "verifier_only_replay",
        "supported_success_signals": [
            "protocol_first_gate_blocked",
            "no_oracle_recovery_observed",
        ],
        "requires_replay_previous_patch": False,
    },
}


def adapter_capabilities() -> dict[str, Any]:
    return {
        "schema": "mas_dx_r_limited_freeze_observation_adapter_capabilities_v1",
        "supported_observation_probes": dict(SUPPORTED_OBSERVATION_PROBES),
        "unsupported_observation_probes": {},
        "claim_boundary": {
            "adapter_only": True,
            "does_not_execute_recovery": True,
            "does_not_add_recovery_credit": True,
        },
    }


def supported_signals_for_shard(shard: dict[str, Any]) -> list[str]:
    probe = str(shard.get("observation_probe", "") or "")
    capability = dict(SUPPORTED_OBSERVATION_PROBES.get(probe, {}) or {})
    return list(capability.get("supported_success_signals", []) or [])


def shard_probe_supported(shard: dict[str, Any]) -> bool:
    probe = str(shard.get("observation_probe", "") or "")
    capability = dict(SUPPORTED_OBSERVATION_PROBES.get(probe, {}) or {})
    if not capability:
        return False
    if str(shard.get("runner_module", "") or "") != str(capability.get("runner_module", "") or ""):
        return False
    if str(shard.get("run_scope", "") or "") != str(capability.get("run_scope", "") or ""):
        return False
    if bool(capability.get("requires_replay_previous_patch", False)) != bool(
        shard.get("replay_previous_patch", False)
    ):
        return False
    return True


def contract_satisfied_by_signals(contract: dict[str, Any], signals: list[str]) -> bool:
    observed = {str(signal) for signal in list(signals or []) if str(signal)}
    any_of = {str(signal) for signal in list(contract.get("any_of", []) or []) if str(signal)}
    all_of = {str(signal) for signal in list(contract.get("all_of", []) or []) if str(signal)}
    if all_of:
        return all_of.issubset(observed)
    if any_of:
        return bool(any_of & observed)
    return False


def adapt_limited_freeze_observation(
    *,
    shard: dict[str, Any],
    raw_row: dict[str, Any],
) -> dict[str, Any]:
    probe = str(shard.get("observation_probe", "") or "")
    if probe == "no_intervention_validation":
        return _adapt_no_intervention(shard=shard, raw_row=raw_row)
    if probe == "patch_state_replay_validation":
        return _adapt_patch_state_replay(shard=shard, raw_row=raw_row)
    if probe == "protocol_first_target_validation":
        return _adapt_protocol_first(shard=shard, raw_row=raw_row)
    return _base_observation(shard=shard, raw_row=raw_row) | {
        "status": "unsupported_observation_probe",
        "observed_signals": [],
        "pre_oracle_observation_status": "not_satisfied",
        "selected_policy_action_admissible": False,
        "blockers": ["unsupported_observation_probe"],
    }


def _adapt_no_intervention(*, shard: dict[str, Any], raw_row: dict[str, Any]) -> dict[str, Any]:
    base = _base_observation(shard=shard, raw_row=raw_row)
    protocol_blocked = _protocol_blocked(raw_row)
    oracle_success = bool(raw_row.get("oracle_success", False))
    error_type = str(raw_row.get("error_type", "") or "")
    stop_reason = str(raw_row.get("stop_reason", "") or "")
    replay_requested = bool(raw_row.get("previous_patch_replay_requested", False))
    replayed = bool(raw_row.get("previous_patch_replayed", False))
    signals: list[str] = []
    status = "inconclusive"
    blockers: list[str] = []

    if error_type == "planned_not_executed" or stop_reason == "planned_not_executed":
        status = "planned_not_executed"
        blockers.append("raw_row_planned_not_executed")
    elif replay_requested or replayed:
        status = "protocol_blocked"
        signals.append("no_intervention_validation_contaminated_by_patch_replay")
        blockers.append("no_intervention_validation_contaminated_by_patch_replay")
    elif oracle_success and not protocol_blocked:
        signals.append("no_intervention_validation_succeeded")
        status = "confirmed"
    elif not oracle_success and not protocol_blocked:
        signals.extend(["no_intervention_validation_failed", "no_oracle_recovery_observed"])
        status = "contradicted"
    elif protocol_blocked:
        signals.append("no_intervention_validation_protocol_blocked")
        status = "protocol_blocked"
        blockers.append("no_intervention_validation_protocol_blocked")
    else:
        blockers.append("no_intervention_validation_signal_inconclusive")

    contract = dict(shard.get("pre_oracle_observation_contract", {}) or {})
    satisfied = contract_satisfied_by_signals(contract, signals)
    return base | {
        "status": status,
        "observed_signals": signals,
        "pre_oracle_observation_status": "satisfied" if satisfied else "not_satisfied",
        "selected_policy_action_admissible": satisfied,
        "protocol_blocked": protocol_blocked or status == "protocol_blocked",
        "blockers": blockers,
    }


def _adapt_patch_state_replay(*, shard: dict[str, Any], raw_row: dict[str, Any]) -> dict[str, Any]:
    base = _base_observation(shard=shard, raw_row=raw_row)
    protocol_blocked = _protocol_blocked(raw_row)
    replay_requested = bool(raw_row.get("previous_patch_replay_requested", False))
    replayed = bool(raw_row.get("previous_patch_replayed", False))
    oracle_success = bool(raw_row.get("oracle_success", False))
    error_type = str(raw_row.get("error_type", "") or "")
    stop_reason = str(raw_row.get("stop_reason", "") or "")
    signals: list[str] = []
    status = "inconclusive"
    blockers: list[str] = []

    if error_type == "planned_not_executed" or stop_reason == "planned_not_executed":
        status = "planned_not_executed"
        blockers.append("raw_row_planned_not_executed")
    elif replay_requested and not replayed:
        signals.append("patch_state_replay_blocked")
        status = "protocol_blocked"
        blockers.append("patch_state_replay_blocked")
    elif replayed and oracle_success and not protocol_blocked:
        signals.append("patch_state_replay_validation_succeeded")
        status = "confirmed"
    elif replayed and not oracle_success and not protocol_blocked:
        signals.extend(["patch_state_replay_validation_failed", "no_oracle_recovery_observed"])
        status = "contradicted"
    elif protocol_blocked:
        signals.append("patch_state_replay_validation_protocol_blocked")
        status = "protocol_blocked"
        blockers.append("patch_state_replay_validation_protocol_blocked")
    else:
        blockers.append("patch_state_replay_signal_inconclusive")

    contract = dict(shard.get("pre_oracle_observation_contract", {}) or {})
    satisfied = contract_satisfied_by_signals(contract, signals)
    return base | {
        "status": status,
        "observed_signals": signals,
        "pre_oracle_observation_status": "satisfied" if satisfied else "not_satisfied",
        "selected_policy_action_admissible": satisfied,
        "protocol_blocked": protocol_blocked or status == "protocol_blocked",
        "blockers": blockers,
    }


def _adapt_protocol_first(*, shard: dict[str, Any], raw_row: dict[str, Any]) -> dict[str, Any]:
    base = _base_observation(shard=shard, raw_row=raw_row)
    protocol_blocked = _protocol_blocked(raw_row)
    oracle_success = bool(raw_row.get("oracle_success", False))
    error_type = str(raw_row.get("error_type", "") or "")
    stop_reason = str(raw_row.get("stop_reason", "") or "")
    signals: list[str] = []
    status = "inconclusive"
    blockers: list[str] = []

    if error_type == "planned_not_executed" or stop_reason == "planned_not_executed":
        status = "planned_not_executed"
        blockers.append("raw_row_planned_not_executed")
    elif protocol_blocked:
        signals.append("protocol_first_gate_blocked")
        if not oracle_success:
            signals.append("no_oracle_recovery_observed")
        status = "confirmed"
    elif oracle_success:
        signals.append("protocol_first_validation_succeeded")
        status = "confirmed"
    else:
        signals.extend(["protocol_first_gate_not_blocked", "no_oracle_recovery_observed"])
        status = "contradicted"

    contract = dict(shard.get("pre_oracle_observation_contract", {}) or {})
    satisfied = contract_satisfied_by_signals(contract, signals)
    return base | {
        "status": status,
        "observed_signals": signals,
        "pre_oracle_observation_status": "satisfied" if satisfied else "not_satisfied",
        "selected_policy_action_admissible": satisfied,
        "protocol_blocked": protocol_blocked,
        "blockers": blockers,
    }


def _base_observation(*, shard: dict[str, Any], raw_row: dict[str, Any]) -> dict[str, Any]:
    action = dict(raw_row.get("action_adherence_observed", {}) or {})
    model_call_count = int(raw_row.get("model_call_count", action.get("model_call_count", 0)) or 0)
    token_cost = float(raw_row.get("token_cost", action.get("token_cost", 0.0)) or 0.0)
    return {
        "schema": "mas_dx_r_limited_freeze_observation_signal_v1",
        "shard_id": str(shard.get("shard_id", "") or ""),
        "instance_id": str(raw_row.get("instance_id", "") or shard.get("instance_id", "") or ""),
        "source_guard": str(shard.get("source_guard", "") or ""),
        "delta_id": str(shard.get("delta_id", "") or ""),
        "freeze_rule_id": str(shard.get("freeze_rule_id", "") or ""),
        "observation_probe": str(shard.get("observation_probe", "") or ""),
        "selected_policy_action": str(shard.get("selected_policy_action", "") or ""),
        "raw_error_type": str(raw_row.get("error_type", "") or ""),
        "raw_stop_reason": str(raw_row.get("stop_reason", "") or ""),
        "raw_oracle_success": bool(raw_row.get("oracle_success", False)),
        "model_call_count": model_call_count,
        "token_cost": token_cost,
        "recovery_credit_allowed": False,
        "claim_boundary": {
            "observation_signal_only": True,
            "does_not_add_recovery_credit": True,
            "requires_downstream_accounting_before_claim": True,
        },
    }


def _protocol_blocked(raw_row: dict[str, Any]) -> bool:
    if bool(raw_row.get("evaluation_protocol_error", False)):
        return True
    error_type = str(raw_row.get("error_type", "") or "")
    if error_type in {"evaluation_protocol_blocked", "previous_patch_replay_blocked"}:
        return True
    protocol_type = str(raw_row.get("evaluation_protocol_error_type", "") or "")
    return bool(protocol_type)

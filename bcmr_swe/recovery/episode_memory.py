"""Episode memory for Counterexample-Aware Recovery.

The store is deliberately simple and deterministic: JSONL append for newly
observed episodes, read-only snapshots for experiment priors, and explicit
leave-one-instance-out filtering by hashed instance id.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import Any

from bcmr_swe.types import RecoveryLedger, SemanticActionType


CAR_METHOD_VERSION = "car_v1"


@dataclass(frozen=True, slots=True)
class ActionPrior:
    action: str
    sample_count: int = 0
    action_success_sample_count: int = 0
    cell_success_sample_count: int = 0
    success_rate: float = 0.0
    trajectory_success_rate: float = 0.0
    trajectory_success_sample_count: int = 0
    trajectory_failure_rate: float = 0.0
    trajectory_failure_sample_count: int = 0
    state_change_rate: float = 0.0
    no_diff_rate: float = 0.0
    no_diff_lower_bound: float = 0.0
    avg_token_cost: float = 0.0
    last_result_mode: str = ""
    last_verifier_delta: str = "unknown"
    source: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "sample_count": int(self.sample_count),
            "action_success_sample_count": int(self.action_success_sample_count),
            "cell_success_sample_count": int(self.cell_success_sample_count),
            "success_rate": float(self.success_rate),
            "trajectory_success_rate": float(self.trajectory_success_rate),
            "trajectory_success_sample_count": int(self.trajectory_success_sample_count),
            "trajectory_failure_rate": float(self.trajectory_failure_rate),
            "trajectory_failure_sample_count": int(self.trajectory_failure_sample_count),
            "state_change_rate": float(self.state_change_rate),
            "no_diff_rate": float(self.no_diff_rate),
            "no_diff_lower_bound": float(self.no_diff_lower_bound),
            "avg_token_cost": float(self.avg_token_cost),
            "last_result_mode": self.last_result_mode,
            "last_verifier_delta": self.last_verifier_delta,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class RecoveryEpisode:
    episode_id: str
    source_run_id: str
    instance_id_hash: str
    counterexample_type: str
    object_chain_hash: str
    object_chain_signature: dict[str, int] | None
    failure_family_manual: str
    action_taken: str
    action_score: float = 0.0
    action_was_first_choice: bool = False
    state_changed: bool = False
    fresh_artifact_produced: bool = False
    no_diff_after: bool = False
    verifier_delta: str = "unknown"
    result_mode: str = ""
    action_succeeded: bool = False
    action_success_credit: bool = False
    action_changed_state: bool = False
    cell_succeeded: bool = False
    cell_outcome_known: bool = False
    token_cost: float = 0.0
    latency_sec: float = 0.0
    success_within_episode: bool = False
    stop_reason: str = ""
    provider_clean: bool = True
    oracle_clean: bool = True
    infra_clean: bool = True
    method_version: str = CAR_METHOD_VERSION
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "car_recovery_episode_v1",
            "episode_id": self.episode_id,
            "source_run_id": self.source_run_id,
            "instance_id_hash": self.instance_id_hash,
            "counterexample_type": self.counterexample_type,
            "object_chain_hash": self.object_chain_hash,
            "object_chain_signature": dict(self.object_chain_signature or {}),
            "failure_family_manual": self.failure_family_manual,
            "action_taken": self.action_taken,
            "action_score": float(self.action_score),
            "action_was_first_choice": bool(self.action_was_first_choice),
            "state_changed": bool(self.state_changed),
            "fresh_artifact_produced": bool(self.fresh_artifact_produced),
            "no_diff_after": bool(self.no_diff_after),
            "verifier_delta": self.verifier_delta,
            "result_mode": self.result_mode,
            "action_succeeded": bool(self.action_succeeded),
            "action_success_credit": bool(self.action_success_credit),
            "action_changed_state": bool(self.action_changed_state),
            "cell_succeeded": bool(self.cell_succeeded),
            "cell_outcome_known": bool(self.cell_outcome_known),
            "token_cost": float(self.token_cost),
            "latency_sec": float(self.latency_sec),
            "success_within_episode": bool(self.success_within_episode),
            "stop_reason": self.stop_reason,
            "provider_clean": bool(self.provider_clean),
            "oracle_clean": bool(self.oracle_clean),
            "infra_clean": bool(self.infra_clean),
            "method_version": self.method_version,
            "created_at": float(self.created_at or 0.0),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecoveryEpisode":
        return cls(
            episode_id=str(data.get("episode_id", "") or ""),
            source_run_id=str(data.get("source_run_id", "") or ""),
            instance_id_hash=str(data.get("instance_id_hash", "") or ""),
            counterexample_type=str(data.get("counterexample_type", "") or ""),
            object_chain_hash=str(data.get("object_chain_hash", "") or ""),
            object_chain_signature={
                _canonical_object_type(key): int(value)
                for key, value in dict(data.get("object_chain_signature", {}) or {}).items()
                if _canonical_object_type(key)
            },
            failure_family_manual=str(data.get("failure_family_manual", "") or ""),
            action_taken=str(data.get("action_taken", "") or ""),
            action_score=_safe_float(data.get("action_score", 0.0)),
            action_was_first_choice=bool(data.get("action_was_first_choice", False)),
            state_changed=bool(data.get("state_changed", data.get("action_changed_state", False))),
            fresh_artifact_produced=bool(data.get("fresh_artifact_produced", False)),
            no_diff_after=bool(data.get("no_diff_after", False)),
            verifier_delta=str(data.get("verifier_delta", "") or "unknown"),
            result_mode=str(data.get("result_mode", "") or data.get("stop_reason", "") or ""),
            action_succeeded=bool(data.get("action_succeeded", data.get("success_within_episode", False))),
            action_success_credit=bool(
                data.get(
                    "action_success_credit",
                    data.get("action_succeeded", data.get("success_within_episode", False)),
                )
            ),
            action_changed_state=bool(data.get("action_changed_state", data.get("state_changed", False))),
            cell_succeeded=bool(data.get("cell_succeeded", False)),
            cell_outcome_known=bool(data.get("cell_outcome_known", False)),
            token_cost=_safe_float(data.get("token_cost", 0.0)),
            latency_sec=_safe_float(data.get("latency_sec", 0.0)),
            success_within_episode=bool(data.get("success_within_episode", False)),
            stop_reason=str(data.get("stop_reason", "") or ""),
            provider_clean=bool(data.get("provider_clean", True)),
            oracle_clean=bool(data.get("oracle_clean", True)),
            infra_clean=bool(data.get("infra_clean", True)),
            method_version=str(data.get("method_version", "") or CAR_METHOD_VERSION),
            created_at=_safe_float(data.get("created_at", 0.0)),
        )


class EpisodeStore:
    def __init__(self, path: str | Path | None = None):
        self.path = self._normalize_optional_path(path)

    @staticmethod
    def _normalize_optional_path(path: str | Path | None) -> Path | None:
        if path is None:
            return None
        text = str(path).strip()
        if not text or text == ".":
            return None
        return Path(text)

    def load(self) -> list[RecoveryEpisode]:
        if self.path is None or not self.path.exists() or self.path.is_dir():
            return []
        episodes: list[RecoveryEpisode] = []
        by_episode_id: dict[str, RecoveryEpisode] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                episode = RecoveryEpisode.from_dict(payload)
                if episode.action_taken:
                    if episode.episode_id:
                        by_episode_id[episode.episode_id] = episode
                    else:
                        episodes.append(episode)
        episodes.extend(by_episode_id.values())
        return episodes

    def append(self, episode: RecoveryEpisode) -> None:
        if self.path is None:
            return
        if self.path.exists() and self.path.is_dir():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(episode.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def stable_hash(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def object_chain_hash_from_ledger(ledger: RecoveryLedger) -> str:
    signature = object_chain_signature_from_ledger(ledger)
    if not signature:
        return stable_hash(str(ledger.active_object_type or "unknown"))
    expanded: list[str] = []
    for object_type in sorted(signature):
        expanded.extend([object_type] * int(signature[object_type]))
    return stable_hash(",".join(expanded))


def object_chain_signature_from_ledger(ledger: RecoveryLedger) -> dict[str, int]:
    structured = dict(ledger.structured_state or {})
    object_chain = structured.get("object_chain_view") or []
    object_types: list[str] = []
    for item in list(object_chain or []):
        if not isinstance(item, dict):
            continue
        object_type = _canonical_object_type(item.get("object_type", "") or item.get("type", ""))
        if object_type:
            object_types.append(object_type)
    if not object_types:
        object_types = [_canonical_object_type(ledger.active_object_type or "unknown")]
    return dict(Counter(item for item in object_types if item))


def object_chain_similarity(left: dict[str, int] | None, right: dict[str, int] | None) -> float:
    left_counts = {
        _canonical_object_type(k): int(v)
        for k, v in dict(left or {}).items()
        if _canonical_object_type(k) and int(v) > 0
    }
    right_counts = {
        _canonical_object_type(k): int(v)
        for k, v in dict(right or {}).items()
        if _canonical_object_type(k) and int(v) > 0
    }
    if not left_counts or not right_counts:
        return 0.0
    keys = set(left_counts) | set(right_counts)
    intersection = sum(min(left_counts.get(key, 0), right_counts.get(key, 0)) for key in keys)
    union = sum(max(left_counts.get(key, 0), right_counts.get(key, 0)) for key in keys)
    return float(intersection) / float(union) if union > 0 else 0.0


def instance_hash_from_ledger(ledger: RecoveryLedger) -> str:
    structured = dict(ledger.structured_state or {})
    metadata = dict(structured.get("metadata", {}) or {})
    local = dict(structured.get("local_region_view", {}) or {})
    return stable_hash(
        metadata.get("instance_id", "")
        or local.get("instance_id", "")
        or ledger.metadata.get("instance_id", "")
    )


def make_recovery_episode(
    ledger: RecoveryLedger,
    *,
    action_taken: str,
    result_mode: str,
    token_cost: float = 0.0,
    latency_sec: float = 0.0,
    state_changed: bool | None = None,
    verifier_delta: str = "",
    action_score: float = 0.0,
    action_was_first_choice: bool = False,
) -> RecoveryEpisode:
    action_value = str(action_taken or "")
    fresh_source_files = fresh_source_files_from_latest_guard(ledger)
    mode = str(result_mode or "")
    no_fresh_source_modes = {
        "no_diff",
        "contract_violation_no_fresh_source",
        "intent_violation_no_fresh_source",
        "wrong_edit_target",
        "budget_exhausted",
    }
    no_diff_after = mode in no_fresh_source_modes and not fresh_source_files
    changed = bool(state_changed) if state_changed is not None else bool(fresh_source_files) or not no_diff_after
    if mode in no_fresh_source_modes and not fresh_source_files:
        changed = False
    structured = dict(ledger.structured_state or {})
    evidence = dict(structured.get("evidence_pack", {}) or {})
    latest_ce = dict(ledger.metadata.get("latest_car_counterexample", {}) or {})
    return RecoveryEpisode(
        episode_id=uuid.uuid4().hex,
        source_run_id=str(ledger.metadata.get("car_source_run_id", "") or ""),
        instance_id_hash=instance_hash_from_ledger(ledger),
        counterexample_type=str(latest_ce.get("counterexample_type", "") or ""),
        object_chain_hash=object_chain_hash_from_ledger(ledger),
        object_chain_signature=object_chain_signature_from_ledger(ledger),
        failure_family_manual=str(evidence.get("failure_family_manual", "") or ""),
        action_taken=action_value,
        action_score=float(action_score or 0.0),
        action_was_first_choice=bool(action_was_first_choice),
        state_changed=changed,
        fresh_artifact_produced=bool(fresh_source_files),
        no_diff_after=no_diff_after,
        verifier_delta=str(verifier_delta or verifier_delta_from_result_mode(mode)),
        result_mode=mode,
        action_succeeded=mode == "strong_source_success",
        action_success_credit=mode == "strong_source_success",
        action_changed_state=changed,
        cell_succeeded=bool(ledger.metadata.get("car_cell_succeeded", False)),
        token_cost=float(token_cost or 0.0),
        latency_sec=float(latency_sec or 0.0),
        success_within_episode=mode == "strong_source_success",
        stop_reason=str(ledger.metadata.get("budget_stop_reason", "") or ""),
        provider_clean=not bool(ledger.metadata.get("provider_error_observed", False)),
        oracle_clean=not bool(ledger.metadata.get("oracle_infra_error", False)),
        infra_clean=not bool(ledger.metadata.get("substrate_error", False)),
        method_version=str(ledger.metadata.get("car_method_version", "") or CAR_METHOD_VERSION),
        created_at=time.time(),
    )


def remember_episode(
    ledger: RecoveryLedger,
    episode: RecoveryEpisode,
    *,
    append_path: str | Path | None = None,
) -> None:
    history = [
        dict(item)
        for item in list(ledger.metadata.get("car_episode_memory", []) or [])
        if isinstance(item, dict)
    ]
    payload = episode.to_dict()
    history.append(payload)
    ledger.metadata["car_episode_memory"] = history[-12:]
    ledger.metadata["latest_car_episode"] = payload
    if append_path:
        EpisodeStore(append_path).append(episode)


def finalize_episodes(
    ledger: RecoveryLedger,
    *,
    cell_succeeded: bool,
    append_path: str | Path | None = None,
    provider_clean: bool | None = None,
    oracle_clean: bool | None = None,
    infra_clean: bool | None = None,
) -> None:
    finalized: list[dict[str, Any]] = []
    for item in list(ledger.metadata.get("car_episode_memory", []) or []):
        if not isinstance(item, dict):
            continue
        payload = dict(item)
        payload["cell_succeeded"] = bool(cell_succeeded)
        payload["cell_outcome_known"] = True
        if provider_clean is not None:
            payload["provider_clean"] = bool(payload.get("provider_clean", True)) and bool(provider_clean)
        if oracle_clean is not None:
            payload["oracle_clean"] = bool(payload.get("oracle_clean", True)) and bool(oracle_clean)
        if infra_clean is not None:
            payload["infra_clean"] = bool(payload.get("infra_clean", True)) and bool(infra_clean)
        finalized.append(payload)
        if append_path:
            EpisodeStore(append_path).append(RecoveryEpisode.from_dict(payload))
    if finalized:
        ledger.metadata["car_episode_memory"] = finalized[-12:]
        ledger.metadata["latest_car_episode"] = finalized[-1]


def action_priors(
    episodes: list[RecoveryEpisode],
    *,
    counterexample_type: str,
    object_chain_hash: str,
    object_chain_signature: dict[str, int] | None = None,
    candidate_actions: list[str],
    exclude_instance_hash: str | None = None,
    method_version: str = CAR_METHOD_VERSION,
    min_samples_for_exact_chain: int = 3,
    similarity_threshold: float = 0.5,
) -> dict[str, ActionPrior]:
    actions = [str(action) for action in candidate_actions]
    target_signature = dict(object_chain_signature or {})
    exact = _filter_episodes(
        episodes,
        counterexample_type=counterexample_type,
        object_chain_hash=object_chain_hash,
        object_chain_signature=target_signature,
        min_object_chain_similarity=1.0,
        exclude_instance_hash=exclude_instance_hash,
        method_version=method_version,
    )
    source = "ce_type_object_chain"
    if len(exact) < int(min_samples_for_exact_chain):
        similar = []
        if target_signature:
            similar = _filter_episodes(
                episodes,
                counterexample_type=counterexample_type,
                object_chain_hash="",
                object_chain_signature=target_signature,
                min_object_chain_similarity=float(similarity_threshold),
                exclude_instance_hash=exclude_instance_hash,
                method_version=method_version,
            )
        if similar:
            exact = similar
            source = "ce_type_object_chain_soft"
        else:
            exact = _filter_episodes(
                episodes,
                counterexample_type=counterexample_type,
                object_chain_hash="",
                object_chain_signature={},
                min_object_chain_similarity=0.0,
                exclude_instance_hash=exclude_instance_hash,
                method_version=method_version,
            )
            source = "ce_type_fallback"
    if len(exact) < int(min_samples_for_exact_chain) and source == "ce_type_object_chain_soft":
        # Keep soft matches even when sparse: they are more specific than pure
        # counterexample-type fallback and can still inform non-rejection scores.
        pass
    elif len(exact) < int(min_samples_for_exact_chain) and source != "ce_type_fallback":
        exact = _filter_episodes(
            episodes,
            counterexample_type=counterexample_type,
            object_chain_hash="",
            object_chain_signature={},
            min_object_chain_similarity=0.0,
            exclude_instance_hash=exclude_instance_hash,
            method_version=method_version,
        )
        source = "ce_type_fallback"
    return {
        action: _prior_for_action(
            action,
            [episode for episode in exact if episode.action_taken == action],
            source=source,
        )
        for action in actions
    }


def local_action_prior(ledger: RecoveryLedger, action: str | SemanticActionType) -> ActionPrior:
    action_value = str(action.value if isinstance(action, SemanticActionType) else action)
    episodes = [
        RecoveryEpisode.from_dict(dict(item))
        for item in list(ledger.metadata.get("car_episode_memory", []) or [])
        if isinstance(item, dict) and str(item.get("action_taken", "") or "") == action_value
    ]
    return _prior_for_action(action_value, episodes, source="episode")


def history_action_priors_from_ledger(
    ledger: RecoveryLedger,
    *,
    counterexample_type: str,
    candidate_actions: list[str],
) -> dict[str, ActionPrior]:
    raw = [
        RecoveryEpisode.from_dict(dict(item))
        for item in list(ledger.metadata.get("car_prior_episodes", []) or [])
        if isinstance(item, dict)
    ]
    return action_priors(
        raw,
        counterexample_type=counterexample_type,
        object_chain_hash=object_chain_hash_from_ledger(ledger),
        object_chain_signature=object_chain_signature_from_ledger(ledger),
        candidate_actions=candidate_actions,
        exclude_instance_hash=instance_hash_from_ledger(ledger),
        method_version=str(ledger.metadata.get("car_method_version", "") or CAR_METHOD_VERSION),
    )


def merge_priors(local: ActionPrior, historical: ActionPrior) -> ActionPrior:
    if historical.sample_count <= 0:
        return local
    if local.sample_count <= 0:
        return historical
    total = local.sample_count + historical.sample_count

    def weighted(left: float, right: float) -> float:
        return (
            (float(left) * float(local.sample_count))
            + (float(right) * float(historical.sample_count))
        ) / float(total)

    action_success_total = local.action_success_sample_count + historical.action_success_sample_count
    success_rate = float(action_success_total) / float(total) if total > 0 else 0.0
    cell_known_total = local.cell_success_sample_count + historical.cell_success_sample_count
    trajectory_success_total = (
        local.trajectory_success_sample_count
        + historical.trajectory_success_sample_count
    )
    trajectory_failure_total = (
        local.trajectory_failure_sample_count
        + historical.trajectory_failure_sample_count
    )
    trajectory_success_rate = (
        float(trajectory_success_total) / float(cell_known_total)
        if cell_known_total > 0
        else 0.0
    )
    trajectory_failure_rate = (
        float(trajectory_failure_total) / float(cell_known_total)
        if cell_known_total > 0
        else 0.0
    )

    return ActionPrior(
        action=local.action,
        sample_count=total,
        action_success_sample_count=action_success_total,
        cell_success_sample_count=local.cell_success_sample_count + historical.cell_success_sample_count,
        success_rate=success_rate,
        trajectory_success_rate=trajectory_success_rate,
        trajectory_success_sample_count=trajectory_success_total,
        trajectory_failure_rate=trajectory_failure_rate,
        trajectory_failure_sample_count=trajectory_failure_total,
        state_change_rate=weighted(local.state_change_rate, historical.state_change_rate),
        no_diff_rate=weighted(local.no_diff_rate, historical.no_diff_rate),
        no_diff_lower_bound=max(local.no_diff_lower_bound, historical.no_diff_lower_bound),
        avg_token_cost=weighted(local.avg_token_cost, historical.avg_token_cost),
        last_result_mode=local.last_result_mode or historical.last_result_mode,
        last_verifier_delta=local.last_verifier_delta or historical.last_verifier_delta,
        source="local+history",
    )


def wilson_lower_bound(successes: int, total: int, *, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    phat = float(successes) / float(total)
    denom = 1.0 + (z * z / float(total))
    centre = phat + (z * z / (2.0 * float(total)))
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * float(total))) / float(total))
    return max(0.0, (centre - margin) / denom)


def fresh_source_files_from_latest_guard(ledger: RecoveryLedger) -> list[str]:
    guard = dict(ledger.last_action_result.get("semantic_guard", {}) or {})
    if not guard:
        guards = [
            dict(item)
            for item in list(ledger.metadata.get("guard_history", []) or [])
            if isinstance(item, dict)
        ]
        guard = guards[-1] if guards else {}
    classes = dict(guard.get("fresh_changed_file_classes", {}) or {})
    return [
        str(item)
        for item in list(classes.get("source_files", []) or [])
        if str(item).strip()
    ]


def verifier_delta_from_result_mode(result_mode: str) -> str:
    mode = str(result_mode or "")
    if mode == "strong_source_success":
        return "improved"
    if mode in {
        "oracle_failed_after_source_edit",
        "contract_violation_after_source_edit",
        "intent_violation_missed_target",
        "intent_violation_revoked_target",
        "intent_violation_too_broad",
        "intent_violation_after_source_edit",
    }:
        return "same"
    if mode in {"no_diff", "contract_violation_no_fresh_source", "intent_violation_no_fresh_source", "wrong_edit_target"}:
        return "same"
    return "unknown"


def _filter_episodes(
    episodes: list[RecoveryEpisode],
    *,
    counterexample_type: str,
    object_chain_hash: str,
    object_chain_signature: dict[str, int] | None,
    min_object_chain_similarity: float,
    exclude_instance_hash: str | None,
    method_version: str,
) -> list[RecoveryEpisode]:
    selected: list[RecoveryEpisode] = []
    for episode in episodes:
        if not episode.provider_clean or not episode.oracle_clean or not episode.infra_clean:
            continue
        if method_version and episode.method_version != method_version:
            continue
        if exclude_instance_hash and episode.instance_id_hash == exclude_instance_hash:
            continue
        if counterexample_type and episode.counterexample_type != counterexample_type:
            continue
        if object_chain_hash and episode.object_chain_hash != object_chain_hash:
            continue
        if object_chain_signature:
            similarity = object_chain_similarity(object_chain_signature, episode.object_chain_signature)
            if similarity < float(min_object_chain_similarity):
                continue
        selected.append(episode)
    return selected


def _prior_for_action(action: str, episodes: list[RecoveryEpisode], *, source: str) -> ActionPrior:
    if not episodes:
        return ActionPrior(action=action, source=source)
    cell_known = [episode for episode in episodes if episode.cell_outcome_known]
    trajectory_successes = sum(1 for episode in cell_known if episode.cell_succeeded)
    trajectory_failures = len(cell_known) - trajectory_successes
    action_successes = sum(1 for episode in episodes if episode.action_success_credit)
    state_changed = sum(1 for episode in episodes if episode.action_changed_state or episode.state_changed)
    no_diff = sum(1 for episode in episodes if episode.no_diff_after)
    costs = [float(episode.token_cost) for episode in episodes if float(episode.token_cost) > 0.0]
    total = len(episodes)
    return ActionPrior(
        action=action,
        sample_count=total,
        action_success_sample_count=action_successes,
        cell_success_sample_count=len(cell_known),
        success_rate=float(action_successes) / float(total),
        trajectory_success_rate=(
            float(trajectory_successes) / float(len(cell_known))
            if cell_known
            else 0.0
        ),
        trajectory_success_sample_count=trajectory_successes,
        trajectory_failure_rate=(
            float(trajectory_failures) / float(len(cell_known))
            if cell_known
            else 0.0
        ),
        trajectory_failure_sample_count=trajectory_failures,
        state_change_rate=float(state_changed) / float(total),
        no_diff_rate=float(no_diff) / float(total),
        no_diff_lower_bound=wilson_lower_bound(no_diff, total),
        avg_token_cost=sum(costs) / float(len(costs)) if costs else 0.0,
        last_result_mode=episodes[-1].result_mode or episodes[-1].stop_reason,
        last_verifier_delta=episodes[-1].verifier_delta,
        source=source,
    )


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _canonical_object_type(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    chars: list[str] = []
    for index, char in enumerate(text):
        if char in {"-", " ", "."}:
            chars.append("_")
        elif char.isupper() and index > 0 and text[index - 1].islower():
            chars.append("_")
            chars.append(char.lower())
        else:
            chars.append(char.lower())
    return "".join(chars).strip("_")

"""Stage-boundary invalidation helpers for MAS recovery events."""

from __future__ import annotations

from typing import Any, Iterable

from bcmr_swe.recovery.structured_state import build_structured_recovery_state_from_failed_state
from bcmr_swe.types import FailedState, OpType, PropagationObject, RecoveryProgram


def _dedupe(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _object_from_any(item: Any) -> PropagationObject | None:
    if isinstance(item, PropagationObject):
        return item
    if isinstance(item, dict):
        try:
            return PropagationObject.from_dict(item)
        except Exception:
            return None
    return None


def _object_chain_from_structured_state(payload: dict[str, Any]) -> list[PropagationObject]:
    if not isinstance(payload, dict):
        return []
    values = payload.get("object_chain_view") or []
    if isinstance(values, dict):
        values = values.get("objects") or []
    chain: list[PropagationObject] = []
    for item in list(values or []):
        obj = _object_from_any(item)
        if obj is not None:
            chain.append(obj)
    return chain


def object_chain_from_failed_state(failed_state: FailedState | None) -> list[PropagationObject]:
    """Best-effort extraction of MAS propagation objects from a failed state."""

    if failed_state is None:
        return []
    metadata = dict(getattr(failed_state, "metadata", {}) or {})
    raw_chain = metadata.get("mas_object_chain") or metadata.get("object_chain_view") or []
    chain: list[PropagationObject] = []
    for item in list(raw_chain or []):
        obj = _object_from_any(item)
        if obj is not None:
            chain.append(obj)
    if chain:
        return chain
    try:
        return list(build_structured_recovery_state_from_failed_state(failed_state).object_chain_view)
    except Exception:
        return []


def object_chain_from_case_context(*, ctx: Any = None, row: dict[str, Any] | None = None, case: Any = None) -> list[PropagationObject]:
    """Extract propagation objects available to live drivers.

    Live replay paths do not always instantiate ``FailedState`` objects. This
    helper reads the same object-chain surface from the case provenance, row
    metadata, or ctx metadata when present.
    """

    candidates: list[Any] = []
    if row:
        candidates.extend(
            [
                row.get("structured_recovery_state"),
                dict(row.get("failed_state_metadata", {}) or {}).get("structured_recovery_state"),
                dict(row.get("failed_state_metadata", {}) or {}).get("mas_object_chain"),
                row.get("mas_object_chain"),
            ]
        )
    if case is not None:
        provenance = dict(getattr(case, "provenance_context", {}) or {})
        candidates.extend(
            [
                provenance.get("structured_recovery_state"),
                dict(provenance.get("failed_state_metadata", {}) or {}).get("structured_recovery_state"),
                dict(provenance.get("failed_state_metadata", {}) or {}).get("mas_object_chain"),
                provenance.get("mas_object_chain"),
            ]
        )
    if ctx is not None:
        for attr in ("structured_recovery_state", "failed_state_metadata", "mas_object_chain"):
            if hasattr(ctx, attr):
                candidates.append(getattr(ctx, attr))
        manifest = getattr(ctx, "manifest", None)
        if isinstance(manifest, dict):
            candidates.extend(
                [
                    manifest.get("structured_recovery_state"),
                    manifest.get("mas_object_chain"),
                    dict(manifest.get("failed_state_metadata", {}) or {}).get("structured_recovery_state"),
                    dict(manifest.get("failed_state_metadata", {}) or {}).get("mas_object_chain"),
                ]
            )

    chain: list[PropagationObject] = []
    for candidate in candidates:
        if isinstance(candidate, dict) and "object_chain_view" in candidate:
            chain.extend(_object_chain_from_structured_state(candidate))
            continue
        if isinstance(candidate, list):
            for item in candidate:
                obj = _object_from_any(item)
                if obj is not None:
                    chain.append(obj)
    deduped: list[PropagationObject] = []
    seen: set[str] = set()
    for obj in chain:
        key = obj.object_id or f"{obj.object_type}:{obj.producer_stage}:{obj.consumer_stage}:{obj.evidence_anchor}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(obj)
    return deduped


def _matches_target(obj: PropagationObject, target: str) -> bool:
    value = str(target or "").strip()
    if not value:
        return False
    payload = dict(obj.payload or {})
    return value in {
        obj.object_id,
        obj.object_type,
        str(payload.get("node_id", "") or ""),
        str(payload.get("fact_key", "") or ""),
        str(payload.get("fact_value_excerpt", "") or ""),
        str(payload.get("path", "") or ""),
    }


def invalidated_objects_for_targets(
    object_chain: list[PropagationObject],
    *,
    invalidated_targets: list[str] | None = None,
    invalidated_object_ids: list[str] | None = None,
    active_object_id: str = "",
    include_all: bool = False,
) -> list[PropagationObject]:
    """Resolve invalidated target strings to typed propagation objects."""

    if not object_chain:
        return []
    object_ids = _dedupe(list(invalidated_object_ids or []) + [active_object_id])
    targets = _dedupe(list(invalidated_targets or []))
    if include_all or "all" in targets:
        return list(object_chain)

    selected: list[PropagationObject] = []
    for obj in object_chain:
        if any(_matches_target(obj, target) for target in object_ids + targets):
            selected.append(obj)
            continue
        payload = dict(obj.payload or {})
        if "latest_patch" in targets and (
            obj.object_type == "SharedFact"
            or str(payload.get("fact_key", "") or "") == "latest_patch"
        ):
            selected.append(obj)
            continue
        if "localized_path" in targets and obj.object_type == "Selection":
            selected.append(obj)
            continue
        if "current_target" in targets and obj.object_type == "Selection":
            selected.append(obj)
            continue
    return selected


def invalidation_delta_for_objects(objects: list[PropagationObject]) -> dict[str, Any]:
    """Return event-log ready invalidated object ids and stage pairs."""

    ids: list[str] = []
    stages: list[tuple[str, str]] = []
    for obj in objects:
        object_id = str(obj.object_id or "").strip()
        producer = str(obj.producer_stage or "").strip()
        consumer = str(obj.consumer_stage or "").strip()
        if not object_id or not producer or not consumer:
            continue
        if object_id in ids:
            continue
        ids.append(object_id)
        stages.append((producer, consumer))
    return {
        "invalidated_object_ids": ids,
        "invalidated_object_stages": stages,
    }


def invalidation_delta_for_targets(
    object_chain: list[PropagationObject],
    *,
    invalidated_targets: list[str] | None = None,
    invalidated_object_ids: list[str] | None = None,
    active_object_id: str = "",
    include_all: bool = False,
) -> dict[str, Any]:
    return invalidation_delta_for_objects(
        invalidated_objects_for_targets(
            object_chain,
            invalidated_targets=invalidated_targets,
            invalidated_object_ids=invalidated_object_ids,
            active_object_id=active_object_id,
            include_all=include_all,
        )
    )


def invalidation_delta_for_program(
    program: RecoveryProgram,
    object_chain: list[PropagationObject],
) -> dict[str, Any]:
    """Infer the propagation-object invalidations represented by a program."""

    targets: list[str] = []
    object_ids: list[str] = []
    include_all = False
    selective_roles: list[str] = []
    for step in program.steps:
        args = dict(step.args or {})
        if step.op == OpType.REVOKE:
            targets.extend([args.get("fact_id", ""), args.get("object_id", ""), args.get("target", "")])
            if args.get("fact_id") == "fact:latest_patch":
                targets.append("latest_patch")
        if step.op == OpType.ROLLBACK:
            anchor = str(args.get("anchor_label", "") or args.get("checkpoint_label", "") or "")
            if anchor == "initial":
                include_all = True
            elif anchor == "post_locate":
                targets.extend(["latest_patch", "current_target"])
            elif anchor == "post_patch":
                targets.append("latest_patch")
        if step.op == OpType.SELECTIVE_REPLAY:
            role = str(args.get("role", "") or "").strip().lower()
            if role:
                selective_roles.append(role)
        object_ids.append(str(args.get("object_id", "") or ""))
        invalidate = args.get("invalidate", []) or []
        if isinstance(invalidate, str):
            targets.append(invalidate)
        else:
            targets.extend(str(item) for item in invalidate)

    # SELECTIVE_REPLAY invalidates the typed objects produced by the
    # named role — i.e., every propagation edge whose producer_stage is
    # that role. The consumer stages of those edges are the boundaries
    # the replay will re-cross, so the deriviation is symmetric with the
    # ROLLBACK/REVOKE logic but keyed on producer_stage rather than on
    # object content.
    if selective_roles and object_chain:
        for obj in object_chain:
            if str(obj.producer_stage or "").strip().lower() in selective_roles:
                object_ids.append(obj.object_id)

    return invalidation_delta_for_targets(
        object_chain,
        invalidated_targets=targets,
        invalidated_object_ids=object_ids,
        include_all=include_all,
    )

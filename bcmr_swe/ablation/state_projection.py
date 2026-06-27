"""State-projection ablation config.

`StateProjectionConfig` gates each view of a `StructuredRecoveryState` and
controls the render format. Callers pass a config into `project(state)` and
never touch the state-building code directly.

The config is frozen — leave-one-out ablations are expressed as
`dataclasses.replace(CFG_FULL, include_object_chain_view=False)`.

Design rules:

- The config is the only input that influences projection output. No globals.
- Every projected payload carries `_cfg_id` at the top level so event logs
  and benchmark rows can be joined back to the cfg without string parsing.
- Render formats are content-preserving when `token_budget` is large enough;
  this is what makes the C1 content-controlled structure ablation valid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from bcmr_swe.types import PropagationObject, StructuredRecoveryState


RenderFormat = Literal["json", "bullets", "prose"]

_ALL_OBJECT_TYPES: frozenset[str] = frozenset({"Selection", "SharedFact", "VerifierVerdict"})


@dataclass(frozen=True, slots=True)
class StateProjectionConfig:
    """Composable gate set for building the LLM-facing recovery state view."""

    cfg_id: str
    include_local_region_view: bool = True
    include_role_aggregate_view: bool = True
    include_object_chain_view: bool = True
    include_replay_anchor_view: bool = True
    include_evidence_pack: bool = True
    include_negative_constraints: bool = True
    include_state_numeric: bool = False
    object_chain_types: frozenset[str] = _ALL_OBJECT_TYPES
    object_chain_max: int = 16
    render_format: RenderFormat = "json"
    token_budget: int = 1500
    role_aggregate_excerpt_chars: int = 600

    def __post_init__(self) -> None:
        assert self.cfg_id and isinstance(self.cfg_id, str), "cfg_id required"
        assert self.object_chain_max >= 0, "object_chain_max must be non-negative"
        assert self.render_format in ("json", "bullets", "prose"), "unknown render_format"
        invalid = self.object_chain_types - _ALL_OBJECT_TYPES
        assert not invalid, f"unknown object_chain_types: {sorted(invalid)}"

    def project(self, state: StructuredRecoveryState) -> dict[str, Any]:
        """Return the LLM-facing payload selected by this config.

        The returned dict always contains a top-level `_cfg_id`, so the event
        log and benchmark row can tag the projection without string-matching.
        """

        assert isinstance(state, StructuredRecoveryState), "project requires StructuredRecoveryState"

        local = _filter_local_region(
            dict(state.local_region_view or {}),
            include_negative_constraints=self.include_negative_constraints,
        )
        evidence = _filter_evidence_pack(
            dict(state.evidence_pack or {}),
            include_negative_constraints=self.include_negative_constraints,
        )
        role_view = _filter_role_aggregate(
            {
                str(k): dict(v)
                for k, v in (state.role_aggregate_view or {}).items()
            },
            excerpt_chars=self.role_aggregate_excerpt_chars,
        )
        objects = _filter_object_chain(
            state.object_chain_view or [],
            keep_types=self.object_chain_types,
            max_items=self.object_chain_max,
        )

        payload: dict[str, Any] = {"_cfg_id": self.cfg_id}
        if self.include_local_region_view:
            payload["local_region_view"] = local
        if self.include_role_aggregate_view:
            payload["role_aggregate_view"] = role_view
        if self.include_object_chain_view:
            payload["object_chain_view"] = [obj.to_dict() for obj in objects]
        if self.include_replay_anchor_view:
            payload["replay_anchor_view"] = dict(state.replay_anchor_view or {})
        if self.include_evidence_pack:
            payload["evidence_pack"] = evidence
        if self.include_state_numeric:
            payload["state_numeric"] = {
                str(k): float(v) for k, v in dict(state.state_numeric or {}).items()
            }

        rendered = _render_payload(payload, fmt=self.render_format)
        return {
            "_cfg_id": self.cfg_id,
            "_render_format": self.render_format,
            "_render_text": rendered,
            "structured_fields": payload,
        }

    def summary_for_event_log(self) -> dict[str, Any]:
        return {
            "cfg_id": self.cfg_id,
            "include_local_region_view": self.include_local_region_view,
            "include_role_aggregate_view": self.include_role_aggregate_view,
            "include_object_chain_view": self.include_object_chain_view,
            "include_replay_anchor_view": self.include_replay_anchor_view,
            "include_evidence_pack": self.include_evidence_pack,
            "include_negative_constraints": self.include_negative_constraints,
            "object_chain_types": sorted(self.object_chain_types),
            "object_chain_max": self.object_chain_max,
            "render_format": self.render_format,
            "token_budget": self.token_budget,
        }


def _filter_local_region(
    local: dict[str, Any], *, include_negative_constraints: bool
) -> dict[str, Any]:
    result = dict(local)
    if not include_negative_constraints:
        result.pop("negative_constraints", None)
    return result


def _filter_evidence_pack(
    evidence: dict[str, Any], *, include_negative_constraints: bool
) -> dict[str, Any]:
    result = dict(evidence)
    if not include_negative_constraints:
        result.pop("negative_constraints", None)
    return result


def _filter_role_aggregate(
    role_view: dict[str, dict[str, Any]], *, excerpt_chars: int
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for role, payload in role_view.items():
        clean = dict(payload)
        for key in ("located_files_excerpt", "plan_excerpt", "patch_excerpt", "verification_excerpt"):
            if key in clean:
                clean[key] = str(clean[key])[:excerpt_chars]
        out[role] = clean
    return out


def _filter_object_chain(
    objects: list[PropagationObject],
    *,
    keep_types: frozenset[str],
    max_items: int,
) -> list[PropagationObject]:
    kept: list[PropagationObject] = []
    for obj in objects:
        if obj.object_type not in keep_types:
            continue
        kept.append(obj)
        if len(kept) >= max_items:
            break
    return kept


def _render_payload(payload: dict[str, Any], *, fmt: RenderFormat) -> str:
    if fmt == "json":
        return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    if fmt == "bullets":
        return _render_bullets(payload)
    if fmt == "prose":
        return _render_prose(payload)
    raise ValueError(f"unknown render_format: {fmt}")


def _render_bullets(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    for section, value in payload.items():
        if section.startswith("_"):
            continue
        lines.append(f"- {section}:")
        lines.extend(_bullet_lines(value, indent=2))
    return "\n".join(lines)


def _bullet_lines(value: Any, *, indent: int) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        out: list[str] = []
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                out.append(f"{prefix}- {k}:")
                out.extend(_bullet_lines(v, indent=indent + 2))
            else:
                out.append(f"{prefix}- {k}: {_scalar(v)}")
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, (dict, list)):
                out.append(f"{prefix}- item:")
                out.extend(_bullet_lines(item, indent=indent + 2))
            else:
                out.append(f"{prefix}- {_scalar(item)}")
        return out
    return [f"{prefix}- {_scalar(value)}"]


def _render_prose(payload: dict[str, Any]) -> str:
    """Natural-language rendering that preserves content but removes all
    structural markup. Used for content-controlled structure ablations."""
    sentences: list[str] = []
    for section, value in payload.items():
        if section.startswith("_"):
            continue
        pretty = section.replace("_", " ")
        sentences.append(f"{pretty.capitalize()}: {_prose_value(value)}")
    return "\n\n".join(sentences)


def _prose_value(value: Any) -> str:
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            parts.append(f"{k.replace('_', ' ')} is {_prose_value(v)}")
        return "; ".join(parts) + "."
    if isinstance(value, list):
        if not value:
            return "none"
        return ", ".join(_prose_value(v) for v in value)
    return _scalar(value)


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "none"
    return str(value)

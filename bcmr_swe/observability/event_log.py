"""JSONL event logger for BCMR recovery runs.

Every significant recovery action emits one event. Events are self-contained
and append-only; the stream can be reconstructed into a relational table by
`duckdb.read_json_auto(...)` or `pandas.read_json(..., lines=True)`.

Design invariants:

- One file per (run_id, case_id). Append-only. No in-place edits.
- Every event carries the schema version; old events are never rewritten.
- `state_delta` is sufficient to reconstruct cumulative state; callers must
  not rely on external files to interpret an event.
- Every event carries the ablation `cfg_id`s, so post-hoc joins can filter
  cells without re-running experiments.

This module deliberately has no external dependencies beyond the standard
library, so it runs in any subprocess / harness environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


EVENT_SCHEMA_VERSION = "bcmr.event.v1"


_REQUIRED_TOP_FIELDS: tuple[str, ...] = (
    "schema_version",
    "ts_iso",
    "ts_epoch_ms",
    "run_id",
    "case_id",
    "method",
    "step_id",
    "event_type",
)

_VALID_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "run_started",
        "case_started",
        "checkpoint_restored",
        "state_projected",
        "action_selected",
        "action_executed",
        "verifier_ran",
        "budget_accounted",
        "invariant_violated",
        "episode_started",
        "episode_finished",
        "reflection_written",
        "case_finished",
        "run_finished",
    }
)


class EventSchemaError(ValueError):
    """Raised when an emitted event does not satisfy the minimum schema."""


def state_digest(payload: Any) -> str:
    """Stable content digest for any JSON-serializable state snapshot."""
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


@dataclass(slots=True)
class _EventLoggerCounters:
    step_id: int = 0
    events_written: int = 0


@dataclass(slots=True)
class RunEventLogger:
    """Context-manager JSONL event emitter.

    Usage:

        with RunEventLogger(out_dir=..., run_id=..., case_id=...,
                            method="bounded_program",
                            state_projection_cfg_id="FULL",
                            action_space_cfg_id="FULL") as log:
            log.emit("case_started", {...})
            step = log.next_step()
            log.emit("action_executed", {...}, parent_step_id=step-1,
                     state_delta={...}, budget={...})
    """

    out_dir: Path
    run_id: str
    case_id: str
    method: str
    state_projection_cfg_id: str
    action_space_cfg_id: str
    extra_ablation: dict[str, Any] = field(default_factory=dict)
    _file_path: Path = field(init=False)
    _fp: Any = field(default=None, init=False)
    _counters: _EventLoggerCounters = field(default_factory=_EventLoggerCounters, init=False)
    _closed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        assert self.run_id, "RunEventLogger requires a non-empty run_id"
        assert self.case_id, "RunEventLogger requires a non-empty case_id"
        assert self.method, "RunEventLogger requires a non-empty method"
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        safe_case = _safe_component(self.case_id)
        safe_method = _safe_component(self.method)
        self._file_path = self.out_dir / f"{self.run_id}__{safe_case}__{safe_method}.jsonl"

    @property
    def file_path(self) -> Path:
        return self._file_path

    @property
    def events_written(self) -> int:
        return self._counters.events_written

    def next_step(self) -> int:
        self._counters.step_id += 1
        return self._counters.step_id

    def __enter__(self) -> "RunEventLogger":
        self._fp = open(self._file_path, "a", encoding="utf-8")
        self.emit(
            "run_started",
            payload={"wall_epoch_ms": _now_ms()},
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.emit(
                "run_finished",
                payload={
                    "wall_epoch_ms": _now_ms(),
                    "events_written": self._counters.events_written,
                    "error": None if exc is None else f"{exc_type.__name__}: {exc}",
                },
            )
        finally:
            if self._fp is not None:
                self._fp.flush()
                os.fsync(self._fp.fileno())
                self._fp.close()
                self._fp = None
            self._closed = True

    def emit(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        parent_step_id: int | None = None,
        state_delta: dict[str, Any] | None = None,
        budget: dict[str, Any] | None = None,
        verifier: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
        pre_state: Any = None,
        post_state: Any = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise EventSchemaError("RunEventLogger is already closed")
        if event_type not in _VALID_EVENT_TYPES:
            raise EventSchemaError(f"unknown event_type: {event_type!r}")

        step_id = self._counters.step_id
        event: dict[str, Any] = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "ts_iso": _now_iso(),
            "ts_epoch_ms": _now_ms(),
            "run_id": self.run_id,
            "case_id": self.case_id,
            "method": self.method,
            "step_id": step_id,
            "parent_step_id": parent_step_id,
            "event_type": event_type,
            "payload": dict(payload or {}),
            "pre_state_digest": state_digest(pre_state) if pre_state is not None else None,
            "post_state_digest": state_digest(post_state) if post_state is not None else None,
            "state_delta": dict(state_delta) if state_delta else None,
            "budget": dict(budget) if budget else None,
            "verifier": dict(verifier) if verifier else None,
            "provenance": dict(provenance) if provenance else None,
            "ablation": {
                "state_projection_cfg_id": self.state_projection_cfg_id,
                "action_space_cfg_id": self.action_space_cfg_id,
                **dict(self.extra_ablation),
            },
        }

        _validate_event(event)
        line = json.dumps(event, ensure_ascii=False, default=str)
        assert self._fp is not None, "event logger file pointer missing"
        self._fp.write(line)
        self._fp.write("\n")
        self._fp.flush()
        self._counters.events_written += 1
        return event


def _validate_event(event: dict[str, Any]) -> None:
    for field_name in _REQUIRED_TOP_FIELDS:
        if field_name not in event:
            raise EventSchemaError(f"missing required field: {field_name}")
    if not isinstance(event["step_id"], int):
        raise EventSchemaError("step_id must be int")
    if event["event_type"] not in _VALID_EVENT_TYPES:
        raise EventSchemaError(f"invalid event_type: {event['event_type']!r}")
    ablation = event.get("ablation")
    if not isinstance(ablation, dict):
        raise EventSchemaError("ablation block is required and must be a dict")
    for key in ("state_projection_cfg_id", "action_space_cfg_id"):
        if key not in ablation or not isinstance(ablation[key], str):
            raise EventSchemaError(f"ablation.{key} must be a string cfg id")


def read_event_log(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield events from a JSONL stream. Raises EventSchemaError on malformed lines."""
    p = Path(path)
    with open(p, "r", encoding="utf-8") as fp:
        for lineno, raw in enumerate(fp, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise EventSchemaError(f"{p}:{lineno}: invalid JSON: {exc}") from exc
            _validate_event(event)
            yield event


def _safe_component(text: str, *, limit: int = 80) -> str:
    safe = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in text)
    return safe[:limit] or "anon"


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _now_ms() -> int:
    return int(time.time() * 1000.0)

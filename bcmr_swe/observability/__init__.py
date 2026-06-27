"""BCMR observability subsystem.

Emits structured, DuckDB-queryable JSONL event streams for every recovery
run. See `bcmr_swe/observability/event_log.py`.
"""

from __future__ import annotations

from bcmr_swe.observability.event_log import (
    EVENT_SCHEMA_VERSION,
    EventSchemaError,
    RunEventLogger,
    read_event_log,
    state_digest,
)
from bcmr_swe.observability.invariants import (
    Invariant,
    InvariantViolation,
    assert_budget_monotonic,
    assert_checkpoint_restored,
    assert_manifest_lock,
    assert_recoverable_denominator_present,
    assert_same_checkpoint_across_methods,
    assert_workspace_reset,
)

__all__ = [
    "EVENT_SCHEMA_VERSION",
    "EventSchemaError",
    "Invariant",
    "InvariantViolation",
    "RunEventLogger",
    "assert_budget_monotonic",
    "assert_checkpoint_restored",
    "assert_manifest_lock",
    "assert_recoverable_denominator_present",
    "assert_same_checkpoint_across_methods",
    "assert_workspace_reset",
    "read_event_log",
    "state_digest",
]

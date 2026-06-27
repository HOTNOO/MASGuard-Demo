"""感知层数据模型"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any


class FaultType(str, enum.Enum):
    HEARTBEAT_LOSS = "heartbeat_loss"
    TOKEN_EXHAUSTED = "token_exhausted"
    NETWORK_ERROR = "network_error"
    HALLUCINATION = "hallucination"
    LOOP = "loop"
    MISALIGNMENT = "misalignment"
    EXTERNAL = "external"
    INJECTED = "injected"


@dataclass
class FaultSignal:
    fault_type: FaultType
    severity: str  # info / warn / error
    agent_id: str | None = None
    message: str | None = None
    suggested_action: str | None = None
    extra: dict[str, Any] | None = None
    node_ids: list[str] | None = None

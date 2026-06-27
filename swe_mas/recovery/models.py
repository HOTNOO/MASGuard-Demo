"""恢复层数据模型与策略"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any
from swe_mas.perception.models import FaultType

class ResourceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Granularity(str, Enum):
    CONCISE = "concise"
    STANDARD = "standard"
    VERBOSE = "verbose"

@dataclass
class ResourceCondition:
    bandwidth: ResourceLevel = ResourceLevel.MEDIUM
    token_left: int | None = None

@dataclass
class RecoveryStrategy:
    granularity: Granularity
    focus_areas: list[str]
    discard_recent_branch: bool = False
    reuse_healthy_intent: bool = True
    max_history: int | None = None

@dataclass
class RecoveryRequest:
    session_id: str
    target_role: str
    fault_type: FaultType
    resource: ResourceCondition
    details: dict[str, Any] | None = None
